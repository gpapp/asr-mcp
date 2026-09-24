import asyncio
import logging
import time
from typing import Optional

import numpy as np
import torch

from asr_mcp.core.model_state import state, LRUCache, GPU_SHRINK_RUN_OPTIONS, run_embedding
from asr_mcp.diarization.clustering import (
    cap_clusters, greedy_merge_clusters, merge_similar_speakers, match_known_speakers_full,
)
from asr_mcp.diarization.segment_ops import (
    collapse_same_speaker_segments, absorb_islands, absorb_minority_speakers,
    eliminate_ghost_speakers,
)
from asr_mcp.speaker.audio import refine_speaker_boundaries
from asr_mcp.speaker.embedding import extract_embedding
from asr_mcp.speaker.vad import split_at_energy_dips, run_vad_chunked, run_vad_onnx, merge_vad_sections
from asr_mcp.speaker.profiling import profile_speakers, relabel_by_pitch

logger = logging.getLogger("asr_mcp.diarization.pipeline")

_embedding_cache = LRUCache(max_size=5000)


class Diarizer:
    def __init__(self, model_state=None, settings=None):
        self._state = model_state or state
        self._settings = settings
        self._cfg = None
        self._raw_vad_sections: list[dict] = []  # uncollapsed VAD, in seconds

    def _load_cfg(self):
        if self._cfg is None:
            from asr_mcp.config import get_config
            self._cfg = get_config()
        return self._cfg

    async def run(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        diarization_threshold: Optional[float] = None,
        vad_threshold: Optional[float] = None,
        known_speakers: Optional[dict] = None,
        progress_callback=None,
    ) -> dict:
        cfg = self._load_cfg()
        threshold = diarization_threshold or cfg.get("diarization", {}).get("default_threshold", 0.35)
        v_threshold = vad_threshold or cfg.get("vad", {}).get("default_threshold", 0.5)
        min_speech_ms = cfg.get("vad", {}).get("min_speech_duration_ms", 250)

        start_time = time.time()

        if not self._state.is_ready:
            return {"error": "Models not loaded. CUDA GPU required.", "segments": [], "total_time_sec": 0}

        # Load audio
        waveform, sample_rate = self._load_audio(audio_path)
        if waveform is None:
            return {"error": "Failed to load audio"}

        waveform_np = waveform.numpy().squeeze()
        audio_duration = waveform.shape[-1] / sample_rate

        if progress_callback:
            await progress_callback({"stage": "Detecting speech regions", "progress": 0.1})

        # Step 1: VAD — preserve raw uncollapsed sections for later boundary refinement
        self._raw_vad_sections = self._run_vad(
            waveform, sample_rate, v_threshold, min_speech_ms, merge_close=False
        )
        if not self._raw_vad_sections:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        # Step 1b: Merge nearby speech regions separated by <0.1s silence for clustering
        speech_ts = merge_vad_sections(self._raw_vad_sections, max_gap_sec=0.1)

        if progress_callback:
            await progress_callback({"stage": "Splitting audio at silence boundaries", "progress": 0.15})

        # Step 2: Energy-dip splitting (split at genuine pauses >1s)
        speech_ts = split_at_energy_dips(
            speech_ts, waveform_np, sample_rate,
            min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
            min_split_piece=2.0,
        )

        # Step 3: Extract one embedding per segment (seconds-based timestamps)
        all_segments = []
        all_embeddings = []
        total_segs = len(speech_ts)
        for seg_idx, ts in enumerate(speech_ts):
            if progress_callback:
                p = 0.15 + 0.45 * ((seg_idx + 1) / max(total_segs, 1))
                await progress_callback({"stage": f"Extracting embeddings ({seg_idx+1}/{total_segs})", "progress": p})
            start_sec = float(ts["start"])
            end_sec = float(ts["end"])
            dur_sec = end_sec - start_sec
            if dur_sec < 0.3:
                continue
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            segment_audio = waveform[..., start_sample:end_sample]
            all_segments.append({
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "duration": round(dur_sec, 3),
            })
            emb = extract_embedding(
                segment_audio, sample_rate, self._state.embedding_session,
            )
            all_embeddings.append(emb)

        if not all_embeddings:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        raw_embeddings = np.array(all_embeddings, dtype=np.float32)

        if progress_callback:
            await progress_callback({"stage": "Clustering speakers", "progress": 0.6})

        # Step 5: Clustering
        from sklearn.cluster import AgglomerativeClustering
        if num_speakers:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, metric="cosine", linkage="average"
            )
        else:
            distance_threshold = cfg.get("diarization", {}).get("distance_threshold", 0.55)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold,
                metric="cosine", linkage="average",
            )
        long_labels = clustering.fit_predict(raw_embeddings)

        # Step 5b: Cap clusters if too many
        max_clusters = cfg.get("diarization", {}).get("max_clusters", 15)
        long_labels = cap_clusters(raw_embeddings, long_labels, max_clusters=max_clusters)

        # Step 6: Greedy merge
        merge_thresh = cfg.get("diarization", {}).get("merge_threshold", 0.45)
        long_labels, cluster_centroids = greedy_merge_clusters(raw_embeddings, long_labels, merge_thresh)

        if progress_callback:
            await progress_callback({"stage": "Building speaker segments", "progress": 0.7})

        # Step 7: Map labels to segments
        merged_segments = []
        for i, (seg, label) in enumerate(zip(all_segments, long_labels)):
            merged_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"Speaker {label + 1}",
                "index": i,
            })

        # Step 8: Collapse + absorb islands
        merged_segments = collapse_same_speaker_segments(merged_segments, max_gap=0.5)
        merged_segments = absorb_islands(merged_segments, min_island_dur=1.0)

        if progress_callback:
            await progress_callback({"stage": "Profiling speakers", "progress": 0.8})

        # Step 9: Speaker profiling
        profiles = profile_speakers(waveform, merged_segments, sample_rate)

        # Step 9b: Merge similar speakers (compare voiceprints, merge close ones)
        merged_segments, cluster_centroids, profiles = merge_similar_speakers(
            merged_segments, raw_embeddings, long_labels,
            cluster_centroids, profiles, cfg,
        )

        # Step 9c: Relabel by pitch (highest pitch = Speaker 1)
        merged_segments, profiles, label_map = relabel_by_pitch(merged_segments, profiles)

        # Renumber cluster_centroids to match relabeled segments
        old_to_new = {}
        for old_name, new_name in label_map.items():
            if old_name.startswith("Speaker ") and new_name.startswith("Speaker "):
                try:
                    old_num = int(old_name.split()[-1]) - 1
                    new_num_val = int(new_name.split()[-1]) - 1
                    old_to_new[old_num] = new_num_val
                except (ValueError, IndexError):
                    pass
        renumbered_centroids = {}
        for old_num, centroid in cluster_centroids.items():
            if old_num in old_to_new:
                renumbered_centroids[old_to_new[old_num]] = centroid
        cluster_centroids = renumbered_centroids

        # Build string-keyed centroids for boundary refinement and known-speaker matching
        string_centroids = {}
        for num, centroid in cluster_centroids.items():
            string_centroids[f"Speaker {num + 1}"] = centroid

        # Step 10: Boundary refinement (uses string-keyed centroids)
        merged_segments = refine_speaker_boundaries(
            merged_segments, waveform, self._state.embedding_session,
            string_centroids, sample_rate,
            embedding_cache=_embedding_cache,
        )

        # Step 11: Known speaker matching (BEFORE ghost elimination so alternatives are populated)
        if known_speakers:
            merged_segments, profiles = match_known_speakers_full(
                merged_segments, all_segments, list(range(len(all_segments))),
                raw_embeddings, cluster_centroids, profiles, known_speakers, cfg,
                renumber=True,
            )
        else:
            # No known speakers: return match_info stub so downstream still works
            pass

        # Step 12: Ghost elimination (uses seg["alternatives"] if populated by step 11)
        merged_segments = eliminate_ghost_speakers(merged_segments, profiles)

        # Step 13: Absorb minority speakers (safeguards matched known voiceprints)
        if known_speakers:
            known_names = set(known_speakers.keys())
        else:
            known_names = set()
        merged_segments = absorb_minority_speakers(
            merged_segments, max_utterance_sec=5.0, min_speaker_dur=8.0,
            protected_speakers=known_names,
        )

        # Step 14: Exact turn boundary refinement using raw VAD sections
        if self._raw_vad_sections:
            merged_segments = await self._refine_turn_boundaries_exact(
                merged_segments, waveform_np, sample_rate, profiles, known_speakers or {}
            )

        if progress_callback:
            await progress_callback({"stage": "diarization_finished", "progress": 1.0})

        total_time = time.time() - start_time

        return {
            "segments": merged_segments,
            "profiles": {k: {kk: vv for kk, vv in v.items() if kk != "mfcc"} for k, v in profiles.items()},
            "total_speakers": len(set(s.get("speaker") for s in merged_segments)),
            "total_time_sec": round(total_time, 2),
            "audio_duration_sec": round(audio_duration, 2),
        }

    def _load_audio(self, audio_path: str) -> tuple:
        try:
            from asr_mcp.voiceprint.utils import load_audio
            return load_audio(audio_path)
        except Exception as e:
            logger.error("Failed to load audio %s: %s", audio_path, e)
            return None, None

    def _run_vad(self, waveform, sample_rate, threshold, min_speech_ms, merge_close: bool = True):
        if self._state.vad_session is not None:
            return run_vad_onnx(
                waveform, self._state.vad_session,
                sample_rate=sample_rate,
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                merge_close=merge_close,
            )
        result = run_vad_chunked(
            waveform, sample_rate=sample_rate,
            threshold=threshold, min_speech_duration_ms=min_speech_ms,
        )
        # run_vad_chunked already returns seconds-based; no merge_close param needed
        return result

    def _get_speaker_ref(
        self,
        speaker_name: str,
        profiles: dict,
        known_speakers: dict,
        waveform_np: np.ndarray,
        sample_rate: int,
        segments: list,
    ) -> Optional[np.ndarray]:
        """Get a speaker reference embedding for boundary attribution.

        Priority: known voiceprint DB embedding > cluster centroid from profiles.
        Falls back to averaging up to 30s of that speaker's own segment audio.
        """
        # 1. Known speaker DB embedding
        clean_name = speaker_name.strip("[]")
        if clean_name in known_speakers:
            emb = known_speakers[clean_name].get("embedding")
            if emb is not None:
                arr = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                return arr / norm if norm > 1e-8 else arr

        # 2. Profile centroid embedding
        profile = profiles.get(speaker_name, {})
        centroid = profile.get("centroid")
        if centroid is not None:
            arr = np.array(centroid, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 1e-8 else arr

        # 3. Average embedding from up to 30s of own segments
        own_segs = [s for s in segments if s.get("speaker") == speaker_name]
        total_dur = 0.0
        audio_chunks = []
        for seg in own_segs:
            if total_dur >= 30.0:
                break
            s = int(float(seg["start"]) * sample_rate)
            e = int(float(seg["end"]) * sample_rate)
            chunk = waveform_np[s:e]
            if len(chunk) > 0:
                audio_chunks.append(chunk)
                total_dur += (e - s) / sample_rate

        if not audio_chunks:
            return None

        combined = np.concatenate(audio_chunks)
        tensor = torch.from_numpy(combined).unsqueeze(0)
        emb_session = self._state.embedding_session
        if emb_session is None:
            return None
        try:
            emb = extract_embedding(tensor, sample_rate, emb_session)
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 1e-8 else arr
        except Exception as exc:
            logger.warning("Failed to embed speaker ref for %s: %s", speaker_name, exc)
            return None

    async def _refine_turn_boundaries_exact(
        self,
        segments: list,
        waveform_np: np.ndarray,
        sample_rate: int,
        profiles: dict,
        known_speakers: dict,
    ) -> list:
        """Refine every speaker-change boundary using VAD-level granular speaker attribution.

        Processes ALL consecutive transitions where the speaker changes — both actual
        gaps (left.end < right.start) and contiguous boundaries (left.end == right.start).
        For contiguous boundaries the search region is expanded ±margin into both segments
        so raw VAD sections straddling the nominal cut point are embedded and attributed.

        Algorithm per transition:
          1. Determine search region: [gap_start - margin, gap_end + margin]
             (for contiguous boundaries gap_start == gap_end, so region is [cut - margin, cut + margin])
          2. Gather raw uncollapsed VAD sections overlapping the region (min 0.1s each)
          3. Batch-embed all sections across ALL transitions in one ONNX pass (with LRU cache)
          4. Attribute each section to left or right speaker via cosine dot product
          5. _best_split: find the split index k that minimises total cost
             (left sections right of k + right sections left of k, weighted by confidence margin)
          6. cut = start of first right-owned section past the nominal boundary,
             clamped to [left.start+eps, right.end-eps]
          7. left["end"] = cut; right["start"] = cut  (contiguous, no overlap/gap)

        Fallback chain: energy-dip centre → geometric midpoint.
        """
        if len(segments) < 2 or not self._raw_vad_sections:
            return segments

        refined = [dict(s) for s in segments]
        emb_session = self._state.embedding_session
        if emb_session is None:
            return refined

        margin = 0.5          # expand region into both segments around contiguous boundary
        min_sec_dur = 0.1     # minimum VAD section duration (seconds) to embed
        audio_dur = len(waveform_np) / sample_rate
        raw = sorted(self._raw_vad_sections, key=lambda s: s["start"])

        # --- Pre-compute speaker reference embeddings (cached per speaker name) ---
        ref_cache: dict[str, Optional[np.ndarray]] = {}

        def _ref(name: str) -> Optional[np.ndarray]:
            if name not in ref_cache:
                ref_cache[name] = self._get_speaker_ref(
                    name, profiles, known_speakers, waveform_np, sample_rate, refined
                )
            return ref_cache[name]

        # --- Collect transitions and their candidate VAD sections ---
        # transitions[i] = (left_idx, nominal_cut_start, nominal_cut_end, [vad_section, ...])
        transitions = []
        for i in range(len(refined) - 1):
            left = refined[i]
            right = refined[i + 1]
            if left.get("speaker") == right.get("speaker"):
                continue

            gap_start = float(left["end"])
            gap_end = float(right["start"])

            # Search region: expand into adjacent segments for contiguous boundaries
            region_start = max(0.0, gap_start - margin)
            region_end = min(audio_dur, gap_end + margin)

            secs = [
                s for s in raw
                if s["end"] > region_start and s["start"] < region_end
                and (s["end"] - s["start"]) >= min_sec_dur
            ]
            transitions.append((i, gap_start, gap_end, secs))

        if not transitions:
            return refined

        # --- Batch-embed all distinct VAD sections across all transitions ---
        # Use identity (start, end tuple) as cache key within this call.
        sec_key_to_emb: dict[tuple, Optional[np.ndarray]] = {}
        all_secs_to_embed: list[dict] = []
        seen_keys: set[tuple] = set()
        for _, _, _, secs in transitions:
            for s in secs:
                key = (s["start"], s["end"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_secs_to_embed.append(s)

        if all_secs_to_embed:
            import hashlib
            fbanks, valid_keys = [], []
            from asr_mcp.speaker.audio import extract_fbank
            for s in all_secs_to_embed:
                a = int(s["start"] * sample_rate)
                b = int(s["end"] * sample_rate)
                b = min(b, len(waveform_np))
                chunk = waveform_np[a:b]
                if len(chunk) < int(min_sec_dur * sample_rate):
                    sec_key_to_emb[(s["start"], s["end"])] = None
                    continue
                t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
                fb = extract_fbank(t, sample_rate)  # [T, 80]
                fb = fb - fb.mean(dim=0, keepdim=True)  # CMN along time dimension
                fbanks.append(fb)
                valid_keys.append((s["start"], s["end"]))

            if fbanks:
                hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in fbanks]
                cached_embs: dict[int, np.ndarray] = {}
                misses: list[int] = []
                for idx, h in enumerate(hashes):
                    hit = _embedding_cache.get(h)
                    if hit is not None:
                        cached_embs[idx] = hit
                    else:
                        misses.append(idx)

                if misses:
                    miss_fbs = [fbanks[idx] for idx in misses]
                    max_len = max(fb.shape[0] for fb in miss_fbs)
                    padded = []
                    for fb in miss_fbs:
                        if fb.shape[0] < max_len:
                            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[0]))
                        padded.append(fb)
                    batch = torch.stack(padded, dim=0).numpy().astype(np.float32)
                    input_name = emb_session.get_inputs()[0].name
                    output_names = [o.name for o in emb_session.get_outputs()]
                    try:
                        from asr_mcp.speaker.embedding import _run_with_cpu_fallback
                        out_list = _run_with_cpu_fallback(emb_session, {input_name: batch}, output_names)
                        out = out_list[0]
                    except Exception as e:
                        logger.warning("Boundary batch embed failed, skipping: %s", e)
                        out = None
                    if out is not None:
                        for li, idx in enumerate(misses):
                            e_vec = out[li].astype(np.float64)
                            nrm = np.linalg.norm(e_vec)
                            if nrm > 1e-8:
                                e_vec = e_vec / nrm
                            _embedding_cache.put(hashes[idx], e_vec)
                            cached_embs[idx] = e_vec

                for idx, key in enumerate(valid_keys):
                    sec_key_to_emb[key] = cached_embs.get(idx)

        # --- Apply cuts ---
        n_refined = 0
        for i, gap_start, gap_end, secs in transitions:
            left = refined[i]
            right = refined[i + 1]

            left_ref = _ref(left["speaker"])
            right_ref = _ref(right["speaker"])

            owners_int: list[int] = []   # 0=left, 1=right
            weights: list[float] = []
            valid_secs: list[dict] = []

            if left_ref is not None and right_ref is not None:
                for s in secs:
                    emb = sec_key_to_emb.get((s["start"], s["end"]))
                    if emb is None:
                        continue
                    sa = float(np.dot(emb, left_ref))
                    sb = float(np.dot(emb, right_ref))
                    owners_int.append(0 if sa >= sb else 1)
                    weights.append(abs(sa - sb) + 1e-3)
                    valid_secs.append(s)

            if valid_secs:
                # _best_split: minimise weighted cost (left sections right of k
                # that are owned by right, plus right sections left of k owned by left)
                n = len(owners_int)
                best_k, best_cost = 0, float("inf")
                for k in range(n + 1):
                    cost = 0.0
                    for j in range(k):
                        if owners_int[j] == 1:   # right section before cut → misclassified
                            cost += weights[j]
                    for j in range(k, n):
                        if owners_int[j] == 0:   # left section after cut → misclassified
                            cost += weights[j]
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k

                if best_k < n:
                    cut = float(valid_secs[best_k]["start"])
                else:
                    cut = gap_end  # all sections belong to left speaker

                # Clamp: must stay within [left.start+eps, right.end-eps]
                cut = max(float(left["start"]) + 0.001, min(cut, float(right["end"]) - 0.001))
                # Also respect original gap bounds for gap-type transitions
                if gap_end > gap_start:
                    cut = max(gap_start, min(cut, gap_end))
            else:
                # No embeddable sections — fallback chain
                if gap_end > gap_start:
                    cut = self._gap_energy_cut(waveform_np, gap_start, gap_end, sample_rate)
                else:
                    # Contiguous boundary: expand to find energy dip
                    search_s = max(0.0, gap_start - margin)
                    search_e = min(audio_dur, gap_end + margin)
                    mid = self._gap_energy_cut(waveform_np, search_s, search_e, sample_rate)
                    cut = max(float(left["start"]) + 0.001,
                              min(mid, float(right["end"]) - 0.001))

            left["end"] = round(cut, 4)
            right["start"] = round(cut, 4)
            n_refined += 1
            logger.debug(
                "Boundary refinement %d: %s->%s cut at %.3fs (sections=%d)",
                i, left.get("speaker"), right.get("speaker"), cut, len(valid_secs)
            )

        logger.info("exact_boundary_refinement_done: %d/%d transitions refined",
                    n_refined, len(transitions))
        return refined

    @staticmethod
    def _gap_energy_cut(
        waveform_np: np.ndarray,
        gap_start: float,
        gap_end: float,
        sample_rate: int,
        frame_ms: float = 20.0,
    ) -> float:
        """Return the quietest energy-dip midpoint in a gap, or its geometric midpoint."""
        s = int(gap_start * sample_rate)
        e = int(gap_end * sample_rate)
        chunk = waveform_np[s:e].astype(np.float32)
        frame_len = int(frame_ms / 1000 * sample_rate)
        if len(chunk) < frame_len * 2:
            return (gap_start + gap_end) / 2.0
        energies = []
        for i in range(0, len(chunk) - frame_len, frame_len):
            frame = chunk[i:i + frame_len]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        if not energies:
            return (gap_start + gap_end) / 2.0
        min_idx = int(np.argmin(energies))
        return gap_start + (min_idx + 0.5) * frame_len / sample_rate
