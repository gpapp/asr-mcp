import asyncio
import gc
import hashlib
import logging
import time
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from asr_mcp.core.model_state import state, GPU_SHRINK_RUN_OPTIONS
from asr_mcp.diarization.clustering import (
    cap_clusters, greedy_merge_clusters, match_known_speakers_full,
    collapse_unknown_speakers_second_pass,
)
from asr_mcp.diarization.overlap import detect_overlaps, build_overlap_segments
from asr_mcp.diarization.segment_ops import (
    collapse_same_speaker_segments, absorb_islands, absorb_minority_speakers,
    eliminate_ghost_speakers,
)
from asr_mcp.speaker.audio import extract_fbank, generate_sliding_windows, refine_speaker_boundaries
from asr_mcp.speaker.embedding import extract_embedding, _run_with_cpu_fallback
from asr_mcp.speaker.vad import split_at_energy_dips, run_vad_chunked, run_vad_onnx, merge_vad_sections
from asr_mcp.speaker.profiling import profile_speakers, relabel_by_pitch

logger = logging.getLogger("asr_mcp.diarization.pipeline")

MIN_EMBED_DURATION = 0.5


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
        threshold = diarization_threshold if diarization_threshold is not None else cfg.get("diarization", {}).get("distance_threshold", 0.35)
        v_threshold = vad_threshold if vad_threshold is not None else cfg.get("vad", {}).get("default_threshold", 0.5)
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

        if progress_callback:
            await progress_callback({"stage": "Extracting features", "progress": 0.2})

        # Step 3: Extract sliding-window fbank features across all speech segments
        all_fbanks, all_segments_meta, embeddable_indices = self._extract_features(
            waveform, speech_ts, sample_rate
        )

        if not all_fbanks:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        if progress_callback:
            await progress_callback({"stage": "Extracting embeddings", "progress": 0.4})

        # Step 4: Batched embedding extraction with cache & CMN
        raw_embeddings = self._extract_embeddings(all_fbanks)

        if progress_callback:
            await progress_callback({"stage": "Clustering speakers", "progress": 0.6})

        # Step 5: Clustering
        long_labels, cluster_centroids = self._cluster_embeddings(
            raw_embeddings, num_speakers, threshold, cfg
        )

        # Step 5b: Assign cluster labels to all segments (including short ones)
        self._assign_labels_to_segments(all_segments_meta, embeddable_indices, long_labels)

        # Step 5c: Detect overlaps
        overlap_cfg = cfg.get("overlap", {})
        if overlap_cfg.get("enabled", True):
            proximity_ratio = overlap_cfg.get("proximity_ratio", 0.08)
            min_distance = overlap_cfg.get("min_distance", 0.40)
            detect_overlaps(
                raw_embeddings,
                cluster_centroids,
                embeddable_indices,
                all_segments_meta,
                proximity_ratio=proximity_ratio,
                min_distance=min_distance,
            )

        if progress_callback:
            await progress_callback({"stage": "Building speaker segments", "progress": 0.7})

        # Step 6: Map labels -> "Speaker N" and build contiguous segments
        merged_segments, speaker_map = self._map_to_speakers(all_segments_meta, cfg)

        # Step 7: Split single-speaker segments vs OVERLAP
        ov_segments = [s for s in merged_segments if s.get("speaker") == "OVERLAP"]
        non_ov = [s for s in merged_segments if s.get("speaker") != "OVERLAP"]

        non_ov = absorb_islands(non_ov)

        non_ov = self._refine_boundaries(
            non_ov, all_segments_meta, embeddable_indices,
            raw_embeddings, waveform, sample_rate
        )

        if progress_callback:
            await progress_callback({"stage": "Profiling speakers", "progress": 0.8})

        # Step 8: Speaker profiling
        profiles = profile_speakers(waveform, non_ov, sample_rate)

        # Step 8b: Relabel by pitch
        non_ov, profiles, pitch_remap = relabel_by_pitch(non_ov, profiles)

        # Inject cluster centroid embeddings into every speaker's profile
        centroid_emb_map = {}
        for raw_cluster, init_name in speaker_map.items():
            if raw_cluster in cluster_centroids:
                centroid_emb_map[init_name] = cluster_centroids[raw_cluster].tolist()
        for init_name, final_name in pitch_remap.items():
            if init_name in centroid_emb_map:
                if final_name not in profiles:
                    profiles[final_name] = {}
                profiles[final_name]["embedding"] = centroid_emb_map[init_name]

        # Step 9: Known speaker matching (BEFORE ghost elimination so alternatives are populated)
        if known_speakers:
            non_ov, profiles = match_known_speakers_full(
                non_ov, all_segments_meta, embeddable_indices,
                raw_embeddings, cluster_centroids, profiles, known_speakers, cfg,
                renumber=True,
            )

        # Step 10: Ghost elimination (uses seg["alternatives"] if populated by matching)
        non_ov = eliminate_ghost_speakers(non_ov, profiles=profiles, ghost_threshold_sec=10.0)

        # Step 11: Absorb minority speakers (safeguards matched known voiceprints)
        known_names = set(known_speakers.keys()) if known_speakers else set()
        non_ov = absorb_minority_speakers(
            non_ov, max_utterance_sec=5.0, min_speaker_dur=8.0,
            protected_speakers=known_names,
        )

        # Step 12: Second-pass re-identification and consolidation of unknown speakers
        try:
            non_ov, profiles = collapse_unknown_speakers_second_pass(
                non_ov, waveform_np, sample_rate,
                known_speakers=known_speakers or {},
                profiles=profiles,
                state=self._state,
                cfg=cfg,
            )
        except Exception as e:
            logger.warning("Second pass unknown collapse failed: %s", e)

        # Step 13: Exact turn boundary refinement using raw VAD sections
        if self._raw_vad_sections:
            try:
                non_ov = await self._refine_turn_boundaries_exact(
                    non_ov, waveform_np, sample_rate, profiles, known_speakers or {}
                )
            except Exception as e:
                logger.warning("Exact turn boundary refinement failed: %s", e)

        # Recombine with overlap segments
        merged_segments = sorted(non_ov + ov_segments, key=lambda x: x["start"])
        resolved = []
        for seg in merged_segments:
            if resolved and seg["start"] < resolved[-1]["end"]:
                mid = (seg["start"] + resolved[-1]["end"]) / 2.0
                resolved[-1]["end"] = mid
                seg["start"] = mid
            if seg["end"] > seg["start"]:
                resolved.append(seg)
        merged_segments = resolved

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
        return run_vad_chunked(
            waveform, sample_rate=sample_rate,
            threshold=threshold, min_speech_duration_ms=min_speech_ms,
        )

    def _extract_features(self, waveform_tensor: torch.Tensor, speech_ts: list[dict], sample_rate: int):
        all_fbanks = []
        all_segments_meta = []
        embeddable_indices = []

        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        for ts in speech_ts:
            start_sample = int(ts["start"] * sample_rate)
            end_sample = int(ts["end"] * sample_rate)
            segment_wav = waveform_tensor[:, start_sample:end_sample]

            windows, start_times = generate_sliding_windows(
                segment_wav, sample_rate, window_sec=2.0, stride_sec=1.2
            )

            for w, rel_start in zip(windows, start_times):
                chunk_duration = w.shape[-1] / sample_rate
                global_start = ts["start"] + rel_start
                global_end = global_start + chunk_duration
                meta_idx = len(all_segments_meta)
                all_segments_meta.append({"start": global_start, "end": global_end})

                if chunk_duration >= MIN_EMBED_DURATION:
                    if w.shape[-1] < 1600:
                        w = torch.nn.functional.pad(w, (0, 1600 - w.shape[-1]))
                    all_fbanks.append(extract_fbank(w, sample_rate))
                    embeddable_indices.append(meta_idx)

        return all_fbanks, all_segments_meta, embeddable_indices

    def _extract_embeddings(self, all_fbanks: list[torch.Tensor]) -> np.ndarray:
        fb_hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in all_fbanks]
        cached_embeddings = {}
        miss_indices = []

        for idx, h in enumerate(fb_hashes):
            cached = getattr(self._state, "embedding_cache", None)
            c_emb = cached.get(h) if cached else None
            if c_emb is not None:
                cached_embeddings[idx] = c_emb
            else:
                miss_indices.append(idx)

        if miss_indices:
            miss_fbanks = [all_fbanks[idx] for idx in miss_indices]
            max_len = max(fb.shape[1] for fb in miss_fbanks)
            padded_fbanks = []
            for fb in miss_fbanks:
                if fb.shape[1] < max_len:
                    fb_padded = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
                else:
                    fb_padded = fb
                padded_fbanks.append(fb_padded)

            batch = torch.stack(padded_fbanks, dim=0)  # [N_miss, 1, max_len, 80]
            cmn_batch = batch - batch.mean(dim=2, keepdim=True)
            batch_fbanks = cmn_batch.squeeze(1).numpy().astype(np.float32)  # [N_miss, max_len, 80]

            computed_embeddings = []
            batch_size = 16
            input_name = self._state.embedding_session.get_inputs()[0].name
            output_name = self._state.embedding_session.get_outputs()[0].name

            for i in range(0, len(batch_fbanks), batch_size):
                audio_input = batch_fbanks[i:i + batch_size]
                out = _run_with_cpu_fallback(
                    self._state.embedding_session, {input_name: audio_input}, [output_name]
                )[0]
                if out.ndim == 3:
                    out = out.mean(axis=1)
                computed_embeddings.append(out)

            computed_embeddings = np.concatenate(computed_embeddings, axis=0)

            for local_idx, idx in enumerate(miss_indices):
                emb = computed_embeddings[local_idx]
                h = fb_hashes[idx]
                if getattr(self._state, "embedding_cache", None):
                    self._state.embedding_cache.put(h, emb)
                cached_embeddings[idx] = emb

        raw_embeddings = np.array([cached_embeddings[idx] for idx in range(len(all_fbanks))], dtype=np.float32)
        if raw_embeddings.ndim == 3:
            raw_embeddings = raw_embeddings.mean(axis=1)

        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        raw_embeddings = raw_embeddings / np.maximum(norms, 1e-12)
        return raw_embeddings

    def _cluster_embeddings(self, raw_embeddings: np.ndarray, num_speakers: Optional[int], threshold: float, cfg: dict):
        if num_speakers is not None:
            clusterer = AgglomerativeClustering(
                n_clusters=num_speakers, metric="cosine", linkage="average"
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                metric="cosine", linkage="average"
            )

        if len(raw_embeddings) > 1:
            long_labels = clusterer.fit_predict(raw_embeddings)
        else:
            long_labels = np.array([0])

        max_clusters = cfg.get("diarization", {}).get("max_clusters", 15)
        long_labels = cap_clusters(raw_embeddings, long_labels, max_clusters=max_clusters)

        n_clusters = len(set(int(l) for l in long_labels))
        if n_clusters > 1 and num_speakers is None:
            merge_threshold = cfg.get("diarization", {}).get("merge_threshold", 0.25)
            long_labels, cluster_centroids = greedy_merge_clusters(
                raw_embeddings, long_labels, merge_threshold
            )
        else:
            cluster_centroids = {}
            for cluster_id in set(long_labels):
                mask = (long_labels == cluster_id)
                mean_emb = raw_embeddings[mask].mean(axis=0)
                norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                cluster_centroids[int(cluster_id)] = norm_emb

        return long_labels, cluster_centroids

    def _assign_labels_to_segments(self, all_segments_meta: list[dict], embeddable_indices: list[int], long_labels: np.ndarray):
        for idx, label in zip(embeddable_indices, long_labels):
            all_segments_meta[idx]["speaker_raw"] = int(label)

        emb_mids = np.array([
            (all_segments_meta[i]["start"] + all_segments_meta[i]["end"]) / 2.0
            for i in embeddable_indices
        ])
        for seg in all_segments_meta:
            if "speaker_raw" not in seg:
                mid = (seg["start"] + seg["end"]) / 2.0
                nearest = int(np.argmin(np.abs(emb_mids - mid)))
                seg["speaker_raw"] = all_segments_meta[embeddable_indices[nearest]]["speaker_raw"]

    def _map_to_speakers(self, all_segments_meta: list[dict], cfg: dict) -> tuple[list[dict], dict]:
        speaker_map: dict[int, str] = {}
        for seg in sorted(all_segments_meta, key=lambda x: x["start"]):
            raw = seg["speaker_raw"]
            if raw not in speaker_map:
                speaker_map[raw] = f"Speaker {len(speaker_map) + 1}"
            seg["speaker"] = speaker_map[raw]

        for seg in all_segments_meta:
            if seg.get("is_overlap", False):
                overs_raw = seg.get("overlap_speakers_raw")
                if overs_raw is not None and len(overs_raw) == 2:
                    seg["overlap_speakers"] = sorted([
                        speaker_map[overs_raw[0]],
                        speaker_map[overs_raw[1]],
                    ])

        overlap_cfg = cfg.get("overlap", {})
        max_gap = overlap_cfg.get("max_speaker_gap", 1.0)
        min_dur = overlap_cfg.get("min_duration_sec", 0.3)
        merged_segments = build_overlap_segments(all_segments_meta, max_gap, min_dur)
        return merged_segments, speaker_map

    def _refine_boundaries(
        self,
        merged_segments: list[dict],
        all_segments_meta: list[dict],
        embeddable_indices: list[int],
        raw_embeddings: np.ndarray,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> list[dict]:
        if len(merged_segments) < 2 or not self._state.embedding_session:
            return merged_segments

        spk_emb_lists: dict[str, list] = {}
        for k, meta_idx in enumerate(embeddable_indices):
            spk = all_segments_meta[meta_idx].get("speaker")
            if spk and spk != "OVERLAP":
                spk_emb_lists.setdefault(spk, []).append(raw_embeddings[k])

        named_centroids_np: dict[str, np.ndarray] = {}
        for spk, embs in spk_emb_lists.items():
            avg = np.mean(embs, axis=0)
            named_centroids_np[spk] = avg / (np.linalg.norm(avg) + 1e-12)

        if named_centroids_np:
            merged_segments = refine_speaker_boundaries(
                merged_segments,
                waveform,
                self._state.embedding_session,
                named_centroids_np,
                sample_rate=sample_rate,
            )
        return merged_segments

    def _get_speaker_ref(
        self,
        speaker_name: str,
        profiles: dict,
        known_speakers: dict,
        waveform_np: np.ndarray,
        sample_rate: int,
        segments: list,
    ) -> Optional[np.ndarray]:
        clean_name = speaker_name.strip("[]")
        if clean_name in known_speakers:
            emb = known_speakers[clean_name].get("embedding")
            if emb is not None:
                arr = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(arr)
                return arr / norm if norm > 1e-8 else arr

        profile = profiles.get(speaker_name, {})
        centroid = profile.get("centroid")
        if centroid is not None:
            arr = np.array(centroid, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 1e-8 else arr

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
        if len(segments) < 2 or not self._raw_vad_sections:
            return segments

        refined = [dict(s) for s in segments]
        emb_session = self._state.embedding_session
        if emb_session is None:
            return refined

        margin = 0.5
        min_sec_dur = 0.1
        audio_dur = len(waveform_np) / sample_rate
        raw = sorted(self._raw_vad_sections, key=lambda s: s["start"])

        ref_cache: dict[str, Optional[np.ndarray]] = {}

        def _ref(name: str) -> Optional[np.ndarray]:
            if name not in ref_cache:
                ref_cache[name] = self._get_speaker_ref(
                    name, profiles, known_speakers, waveform_np, sample_rate, refined
                )
            return ref_cache[name]

        transitions = []
        for i in range(len(refined) - 1):
            left = refined[i]
            right = refined[i + 1]
            if left.get("speaker") == right.get("speaker"):
                continue

            gap_start = float(left["end"])
            gap_end = float(right["start"])

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
            fbanks, valid_keys = [], []
            for s in all_secs_to_embed:
                a = int(s["start"] * sample_rate)
                b = int(s["end"] * sample_rate)
                b = min(b, len(waveform_np))
                chunk = waveform_np[a:b]
                if len(chunk) < int(min_sec_dur * sample_rate):
                    sec_key_to_emb[(s["start"], s["end"])] = None
                    continue
                t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
                fb = extract_fbank(t, sample_rate)  # [1, T, 80]
                fb = fb - fb.mean(dim=2, keepdim=True)  # CMN
                fbanks.append(fb)
                valid_keys.append((s["start"], s["end"]))

            if fbanks:
                hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in fbanks]
                cached_embs: dict[int, np.ndarray] = {}
                misses: list[int] = []
                for idx, h in enumerate(hashes):
                    hit = getattr(self._state, "embedding_cache", None)
                    c_hit = hit.get(h) if hit else None
                    if c_hit is not None:
                        cached_embs[idx] = c_hit
                    else:
                        misses.append(idx)

                if misses:
                    miss_fbs = [fbanks[idx] for idx in misses]
                    max_len = max(fb.shape[1] for fb in miss_fbs)
                    padded = []
                    for fb in miss_fbs:
                        if fb.shape[1] < max_len:
                            padded.append(torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1])))
                        else:
                            padded.append(fb)
                    batch = torch.stack(padded, dim=0).squeeze(1).numpy().astype(np.float32)  # [N_miss, max_len, 80]
                    input_name = emb_session.get_inputs()[0].name
                    output_names = [o.name for o in emb_session.get_outputs()]
                    try:
                        computed_outs = []
                        batch_size = 16
                        for b_start in range(0, len(batch), batch_size):
                            b_inp = batch[b_start:b_start + batch_size]
                            out_chunk = _run_with_cpu_fallback(emb_session, {input_name: b_inp}, output_names)[0]
                            if out_chunk.ndim == 3:
                                out_chunk = out_chunk.mean(axis=1)
                            computed_outs.append(out_chunk)
                        out = np.concatenate(computed_outs, axis=0) if computed_outs else None
                    except Exception as e:
                        logger.warning("Boundary batch embed failed, skipping: %s", e)
                        out = None
                    if out is not None:
                        for li, idx in enumerate(misses):
                            e_vec = out[li].astype(np.float64)
                            nrm = np.linalg.norm(e_vec)
                            if nrm > 1e-8:
                                e_vec = e_vec / nrm
                            if getattr(self._state, "embedding_cache", None):
                                self._state.embedding_cache.put(hashes[idx], e_vec)
                            cached_embs[idx] = e_vec

                for idx, key in enumerate(valid_keys):
                    sec_key_to_emb[key] = cached_embs.get(idx)

        n_refined = 0
        for i, gap_start, gap_end, secs in transitions:
            left = refined[i]
            right = refined[i + 1]

            left_ref = _ref(left["speaker"])
            right_ref = _ref(right["speaker"])

            owners_int: list[int] = []
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
                n = len(owners_int)
                best_k, best_cost = 0, float("inf")
                for k in range(n + 1):
                    cost = 0.0
                    for j in range(k):
                        if owners_int[j] == 1:
                            cost += weights[j]
                    for j in range(k, n):
                        if owners_int[j] == 0:
                            cost += weights[j]
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k

                if best_k < n:
                    cut = float(valid_secs[best_k]["start"])
                else:
                    cut = gap_end

                cut = max(float(left["start"]) + 0.001, min(cut, float(right["end"]) - 0.001))
                if gap_end > gap_start:
                    cut = max(gap_start, min(cut, gap_end))
            else:
                if gap_end > gap_start:
                    cut = self._gap_energy_cut(waveform_np, gap_start, gap_end, sample_rate)
                else:
                    search_s = max(0.0, gap_start - margin)
                    search_e = min(audio_dur, gap_end + margin)
                    mid = self._gap_energy_cut(waveform_np, search_s, search_e, sample_rate)
                    cut = max(float(left["start"]) + 0.001,
                              min(mid, float(right["end"]) - 0.001))

            left["end"] = round(cut, 4)
            right["start"] = round(cut, 4)
            n_refined += 1

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
