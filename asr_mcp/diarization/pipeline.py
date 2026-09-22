import asyncio
import logging
import time
from typing import Optional

import numpy as np
import torch

from asr_mcp.core.model_state import state, LRUCache
from asr_mcp.diarization.clustering import (
    cap_clusters, greedy_merge_clusters, merge_similar_speakers, match_known_speakers_full,
)
from asr_mcp.diarization.segment_ops import (
    collapse_same_speaker_segments, absorb_islands, absorb_minority_speakers,
    eliminate_ghost_speakers,
)
from asr_mcp.speaker.audio import refine_speaker_boundaries
from asr_mcp.speaker.embedding import extract_embedding
from asr_mcp.speaker.vad import split_at_energy_dips, run_vad_chunked, run_vad_onnx
from asr_mcp.speaker.profiling import profile_speakers, relabel_by_pitch

logger = logging.getLogger("asr_mcp.diarization.pipeline")

_embedding_cache = LRUCache(max_size=5000)


class Diarizer:
    def __init__(self, model_state=None, settings=None):
        self._state = model_state or state
        self._settings = settings
        self._cfg = None

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

        if progress_callback:
            await progress_callback({"stage": "Detecting speech regions", "progress": 0.1})

        # Step 1: VAD
        speech_ts = self._run_vad(waveform, sample_rate, v_threshold, min_speech_ms)
        if not speech_ts:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        # Step 1b: Merge nearby speech regions separated by <1s silence
        speech_ts = self._merge_nearby_speech(speech_ts, sample_rate, max_gap_sec=1.0)

        if progress_callback:
            await progress_callback({"stage": "Splitting audio at silence boundaries", "progress": 0.15})

        # Step 2: Energy-dip splitting (only split at genuine pauses >1s)
        speech_ts = split_at_energy_dips(
            speech_ts, waveform.numpy().squeeze(), sample_rate,
            min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
            min_split_piece=2.0,
        )

        # Step 3: Extract one embedding per segment (energy-dip bounded)
        all_segments = []
        all_embeddings = []
        total_segs = len(speech_ts)
        for seg_idx, ts in enumerate(speech_ts):
            if progress_callback:
                p = 0.15 + 0.45 * ((seg_idx + 1) / max(total_segs, 1))
                await progress_callback({"stage": f"Extracting embeddings ({seg_idx+1}/{total_segs})", "progress": p})
            start_sample = ts["start"]
            end_sample = ts["end"]
            dur_sec = (end_sample - start_sample) / sample_rate
            if dur_sec < 0.3:
                continue
            segment_audio = waveform[..., start_sample:end_sample]
            all_segments.append({
                "start": round(start_sample / sample_rate, 3),
                "end": round(end_sample / sample_rate, 3),
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

        # Step 8: Map labels to segments (energy-dip boundaries)
        merged_segments = []
        for i, (seg, label) in enumerate(zip(all_segments, long_labels)):
            merged_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"Speaker {label + 1}",
                "index": i,
            })

        # Step 9: Collapse + absorb islands + absorb minority speakers
        merged_segments = collapse_same_speaker_segments(merged_segments, max_gap=0.5)
        merged_segments = absorb_islands(merged_segments, min_island_dur=1.0)
        merged_segments = absorb_minority_speakers(merged_segments, min_speaker_dur=3.0)

        if progress_callback:
            await progress_callback({"stage": "Profiling speakers", "progress": 0.8})

        # Step 10: Speaker profiling
        profiles = profile_speakers(waveform, merged_segments, sample_rate)

        # Step 10b: Merge similar speakers (compare voiceprints, merge close ones)
        merged_segments, cluster_centroids, profiles = merge_similar_speakers(
            merged_segments, raw_embeddings, long_labels,
            cluster_centroids, profiles, cfg,
        )

        # Step 10c: Relabel by pitch (highest pitch = Speaker 1)
        merged_segments, profiles, label_map = relabel_by_pitch(merged_segments, profiles)

        # Renumber cluster_centroids to match relabeled segments (0, 1, 2...)
        # label_map maps old names -> new names. Build new sequential centroids.
        old_to_new = {}  # old integer label -> new integer label
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

        # Step 11: Boundary refinement (uses string-keyed centroids)
        merged_segments = refine_speaker_boundaries(
            merged_segments, waveform, self._state.embedding_session,
            string_centroids, sample_rate,
            embedding_cache=_embedding_cache,
        )

        # Step 12: Ghost elimination
        merged_segments = eliminate_ghost_speakers(merged_segments, profiles)

        # Step 13: Known speaker matching
        if known_speakers:
            merged_segments, match_info = match_known_speakers_full(
                merged_segments, all_segments, list(range(len(all_segments))),
                raw_embeddings, cluster_centroids, profiles, known_speakers, cfg,
                renumber=True,
            )

        if progress_callback:
            await progress_callback({"stage": "diarization_finished", "progress": 1.0})

        total_time = time.time() - start_time

        return {
            "segments": merged_segments,
            "profiles": {k: {kk: vv for kk, vv in v.items() if kk != "mfcc"} for k, v in profiles.items()},
            "total_speakers": len(set(s.get("speaker") for s in merged_segments)),
            "total_time_sec": round(total_time, 2),
            "audio_duration_sec": round(waveform.shape[-1] / sample_rate, 2),
        }

    def _load_audio(self, audio_path: str) -> tuple:
        try:
            from asr_mcp.voiceprint.utils import load_audio
            return load_audio(audio_path)
        except Exception as e:
            logger.error("Failed to load audio %s: %s", audio_path, e)
            return None, None
    def _run_vad(self, waveform, sample_rate, threshold, min_speech_ms):
        if self._state.vad_session is not None:
            return run_vad_onnx(
                waveform, self._state.vad_session,
                sample_rate=sample_rate,
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
            )
        return run_vad_chunked(
            waveform, sample_rate=sample_rate,
            threshold=threshold, min_speech_duration_ms=min_speech_ms,
        )

    @staticmethod
    def _merge_nearby_speech(speech_ts: list, sample_rate: int, max_gap_sec: float = 1.0) -> list:
        if len(speech_ts) <= 1:
            return speech_ts
        max_gap_samples = int(max_gap_sec * sample_rate)
        merged = [speech_ts[0].copy()]
        for seg in speech_ts[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= max_gap_samples:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        logger.info("Merged VAD: %d regions -> %d (gap<%.1fs)", len(speech_ts), len(merged), max_gap_sec)
        return merged
