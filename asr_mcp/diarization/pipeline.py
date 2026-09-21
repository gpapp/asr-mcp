import asyncio
import hashlib
import logging
import time
from typing import Optional

import numpy as np
import torch

from asr_mcp.core.model_state import state, LRUCache
from asr_mcp.diarization.clustering import (
    greedy_merge_clusters, match_known_speakers_full,
)
from asr_mcp.diarization.segment_ops import (
    collapse_same_speaker_segments, absorb_islands, eliminate_ghost_speakers,
)
from asr_mcp.speaker.audio import extract_fbank, generate_sliding_windows, refine_speaker_boundaries
from asr_mcp.speaker.embedding import extract_embedding, batch_embed_files
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
            await progress_callback({"stage": "vad", "progress": 0.1})

        # Step 1: VAD
        speech_ts = self._run_vad(waveform, sample_rate, v_threshold, min_speech_ms)
        if not speech_ts:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        # Step 1b: Merge nearby speech regions separated by <1s silence
        speech_ts = self._merge_nearby_speech(speech_ts, sample_rate, max_gap_sec=1.0)

        if progress_callback:
            await progress_callback({"stage": "features", "progress": 0.2})

        # Step 2: Energy-dip splitting (only split at genuine pauses >1s)
        speech_ts = split_at_energy_dips(
            speech_ts, waveform.numpy().squeeze(), sample_rate,
            min_segment_dur=3.0, dip_ratio=0.35, min_dip_dur=0.5,
            min_split_piece=2.0,
        )

        # Step 3: Sliding windows + feature extraction
        all_windows = []
        all_fbanks = []
        for ts in speech_ts:
            start_sample = ts["start"]
            end_sample = ts["end"]
            segment_audio = waveform[..., start_sample:end_sample]
            windows = generate_sliding_windows(segment_audio, sample_rate, window_sec=3.0, stride_sec=2.5)
            for w in windows:
                w["abs_start"] = start_sample + w["start_sample"]
                w["abs_end"] = start_sample + w["end_sample"]
                w["speech_ts"] = ts
            all_windows.extend(windows)
            for w in windows:
                w_audio = waveform[..., w["abs_start"]:w["abs_end"]]
                fbank = extract_fbank(w_audio, sample_rate)
                fbank = fbank - fbank.mean(dim=0, keepdim=True)
                all_fbanks.append(fbank)

        if not all_fbanks:
            return {"segments": [], "total_time_sec": round(time.time() - start_time, 2)}

        if progress_callback:
            await progress_callback({"stage": "embedding", "progress": 0.4})

        # Step 4: Extract embeddings from pre-computed fbanks
        raw_embeddings = []
        for fbank in all_fbanks:
            fbank_np = fbank.numpy().astype(np.float32)
            fbank_hash = hashlib.md5(fbank_np.tobytes()).hexdigest()
            cached = _embedding_cache.get(fbank_hash)
            if cached is not None:
                raw_embeddings.append(cached)
            else:
                # Feed fbank directly to ONNX embedding model
                fbank_feed = fbank_np[np.newaxis]  # (1, frames, n_mels)
                input_name = self._state.embedding_session.get_inputs()[0].name
                output_name = self._state.embedding_session.get_outputs()[0].name
                emb = self._state.embedding_session.run(
                    [output_name], {input_name: fbank_feed}
                )[0]
                if emb.ndim == 3:
                    emb = emb.mean(axis=1)  # Mean pool over frames
                emb = emb.reshape(1, -1) if emb.ndim == 1 else emb
                norm = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / (norm + 1e-8)
                emb = emb.squeeze().astype(np.float32)
                _embedding_cache.put(fbank_hash, emb)
                raw_embeddings.append(emb)

        raw_embeddings = np.array(raw_embeddings, dtype=np.float32)

        if progress_callback:
            await progress_callback({"stage": "clustering", "progress": 0.6})

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

        # Step 6: Greedy merge
        merge_thresh = cfg.get("diarization", {}).get("merge_threshold", 0.45)
        long_labels, cluster_centroids = greedy_merge_clusters(raw_embeddings, long_labels, merge_thresh)

        if progress_callback:
            await progress_callback({"stage": "segments", "progress": 0.7})

        # Step 8: Map labels to segments
        merged_segments = self._map_labels_to_segments(
            all_windows, long_labels, cluster_centroids, sample_rate,
        )

        # Step 9: Collapse + absorb islands
        merged_segments = collapse_same_speaker_segments(merged_segments, max_gap=0.5)
        merged_segments = absorb_islands(merged_segments, min_island_dur=1.0)

        # Step 10: Boundary refinement
        merged_segments = refine_speaker_boundaries(
            merged_segments, waveform, self._state.embedding_session,
            cluster_centroids, sample_rate,
            embedding_cache=_embedding_cache,
        )

        if progress_callback:
            await progress_callback({"stage": "profiling", "progress": 0.8})

        # Step 11: Speaker profiling
        profiles = profile_speakers(waveform, merged_segments, sample_rate)
        merged_segments, profiles, label_map = relabel_by_pitch(merged_segments, profiles)

        # Step 12: Ghost elimination
        merged_segments = eliminate_ghost_speakers(merged_segments, profiles)

        # Step 13: Known speaker matching
        if known_speakers:
            merged_segments, match_info = match_known_speakers_full(
                merged_segments, all_windows, list(range(len(all_windows))),
                raw_embeddings, cluster_centroids, profiles, known_speakers, cfg,
            )

        if progress_callback:
            await progress_callback({"stage": "done", "progress": 1.0})

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

    def _map_labels_to_segments(self, windows, labels, centroids, sample_rate):
        segments = []
        for i, (w, label) in enumerate(zip(windows, labels)):
            segments.append({
                "start": round(w["abs_start"] / sample_rate, 3),
                "end": round(w["abs_end"] / sample_rate, 3),
                "speaker": f"Speaker {label + 1}",
                "index": i,
            })
        return segments
