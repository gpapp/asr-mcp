"""Audio feature extraction utilities for speaker processing."""
import hashlib
import logging
from typing import Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("asr_mcp.speaker.audio")

FBANK_N_FILTERS = 80
FBANK_N_FFT = 512
FBANK_SAMPLE_RATE = 16000


def extract_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extracts 80-dim log-mel filterbanks from waveform matching WeSpeaker expectations.

    Note: CMN is NOT applied here - it's applied per sub-segment in extract_embedding.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Scale waveform to 16-bit PCM range for kaldi.fbank
    waveform = waveform * 32768.0

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=FBANK_N_FILTERS,
        frame_length=25,
        frame_shift=10,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        dither=0.0,
        window_type="hamming",
    )
    return fbank.unsqueeze(0)  # [1, frames, 80]


def generate_sliding_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    window_sec: float = 3.0,
    stride_sec: float = 1.5,
) -> tuple[list[torch.Tensor], list[float]]:
    """Generates overlapping sliding windows from a continuous waveform."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = waveform.shape[-1]

    windows: list[torch.Tensor] = []
    start_times: list[float] = []

    if total_samples < window_samples:
        return [waveform], [0.0]

    for start in range(0, total_samples - window_samples + 1, stride_samples):
        windows.append(waveform[:, start:start + window_samples])
        start_times.append(start / sample_rate)

    # Handle the last remaining chunk if it doesn't align perfectly
    last_start = len(windows) * stride_samples if windows else 0
    if last_start < total_samples and (total_samples - last_start) > (sample_rate * 0.1):  # min 0.1s
        windows.append(waveform[:, last_start:])
        start_times.append(last_start / sample_rate)

    return windows, start_times


def refine_speaker_boundaries(
    segments: list[dict],
    waveform: torch.Tensor,
    embedding_session,
    cluster_centroids: dict,
    sample_rate: int = 16000,
    search_sec: float = 1.2,
    sub_window_sec: float = 0.5,
    sub_stride_sec: float = 0.2,
    min_segment_dur: float = 0.3,
) -> list[dict]:
    """Refine speaker boundaries after initial clustering.

    At each transition point between two different speakers, re-examines the audio
    in a ±search_sec window around the boundary using fine sub-windows.
    All sub-windows across all transitions are collected and embedded in a
    batched ONNX call with caching.
    """
    if not segments or len(segments) < 2:
        return segments

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    sub_samples = int(sub_window_sec * sample_rate)
    stride_samples = int(sub_stride_sec * sample_rate)
    min_sub_samples = int(0.3 * sample_rate)  # 0.3s minimum embeddable length
    total_samples = waveform.shape[-1]

    # Pre-build centroid matrix for fast cosine similarity
    spk_labels = [s for s in cluster_centroids.keys() if s != "OVERLAP"]
    if not spk_labels:
        return segments

    centroid_matrix = np.stack([
        np.array(cluster_centroids[s], dtype=np.float32) for s in spk_labels
    ])  # [N_speakers, D]

    # ------------------------------------------------------------------ #
    # Pass 1: collect every sub-window chunk across all transitions
    # ------------------------------------------------------------------ #
    transition_meta = []
    all_fbanks = []
    slot_map = []

    refined = [dict(s) for s in segments]

    for i in range(len(refined) - 1):
        left = refined[i]
        right = refined[i + 1]
        if left.get("speaker") == right.get("speaker"):
            continue
        if left.get("speaker") not in cluster_centroids or right.get("speaker") not in cluster_centroids:
            continue

        nominal_boundary = left["end"]
        search_start = max(0.0, nominal_boundary - search_sec)
        search_end = min(total_samples / sample_rate, nominal_boundary + search_sec)
        region_start = int(search_start * sample_rate)
        region_end = int(search_end * sample_rate)

        t_idx = len(transition_meta)
        transition_meta.append((
            i, nominal_boundary, search_start, search_end,
            left["speaker"], right["speaker"],
        ))

        pos = region_start
        while pos + min_sub_samples <= region_end:
            end_pos = min(pos + sub_samples, total_samples)
            chunk = waveform[:, pos:end_pos]
            if chunk.shape[-1] >= min_sub_samples:
                if chunk.shape[-1] < sub_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, sub_samples - chunk.shape[-1]))
                fb = extract_fbank(chunk, sample_rate)  # [1, T, 80]
                fb = fb - fb.mean(dim=1, keepdim=True)  # CMN
                all_fbanks.append(fb)
                center_t = (pos + min(pos + sub_samples, end_pos)) / 2 / sample_rate
                slot_map.append((t_idx, center_t))
            pos += stride_samples

    if not all_fbanks:
        return refined

    # ------------------------------------------------------------------ #
    # Pass 2: batched ONNX inference over all sub-windows with cache
    # ------------------------------------------------------------------ #
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.embedding import _run_with_cpu_fallback

    fb_hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in all_fbanks]
    cached_embeddings = {}
    miss_indices = []

    for idx, h in enumerate(fb_hashes):
        cached = getattr(state, "embedding_cache", None)
        c_emb = cached.get(h) if cached else None
        if c_emb is not None:
            cached_embeddings[idx] = c_emb
        else:
            miss_indices.append(idx)

    if miss_indices:
        miss_fbanks = [all_fbanks[idx] for idx in miss_indices]
        max_len = max(fb.shape[1] for fb in miss_fbanks)
        padded = []
        for fb in miss_fbanks:
            if fb.shape[1] < max_len:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
            padded.append(fb.squeeze(0))

        batch = torch.stack(padded).numpy().astype(np.float32)  # [N_miss, max_len, 80]
        input_name = embedding_session.get_inputs()[0].name
        output_name = embedding_session.get_outputs()[0].name
        computed_chunks = []
        batch_size = 16
        for b_start in range(0, len(batch), batch_size):
            b_inp = batch[b_start:b_start + batch_size]
            out_c = _run_with_cpu_fallback(
                embedding_session, {input_name: b_inp}, [output_name]
            )[0]
            if out_c.ndim == 3:
                out_c = out_c.mean(axis=1)
            computed_chunks.append(out_c)
        raw_embs_miss = np.concatenate(computed_chunks, axis=0) if computed_chunks else np.empty((0, 192), dtype=np.float32)

        for local_idx, idx in enumerate(miss_indices):
            emb = raw_embs_miss[local_idx]
            h = fb_hashes[idx]
            if getattr(state, "embedding_cache", None):
                state.embedding_cache.put(h, emb)
            cached_embeddings[idx] = emb

    raw_embs = np.array([cached_embeddings[idx] for idx in range(len(all_fbanks))], dtype=np.float32)
    if raw_embs.ndim == 3:
        raw_embs = raw_embs.mean(axis=1)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    embs = raw_embs / np.maximum(norms, 1e-12)

    # Cosine distance to centroids
    dists = 1.0 - (embs @ centroid_matrix.T)  # [N, S]
    nearest = [spk_labels[int(np.argmin(d))] for d in dists]

    # ------------------------------------------------------------------ #
    # Pass 3: group results back per transition, then update boundaries
    # ------------------------------------------------------------------ #
    candidates_per_transition: dict[int, list] = {
        t: [] for t in range(len(transition_meta))
    }
    for k, (t_idx, center_t) in enumerate(slot_map):
        candidates_per_transition[t_idx].append((center_t, nearest[k]))

    for t_idx, (seg_i, nominal_boundary, search_start, search_end,
                left_spk, right_spk) in enumerate(transition_meta):
        candidates = candidates_per_transition[t_idx]
        if not candidates:
            continue

        last_left_t = search_start
        first_right_t = search_end

        for center_t, spk in candidates:
            if spk == left_spk:
                last_left_t = center_t
        for center_t, spk in candidates:
            if spk == right_spk and center_t > last_left_t:
                first_right_t = center_t
                break

        new_boundary = round((last_left_t + first_right_t) / 2.0, 4)

        left = refined[seg_i]
        right = refined[seg_i + 1]
        if (abs(new_boundary - nominal_boundary) > 0.05 and
                new_boundary - left["start"] >= min_segment_dur and
                right["end"] - new_boundary >= min_segment_dur):
            refined[seg_i] = dict(left, end=new_boundary)
            refined[seg_i + 1] = dict(right, start=new_boundary)

    refined = [s for s in refined if s["end"] - s["start"] >= min_segment_dur]
    return refined
