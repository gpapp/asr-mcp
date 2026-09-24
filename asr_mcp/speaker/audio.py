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
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, num_mel_bins=FBANK_N_FILTERS,
        sample_frequency=sample_rate,
    )
    return fbank


def generate_sliding_windows(
    waveform: torch.Tensor, sample_rate: int,
    window_sec: float = 2.0, stride_sec: float = 1.2,
) -> list[dict]:
    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = waveform.shape[-1]
    windows = []
    start = 0
    idx = 0
    while start + window_samples <= total_samples:
        end = start + window_samples
        windows.append({
            "index": idx,
            "start_sec": round(start / sample_rate, 3),
            "end_sec": round(end / sample_rate, 3),
            "start_sample": start,
            "end_sample": end,
        })
        start += stride_samples
        idx += 1
    return windows


def refine_speaker_boundaries(
    segments: list[dict],
    waveform: torch.Tensor,
    embedding_session,
    cluster_centroids: dict,
    sample_rate: int = 16000,
    search_sec: float = 1.0,
    sub_window_sec: float = 0.5,
    sub_stride_sec: float = 0.1,
    min_segment_dur: float = 0.3,
    embedding_cache=None,
) -> list[dict]:
    """Refine every speaker-change boundary using sub-window embedding allegiance.

    For each transition between prev and curr:
      1. Evaluate sub-windows in a 2*search_sec region around the boundary.
      2. Assign each sub-window to the speaker whose centroid it is closer to.
      3. Find the allegiance-switch point (first sub-window closer to curr).
      4. Set BOTH prev["end"] and curr["start"] to that point (contiguous, no gaps/overlaps).

    Fixes vs original:
    - Both prev["end"] and curr["start"] are updated (not just curr["start"]).
    - Allegiance metric: correctly identifies the crossover sub-window.
    - No segment duplication on degenerate end<=start.
    """
    if not segments or len(segments) < 2:
        return segments

    from asr_mcp.speaker.embedding import extract_embedding

    refined = [dict(s) for s in segments]

    for i in range(1, len(refined)):
        prev = refined[i - 1]
        curr = refined[i]

        boundary = (prev["end"] + curr["start"]) / 2.0  # nominal midpoint if gap exists
        if curr.get("speaker") == prev.get("speaker"):
            continue

        prev_emb = cluster_centroids.get(prev.get("speaker"))
        curr_emb = cluster_centroids.get(curr.get("speaker"))
        if prev_emb is None or curr_emb is None:
            continue

        search_start = max(0.0, boundary - search_sec)
        search_end = boundary + search_sec

        if search_end - search_start < sub_window_sec:
            continue

        sub_samples_start = int(search_start * sample_rate)
        sub_samples_end = int(search_end * sample_rate)
        sub_audio = waveform[..., sub_samples_start:sub_samples_end]

        sub_windows = generate_sliding_windows(
            sub_audio, sample_rate,
            window_sec=sub_window_sec, stride_sec=sub_stride_sec,
        )

        # Collect allegiance for each sub-window
        allegiances = []  # list of ("prev" | "curr", abs_time_sec)
        for sw in sub_windows:
            sw_audio = sub_audio[..., sw["start_sample"]:sw["end_sample"]]
            if sw_audio.shape[-1] < int(sub_window_sec * sample_rate * 0.5):
                continue

            audio_np = sw_audio.numpy().squeeze()
            sw_hash = hashlib.md5(audio_np.tobytes()).hexdigest()

            if embedding_cache:
                cached = embedding_cache.get(sw_hash)
                if cached is not None:
                    emb = cached
                else:
                    emb = extract_embedding(sw_audio, sample_rate, embedding_session)
                    embedding_cache.put(sw_hash, emb)
            else:
                emb = extract_embedding(sw_audio, sample_rate, embedding_session)

            emb_arr = np.array(emb, dtype=np.float32)
            p_arr = np.array(prev_emb, dtype=np.float32)
            c_arr = np.array(curr_emb, dtype=np.float32)

            norm_e = np.linalg.norm(emb_arr)
            norm_p = np.linalg.norm(p_arr)
            norm_c = np.linalg.norm(c_arr)

            d_prev = 1.0 - float(np.dot(emb_arr, p_arr) / (norm_e * norm_p + 1e-8))
            d_curr = 1.0 - float(np.dot(emb_arr, c_arr) / (norm_e * norm_c + 1e-8))

            sw_abs_start = search_start + sw["start_sec"]
            sw_abs_mid = sw_abs_start + sub_window_sec / 2.0
            allegiances.append(("curr" if d_curr < d_prev else "prev", sw_abs_mid))

        if not allegiances:
            continue

        # Find the first sub-window closer to curr speaker after boundary
        new_boundary = None
        for owner, t in allegiances:
            if owner == "curr" and t >= boundary - sub_window_sec:
                new_boundary = t - sub_window_sec / 2.0
                break

        if new_boundary is None:
            # All sub-windows belong to prev — keep boundary as-is
            continue

        # Clamp to ensure segments stay valid
        new_boundary = max(prev["start"] + min_segment_dur,
                           min(new_boundary, curr["end"] - min_segment_dur))
        new_boundary = round(new_boundary, 3)

        if new_boundary > prev["start"] and new_boundary < curr["end"]:
            prev["end"] = new_boundary   # update BOTH endpoints
            curr["start"] = new_boundary

    # Final sanity pass: ensure no segment has end <= start (discard rather than duplicate)
    valid = []
    for seg in refined:
        if seg.get("end", 0) > seg.get("start", 0):
            valid.append(seg)

    # Collapse same-speaker
    if not valid:
        return refined
    collapsed = [valid[0]]
    for seg in valid[1:]:
        if seg.get("speaker") == collapsed[-1].get("speaker"):
            collapsed[-1]["end"] = seg.get("end", collapsed[-1].get("end"))
        else:
            collapsed.append(seg)

    return collapsed
