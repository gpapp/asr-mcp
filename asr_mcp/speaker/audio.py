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
    search_sec: float = 1.2,
    sub_window_sec: float = 0.5,
    sub_stride_sec: float = 0.2,
    min_segment_dur: float = 0.3,
    embedding_cache=None,
) -> list[dict]:
    if not segments or len(segments) < 2:
        return segments

    from asr_mcp.speaker.embedding import extract_embedding

    refined = [segments[0].copy()]

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        boundary = curr["start"]

        search_start = max(prev["start"], boundary - search_sec)
        search_end = min(curr["end"], boundary + search_sec)

        if search_end - search_start < sub_window_sec:
            refined.append(curr.copy())
            continue

        sub_samples_start = int(search_start * sample_rate)
        sub_samples_end = int(search_end * sample_rate)
        sub_audio = waveform[..., sub_samples_start:sub_samples_end]

        sub_windows = generate_sliding_windows(
            sub_audio, sample_rate,
            window_sec=sub_window_sec, stride_sec=sub_stride_sec,
        )

        best_switch = boundary
        best_dist = float("inf")

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

            prev_emb = cluster_centroids.get(prev.get("speaker"))
            curr_emb = cluster_centroids.get(curr.get("speaker"))

            if prev_emb is not None and curr_emb is not None:
                d_prev = 1.0 - float(np.dot(emb, prev_emb) / (np.linalg.norm(emb) * np.linalg.norm(prev_emb) + 1e-8))
                d_curr = 1.0 - float(np.dot(emb, curr_emb) / (np.linalg.norm(emb) * np.linalg.norm(curr_emb) + 1e-8))

                if d_prev < d_curr:
                    dist = d_prev
                else:
                    dist = -d_curr

                if dist < best_dist:
                    best_dist = dist
                    best_switch = (search_start + sw["start_sec"] + sub_window_sec / 2)

        refined.append({
            **curr,
            "start": round(best_switch, 3),
        })

    for i in range(1, len(refined)):
        if refined[i]["start"] < refined[i - 1]["start"]:
            refined[i]["start"] = refined[i - 1]["start"]
        if refined[i].get("end", 0) <= refined[i]["start"]:
            refined[i] = refined[i - 1]

    collapsed = [refined[0]]
    for seg in refined[1:]:
        if seg.get("speaker") == collapsed[-1].get("speaker"):
            collapsed[-1]["end"] = seg.get("end", collapsed[-1].get("end"))
        else:
            collapsed.append(seg)

    return collapsed
