import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("asr_mcp.speaker.vad")


def split_at_energy_dips(
    speech_ts: list[dict],
    waveform: "np.ndarray",
    sample_rate: int = 16000,
    min_segment_dur: float = 5.0,
    frame_ms: float = 20.0,
    dip_ratio: float = 0.35,
    min_dip_dur: float = 0.15,
    min_split_piece: float = 1.0,
) -> list[dict]:
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    frame_len = int(frame_ms / 1000 * sample_rate)
    results = []

    for seg in speech_ts:
        start = seg["start"]
        end = seg["end"]
        duration = (end - start) / sample_rate

        if duration < min_segment_dur:
            results.append(seg)
            continue

        segment_audio = waveform[start:end].astype(np.float32)
        energies = []
        for i in range(0, len(segment_audio) - frame_len, frame_len):
            frame = segment_audio[i:i + frame_len]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        if not energies:
            results.append(seg)
            continue

        max_energy = max(energies)
        if max_energy < 1e-8:
            results.append(seg)
            continue

        threshold_energy = max_energy * dip_ratio
        in_dip = False
        dip_start = 0
        splits = []

        for i, e in enumerate(energies):
            t = start + i * frame_len
            if e < threshold_energy:
                if not in_dip:
                    in_dip = True
                    dip_start = t
            else:
                if in_dip:
                    dip_end = t
                    dip_duration = (dip_end - dip_start) / sample_rate
                    if dip_duration >= min_dip_dur:
                        split_point = (dip_start + dip_end) // 2
                        left_dur = (split_point - start) / sample_rate
                        right_dur = (end - split_point) / sample_rate
                        if left_dur >= min_split_piece and right_dur >= min_split_piece:
                            splits.append(split_point)
                    in_dip = False

        if not splits:
            results.append(seg)
            continue

        split_points = [start] + splits + [end]
        for j in range(len(split_points) - 1):
            s = split_points[j]
            e = split_points[j + 1]
            if (e - s) / sample_rate >= min_split_piece:
                results.append({"start": s, "end": e})

    results.sort(key=lambda x: x["start"])
    return results


def run_vad_chunked(
    waveform_tensor: torch.Tensor,
    vad_model=None,
    get_speech_timestamps=None,
    sample_rate: int = 16000,
    chunk_duration: int = 30,
    overlap: int = 5,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
) -> list[dict]:
    if vad_model is not None and get_speech_timestamps is not None:
        return get_speech_timestamps(
            waveform_tensor, vad_model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
        )

    total_samples = waveform_tensor.shape[-1]
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    all_ts = []

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform_tensor[..., start:end]

        chunk_ts = _run_vad_simple(chunk, threshold, min_speech_duration_ms, sample_rate)
        for ts in chunk_ts:
            all_ts.append({
                "start": ts["start"] + start,
                "end": ts["end"] + start,
            })

        start = end - overlap_samples
        if start + overlap_samples >= total_samples:
            break

    return all_ts


def _run_vad_simple(
    waveform: torch.Tensor,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    sample_rate: int = 16000,
) -> list[dict]:
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    frame_size = 512
    hop_size = 256
    audio_np = waveform.numpy().astype(np.float32)

    speech_frames = []
    for i in range(0, len(audio_np) - frame_size, hop_size):
        frame = audio_np[i:i + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        speech_frames.append({
            "start": i,
            "end": i + frame_size,
            "energy": energy,
        })

    if not speech_frames:
        return []

    energies = [f["energy"] for f in speech_frames]
    median_energy = np.median(energies)
    threshold_energy = median_energy * threshold

    speech_regions = []
    in_speech = False
    region_start = 0

    for f in speech_frames:
        if f["energy"] > threshold_energy:
            if not in_speech:
                in_speech = True
                region_start = f["start"]
        else:
            if in_speech:
                in_speech = False
                duration_ms = (f["start"] - region_start) / sample_rate * 1000
                if duration_ms >= min_speech_duration_ms:
                    speech_regions.append({
                        "start": region_start,
                        "end": f["start"],
                    })

    if in_speech:
        last = speech_frames[-1]
        duration_ms = (last["end"] - region_start) / sample_rate * 1000
        if duration_ms >= min_speech_duration_ms:
            speech_regions.append({
                "start": region_start,
                "end": last["end"],
            })

    return speech_regions


def run_vad_onnx(
    waveform: torch.Tensor,
    vad_session,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
) -> list[dict]:
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    audio_np = waveform.numpy().astype(np.float32)
    frame_size = 512
    min_speech_samples = int(min_speech_duration_ms / 1000 * sample_rate)

    input_name = vad_session.get_inputs()[0].name
    input_names = [inp.name for inp in vad_session.get_inputs()]

    h = np.zeros((2, 1, 128), dtype=np.float32)
    c = np.zeros((2, 1, 128), dtype=np.float32)
    sr_np = np.array([sample_rate], dtype=np.int64)

    speech_probs = []
    for i in range(0, len(audio_np) - frame_size, frame_size):
        frame = audio_np[i:i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        feed = {input_name: frame[np.newaxis]}
        if "state" in input_names:
            feed["state"] = h
        if "sr" in input_names:
            feed["sr"] = sr_np
        outputs = vad_session.run(None, feed)
        prob = outputs[0]
        if len(outputs) >= 3:
            h = outputs[1]
            c = outputs[2]
        speech_probs.append({
            "start": i,
            "end": i + frame_size,
            "prob": float(prob[0][0]) if prob.ndim > 0 else float(prob[0]),
        })

    in_speech = False
    region_start = 0
    results = []

    for p in speech_probs:
        if p["prob"] >= threshold:
            if not in_speech:
                in_speech = True
                region_start = p["start"]
        else:
            if in_speech:
                in_speech = False
                if (p["start"] - region_start) >= min_speech_samples:
                    results.append({"start": region_start, "end": p["start"]})

    if in_speech:
        last = speech_probs[-1]
        if (last["end"] - region_start) >= min_speech_samples:
            results.append({"start": region_start, "end": last["end"]})

    return results
