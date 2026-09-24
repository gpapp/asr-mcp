import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("asr_mcp.speaker.vad")


def merge_vad_sections(segments: list[dict], max_gap_sec: float = 0.1) -> list[dict]:
    """Fuse VAD sections separated by <= max_gap_sec of silence (sorted).

    Args:
        segments: List of {"start": float, "end": float} in seconds.
        max_gap_sec: Maximum gap in seconds to merge across.
    """
    if not segments:
        return []
    segs = sorted(segments, key=lambda x: x["start"])
    merged = [dict(segs[0])]
    for seg in segs[1:]:
        if seg["start"] <= merged[-1]["end"] + max_gap_sec:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


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
    """Split long VAD segments at local energy dips.

    Operates on float-second timestamps: {"start": float, "end": float}.
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    frame_samples = int(frame_ms / 1000 * sample_rate)
    hop_samples = frame_samples
    results = []

    for seg in speech_ts:
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        duration = end_sec - start_sec

        if duration < min_segment_dur:
            results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
            continue

        start_samp = int(start_sec * sample_rate)
        end_samp = int(end_sec * sample_rate)
        segment_audio = waveform[start_samp:end_samp].astype(np.float32)

        energies = []
        for i in range(0, len(segment_audio) - frame_samples, hop_samples):
            frame = segment_audio[i:i + frame_samples]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        if not energies:
            results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
            continue

        median_energy = np.median(energies)
        if median_energy < 1e-8:
            results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
            continue

        threshold_energy = median_energy * dip_ratio
        is_dip = np.array(energies) < threshold_energy

        split_times = []  # seconds relative to segment start
        i = 0
        while i < len(is_dip):
            if is_dip[i]:
                j = i
                while j < len(is_dip) and is_dip[j]:
                    j += 1
                dip_len = j - i
                dip_dur = dip_len * hop_samples / sample_rate
                if dip_dur >= min_dip_dur:
                    center = (i + dip_len // 2) * hop_samples / sample_rate
                    split_times.append(center)
                i = j
            else:
                i += 1

        if not split_times:
            results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
            continue

        boundaries = [0.0] + split_times + [duration]
        pieces = []
        for k in range(len(boundaries) - 1):
            piece_start = start_sec + boundaries[k]
            piece_end = start_sec + boundaries[k + 1]
            if (piece_end - piece_start) >= min_split_piece:
                pieces.append({"start": round(piece_start, 4), "end": round(piece_end, 4)})

        if len(pieces) <= 1:
            results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
        else:
            results.extend(pieces)

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
    """Run VAD on audio in chunks; returns seconds-based segments."""
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
                "start": ts["start"] + start / sample_rate,
                "end": ts["end"] + start / sample_rate,
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
    """Energy-based fallback VAD returning seconds-based segments."""
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
            "start": i / sample_rate,
            "end": (i + frame_size) / sample_rate,
            "energy": energy,
        })

    if not speech_frames:
        return []

    energies = [f["energy"] for f in speech_frames]
    median_energy = np.median(energies)
    threshold_energy = median_energy * threshold

    speech_regions = []
    in_speech = False
    region_start = 0.0

    for f in speech_frames:
        if f["energy"] > threshold_energy:
            if not in_speech:
                in_speech = True
                region_start = f["start"]
        else:
            if in_speech:
                in_speech = False
                duration_ms = (f["start"] - region_start) * 1000
                if duration_ms >= min_speech_duration_ms:
                    speech_regions.append({
                        "start": region_start,
                        "end": f["start"],
                    })

    if in_speech:
        last = speech_frames[-1]
        duration_ms = (last["end"] - region_start) * 1000
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
    merge_close: bool = True,
) -> list[dict]:
    """Run Silero VAD ONNX session frame-by-frame on the waveform.

    Returns segments as {"start": float, "end": float} in **seconds**.

    Args:
        merge_close: When True (default) fuse speech sections separated by
            <=0.1s silence — appropriate for downstream processing.
            When False return raw per-frame speech regions uncollapsed,
            which is needed for exact turn-boundary attribution.
    """
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    audio_np = waveform.numpy().astype(np.float32)
    frame_size = 512
    min_speech_samples = int(min_speech_duration_ms / 1000 * sample_rate)
    min_silence_frames = int(100 / 32)  # 100ms silence threshold to split segments

    input_names = [inp.name for inp in vad_session.get_inputs()]
    input_name = input_names[0]

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
            "start_idx": i,
            "end_idx": i + frame_size,
            "prob": float(prob[0][0]) if prob.ndim > 0 else float(prob[0]),
        })

    # Convert to seconds-based segments using trigger/silence logic
    triggered = False
    temp_end = 0
    current_speech: dict = {}
    speech_segments = []

    for p in speech_probs:
        if p["prob"] >= threshold:
            if not triggered:
                triggered = True
                current_speech = {"start": p["start_idx"] / sample_rate}
            temp_end = 0
        elif triggered:
            temp_end += 1
            if temp_end >= min_silence_frames:
                triggered = False
                end_sec = (speech_probs[speech_probs.index(p) - temp_end + 1]["start_idx"]) / sample_rate
                if "start" in current_speech:
                    dur_samples = end_sec * sample_rate - current_speech["start"] * sample_rate
                    if dur_samples >= min_speech_samples:
                        current_speech["end"] = end_sec
                        speech_segments.append(current_speech)
                current_speech = {}

    if triggered and "start" in current_speech:
        last = speech_probs[-1]
        end_sec = last["end_idx"] / sample_rate
        dur_samples = end_sec * sample_rate - current_speech["start"] * sample_rate
        if dur_samples >= min_speech_samples:
            current_speech["end"] = end_sec
            speech_segments.append(current_speech)

    if not speech_segments:
        return []

    speech_segments.sort(key=lambda x: x["start"])

    if not merge_close:
        return [dict(s) for s in speech_segments]

    return merge_vad_sections(speech_segments, max_gap_sec=0.1)
