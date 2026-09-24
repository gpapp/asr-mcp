import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from asr_mcp.speaker.embedding import compute_pitch, compute_energy

logger = logging.getLogger("asr_mcp.speaker.profiling")


def profile_speakers(
    waveform: torch.Tensor,
    merged_segments: list[dict],
    sample_rate: int = 16000,
) -> dict[str, dict]:
    profiles = {}
    speakers = set(seg.get("speaker", "UNKNOWN") for seg in merged_segments)

    for speaker in speakers:
        speaker_segments = [s for s in merged_segments if s.get("speaker") == speaker]
        if not speaker_segments:
            continue

        all_audio = []
        total_duration = 0.0

        for seg in speaker_segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[-1], end_sample)
            if end_sample > start_sample:
                audio_chunk = waveform[..., start_sample:end_sample]
                all_audio.append(audio_chunk)
                total_duration += (end_sample - start_sample) / sample_rate

        if not all_audio:
            continue

        combined = torch.cat(all_audio, dim=-1)

        pitch_hz, pitch_std = compute_pitch(combined, sample_rate)
        energy_rms = compute_energy(combined)

        profiles[speaker] = {
            "pitch_hz": pitch_hz,
            "pitch_std": pitch_std,
            "energy_rms": energy_rms,
            "total_speech_sec": total_duration,
        }

    return profiles


def relabel_by_pitch(
    merged_segments: list[dict],
    profiles: dict[str, dict],
) -> Tuple[list[dict], dict[str, dict], dict[str, str]]:
    if not profiles:
        return merged_segments, profiles, {}

    sorted_speakers = sorted(
        profiles.keys(),
        key=lambda s: profiles[s].get("pitch_hz", 0) if profiles[s].get("pitch_hz", 0) > 0 else 9999,
    )

    label_map = {}
    for new_idx, old_label in enumerate(sorted_speakers):
        new_label = f"Speaker {new_idx + 1}"
        label_map[old_label] = new_label

    new_segments = []
    for seg in merged_segments:
        old_speaker = seg.get("speaker", "UNKNOWN")
        new_seg = {**seg, "speaker": label_map.get(old_speaker, old_speaker)}
        new_segments.append(new_seg)

    new_profiles = {}
    for old_label, new_label in label_map.items():
        if old_label in profiles:
            new_profiles[new_label] = profiles[old_label]

    return new_segments, new_profiles, label_map
