import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("asr_mcp.speaker.profiling")


def _extract_mfcc_stats(chunk: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    try:
        import torchaudio.compliance.kaldi as kaldi
        waveform = torch.from_numpy(chunk).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mfcc = kaldi.mfcc(waveform, sample_frequency=sr, num_ceps=n_mfcc, num_mel_bins=26)
        mfcc_np = mfcc.numpy()
        stats = {}
        for i in range(min(n_mfcc, mfcc_np.shape[1])):
            stats[f"mfcc{i}_mean"] = float(np.mean(mfcc_np[:, i]))
            stats[f"mfcc{i}_std"] = float(np.std(mfcc_np[:, i]))
        return stats
    except Exception as e:
        logger.warning("MFCC extraction failed: %s", e)
        return {f"mfcc{i}_mean": 0.0 for i in range(n_mfcc)} | {f"mfcc{i}_std": 0.0 for i in range(n_mfcc)}


def _extract_spectral_features(chunk: np.ndarray, sr: int) -> dict:
    try:
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(chunk, fs=sr, nperseg=512, noverlap=256)
        freqs = f[:, np.newaxis]
        spectral_centroid = float(np.sum(freqs * Sxx) / (np.sum(Sxx) + 1e-8))
        cumulative = np.cumsum(Sxx, axis=0)
        total = cumulative[-1]
        rolloff_idx = np.searchsorted(cumulative[:, 0], 0.85 * total[0])
        spectral_rolloff = float(f[min(rolloff_idx, len(f) - 1)])
        return {"spectral_centroid": spectral_centroid, "spectral_rolloff": spectral_rolloff}
    except Exception as e:
        logger.warning("Spectral extraction failed: %s", e)
        return {"spectral_centroid": 0.0, "spectral_rolloff": 0.0}


def profile_speakers(
    waveform: torch.Tensor,
    merged_segments: list[dict],
    sample_rate: int = 16000,
) -> dict[str, dict]:
    from asr_mcp.speaker.embedding import compute_pitch, compute_energy

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

        audio_np = combined.numpy().squeeze() if combined.dim() > 1 else combined.numpy()
        spectral = _extract_spectral_features(audio_np, sample_rate)
        mfcc_stats = _extract_mfcc_stats(audio_np, sample_rate)

        profiles[speaker] = {
            "pitch_hz": pitch_hz,
            "pitch_std": pitch_std,
            "energy_rms": energy_rms,
            "spectral_centroid": spectral.get("spectral_centroid", 0.0),
            "spectral_rolloff": spectral.get("spectral_rolloff", 0.0),
            "total_speech_sec": total_duration,
            "mfcc": mfcc_stats,
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
        key=lambda s: profiles[s].get("pitch_hz", 0),
    )

    label_map = {}
    for new_idx, old_label in enumerate(sorted_speakers):
        new_label = f"SPEAKER_{new_idx:02d}"
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
