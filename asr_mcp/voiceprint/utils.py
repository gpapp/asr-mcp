import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("asr_mcp.voiceprint.utils")

SAMPLE_RATE = 16000


def load_audio(wav_path: str, target_sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        waveform = torch.from_numpy(data).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        return waveform, target_sr
    return torch.from_numpy(data).unsqueeze(0).float(), sr


def parse_time(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def ensure_wav(file_path: Path, output_dir: Optional[Path] = None) -> Path:
    if file_path.suffix.lower() == ".wav":
        return file_path
    out_dir = output_dir or Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_path.stem}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(file_path), "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-acodec", "pcm_s16le", str(out_path)],
            capture_output=True, check=True,
        )
        return out_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error("ffmpeg conversion failed: %s", e)
        raise


def load_audio_segment(wav_path: str, start_sec: float, end_sec: float):
    waveform, sr = load_audio(wav_path)
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = int(end_sec * SAMPLE_RATE)
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[-1], end_sample)
    return waveform[..., start_sample:end_sample], SAMPLE_RATE


def extract_speaker_audio(
    wav_path: str,
    segments: list,
    speaker_name: str,
    output_path: str,
    min_duration: float = 1.5,
) -> bool:
    try:
        all_chunks = []
        for seg in segments:
            if seg.get("speaker") != speaker_name:
                continue
            duration = seg.get("end", 0) - seg.get("start", 0)
            if duration < min_duration:
                continue
            chunk, _ = load_audio_segment(wav_path, seg["start"], seg["end"])
            all_chunks.append(chunk)

        if not all_chunks:
            logger.warning("No segments found for speaker %s", speaker_name)
            return False

        combined = torch.cat(all_chunks, dim=-1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, combined.numpy().squeeze(), SAMPLE_RATE)
        logger.info("Extracted %.1fs of audio for %s -> %s",
                     combined.shape[-1] / SAMPLE_RATE, speaker_name, output_path)
        return True
    except Exception as e:
        logger.error("Failed to extract audio for %s: %s", speaker_name, e)
        return False
