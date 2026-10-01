import hashlib
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


def generate_segment_hash(audio_path: str) -> str:
    """Generate a short 6-char alphanumeric hash from audio filename (stable across runs)."""
    key = Path(audio_path).stem
    hash_int = int(hashlib.md5(key.encode()).hexdigest(), 16)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result = []
    for _ in range(6):
        hash_int, idx = divmod(hash_int, 32)
        result.append(chars[idx])
    return "".join(result)


def format_time_short(seconds: float) -> str:
    """Format seconds as MM-SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}-{secs:02d}"


def load_audio(wav_path: str, target_sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, int]:
    p = Path(wav_path)
    if p.suffix.lower() != ".wav":
        wav_path = str(ensure_wav(p))
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
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(file_path), "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-acodec", "pcm_s16le", str(out_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.error("ffmpeg failed for %s (rc=%d): %s", file_path.name, result.returncode, stderr[-500:])
            raise RuntimeError(f"ffmpeg conversion failed for {file_path.name}: {stderr[-200:]}")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced empty output for {file_path.name}")
        return out_path
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — install ffmpeg for audio format conversion")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out converting {file_path.name} (>120s)")


def convert_to_wav(input_path: str, output_dir: Optional[Path] = None) -> Path:
    """Convert any audio/video format to WAV. Returns path to WAV file.
    Caller is responsible for cleanup."""
    p = Path(input_path)
    if p.suffix.lower() == ".wav":
        return p
    return ensure_wav(p, output_dir)


def slice_audio(waveform: torch.Tensor, start_sec: float, end_sec: float,
                sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """In-memory slice of an ALREADY-LOADED waveform.

    Split out from :func:`load_audio_segment` so a caller that needs many
    segments of one file can decode the file once (see
    :class:`AudioSegmentSource`).  Semantics are identical to the old
    inline arithmetic: the slice is clamped to the waveform and an end before
    the start yields an empty tensor rather than an exception.
    """
    total = int(waveform.shape[-1])
    start_sample = max(0, int(start_sec * sample_rate))
    end_sample = min(total, int(end_sec * sample_rate))
    if end_sample < start_sample:
        end_sample = start_sample
    return waveform[..., start_sample:end_sample]


class AudioSegmentSource:
    """One decode per source file; every segment after that is a memory slice.

    ``load_audio_segment`` decoded the **entire** file on every call, so a
    cluster with 89 snippets cost 89 whole-file decodes (docs/plans/
    boundary-and-voice-quality-plan.md, F4).  A single request builds one of
    these and reuses it, which turns that into one decode plus 89 slices.

    ``loads`` counts the actual decodes so a test can assert it is 1.
    """

    def __init__(self, wav_path: str, target_sr: int = SAMPLE_RATE):
        self.path = wav_path
        self.target_sr = int(target_sr)
        self.loads = 0
        self._waveform = None
        self._sr = self.target_sr

    @property
    def loaded(self) -> bool:
        return self._waveform is not None

    def waveform(self) -> tuple[torch.Tensor, int]:
        if self._waveform is None:
            self.loads += 1
            self._waveform, self._sr = load_audio(self.path, self.target_sr)
        return self._waveform, self._sr

    def segment(self, start_sec: float, end_sec: float) -> tuple[torch.Tensor, int]:
        waveform, sr = self.waveform()
        return slice_audio(waveform, start_sec, end_sec, sr), sr


def load_audio_segment(wav_path: str, start_sec: float, end_sec: float):
    return AudioSegmentSource(wav_path).segment(start_sec, end_sec)


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
        sf.write(output_path, combined.numpy().squeeze(), SAMPLE_RATE, format="FLAC")
        logger.info("Extracted %.1fs of audio for %s -> %s",
                     combined.shape[-1] / SAMPLE_RATE, speaker_name, output_path)
        return True
    except Exception as e:
        logger.error("Failed to extract audio for %s: %s", speaker_name, e)
        return False
