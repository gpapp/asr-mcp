import asyncio
import logging
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional

import numpy as np
import librosa
from scipy.signal import spectrogram

from asr_mcp.core.model_state import state

logger = logging.getLogger("asr_mcp.core.transcriber")

_mel_filterbank_cache: Optional[np.ndarray] = None
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
PREEMPHASIS = 0.97
DITHER = 1e-5


def _get_mel_filterbank() -> np.ndarray:
    global _mel_filterbank_cache
    if _mel_filterbank_cache is not None:
        return _mel_filterbank_cache
    _mel_filterbank_cache = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        fmin=0.0, fmax=SAMPLE_RATE / 2, norm="slaney",
    ).astype(np.float32)
    return _mel_filterbank_cache


def _preemphasis(audio: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    return np.concatenate([[audio[0]], audio[1:] - coeff * audio[:-1]])


def _compute_mel_spectrogram_fast(audio: np.ndarray) -> np.ndarray:
    if DITHER > 0:
        rng = np.random.RandomState(abs(hash(audio.ctypes.data) % (2**31)))
        audio = audio + rng.randn(len(audio)).astype(np.float32) * DITHER

    audio = _preemphasis(audio)

    win = np.hanning(WIN_LENGTH)

    f, t, Sxx = spectrogram(
        audio, fs=SAMPLE_RATE, nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT,
        window=win, scaling="spectrum",
    )

    Sxx = np.maximum(Sxx, 1e-10)
    mel_fb = _get_mel_filterbank()
    mel_spec = mel_fb @ Sxx
    log_mel = np.log(mel_spec + 1e-8).astype(np.float32)

    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-5
    log_mel = ((log_mel - mean) / std).astype(np.float32)

    return log_mel.T


@contextmanager
def inference_timeout(seconds: int):
    if sys.platform != "win32" and hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise TimeoutError(f"Inference timed out after {seconds}s")
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


def _trim_partial_prefix(before: str, unit: str) -> str:
    if not before:
        return ""
    trimmed = before.rstrip()
    if unit == "word" and trimmed and not trimmed[-1].isspace():
        parts = trimmed.rsplit(" ", 1)
        if len(parts) > 1:
            return parts[0] + " "
        return ""
    return trimmed + " " if trimmed else ""


def _trim_partial_suffix(after: str, unit: str) -> str:
    if not after:
        return ""
    trimmed = after.lstrip()
    if unit == "word" and trimmed and not trimmed[0].isspace():
        parts = trimmed.split(" ", 1)
        if len(parts) > 1:
            return " " + parts[1]
        return ""
    return " " + trimmed if trimmed else ""


def clean_transcript(text: str) -> str:
    pattern = r'(.{2,}?)\1{2,}'
    cleaned = re.sub(pattern, r'\1', text)
    cleaned = re.sub(r'\[inaudible\](\s*\[inaudible\])+', '[inaudible]', cleaned)
    return cleaned.strip()


def transcribe_audio_sync(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None,
) -> dict:
    start_time = time.time()

    if audio is not None and mel_spectrogram is None:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        mel_spectrogram = _compute_mel_spectrogram_fast(audio)

    if mel_spectrogram is None:
        return {"text": "", "error": "No audio provided"}

    try:
        encoder_input_name = state.encoder_session.get_inputs()[0].name
        encoder_outputs = state.encoder_session.run(
            None, {encoder_input_name: mel_spectrogram[np.newaxis]}
        )
    except Exception as e:
        logger.error("Encoder inference failed: %s", e)
        return {"text": "", "error": str(e)}

    input_ids = state.prompt_ids if prefix_ids is None else np.array([prefix_ids], dtype=np.int64)

    if past_kv_cache_ort is not None:
        past = past_kv_cache_ort
    else:
        past = {}
        for i in range(len(state.encoder_session.get_outputs()) // 2):
            past[f"past_key_values.{i}.key"] = np.zeros((1, 8, 0, 128), dtype=np.float32)
            past[f"past_key_values.{i}.value"] = np.zeros((1, 8, 0, 128), dtype=np.float32)

    generated_tokens = []
    max_new = state.settings.max_new_tokens if state.settings else 448

    for step in range(max_new):
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_outputs[0],
        }
        for k, v in past.items():
            decoder_inputs[k] = v

        try:
            outputs = state.decoder_session.run(None, decoder_inputs)
        except Exception as e:
            logger.error("Decoder step %d failed: %s", step, e)
            break

        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        past = {}
        output_names = [o.name for o in state.decoder_session.get_outputs()]
        for i, name in enumerate(output_names[1:]):
            past[name] = outputs[i + 1]

        if next_token == state.tokens.get("eos_token_id", 2):
            break

        if next_token >= 3 and next_token < len(state.tokens.get("added_tokens_decoder", {})):
            generated_tokens.append(next_token)
        else:
            generated_tokens.append(next_token)

        input_ids = np.array([[next_token]], dtype=np.int64)

    text = state.tokens.decode(generated_tokens) if state.tokens else ""
    text = clean_transcript(text)

    inference_time = time.time() - start_time
    audio_duration = len(mel_spectrogram[0]) * HOP_LENGTH / SAMPLE_RATE if mel_spectrogram is not None else 0

    return {
        "text": text,
        "audio_duration_sec": round(audio_duration, 2),
        "inference_time_sec": round(inference_time, 2),
        "tokens_generated": len(generated_tokens),
    }


async def transcribe_audio_async(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None,
) -> dict:
    loop = asyncio.get_event_loop()
    from asr_mcp.core.model_state import executor as _executor
    pool = _executor or ThreadPoolExecutor(max_workers=2)
    return await loop.run_in_executor(
        pool,
        lambda: transcribe_audio_sync(
            audio=audio, language=language, timeout_sec=timeout_sec,
            mel_spectrogram=mel_spectrogram,
            past_kv_cache_ort=past_kv_cache_ort,
            prefix_ids=prefix_ids,
        ),
    )
