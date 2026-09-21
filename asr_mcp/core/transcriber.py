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

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128

MAX_ENCODER_SEC = 30.0


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


def _parse_encoder_outputs(encoder_outputs, encoder_session):
    enc_output_names = [o.name for o in encoder_session.get_outputs()]
    cross_kv = {}
    hidden_states = None

    if len(encoder_outputs) == 2 and "cross" in str(enc_output_names).lower():
        cross_k = encoder_outputs[0]
        cross_v = encoder_outputs[1]
        num_layers_enc = cross_k.shape[0]
        logger.debug("Encoder cross KV stacked format: layers=%d, shape=%s", num_layers_enc, cross_k.shape)

        for i in range(num_layers_enc):
            k = cross_k[i].reshape(1, NUM_HEADS, -1, HEAD_DIM).astype(np.float32)
            v = cross_v[i].reshape(1, NUM_HEADS, -1, HEAD_DIM).astype(np.float32)
            cross_kv[f"past_key_values.{i}.encoder.key"] = k
            cross_kv[f"past_key_values.{i}.encoder.value"] = v
    else:
        for name, val in zip(enc_output_names, encoder_outputs):
            val = val.astype(np.float32)
            if "encoder" in name.lower() and "key" not in name.lower() and "value" not in name.lower():
                hidden_states = val
            if "key" in name.lower() and "encoder" in name.lower():
                cross_kv[name] = val
            elif "value" in name.lower() and "encoder" in name.lower():
                cross_kv[name] = val
            else:
                hidden_states = val

    if hidden_states is None and not cross_kv:
        hidden_states = encoder_outputs[0].astype(np.float32)

    if not cross_kv:
        for i in range(NUM_LAYERS):
            cross_kv[f"past_key_values.{i}.encoder.key"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            cross_kv[f"past_key_values.{i}.encoder.value"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )

    return hidden_states, cross_kv


def _init_self_kv_cache():
    self_kv = {}
    for i in range(NUM_LAYERS):
        self_kv[f"past_key_values.{i}.decoder.key"] = np.zeros(
            (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
        )
        self_kv[f"past_key_values.{i}.decoder.value"] = np.zeros(
            (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
        )
    return self_kv


def _build_decoder_inputs(dec_input_names, input_ids, position, cross_kv, self_kv, encoder_hidden_states=None, num_logits_to_keep=1):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    inputs = {}
    inputs["input_ids"] = input_ids.astype(np.int64)
    inputs["attention_mask"] = np.ones((batch_size, seq_len), dtype=np.int64)
    inputs["position_ids"] = np.arange(position, position + seq_len, dtype=np.int64).reshape(1, -1)
    inputs["num_logits_to_keep"] = np.array(num_logits_to_keep, dtype=np.int64)

    if encoder_hidden_states is not None:
        inputs["encoder_hidden_states"] = encoder_hidden_states.astype(np.float32)

    inputs.update(cross_kv)
    inputs.update(self_kv)

    feed = {}
    for name in dec_input_names:
        if name in inputs:
            feed[name] = inputs[name]

    return feed


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

    max_frames = int(MAX_ENCODER_SEC * (SAMPLE_RATE / HOP_LENGTH))

    try:
        encoder_input_name = state.encoder_session.get_inputs()[0].name
        if mel_spectrogram.shape[0] <= max_frames:
            encoder_outputs = state.encoder_session.run(
                None, {encoder_input_name: mel_spectrogram[np.newaxis]}
            )
        else:
            enc_output_names = [o.name for o in state.encoder_session.get_outputs()]
            all_parts = {name: [] for name in enc_output_names}
            overlap_frames = max_frames // 4
            pos = 0

            while pos < mel_spectrogram.shape[0]:
                end = min(pos + max_frames, mel_spectrogram.shape[0])
                chunk = mel_spectrogram[pos:end]
                chunk_out = state.encoder_session.run(
                    None, {encoder_input_name: chunk[np.newaxis]}
                )
                for name, val in zip(enc_output_names, chunk_out):
                    arr = val.astype(np.float32)
                    if arr.ndim >= 3 and pos > 0:
                        trim = min(overlap_frames, arr.shape[-2])
                        arr = arr[..., trim:, :]
                    all_parts[name].append(arr)
                pos = end - overlap_frames if end < mel_spectrogram.shape[0] else mel_spectrogram.shape[0]

            encoder_outputs = []
            for name in enc_output_names:
                parts = all_parts[name]
                arr = parts[0]
                if arr.ndim >= 3 and arr.shape[-2] > 1:
                    encoder_outputs.append(np.concatenate(parts, axis=-2))
                else:
                    encoder_outputs.append(parts[-1])
            logger.info("Chunked encoder for %d mel frames", mel_spectrogram.shape[0])
    except Exception as e:
        logger.error("Encoder inference failed: %s", e)
        return {"text": "", "error": str(e)}

    encoder_hidden_states, cross_kv = _parse_encoder_outputs(encoder_outputs, state.encoder_session)

    if past_kv_cache_ort is not None:
        self_kv = past_kv_cache_ort.get("self_kv", _init_self_kv_cache())
    else:
        self_kv = _init_self_kv_cache()

    dec_input_names = [inp.name for inp in state.decoder_session.get_inputs()]
    dec_output_names = [out.name for out in state.decoder_session.get_outputs()]

    if "encoder_hidden_states" not in dec_input_names:
        encoder_hidden_states = None

    if prefix_ids is not None:
        input_ids = np.array([prefix_ids], dtype=np.int64)
    else:
        input_ids = np.array([state.prompt_ids], dtype=np.int64)

    position = 0
    generated_tokens = []
    max_new = state.settings.max_new_tokens if state.settings else 448

    for step in range(max_new):
        feed = _build_decoder_inputs(
            dec_input_names, input_ids, position, cross_kv, self_kv,
            encoder_hidden_states=encoder_hidden_states,
        )

        try:
            outputs = state.decoder_session.run(None, feed)
        except Exception as e:
            logger.error("Decoder step %d failed: %s", step, e)
            logger.error("Feed keys: %s", list(feed.keys()))
            for k, v in feed.items():
                logger.error("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)
            logger.error("Expected decoder input names: %s", dec_input_names)
            break

        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))

        new_self_kv = {}
        for i, name in enumerate(dec_output_names):
            if name == "logits":
                continue
            # Map HuggingFace ONNX output names (present.*) back to input names (past_key_values.*)
            mapped_name = name.replace("present.", "past_key_values.") if name.startswith("present.") else name
            new_self_kv[mapped_name] = outputs[i]
        if new_self_kv:
            self_kv = new_self_kv

        if next_token == state.eos_token_id:
            break

        generated_tokens.append(next_token)
        input_ids = np.array([[next_token]], dtype=np.int64)
        position += 1

    if state.tokenizer:
        text = state.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        text = "".join(chr(t) if 32 <= t < 127 else "" for t in generated_tokens)
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
