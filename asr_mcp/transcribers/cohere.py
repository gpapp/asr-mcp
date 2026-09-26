"""Cohere Transcribe ASR backend (ONNX Runtime)."""

import asyncio
import logging
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import onnxruntime as ort
from scipy.signal import spectrogram

from asr_mcp.transcribers.base import ASRBackend
from asr_mcp.config.settings import Settings
from asr_mcp.core.model_loader import (
    get_session_options,
    _get_providers,
    ensure_model,
)
from asr_mcp.core.model_state import (
    is_gpu_oom,
    log_gpu_memory,
    GPU_SHRINK_RUN_OPTIONS,
)

logger = logging.getLogger("asr_mcp.transcribers.cohere")

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 160
PREEMPHASIS = 0.97
DITHER = 0.0

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128

MAX_ENCODER_SEC = 30.0


def _preemphasis(audio: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    return np.concatenate([[audio[0]], audio[1:] - coeff * audio[:-1]])


def _trim_partial_prefix(before: str, unit: str) -> str:
    words = unit.lower().split()
    b = before.rstrip()
    b_lower = b.lower()
    for start in range(len(words)):
        suffix = " ".join(words[start:])
        if b_lower.endswith(suffix):
            return b[: len(b) - len(suffix)].rstrip()
    return b


def _trim_partial_suffix(after: str, unit: str) -> str:
    words = unit.lower().split()
    a = after.lstrip()
    a_lower = a.lower()
    for end in range(len(words), 0, -1):
        prefix = " ".join(words[:end])
        if a_lower.startswith(prefix):
            return a[len(prefix):].lstrip()
    return a


_LOOP_RE = re.compile(r'(.{4,120}?)(?:\s+\1){2,}', re.IGNORECASE)


def clean_transcript(text: str) -> str:
    text = re.sub(r'\b(\w{3,})\s+\1\b', r'\1', text)
    text = re.sub(r'\b(\w{2})\s+\1\b', r'\1', text)
    prev = None
    while prev != text:
        prev = text
        m = _LOOP_RE.search(text)
        if not m:
            break
        unit = m.group(1)
        before = _trim_partial_prefix(text[:m.start()], unit)
        after = _trim_partial_suffix(text[m.end():], unit)
        parts = [p for p in (before, '[inaudible]', after) if p]
        text = ' '.join(parts)
    text = re.sub(r'(\[inaudible\]\s*){2,}', '[inaudible] ', text)
    text = re.sub(
        r'\[inaudible\]\s+(?:\w[\w\s,\']{0,80}?)\s+\[inaudible\]',
        '[inaudible]',
        text,
    )
    return text.strip()


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


class CohereBackend(ASRBackend):
    """ONNX Cohere Transcribe backend; owns encoder/decoder/tokenizer sessions."""

    name = "cohere"

    def __init__(self):
        super().__init__()
        self.encoder_session = None
        self.decoder_session = None
        self.tokenizer = None
        self.tokens = None
        self.prompt_ids = None
        self.eos_token_id = 3
        self.settings = None
        self._cpu_encoder_session = None
        self._mel_filterbank_cache = None

    # ---------- model loading ----------

    def load(self, settings: Settings) -> None:
        so = get_session_options(settings)

        if self.encoder_session is None:
            model_dir = ensure_model(settings)
            encoder_file = f"onnx/encoder_model{settings.encoder_model_type}.onnx"
            enc_providers = _get_providers(settings, "encoder")
            logger.info("Loading encoder session (providers=%s)", enc_providers)
            self.encoder_session = ort.InferenceSession(
                str(model_dir / encoder_file), sess_options=so, providers=enc_providers
            )
            logger.info("Encoder inputs: %s", [i.name for i in self.encoder_session.get_inputs()])
            logger.info("Encoder outputs: %s", [o.name for o in self.encoder_session.get_outputs()])

        if self.decoder_session is None:
            model_dir = ensure_model(settings)
            decoder_file = f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx"
            logger.info("Loading decoder session (CPU only)")
            self.decoder_session = ort.InferenceSession(
                str(model_dir / decoder_file), sess_options=so, providers=["CPUExecutionProvider"]
            )
            logger.info("Decoder inputs: %s", [i.name for i in self.decoder_session.get_inputs()])
            logger.info("Decoder outputs: %s", [o.name for o in self.decoder_session.get_outputs()])

        if self.tokenizer is None:
            model_dir = ensure_model(settings)
            self.tokenizer = _load_tokenizer(model_dir)
            if self.tokenizer:
                self.prompt_ids = _build_prompt_ids(self.tokenizer)
                token_to_id = self.tokenizer.get_vocab()
                self.tokens = {v: k for k, v in token_to_id.items()}
                self.eos_token_id = token_to_id.get("endoftext", 3)
                logger.info("Decoder prompt token IDs: %s", self.prompt_ids)
                logger.info("EOS token ID: %d", self.eos_token_id)
            else:
                self.prompt_ids = [13764, 13902, 14190, 14021, 14074, 14254, 13912]
                self.eos_token_id = 3
                logger.warning("Tokenizer not loaded, using hardcoded prompt IDs: %s", self.prompt_ids)

        self.settings = settings
        self.loaded = True
        logger.info("Cohere backend loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return bool(
            self.loaded
            and self.encoder_session is not None
            and self.decoder_session is not None
            and self.tokenizer is not None
        )

    def unload(self) -> None:
        import gc
        self.encoder_session = None
        self.decoder_session = None
        self.tokenizer = None
        self.tokens = None
        self._cpu_encoder_session = None
        self.loaded = False
        gc.collect()

    def unload_encoder(self) -> None:
        """Free only the encoder session (diarize never uses it)."""
        import gc
        if self.encoder_session is None:
            return
        logger.info("Unloading encoder session (VRAM)")
        self.encoder_session = None
        gc.collect()
        log_gpu_memory("encoder unloaded")

    def reload_encoder(self, force_cpu: bool = False) -> None:
        """Rebuild the encoder session from scratch (arena-reset on OOM)."""
        import gc
        so = get_session_options(self.settings)
        model_dir = Path(self.settings.model_dir)
        encoder_file = f"onnx/encoder_model{self.settings.encoder_model_type}.onnx"
        providers = ["CPUExecutionProvider"] if force_cpu else _get_providers(self.settings, "encoder")
        self.encoder_session = None
        gc.collect()
        self.encoder_session = ort.InferenceSession(
            str(model_dir / encoder_file), sess_options=so, providers=providers
        )

    def _get_cpu_encoder_session(self) -> ort.InferenceSession:
        if self._cpu_encoder_session is None:
            model_dir = Path(self.settings.model_dir)
            encoder_file = f"onnx/encoder_model{self.settings.encoder_model_type}.onnx"
            model_path = str(model_dir / encoder_file)
            cpu_so = get_session_options(self.settings)
            logger.warning("Loading CPU encoder fallback session: %s", model_path)
            self._cpu_encoder_session = ort.InferenceSession(
                model_path, sess_options=cpu_so, providers=["CPUExecutionProvider"],
            )
        return self._cpu_encoder_session

    def _run_encoder(self, feed: dict):
        """Run encoder on GPU with arena-reset retry, then CPU fallback on OOM."""
        try:
            return self.encoder_session.run(None, feed, run_options=GPU_SHRINK_RUN_OPTIONS)
        except Exception as e:
            if not is_gpu_oom(e):
                raise
            logger.warning("GPU OOM on encoder, reloading fresh arena: %s", e)
            log_gpu_memory("encoder OOM before reload")
            try:
                self.reload_encoder()
                log_gpu_memory("encoder OOM after reload")
                return self.encoder_session.run(None, feed, run_options=GPU_SHRINK_RUN_OPTIONS)
            except Exception as e2:
                if not is_gpu_oom(e2):
                    raise
                logger.warning("GPU OOM persisted after arena reset, falling back to CPU encoder: %s", e2)
                log_gpu_memory("encoder CPU fallback")
                return self._get_cpu_encoder_session().run(None, feed)

    # ---------- feature extraction ----------

    def _get_mel_filterbank(self) -> np.ndarray:
        if self._mel_filterbank_cache is not None:
            return self._mel_filterbank_cache
        self._mel_filterbank_cache = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        ).astype(np.float32)
        return self._mel_filterbank_cache

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        if DITHER > 0:
            rng = np.random.RandomState(abs(hash(audio.ctypes.data) % (2**31)))
            audio = audio + rng.randn(len(audio)).astype(np.float32) * DITHER

        audio = _preemphasis(audio)

        f, t, Sxx = spectrogram(
            audio, fs=SAMPLE_RATE, window='hann',
            nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
            return_onesided=True, mode='magnitude',
        )

        mel_fb = self._get_mel_filterbank()
        mel_spec = mel_fb @ Sxx

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mean = mel_spec_db.mean(axis=1, keepdims=True)
        std = mel_spec_db.std(axis=1, keepdims=True) + 1e-8
        mel_spec_db = ((mel_spec_db - mean) / std).astype(np.float32)

        return mel_spec_db.T

    # ---------- decode helpers ----------

    def _parse_encoder_outputs(self, encoder_outputs, encoder_session):
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
                low = name.lower()
                if ("encoder" in low and "key" not in low and "value" not in low) or \
                   low in ("hidden_states", "last_hidden_state", "encoder_output", "encoder_out", "memory", "context"):
                    hidden_states = val
                elif "key" in low and "encoder" in low:
                    cross_kv[name] = val
                elif "value" in low and "encoder" in low:
                    cross_kv[name] = val

        if hidden_states is None:
            hidden_states = encoder_outputs[0].astype(np.float32)
            logger.warning("No explicit hidden_states output found; using encoder_outputs[0] as hidden_states")

        if not cross_kv:
            for i in range(NUM_LAYERS):
                cross_kv[f"past_key_values.{i}.encoder.key"] = np.zeros(
                    (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
                )
                cross_kv[f"past_key_values.{i}.encoder.value"] = np.zeros(
                    (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
                )

        return hidden_states, cross_kv

    def _init_self_kv_cache(self):
        self_kv = {}
        for i in range(NUM_LAYERS):
            self_kv[f"past_key_values.{i}.decoder.key"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            self_kv[f"past_key_values.{i}.decoder.value"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
        return self_kv

    def _build_decoder_inputs(self, dec_input_names, input_ids, position, past_seq_len, tokens_this_call, cross_kv, self_kv, encoder_hidden_states=None, enc_hs_input_name=None, num_logits_to_keep=1):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        total_seq_len = past_seq_len + tokens_this_call + seq_len

        inputs = {}
        inputs["input_ids"] = input_ids.astype(np.int64)
        inputs["attention_mask"] = np.ones((batch_size, total_seq_len), dtype=np.int64)
        inputs["position_ids"] = np.arange(position, position + seq_len, dtype=np.int64).reshape(1, -1)
        inputs["num_logits_to_keep"] = np.array(num_logits_to_keep, dtype=np.int64)

        if encoder_hidden_states is not None and enc_hs_input_name is not None:
            inputs[enc_hs_input_name] = encoder_hidden_states.astype(np.float32)

        inputs.update(cross_kv)
        inputs.update(self_kv)

        feed = {}
        for name in dec_input_names:
            if name in inputs:
                feed[name] = inputs[name]

        return feed

    def _plan_window_bounds(self, mel: np.ndarray, window_frames: int, min_tail: int = 300, snap: int = 100) -> list[int]:
        total = mel.shape[0]
        bounds = [0]
        pos = 0
        while total - pos > window_frames:
            target = pos + window_frames
            if total - target < min_tail:
                break
            lo = max(pos + window_frames // 2, target - snap)
            energies = mel[lo:target].sum(axis=1)
            cut = lo + int(np.argmin(energies))
            if cut <= pos:
                cut = target
            bounds.append(cut)
            pos = cut
        bounds.append(total)
        return bounds

    def _transcribe_windowed(
        self,
        mel: np.ndarray,
        language: str = "en",
        timeout_sec: int = 120,
        max_frames: Optional[int] = None,
        progress_cb: Optional[callable] = None,
    ) -> dict:
        if max_frames is None:
            max_frames = int(MAX_ENCODER_SEC * (SAMPLE_RATE / HOP_LENGTH))
        bounds = self._plan_window_bounds(mel, max_frames)
        n_windows = len(bounds) - 1
        logger.info("Windowing long audio: %d frames -> %d windows", mel.shape[0], n_windows)

        segments_out = []
        text_parts = []
        tokens_total = 0
        inference_total = 0.0
        errors = []

        for i in range(n_windows):
            s, e = bounds[i], bounds[i + 1]
            off = s * HOP_LENGTH / SAMPLE_RATE
            if progress_cb:
                try:
                    progress_cb(i + 1, n_windows)
                except Exception:
                    logger.debug("progress_cb failed", exc_info=True)
            r = self.transcribe_audio_sync(
                mel_spectrogram=mel[s:e],
                language=language,
                timeout_sec=timeout_sec,
                _no_window=True,
            )
            if r.get("error"):
                errors.append(f"window {i} ({off:.1f}s): {r['error']}")
                logger.error("Window %d transcription failed: %s", i, r["error"])
                continue
            if r.get("text"):
                text_parts.append(r["text"].strip())
            for seg in r.get("segments") or []:
                segments_out.append({
                    "start": round(seg["start"] + off, 3),
                    "end": round(seg["end"] + off, 3),
                    "text": seg["text"],
                })
            tokens_total += r.get("tokens_generated", 0)
            inference_total += r.get("inference_time_sec", 0.0)

        audio_duration = mel.shape[0] * HOP_LENGTH / SAMPLE_RATE
        text = " ".join(p for p in text_parts if p).strip()

        result = {
            "text": text,
            "segments": segments_out if segments_out else None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(inference_total, 2),
            "tokens_generated": tokens_total,
        }
        if errors:
            result["error"] = "; ".join(errors)
        return result

    # ---------- transcription ----------

    def transcribe_audio_sync(
        self,
        audio: Optional[np.ndarray] = None,
        language: str = "en",
        timeout_sec: int = 120,
        mel_spectrogram: Optional[np.ndarray] = None,
        past_kv_cache_ort: Optional[dict] = None,
        prefix_ids: Optional[list[int]] = None,
        _no_window: bool = False,
        progress_cb: Optional[callable] = None,
        context: str = "",
    ) -> dict:
        start_time = time.time()

        if audio is not None and mel_spectrogram is None:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            mel_spectrogram = self.compute_mel_spectrogram(audio)

        if mel_spectrogram is None:
            return {"text": "", "error": "No audio provided"}

        max_frames = int(MAX_ENCODER_SEC * (SAMPLE_RATE / HOP_LENGTH))

        if (
            not _no_window
            and past_kv_cache_ort is None
            and prefix_ids is None
            and mel_spectrogram.shape[0] > max_frames
        ):
            return self._transcribe_windowed(
                mel_spectrogram, language=language, timeout_sec=timeout_sec,
                max_frames=max_frames, progress_cb=progress_cb,
            )

        enc_input_names = [inp.name for inp in self.encoder_session.get_inputs()]
        encoder_input_name = enc_input_names[0]
        encoder_feed_value = mel_spectrogram[np.newaxis]

        if encoder_input_name in ("input_features", "mel", "mel_spectrogram"):
            encoder_feed_value = encoder_feed_value.astype(np.float32)

        try:
            if mel_spectrogram.shape[0] <= max_frames:
                encoder_outputs = self._run_encoder(
                    {encoder_input_name: encoder_feed_value}
                )
            else:
                enc_output_names = [o.name for o in self.encoder_session.get_outputs()]
                all_parts = {name: [] for name in enc_output_names}
                overlap_frames = max_frames // 4
                pos = 0

                while pos < mel_spectrogram.shape[0]:
                    end = min(pos + max_frames, mel_spectrogram.shape[0])
                    chunk = mel_spectrogram[pos:end]
                    chunk_out = self._run_encoder(
                        {encoder_input_name: chunk[np.newaxis] if encoder_input_name != "input_features" else chunk[np.newaxis].astype(np.float32)}
                    )
                    for name, val in zip(enc_output_names, chunk_out):
                        arr = val.astype(np.float32)
                        if arr.ndim >= 3 and pos > 0:
                            in_len = end - pos
                            out_seq = arr.shape[-2]
                            if out_seq > 0 and in_len > 0:
                                ratio = out_seq / in_len
                                trim = min(int(round(overlap_frames * ratio)), out_seq)
                                if trim > 0:
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

        encoder_hidden_states, cross_kv = self._parse_encoder_outputs(encoder_outputs, self.encoder_session)

        if past_kv_cache_ort is not None:
            self_kv = past_kv_cache_ort.get("self_kv", self._init_self_kv_cache())
        else:
            self_kv = self._init_self_kv_cache()

        dec_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        dec_output_names = [out.name for out in self.decoder_session.get_outputs()]

        enc_hs_input_name = None
        for candidate in ("encoder_hidden_states", "encoder_hidden_state", "encoder_output", "encoder_out", "memory", "context"):
            if candidate in dec_input_names:
                enc_hs_input_name = candidate
                break
        if enc_hs_input_name is None:
            logger.warning("Decoder has no encoder_hidden_states input; cross-attention will not work. Inputs: %s", dec_input_names)
            encoder_hidden_states = None
        else:
            logger.info("Using encoder hidden states input: %s", enc_hs_input_name)

        if prefix_ids is not None:
            input_ids = np.array([prefix_ids], dtype=np.int64)
        else:
            input_ids = np.array([self.prompt_ids], dtype=np.int64)

        position = 0
        past_seq_len = 0
        tokens_this_call = 0
        generated_tokens = []
        max_new = self.settings.max_new_tokens if self.settings else 448

        for step in range(max_new):
            seq_len = input_ids.shape[1]
            feed = self._build_decoder_inputs(
                dec_input_names, input_ids, position, past_seq_len, tokens_this_call,
                cross_kv, self_kv,
                encoder_hidden_states=encoder_hidden_states,
                enc_hs_input_name=enc_hs_input_name,
            )

            try:
                outputs = self.decoder_session.run(None, feed)
            except Exception as e:
                logger.error("Decoder step %d failed: %s", step, e)
                logger.error("Feed keys: %s", list(feed.keys()))
                for k, v in feed.items():
                    logger.error("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)
                logger.error("Expected decoder input names: %s", dec_input_names)
                break

            logits = outputs[0]
            next_token = int(np.argmax(logits[0, -1, :]))

            if logger.isEnabledFor(logging.DEBUG):
                if self.tokenizer:
                    tok_str = self.tokenizer.decode([next_token], skip_special_tokens=False)
                else:
                    tok_str = chr(next_token) if 32 <= next_token < 127 else f"<{next_token}>"
                logger.debug("Step %d: token=%d (%s), logits_shape=%s", step, next_token, tok_str, logits.shape)

            new_self_kv = {}
            for i, name in enumerate(dec_output_names):
                if name == "logits":
                    continue
                mapped_name = name.replace("present.", "past_key_values.") if name.startswith("present.") else name
                new_self_kv[mapped_name] = outputs[i]
            if new_self_kv:
                self_kv = new_self_kv

            if next_token == self.eos_token_id:
                break

            generated_tokens.append(next_token)
            input_ids = np.array([[next_token]], dtype=np.int64)
            tokens_this_call += seq_len
            position += seq_len

        audio_duration = mel_spectrogram.shape[0] * HOP_LENGTH / SAMPLE_RATE if mel_spectrogram is not None else 0

        segments_out = []
        if self.tokenizer and self.tokens:
            token_to_id = self.tokenizer.get_vocab()
            SPLIT_TOKEN_BASE = token_to_id.get("<|spltoken0|>", -1)
            NUM_SPLIT_BINS = 34

            seg_text_parts = []
            current_seg_start = 0.0
            pending_start = 0.0

            def _flush_segment(seg_end: float):
                nonlocal seg_text_parts, current_seg_start, pending_start
                seg_text = "".join(seg_text_parts).strip()
                seg_text_parts = []
                if seg_text:
                    cleaned = clean_transcript(seg_text)
                    if cleaned.strip():
                        segments_out.append({
                            "start": round(pending_start, 3),
                            "end": round(seg_end, 3),
                            "text": cleaned.strip(),
                        })
                        pending_start = seg_end
                current_seg_start = seg_end

            for tok_id in generated_tokens:
                tok_str = self.tokens.get(tok_id, "")
                if tok_str.startswith("<|"):
                    if SPLIT_TOKEN_BASE != -1 and SPLIT_TOKEN_BASE <= tok_id < SPLIT_TOKEN_BASE + NUM_SPLIT_BINS:
                        seg_end = audio_duration * (tok_id - SPLIT_TOKEN_BASE) / NUM_SPLIT_BINS
                        _flush_segment(seg_end)
                    continue
                seg_text_parts.append(tok_str.replace("\u2581", " "))

            _flush_segment(audio_duration)

            if segments_out:
                text = " ".join(s["text"] for s in segments_out)
            else:
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                text = clean_transcript(text)
        else:
            text = "".join(chr(t) if 32 <= t < 127 else "" for t in generated_tokens)
            text = clean_transcript(text)

        inference_time = time.time() - start_time

        return {
            "text": text,
            "segments": segments_out if segments_out else None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(inference_time, 2),
            "tokens_generated": len(generated_tokens),
        }

    async def transcribe_audio_async(
        self,
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
            lambda: self.transcribe_audio_sync(
                audio=audio, language=language, timeout_sec=timeout_sec,
                mel_spectrogram=mel_spectrogram,
                past_kv_cache_ort=past_kv_cache_ort,
                prefix_ids=prefix_ids,
            ),
        )


def _load_tokenizer(model_dir: Path):
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        logger.warning("tokenizer.json not found at %s", tokenizer_path)
        return None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info("Loaded tokenizer from %s (vocab_size=%d)", tokenizer_path, tokenizer.get_vocab_size())
        return tokenizer
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        return None


def _build_prompt_ids(tokenizer, language: str = "en") -> list[int]:
    token_to_id = tokenizer.get_vocab()
    lang_token = f"<|{language}|>"
    prompt_tokens = [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        lang_token,
        lang_token,
        "<|pnc|>",
        "<|noitn|>",
        "<|timestamp|>",
        "<|nodiarize|>",
    ]
    return [token_to_id[t] for t in prompt_tokens if t in token_to_id]