"""Qwen3-ASR backend (transformers / qwen-asr package)."""

import logging
import time
from typing import Optional

import numpy as np

from asr_mcp.transcribers.base import ASRBackend
from asr_mcp.config.settings import Settings

logger = logging.getLogger("asr_mcp.transcribers.qwen3")

SAMPLE_RATE = 16000

DEFAULT_CHUNK_SEC = 30.0
MIN_CHUNK_SEC = 10.0
CONTEXT_TAIL_CHARS = 240

# ISO 639-1 -> Qwen canonical language name (None = auto-detect).
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "tl": "Filipino",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def _to_canonical_language(language: Optional[str]) -> Optional[str]:
    """Map internal language code to a Qwen canonical name (None = auto)."""
    if not language or str(language).lower() in ("none", "auto", "auto-detect"):
        return None
    return LANGUAGE_MAP.get(str(language).strip().lower())


def _context_tail(text, max_chars=CONTEXT_TAIL_CHARS):
    """Bounded tail of *text* carried into the next chunk as decode context."""
    if not text:
        return ""
    t = " ".join(str(text).split())
    return t[-max_chars:].strip() if len(t) > max_chars else t


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR backend wrapping qwen_asr.Qwen3ASRModel with ForcedAligner."""

    name = "qwen3-asr"

    def __init__(self):
        super().__init__()
        self.recognizer = None  # Qwen3ASRModel (includes forced aligner)
        self.settings = None
        self._chunk_sec = DEFAULT_CHUNK_SEC

    def load(self, settings: Settings) -> None:
        if self.loaded:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            dtype = getattr(torch, settings.qwen_torch_dtype, torch.float16)
            use_cuda = torch.cuda.is_available()
            device_map = settings.cuda_device if use_cuda else "cpu"
            if not use_cuda:
                logger.warning("CUDA not available; Qwen3-ASR will run on CPU")
                dtype = torch.float32

            model_dir = self._ensure_local_model(
                settings.qwen_model_name, settings.qwen_model_dir, settings.hf_token
            )
            aligner_dir = self._ensure_local_model(
                settings.qwen_forced_aligner_name, settings.qwen_forced_aligner_dir, settings.hf_token
            )

            model_kwargs = dict(
                dtype=dtype,
                attn_implementation="sdpa",
                device_map=device_map,
                max_inference_batch_size=settings.qwen_max_inference_batch_size,
                max_new_tokens=settings.qwen_max_new_tokens,
            )
            if settings.qwen_quantize_4bit and use_cuda:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            base_aligner_kwargs = dict(dtype=dtype, device_map=device_map)
            aligner_attempts = [base_aligner_kwargs]
            if getattr(settings, "qwen_aligner_quantize_4bit", True) and use_cuda:
                from transformers import BitsAndBytesConfig
                quant_aligner = dict(base_aligner_kwargs)
                quant_aligner["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                aligner_attempts = [quant_aligner, base_aligner_kwargs]

            logger.info(
                "Loading Qwen3-ASR model %s from %s (dtype=%s, device=%s, quantize_4bit=%s, "
                "aligner_4bit=%s)",
                settings.qwen_model_name, model_dir, dtype, device_map,
                settings.qwen_quantize_4bit, len(aligner_attempts) == 2,
            )

            self.recognizer = None
            last_err = None
            for i, aligner_kwargs in enumerate(aligner_attempts):
                try:
                    self.recognizer = Qwen3ASRModel.from_pretrained(
                        str(model_dir),
                        forced_aligner=str(aligner_dir),
                        forced_aligner_kwargs=aligner_kwargs,
                        **model_kwargs,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    self.recognizer = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if i < len(aligner_attempts) - 1:
                        logger.warning(
                            "Qwen3 forced aligner 4-bit load failed (%s); "
                            "retrying unquantized in %s", e, dtype,
                        )
            if self.recognizer is None:
                raise last_err if last_err else RuntimeError("Qwen3-ASR load failed")

            self.settings = settings
            self.loaded = True
            logger.info("Qwen3-ASR backend loaded successfully")
        except Exception as e:
            self.recognizer = None
            self.loaded = False
            logger.exception("Qwen3-ASR model load failed: %s", e)
            raise

    @staticmethod
    def _ensure_local_model(repo_id: str, target_dir, hf_token: Optional[str]):
        """Download the HF repo into `models/` (like cohere) and return the local path."""
        from pathlib import Path
        from huggingface_hub import snapshot_download
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if not Qwen3Backend._model_dir_complete(target):
            logger.info("Downloading %s -> %s", repo_id, target)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                token=hf_token,
            )
        return target

    @staticmethod
    def _model_dir_complete(target) -> bool:
        """True when the local dir has a config + at least one weights file."""
        from pathlib import Path
        target = Path(target)
        if not (target / "config.json").exists():
            return False
        for p in target.iterdir():
            if p.is_file() and p.suffix in (".safetensors", ".bin", ".gguf", ".onnx"):
                return True
        return False

    def unload(self) -> None:
        import gc
        if self.recognizer is not None:
            try:
                import torch
                del self.recognizer
                self.recognizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                self.recognizer = None
        self.loaded = False
        gc.collect()

    @staticmethod
    def _free_cuda_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("empty_cache failed", exc_info=True)

    def _snap_cut(self, arr, ideal, lo, hi, band_sec=1.5):
        """Sample index of the quietest frame near *ideal*, clamped to [lo, hi]."""
        target = int(ideal)
        lo_i = max(int(lo), target - int(band_sec * SAMPLE_RATE))
        hi_i = min(int(hi), target + int(band_sec * SAMPLE_RATE))
        if hi_i - lo_i < int(0.2 * SAMPLE_RATE):
            return max(int(lo), min(target, int(hi)))
        seg = arr[lo_i:hi_i]
        win = max(1, int(0.05 * SAMPLE_RATE))
        n = len(seg) // win
        if n < 1:
            return max(int(lo), min(target, int(hi)))
        rms = np.sqrt(
            (seg[:n * win].astype(np.float64).reshape(n, win) ** 2).mean(axis=1) + 1e-12
        )
        return lo_i + int(np.argmin(rms)) * win + win // 2

    def _decode(self, arr, language, context):
        """One recognizer call. Returns (text, segments). Raises on failure."""
        call_kwargs = {
            "language": _to_canonical_language(language),
            "return_time_stamps": True,
        }
        ctx = (context or "").strip()
        if ctx:
            call_kwargs["context"] = ctx

        results = self.recognizer.transcribe(audio=(arr, SAMPLE_RATE), **call_kwargs)
        if not results:
            return "", []

        r = results[0]
        text = getattr(r, "text", "") or ""
        segments_out = []
        time_stamps = getattr(r, "time_stamps", None)
        items = getattr(time_stamps, "items", None)
        if items:
            for it in items:
                seg_text = getattr(it, "text", "") or ""
                seg_start = float(getattr(it, "start_time", 0.0) or 0.0)
                seg_end = float(getattr(it, "end_time", seg_start) or seg_start)
                if seg_text.strip():
                    segments_out.append({
                        "start": round(seg_start, 3),
                        "end": round(seg_end, 3),
                        "text": seg_text.strip(),
                    })

        if not segments_out and text.strip():
            dur = len(arr) / SAMPLE_RATE
            segments_out = [{
                "start": 0.0,
                "end": round(dur, 3),
                "text": text.strip(),
            }]
        return text.strip(), segments_out

    def _transcribe_chunked(
        self,
        arr,
        language="en",
        context="",
        progress_cb=None,
    ):
        """Decode *arr* in memory-bounded chunks with OOM backoff.

        VRAM for the encoder + forced aligner scales with input length, so a
        turn that fits in one call can OOM on a small card. Each chunk is
        decoded independently, timestamps are offset by the chunk start, and
        the previous chunk's text tail is carried as decode context. On CUDA
        OOM the chunk size is halved and the same position is retried, so a
        long turn degrades into smaller windows instead of returning nothing.
        """
        from asr_mcp.core.model_state import is_gpu_oom

        total = len(arr)
        audio_duration = total / SAMPLE_RATE
        chunk_sec = self._chunk_sec
        est_windows = max(1, int(np.ceil(audio_duration / max(chunk_sec, 1.0))))

        text_parts = []
        segments_out = []
        errors = []
        pos = 0
        window = 0

        while pos < total:
            remaining = (total - pos) / SAMPLE_RATE
            if remaining <= chunk_sec:
                end = total
            else:
                end = self._snap_cut(arr, pos + chunk_sec * SAMPLE_RATE, pos, total)
                if end <= pos:
                    end = min(total, pos + int(chunk_sec * SAMPLE_RATE))

            piece = arr[pos:end]
            try:
                text, segs = self._decode(piece, language, context)
            except Exception as e:
                if is_gpu_oom(e) and chunk_sec > MIN_CHUNK_SEC:
                    chunk_sec = max(MIN_CHUNK_SEC, chunk_sec / 2.0)
                    self._chunk_sec = chunk_sec
                    self._free_cuda_cache()
                    logger.warning(
                        "Qwen3 CUDA OOM at %.1fs; reducing chunk size to %.1fs",
                        pos / SAMPLE_RATE, chunk_sec,
                    )
                    continue
                logger.error("Qwen3-ASR chunk failed at %.1fs: %s", pos / SAMPLE_RATE, e)
                errors.append("chunk at %.1fs: %s" % (pos / SAMPLE_RATE, e))
                self._free_cuda_cache()
                pos = end
                continue

            window += 1
            offset = pos / SAMPLE_RATE
            if text.strip():
                text_parts.append(text.strip())
                context = _context_tail(text)
            for seg in segs:
                new_seg = dict(seg)
                new_seg["start"] = round(float(seg.get("start", 0.0)) + offset, 3)
                new_seg["end"] = round(float(seg.get("end", 0.0)) + offset, 3)
                segments_out.append(new_seg)

            if progress_cb and est_windows > 1:
                try:
                    progress_cb(min(window, est_windows), est_windows)
                except Exception:
                    logger.debug("progress_cb failed", exc_info=True)

            pos = end

        return {
            "text": " ".join(text_parts).strip(),
            "segments": segments_out,
            "errors": errors,
        }

    def transcribe_audio_sync(
        self,
        audio=None,
        language: str = "en",
        timeout_sec: float = 120,
        mel_spectrogram=None,
        past_kv_cache_ort=None,
        prefix_ids=None,
        _no_window: bool = False,
        progress_cb=None,
        context: str = "",
    ) -> dict:
        start_time = time.time()

        if audio is None:
            return {"text": "", "error": "No audio provided (Qwen3 requires raw audio, not mel)"}

        if not self.loaded or self.recognizer is None:
            return {"text": "", "error": "Qwen3 backend not loaded"}

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / np.max(np.abs(arr))

        audio_duration = len(arr) / SAMPLE_RATE

        try:
            out = self._transcribe_chunked(
                arr, language=language, context=context, progress_cb=progress_cb,
            )
        except Exception as e:
            logger.error("Qwen3-ASR transcription failed: %s", e)
            return {
                "text": "",
                "segments": None,
                "audio_duration_sec": round(audio_duration, 2),
                "inference_time_sec": round(time.time() - start_time, 2),
                "tokens_generated": 0,
                "error": str(e),
            }

        text = out["text"]
        segments_out = out["segments"]
        if not text and not segments_out and not out["errors"]:
            return {
                "text": "",
                "segments": None,
                "audio_duration_sec": round(audio_duration, 2),
                "inference_time_sec": round(time.time() - start_time, 2),
                "tokens_generated": 0,
                "error": "No transcription produced",
            }

        result = {
            "text": text,
            "segments": segments_out if segments_out else None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(time.time() - start_time, 2),
            "tokens_generated": 0,
        }
        if out["errors"]:
            result["error"] = "; ".join(out["errors"])
        return result