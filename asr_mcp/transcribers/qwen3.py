"""Qwen3-ASR backend (transformers / qwen-asr package)."""

import logging
import time
from typing import Optional

import numpy as np

from asr_mcp.transcribers.base import ASRBackend
from asr_mcp.config.settings import Settings

logger = logging.getLogger("asr_mcp.transcribers.qwen3")

SAMPLE_RATE = 16000

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


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR backend wrapping qwen_asr.Qwen3ASRModel with ForcedAligner."""

    name = "qwen3-asr"

    def __init__(self):
        super().__init__()
        self.recognizer = None  # Qwen3ASRModel (includes forced aligner)
        self.settings = None

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
                model_kwargs["load_in_4bit"] = True

            aligner_kwargs = dict(dtype=dtype, device_map=device_map)

            logger.info(
                "Loading Qwen3-ASR model %s (dtype=%s, device=%s, quantize_4bit=%s)",
                settings.qwen_model_name, dtype, device_map, settings.qwen_quantize_4bit,
            )
            self.recognizer = Qwen3ASRModel.from_pretrained(
                settings.qwen_model_name,
                forced_aligner=settings.qwen_forced_aligner_name,
                forced_aligner_kwargs=aligner_kwargs,
                **model_kwargs,
            )
            self.settings = settings
            self.loaded = True
            logger.info("Qwen3-ASR backend loaded successfully")
        except Exception as e:
            self.recognizer = None
            self.loaded = False
            logger.exception("Qwen3-ASR model load failed: %s", e)
            raise

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
            results = self.recognizer.transcribe(
                audio=(arr, SAMPLE_RATE),
                language=_to_canonical_language(language),
                return_time_stamps=True,
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

        if not results:
            return {
                "text": "",
                "segments": None,
                "audio_duration_sec": round(audio_duration, 2),
                "inference_time_sec": round(time.time() - start_time, 2),
                "tokens_generated": 0,
                "error": "No transcription produced",
            }

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
            segments_out = [{
                "start": 0.0,
                "end": round(audio_duration, 3),
                "text": text.strip(),
            }]

        return {
            "text": text.strip(),
            "segments": segments_out if segments_out else None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(time.time() - start_time, 2),
            "tokens_generated": 0,
        }