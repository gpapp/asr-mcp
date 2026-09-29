"""Whisper backend (faster-whisper / CTranslate2).

Defaults to large-v3-turbo and is **quantized** (int8_float16 on CUDA, int8 on
CPU) so the ~1.6GB fp16 weights fit next to the diarization models on a 4GB
card (~1.0GB VRAM measured on a GTX 1650). On CUDA load failure the backend
escalates quantization and finally falls back to CPU; a CUDA OOM during
transcription rebuilds the model on CPU (int8) and retries once.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from asr_mcp.config.settings import Settings
from asr_mcp.transcribers.base import ASRBackend

logger = logging.getLogger("asr_mcp.transcribers.whisper")

SAMPLE_RATE = 16000
WINDOW_SEC = 30.0  # Whisper decodes in 30s windows — progress granularity

# Default initial_prompt. Without one, long-form / room-audio decodes can lock into
# a lowercase, unpunctuated style from the very first segment, and
# condition_on_previous_text then perpetuates it for the whole file (observed:
# 0/251 segments punctuated on a 34-min meeting recording). A punctuated anchor
# fixes degraded audio (sentence density 4.6 -> 17.5 per 1000 letters) and leaves
# clean English unchanged; Hungarian output stays 98.5% identical with MORE
# punctuation and no English word injection (verified on a 90s hu clip).
STYLE_ANCHOR = (
    "The meeting started at ten o'clock and we reviewed the delivery plan first. "
    "Anna explained that the staging environment is ready, but the migration script "
    "still needs review. We agreed to postpone the release until Thursday so that "
    "QA has two full days for regression testing. Marcus will update the ticket "
    "with the new deadline and notify the support team."
)

# Compute types that need GPU fp16 arithmetic (CT2 rejects them on CPU).
_GPU_ONLY_COMPUTE = {"int8_float16", "int4_float16"}


def _device_index(cuda_device) -> int:
    """Ordinal from a 'cuda:N' / 'N' style setting (default 0)."""
    try:
        return int(str(cuda_device).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0


class WhisperBackend(ASRBackend):
    """faster-whisper CTranslate2 backend (quantized to fit small GPUs)."""

    name = "whisper"

    # Static (code, display-name) list — see ASRBackend.LANGUAGES.
    LANGUAGES = [
        ("af", "Afrikaans"),
        ("sq", "Albanian"),
        ("am", "Amharic"),
        ("ar", "Arabic"),
        ("hy", "Armenian"),
        ("as", "Assamese"),
        ("az", "Azerbaijani"),
        ("ba", "Bashkir"),
        ("eu", "Basque"),
        ("be", "Belarusian"),
        ("bn", "Bengali"),
        ("bs", "Bosnian"),
        ("br", "Breton"),
        ("bg", "Bulgarian"),
        ("my", "Burmese"),
        ("yue", "Cantonese"),
        ("ca", "Catalan"),
        ("zh", "Chinese"),
        ("hr", "Croatian"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("nl", "Dutch"),
        ("en", "English"),
        ("et", "Estonian"),
        ("fo", "Faroese"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("gl", "Galician"),
        ("ka", "Georgian"),
        ("de", "German"),
        ("el", "Greek"),
        ("gu", "Gujarati"),
        ("ht", "Haitian Creole"),
        ("ha", "Hausa"),
        ("haw", "Hawaiian"),
        ("he", "Hebrew"),
        ("hi", "Hindi"),
        ("hu", "Hungarian"),
        ("is", "Icelandic"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("jw", "Javanese"),
        ("kn", "Kannada"),
        ("kk", "Kazakh"),
        ("km", "Khmer"),
        ("ko", "Korean"),
        ("lo", "Lao"),
        ("la", "Latin"),
        ("lv", "Latvian"),
        ("ln", "Lingala"),
        ("lt", "Lithuanian"),
        ("lb", "Luxembourgish"),
        ("mk", "Macedonian"),
        ("mg", "Malagasy"),
        ("ms", "Malay"),
        ("ml", "Malayalam"),
        ("mt", "Maltese"),
        ("mi", "Maori"),
        ("mr", "Marathi"),
        ("mn", "Mongolian"),
        ("ne", "Nepali"),
        ("no", "Norwegian"),
        ("nn", "Norwegian Nynorsk"),
        ("oc", "Occitan"),
        ("ps", "Pashto"),
        ("fa", "Persian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("pa", "Punjabi"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("sa", "Sanskrit"),
        ("sr", "Serbian"),
        ("sn", "Shona"),
        ("sd", "Sindhi"),
        ("si", "Sinhala"),
        ("sk", "Slovak"),
        ("sl", "Slovenian"),
        ("so", "Somali"),
        ("es", "Spanish"),
        ("su", "Sundanese"),
        ("sw", "Swahili"),
        ("sv", "Swedish"),
        ("tl", "Tagalog"),
        ("tg", "Tajik"),
        ("ta", "Tamil"),
        ("tt", "Tatar"),
        ("te", "Telugu"),
        ("th", "Thai"),
        ("bo", "Tibetan"),
        ("tr", "Turkish"),
        ("tk", "Turkmen"),
        ("uk", "Ukrainian"),
        ("ur", "Urdu"),
        ("uz", "Uzbek"),
        ("vi", "Vietnamese"),
        ("cy", "Welsh"),
        ("yi", "Yiddish"),
        ("yo", "Yoruba"),
    ]

    def __init__(self):
        super().__init__()
        self.model = None
        self.settings: Optional[Settings] = None
        self.model_path: Optional[Path] = None
        self._device = "cpu"
        self._device_index = 0
        self._compute_type = "int8"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_compute_type(requested: str, use_cuda: bool) -> str:
        """'auto' → int8_float16 on CUDA, int8 on CPU (quantized by default)."""
        req = str(requested or "auto").strip().lower()
        if req in ("", "auto"):
            return "int8_float16" if use_cuda else "int8"
        return req

    @staticmethod
    def _ensure_local_model(model_spec: str, base_dir, hf_token: Optional[str]) -> Path:
        """Resolve *model_spec* to a local model dir, downloading it once.

        Accepts an existing local directory, a faster-whisper size name
        (e.g. 'large-v3-turbo') or a HuggingFace repo id. Downloads land in
        ``<base_dir>/<sanitized spec>`` so the mounted models volume caches them.
        """
        spec = (model_spec or "").strip()
        direct = Path(spec).expanduser()
        if direct.is_dir():
            return direct
        target = Path(base_dir) / spec.replace("/", "--")
        if (target / "config.json").exists() and (target / "model.bin").exists():
            return target
        from faster_whisper.utils import download_model

        target.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Whisper model %s -> %s", spec, target)
        path = download_model(spec, output_dir=str(target), use_auth_token=hf_token or None)
        return Path(path)

    def load(self, settings: Settings) -> None:
        if self.loaded:
            return
        import ctranslate2

        use_cuda = ctranslate2.get_cuda_device_count() > 0
        if not use_cuda:
            logger.warning("CUDA not available; Whisper will run on CPU")
        compute = self._resolve_compute_type(settings.whisper_compute_type, use_cuda)

        try:
            self.model_path = self._ensure_local_model(
                settings.whisper_model, settings.whisper_model_dir, settings.hf_token
            )
        except Exception as e:
            self.loaded = False
            logger.exception("Whisper model download failed: %s", e)
            raise

        attempts: list[tuple[str, int, str]] = []
        if use_cuda:
            idx = _device_index(settings.cuda_device)
            attempts.append(("cuda", idx, compute))
            if compute != "int8":
                attempts.append(("cuda", idx, "int8"))
        cpu_compute = "int8" if compute in _GPU_ONLY_COMPUTE else compute
        attempts.append(("cpu", 0, cpu_compute))

        last_err: Optional[Exception] = None
        for device, idx, ctype in attempts:
            try:
                self._create_model(settings, device, idx, ctype)
                return
            except Exception as e:
                last_err = e
                self.model = None
                self.loaded = False
                logger.warning("Whisper load failed on %s/%s: %s", device, ctype, e)
                self._free_cuda_cache()
                gc.collect()
        raise last_err if last_err else RuntimeError("Whisper backend load failed")

    def _create_model(self, settings: Settings, device: str, device_index: int, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model %s from %s (device=%s, compute_type=%s)",
            settings.whisper_model, self.model_path, device, compute_type,
        )
        self.model = WhisperModel(
            str(self.model_path),
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=max(0, int(settings.whisper_cpu_threads)),
            num_workers=1,
        )
        self._device = device
        self._device_index = device_index
        self._compute_type = compute_type
        self.settings = settings
        self.loaded = True
        logger.info("Whisper backend loaded (compute_type=%s)", compute_type)

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
        self.loaded = False
        self._free_cuda_cache()
        gc.collect()

    def _fallback_to_cpu(self) -> None:
        """Drop the GPU model and rebuild it quantized on CPU."""
        settings = self.settings
        self.model = None
        self.loaded = False
        self._free_cuda_cache()
        gc.collect()
        self._create_model(settings, "cpu", 0, "int8")

    @staticmethod
    def _free_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("empty_cache failed", exc_info=True)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _run(
        self,
        arr: np.ndarray,
        language: str,
        context: str,
        progress_cb,
        pre_segmented: bool = False,
    ) -> dict:
        from asr_mcp.core.model_state import is_gpu_oom

        requested = str(language or "").strip().lower() or "auto"
        lang = requested
        if lang in ("", "none", "auto", "auto-detect"):
            lang = None

        # pre_segmented=True: the caller already cut this audio into a single
        # speaker turn (live streaming).  faster-whisper's built-in Silero VAD
        # would re-segment that turn a second time, and previous-text
        # conditioning would carry the *previous* speaker's words into this
        # one.  Both are disabled so the turn is decoded as-is.
        vad_filter = bool(self.settings.whisper_vad_filter) and not pre_segmented
        condition_prev = not pre_segmented

        kwargs = dict(
            language=lang,
            beam_size=max(1, int(self.settings.whisper_beam_size)),
            vad_filter=vad_filter,
            # Keep timestamp tokens: segment-level times feed speaker attribution.
            without_timestamps=False,
            condition_on_previous_text=condition_prev,
        )
        ctx = (context or "").strip()
        # Carry-over context wins when present; otherwise anchor the style so the
        # first segment is decoded punctuated/capitalized (see STYLE_ANCHOR).
        kwargs["initial_prompt"] = ctx or STYLE_ANCHOR

        seg_iter, info = self.model.transcribe(arr, **kwargs)

        audio_duration = len(arr) / SAMPLE_RATE
        est_windows = max(1, int(math.ceil(audio_duration / WINDOW_SEC)))

        segments_out: list[dict] = []
        texts: list[str] = []
        errors: list[str] = []
        reported = 0
        last_win = 0
        cursor = 0.0

        while True:
            try:
                seg = next(seg_iter)
            except StopIteration:
                break
            except Exception as e:
                if is_gpu_oom(e):
                    raise
                errors.append("decode failed at %.1fs: %s" % (cursor, e))
                logger.error("Whisper decode failed at %.1fs: %s", cursor, e)
                break

            txt = (seg.text or "").strip()
            start = float(seg.start or 0.0)
            end = float(seg.end or start)
            cursor = end
            if txt:
                conf = None
                alp = getattr(seg, "avg_logprob", None)
                if alp is not None:
                    try:
                        conf = max(0.0, min(1.0, math.exp(float(alp))))
                    except (ValueError, OverflowError):
                        conf = None
                segments_out.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": txt,
                    "confidence": round(conf, 3) if conf is not None else None,
                })
                texts.append(txt)

            if progress_cb and est_windows > 1:
                win = min(est_windows, max(1, int(math.ceil(end / WINDOW_SEC))))
                last_win = max(last_win, win)
                try:
                    progress_cb(last_win, est_windows, " ".join(texts), segments_out[reported:])
                    reported = len(segments_out)
                except Exception:
                    logger.debug("progress_cb failed", exc_info=True)

        if progress_cb and est_windows > 1 and last_win < est_windows:
            try:
                progress_cb(est_windows, est_windows, " ".join(texts), segments_out[reported:])
            except Exception:
                logger.debug("progress_cb failed", exc_info=True)

        text = " ".join(texts).strip()
        if not segments_out and text:
            segments_out = [{"start": 0.0, "end": round(audio_duration, 3), "text": text}]

        logger.info(
            "Whisper finished: requested=%s lang=%s p=%.2f speech=%.1fs/%.1fs segments=%d device=%s/%s",
            requested,
            getattr(info, "language", "?"),
            float(getattr(info, "language_probability", 0.0) or 0.0),
            float(getattr(info, "duration_after_vad", audio_duration) or 0.0),
            audio_duration, len(segments_out), self._device, self._compute_type,
        )
        return {"text": text, "segments": segments_out, "errors": errors}

    @staticmethod
    def _error_result(message: str, audio_duration: float, start_time: float) -> dict:
        return {
            "text": "",
            "segments": None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(time.time() - start_time, 2),
            "tokens_generated": 0,
            "error": str(message),
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
        pre_segmented: bool = False,
    ) -> dict:
        start_time = time.time()

        if audio is None:
            return {"text": "", "error": "No audio provided (Whisper requires raw audio, not mel)"}
        if not self.loaded or self.model is None:
            return {"text": "", "error": "Whisper backend not loaded"}

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.size and float(np.max(np.abs(arr))) > 1.0:
            arr = arr / float(np.max(np.abs(arr)))
        audio_duration = len(arr) / SAMPLE_RATE

        out = None
        try:
            out = self._run(arr, language=language, context=context, progress_cb=progress_cb,
                            pre_segmented=pre_segmented)
        except Exception as e:
            from asr_mcp.core.model_state import is_gpu_oom

            if is_gpu_oom(e) and self._device == "cuda":
                logger.warning("Whisper CUDA OOM (%s); retrying on CPU (int8)", e)
                try:
                    self._fallback_to_cpu()
                    out = self._run(arr, language=language, context=context,
                                    progress_cb=progress_cb, pre_segmented=pre_segmented)
                except Exception as e2:
                    logger.error("Whisper transcription failed on CPU: %s", e2)
                    return self._error_result(e2, audio_duration, start_time)
            else:
                logger.error("Whisper transcription failed: %s", e)
                return self._error_result(e, audio_duration, start_time)

        if not out["text"] and not out["segments"] and not out["errors"]:
            return self._error_result("No transcription produced", audio_duration, start_time)

        result = {
            "text": out["text"],
            "segments": out["segments"] if out["segments"] else None,
            "audio_duration_sec": round(audio_duration, 2),
            "inference_time_sec": round(time.time() - start_time, 2),
            "tokens_generated": 0,
        }
        if out["errors"]:
            result["error"] = "; ".join(out["errors"])
        return result
