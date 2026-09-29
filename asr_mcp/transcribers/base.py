"""Pluggable ASR backend interface.

Each backend owns its model sessions/models and implements the same surface
so the rest of the app (transcriber facade, model_state lifecycle, phase
unloads) is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from asr_mcp.config.settings import Settings


class ASRBackend(ABC):
    """Abstract ASR backend.

    A backend must be lazily loadable (load() may be called from the model
    lifecycle after TTL unloads) and must expose compute_mel_spectrogram /
    transcribe_audio_sync. Backends may ignore args that don't apply to
    them (e.g. Qwen3 ignores past_kv_cache_ort / prefix_ids / _no_window).
    """

    name: str = "base"

    # Static (code, display-name) pairs the backend supports for forced-language
    # transcription. Populated by each concrete backend as a class attribute so
    # the list is available WITHOUT loading the model.
    LANGUAGES: list[tuple[str, str]] = []

    # Whether language="auto" performs real detection (whisper/qwen) or falls
    # back to a forced default (cohere has no <|auto|> prompt token).
    SUPPORTS_AUTO: bool = True

    @classmethod
    def language_list(cls) -> list[dict]:
        """Languages as [{"code", "name"}] for API responses."""
        return [{"code": code, "name": display} for code, display in cls.LANGUAGES]

    def __init__(self) -> None:
        self.loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @abstractmethod
    def load(self, settings: Settings) -> None:
        """Load the model(s) for this backend. Idempotent via self.loaded."""

    @abstractmethod
    def unload(self) -> None:
        """Free all GPU memory owned by this backend."""

    def unload_encoder(self) -> None:
        """Free the ASR model only (called at diarize phase start)."""
        self.unload()

    def compute_mel_spectrogram(self, audio) -> object:
        """Return backend-specific acoustic features for `audio` (16k mono float32)."""
        raise NotImplementedError(f"{self.name} does not expose mel spectrograms")

    @abstractmethod
    def transcribe_audio_sync(
        self,
        audio=None,
        language: str = "en",
        timeout_sec: float = 120,
        mel_spectrogram=None,
        past_kv_cache_ort=None,
        prefix_ids=None,
        _no_window: bool = False,
        progress_cb: Optional[Callable[[dict], None]] = None,
        context: str = "",
        pre_segmented: bool = False,
    ) -> dict:
        """Transcribe audio (or precomputed features). Returns:
        {text, segments, audio_duration_sec, inference_time_sec,
         tokens_generated, [error]}

        ``context`` is an optional short carry-over transcript that a backend
        may use as a decoding hint. Backends without a context slot ignore it.

        ``pre_segmented`` marks audio the caller already cut into a single
        speaker turn (live streaming). Backends with their own internal VAD or
        previous-text conditioning must not re-segment or re-condition it.
        """