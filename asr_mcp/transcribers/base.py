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
    ) -> dict:
        """Transcribe audio (or precomputed features). Returns:
        {text, segments, audio_duration_sec, inference_time_sec,
         tokens_generated, [error]}
        """