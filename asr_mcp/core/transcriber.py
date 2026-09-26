"""ASR transcriber facade.

Thin dispatch layer over the active ASR backend (see
``asr_mcp.transcribers``). The backend is owned by ``ModelState.backend``;
this module keeps the historical ``transcribe_audio_sync`` /
``transcribe_audio_async`` public entry points working regardless of the
configured ``TRANSCRIBE_ASR_MODEL``.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from asr_mcp.core.model_state import state

logger = logging.getLogger("asr_mcp.core.transcriber")


def _backend():
    if state.backend is None:
        raise RuntimeError(
            "No ASR backend loaded. Call state.ensure_ready() or state.reload_models() first."
        )
    return state.backend


def transcribe_audio_sync(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None,
    _no_window: bool = False,
    progress_cb=None,
    context: str = "",
) -> dict:
    """Transcribe audio (or a precomputed mel) with the active ASR backend."""
    return _backend().transcribe_audio_sync(
        audio=audio,
        language=language,
        timeout_sec=timeout_sec,
        mel_spectrogram=mel_spectrogram,
        past_kv_cache_ort=past_kv_cache_ort,
        prefix_ids=prefix_ids,
        _no_window=_no_window,
        progress_cb=progress_cb,
        context=context,
    )


def _compute_mel_spectrogram_fast(audio: np.ndarray) -> np.ndarray:
    """Backend-specific mel computation (Cohere backend only)."""
    return _backend().compute_mel_spectrogram(audio)


async def transcribe_audio_async(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None,
    context: str = "",
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
            context=context,
        ),
    )