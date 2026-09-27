"""Pluggable ASR backends.

Selectable via TRANSCRIBE_ASR_MODEL env var ("cohere" | "qwen3-asr" | "whisper").
"""

from asr_mcp.transcribers.base import ASRBackend
from asr_mcp.transcribers.cohere import CohereBackend
from asr_mcp.transcribers.qwen3 import Qwen3Backend
from asr_mcp.transcribers.whisper import WhisperBackend

BACKENDS_BY_NAME = {
    "cohere": CohereBackend,
    "qwen3-asr": Qwen3Backend,
    "whisper": WhisperBackend,
}


def resolve_backend_name(settings) -> str:
    """Normalized backend key for settings.asr_model (lowercase, default cohere).

    ModelState uses this for its 'backend still matches settings?' check so a
    case/whitespace mismatch can't trigger a reload loop.
    """
    return (getattr(settings, "asr_model", None) or "cohere").strip().lower()


def get_backend(settings) -> ASRBackend:
    """Return a fresh backend instance selected by settings.asr_model."""
    name = resolve_backend_name(settings)
    cls = BACKENDS_BY_NAME.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown TRANSCRIBE_ASR_MODEL {settings.asr_model!r}; "
            f"expected one of {sorted(BACKENDS_BY_NAME)}"
        )
    return cls()