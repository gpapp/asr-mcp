"""Pluggable ASR backends.

Selectable via TRANSCRIBE_ASR_MODEL env var ("cohere" | "qwen3-asr").
"""

from asr_mcp.transcribers.base import ASRBackend
from asr_mcp.transcribers.cohere import CohereBackend
from asr_mcp.transcribers.qwen3 import Qwen3Backend

BACKENDS_BY_NAME = {
    "cohere": CohereBackend,
    "qwen3-asr": Qwen3Backend,
}


def get_backend(settings) -> ASRBackend:
    """Return a fresh backend instance selected by settings.asr_model."""
    name = (settings.asr_model or "cohere").strip().lower()
    cls = BACKENDS_BY_NAME.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown TRANSCRIBE_ASR_MODEL {settings.asr_model!r}; "
            f"expected one of {sorted(BACKENDS_BY_NAME)}"
        )
    return cls()