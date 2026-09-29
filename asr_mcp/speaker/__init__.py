"""Speaker subpackage.

Imports are resolved lazily (PEP 562) so that dependency-free modules such as
``asr_mcp.speaker.uncertainty`` — the single source of truth for the uncertain
speaker policy — can be imported without pulling in torch / onnxruntime.  The
ML stack is imported on first attribute access instead, which keeps the unit
tests runnable in a lightweight environment.
"""

_EXPORTS = {
    "SpeakerService": "asr_mcp.speaker.service",
    "extract_fbank": "asr_mcp.speaker.audio",
    "generate_sliding_windows": "asr_mcp.speaker.audio",
    "refine_speaker_boundaries": "asr_mcp.speaker.audio",
    "extract_embedding": "asr_mcp.speaker.embedding",
    "batch_embed_files": "asr_mcp.speaker.embedding",
    "normalize_embedding": "asr_mcp.speaker.embedding",
    "compute_pitch": "asr_mcp.speaker.embedding",
    "compute_energy": "asr_mcp.speaker.embedding",
    "split_at_energy_dips": "asr_mcp.speaker.vad",
    "run_vad_chunked": "asr_mcp.speaker.vad",
    "run_vad_onnx": "asr_mcp.speaker.vad",
    "compute_distance": "asr_mcp.speaker.matcher",
    "find_best_match": "asr_mcp.speaker.matcher",
    "match_clusters": "asr_mcp.speaker.matcher",
    "merge_matched_clusters": "asr_mcp.speaker.matcher",
    "profile_speakers": "asr_mcp.speaker.profiling",
    "relabel_by_pitch": "asr_mcp.speaker.profiling",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals()) + __all__))
