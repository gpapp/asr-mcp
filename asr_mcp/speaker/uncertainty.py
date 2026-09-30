"""Single policy for uncertain speaker attribution.

Policy (see AGENTS.md lesson 30): suppress the speaker *identity*, not the
spoken content.  A segment whose speaker cannot be established keeps its text
and interval but reports ``speaker=None`` plus an explicit uncertainty flag,
instead of being force-fitted to the nearest/temporally-closest speaker.

Every module that emits or consumes speaker labels goes through this helper so
the rule is defined in exactly one place:

* ``diarization/segment_ops.py`` — ghost/minority cleanup
* ``api/asr_router.py``        — turn attribution
* ``voiceprint/service.py``    — auto-collection eligibility
* ``streaming/handler.py``     — live transcription

This module is dependency-free (no torch/onnxruntime) so it can be unit
tested in isolation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("asr_mcp.speaker.uncertainty")

UNKNOWN_SPEAKER = "UNKNOWN"
OVERLAP_SPEAKER = "OVERLAP"

#: Generic cluster labels produced by the diarizer; never a real identity.
#: Matches both the canonical "Speaker 3" and the legacy zero-indexed
#: "SPEAKER_01" (which must never be emitted, but may exist in old DBs).
_GENERIC_RE = re.compile(r"^(?:SPEAKER|Speaker)[\s_]*\d+$", re.IGNORECASE)
#: Legacy/auto-generated names that must never be treated as an identity.
_TIMESTAMPED_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_"
    r"|_\d{2}-\d{2}-\d{2}_(?:SPEAKER|Speaker)\s*\d+$)",
    re.IGNORECASE,
)

# Speaker attribution sources
SOURCE_KNOWN_VOICEPRINT = "known_voiceprint"
SOURCE_DIARIZATION = "diarization_cluster"
SOURCE_UNKNOWN = "unknown"

# Fallback thresholds, overridden by config/thresholds.json "uncertainty".
_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # Minimum diarization match confidence to accept a speaker identity.
    "min_speaker_confidence": 0.35,
    # Minimum embedding-distance margin between best and runner-up match.
    "min_match_margin": 0.02,
    # Speech shorter than this is never auto-attributed from a single embedding.
    "min_attribution_dur_sec": 0.5,
    # How much of an ASR segment may fall outside its attributed turn before
    # the attribution is considered a boundary crossing (uncertain).
    "max_boundary_cross_sec": 0.3,
    # Relative share of a segment outside its turn that still counts as a cross.
    "max_boundary_cross_ratio": 0.34,
    # Only spans at most this long are tested for a boundary crossing.  Longer
    # spans (coarse backends emit one segment per ~30s window) are never inside
    # a single turn, so the test degenerates; they are gated by
    # ``low_span_turn_overlap`` instead.
    "max_boundary_cross_span_sec": 3.0,
    # Same-speaker merge gaps (turn prep and result merging).
    "turn_merge_gap_sec": 1.0,
    "result_merge_gap_sec": 1.0,
    # Keep the recovered text when the speaker is unknown (product policy).
    "retain_uncertain_text": True,
    # Never promote UNKNOWN/OVERLAP/generic labels to a real voiceprint.
    "auto_collect_requires_confidence": True,
    # Suppress (rather than reassign) ghost / minority speakers.
    "suppress_ghost_speakers": True,
    "suppress_minority_speakers": True,
    # A short-duration speaker is still a real participant if it holds at least
    # this share of the recording's speech.
    "ghost_max_share": 0.25,
}


def _cfg() -> dict[str, Any]:
    """Merged uncertainty config: thresholds.json values over the defaults."""
    merged = dict(_DEFAULTS)
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        section = raw.get("uncertainty") or {}
        if isinstance(section, dict):
            merged.update(section)
    except Exception:  # config unavailable — defaults are authoritative
        pass
    return merged


def setting(key: str, default: Any = None) -> Any:
    return _cfg().get(key, _DEFAULTS.get(key, default))


def policy_enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def retain_uncertain_text() -> bool:
    return bool(_cfg().get("retain_uncertain_text", True))


# ── Label classification ────────────────────────────────────────────────────

def is_generic_speaker(name: Optional[str]) -> bool:
    """True for auto-generated cluster labels ("Speaker 3", "SPEAKER_01")."""
    if not name:
        return False
    if _GENERIC_RE.match(name.strip()):
        return True
    return bool(_TIMESTAMPED_RE.match(name.strip()))


def is_unknown_label(name: Optional[str]) -> bool:
    """True for labels that carry no identity at all."""
    if name is None:
        return True
    n = str(name).strip()
    return n == "" or n.upper() == UNKNOWN_SPEAKER


def is_uncertain_label(name: Optional[str]) -> bool:
    """True when the label cannot be presented as a confident identity."""
    if is_unknown_label(name):
        return True
    if str(name).strip() == OVERLAP_SPEAKER:
        return True
    return is_generic_speaker(name)


def is_collectible_label(name: Optional[str]) -> bool:
    """Only a real, named identity may feed voiceprint auto-collection."""
    return not is_uncertain_label(name)


def has_speaker_identity(name: Optional[str]) -> bool:
    """True when *name* denotes an actual speaker slot (named OR generic cluster).

    Unlike :func:`is_collectible_label` this accepts generic "Speaker N"
    cluster labels: the diarizer legitimately produces them, they are just
    weaker evidence than a matched voiceprint (see ``speaker_source``).
    """
    if is_unknown_label(name):
        return False
    return str(name).strip() != OVERLAP_SPEAKER


# ── Confidence helpers ──────────────────────────────────────────────────────

def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_confident_attribution(segment: Any, cfg: Optional[dict] = None) -> bool:
    """True when *segment* may carry a confident speaker label.

    Checks (all must hold):
      1. the uncertainty policy is enabled;
      2. the label is a real identity (not UNKNOWN / OVERLAP / generic);
      3. no explicit ``uncertain`` flag;
      4. speaker confidence >= ``min_speaker_confidence``;
      5. voiced duration >= ``min_attribution_dur_sec``.
    """
    c = cfg if cfg is not None else _cfg()
    if not c.get("enabled", True):
        return True

    get = segment.get if isinstance(segment, dict) else lambda k, d=None: getattr(segment, k, d)
    name = get("speaker", None)
    if not has_speaker_identity(name):
        return False
    if get("uncertain", False):
        return False

    min_conf = float(c.get("min_speaker_confidence", _DEFAULTS["min_speaker_confidence"]))
    conf = _as_float(get("speaker_confidence", None))
    if conf is not None and conf < min_conf:
        return False

    min_margin = float(c.get("min_match_margin", _DEFAULTS["min_match_margin"]))
    margin = _as_float(get("speaker_margin", None))
    if margin is not None and margin < min_margin:
        return False

    dur = _as_float(get("end", 0.0))
    dur_s = _as_float(get("start", 0.0))
    if dur is not None and dur_s is not None:
        span = dur - dur_s
        min_dur = float(c.get("min_attribution_dur_sec", _DEFAULTS["min_attribution_dur_sec"]))
        if 0 < span < min_dur:
            return False
    return True


def suppress_uncertain_speaker(
    segment: dict,
    reason: str = "low_confidence",
    source: Optional[str] = None,
) -> dict:
    """Return a copy of *segment* with the identity stripped.

    The interval and any text are preserved (unless the product policy says
    uncertain speech must be discarded), and the result is explicitly marked.
    """
    out = dict(segment)
    out["speaker"] = None
    out["speaker_confidence"] = 0.0
    out["speaker_source"] = source or SOURCE_UNKNOWN
    out["uncertain"] = True
    out["attribution_reason"] = reason
    out["original_speaker"] = segment.get("speaker")
    if not retain_uncertain_text():
        out["text"] = ""
    return out


def mark_confident(segment: dict, source: str = SOURCE_DIARIZATION) -> dict:
    """Annotate a segment whose identity is accepted (no-op fields)."""
    out = dict(segment)
    out["uncertain"] = False
    out.setdefault("speaker_source", source)
    out.setdefault("speaker_confidence", 1.0)
    out.pop("attribution_reason", None)
    return out


def apply_identity(
    segments: list,
    old_name: Optional[str],
    new_name: str,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    margin: Optional[float] = None,
    match_dist: Optional[float] = None,
) -> int:
    """Rename *old_name* -> *new_name* on matching segments, stamping evidence.

    Renaming a cluster is an *assertion about identity*, and the strength of
    that assertion has to travel with the data.  Without it every renamed
    segment reaches the API with ``speaker_confidence`` unset, and the
    attribution layer falls back to the geometric overlap ratio
    (``inside / dur``, i.e. 1.0 for a span fully inside its turn) — so a
    voiceprint match that was only 0.39 confident was reported as 1.0.

    Returns the number of segments stamped.
    """
    src = source or confidence_source_for(new_name)
    n = 0
    for seg in segments:
        if not isinstance(seg, dict) or seg.get("speaker") != old_name:
            continue
        seg["speaker"] = new_name
        seg["uncertain"] = False
        seg["speaker_source"] = src
        seg.pop("attribution_reason", None)
        if confidence is not None:
            seg["speaker_confidence"] = round(float(confidence), 3)
        if margin is not None:
            seg["speaker_margin"] = round(float(margin), 4)
        if match_dist is not None:
            seg["speaker_match_dist"] = round(float(match_dist), 4)
        n += 1
    return n


def confidence_source_for(name: Optional[str]) -> str:
    """Diarization clusters are weaker evidence than a DB voiceprint match."""
    if not has_speaker_identity(name):
        return SOURCE_UNKNOWN
    if is_generic_speaker(name):
        return SOURCE_DIARIZATION
    return SOURCE_KNOWN_VOICEPRINT


def eligible_for_auto_collect(segment: Any) -> bool:
    """Voiceprint auto-collection gate.

    Rejects UNKNOWN / OVERLAP / generic "Speaker N" labels, explicitly
    uncertain segments, low-confidence attributions and segments whose label
    was suppressed downstream.
    """
    c = _cfg()
    if is_uncertain_label(segment.get("speaker") if isinstance(segment, dict) else None):
        return False
    if not is_confident_attribution(segment, c):
        return False
    if c.get("auto_collect_requires_confidence", True):
        conf = _as_float(
            segment.get("speaker_confidence") if isinstance(segment, dict) else None
        )
        if conf is not None and conf < float(c.get("min_speaker_confidence", 0.35)):
            return False
    return True


def eligible_for_learning(segment: Any) -> bool:
    """Gate for LEARNING a new speaker from an unidentified cluster.

    Deliberately more permissive than :func:`eligible_for_auto_collect` in
    exactly one way: a generic ``Speaker N`` label is accepted, because the
    whole point is to learn someone who has no name yet.

    Everything that makes a label *unreliable* still rejects it: UNKNOWN,
    OVERLAP, explicitly-suppressed segments, and low-confidence attributions.
    Learning from a segment whose speaker identity the pipeline itself
    distrusted would poison the new profile with the same wrong audio.
    """
    name = segment.get("speaker") if isinstance(segment, dict) else None
    # NOT is_uncertain_label(): that also rejects generic "Speaker N", which is
    # the one label class learning exists to handle. Only the labels that carry
    # no speaker at all are rejected here.
    if not name or is_unknown_label(name) or str(name).strip() == OVERLAP_SPEAKER:
        return False
    if isinstance(segment, dict) and segment.get("uncertain"):
        return False
    c = _cfg()
    if c.get("auto_collect_requires_confidence", True):
        conf = _as_float(segment.get("speaker_confidence")
                         if isinstance(segment, dict) else None)
        # A generic cluster label carries no speaker_confidence at all; absence
        # is not evidence of doubt, so only an explicit low value rejects.
        if conf is not None and conf < float(c.get("min_speaker_confidence", 0.35)):
            return False
    return True


__all__ = [
    "UNKNOWN_SPEAKER",
    "OVERLAP_SPEAKER",
    "SOURCE_KNOWN_VOICEPRINT",
    "SOURCE_DIARIZATION",
    "SOURCE_UNKNOWN",
    "setting",
    "policy_enabled",
    "retain_uncertain_text",
    "is_generic_speaker",
    "is_unknown_label",
    "is_uncertain_label",
    "is_collectible_label",
    "has_speaker_identity",
    "is_confident_attribution",
    "suppress_uncertain_speaker",
    "mark_confident",
    "apply_identity",
    "confidence_source_for",
    "eligible_for_auto_collect",
    "eligible_for_learning",
]
