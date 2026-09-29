"""Live-turn speaker attribution.

Offline attribution maps ASR items onto diarization turns, but a live turn
arrives *before* any diarization exists.  Two signals are available instead,
and they are independent of each other:

* **the input channel** the turn came from — the client's microphone carries
  the local user, a loopback of the default output device carries whatever the
  machine played (the other side of a call, media, room speakers);
* **the stored voiceprints** of the authenticated user.

Neither is a diarization, and the distinction matters: a mic turn is *labelled*
local, never *identified* by embedding it, because the mic also picks up the
far end acoustically.  Identifying it would compare the local user's voice
against themselves and produce a confident, wrong answer.

The uncertainty policy is the same one the file path uses
(:mod:`asr_mcp.speaker.uncertainty`): anything not confidently attributable is
emitted with ``speaker=None`` and ``uncertain=True``.  A live turn is typically
1-3 s, which is at the low end for ECAPA, so a higher match bar than the file
path is the honest default rather than a bug.

Pure and dependency-free apart from lazy imports, so the gating logic is
unit-testable without a model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from asr_mcp.streaming.turn_detector import SAMPLE_RATE

logger = logging.getLogger("asr_mcp.streaming.attribution")

CHANNEL_MIC = 0
CHANNEL_SPEAKER = 1

#: Label used for the local user.  Chosen over "Speaker 1" so a transcript is
#: readable, but deliberately NOT a real name — the server cannot know the
#: user's name from their voice, only the client can.
LOCAL_SPEAKER_LABEL = os.environ.get("TRANSCRIBE_LIVE_LOCAL_LABEL", "You")

_DEFAULTS = {
    # Live turns are short; ECAPA is far less reliable below a few seconds.
    "min_match_confidence": 0.60,
    "min_match_margin": 0.05,
    "min_turn_sec": 0.5,
    "max_turn_sec": 30.0,
    "match_voiceprints": True,
}


def config() -> dict:
    """Live-attribution section of thresholds.json, merged over the defaults."""
    merged = dict(_DEFAULTS)
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        merged.update(raw.get("live_attribution") or {})
    except Exception as e:  # pragma: no cover - config unavailable
        logger.warning("Live attribution config unavailable, using defaults: %s", e)
    return merged


def _unknown(reason: str, conf: float = 0.0, margin: float = 0.0,
             match_dist: float = -1.0):
    return {
        "speaker": None,
        "speaker_source": "unknown",
        "speaker_confidence": round(float(conf), 3),
        "speaker_margin": round(float(margin), 4),
        "speaker_match_dist": round(float(match_dist), 4) if match_dist >= 0 else None,
        "uncertain": True,
        "attribution_reason": reason,
    }


def _identified(name: str, conf: float, margin: float, match_dist: float,
                source: str = "known_voiceprint"):
    return {
        "speaker": name,
        "speaker_source": source,
        "speaker_confidence": round(float(conf), 3),
        "speaker_margin": round(float(margin), 4),
        "speaker_match_dist": round(float(match_dist), 4),
        "uncertain": False,
        "attribution_reason": None,
    }


def attribute_live_turn(
    turn,
    channel: int,
    voiceprints: Optional[dict] = None,
    cfg: Optional[dict] = None,
    embed_fn=None,
    pitch_fn=None,
    energy_fn=None,
) -> dict:
    """Resolve a live turn to a speaker identity dict.

    ``turn`` only needs a ``duration_sec`` attribute.  ``embed_fn`` /
    ``pitch_fn`` / ``energy_fn`` are injectable so tests can exercise the
    gating logic without loading ECAPA or running a model.
    """
    cfg = cfg or config()
    duration = float(getattr(turn, "duration_sec", 0.0) or 0.0)

    if duration < float(cfg["min_turn_sec"]):
        return _unknown("live_turn_too_short", conf=0.0)
    if duration > float(cfg["max_turn_sec"]):
        return _unknown("live_turn_too_long", conf=0.0)

    if channel == CHANNEL_MIC:
        # The local user's own voice; identity comes from the input device, not
        # from an embedding (see module docstring).
        return _identified(
            LOCAL_SPEAKER_LABEL, conf=1.0, margin=1.0, match_dist=0.0,
            source="input_device",
        )

    if not bool(cfg.get("match_voiceprints", True)):
        return _unknown("live_match_disabled")
    if not voiceprints:
        return _unknown("no_voiceprints")
    if embed_fn is None:
        # Not a refusal: the caller simply did not provide the model hooks.
        return _unknown("live_match_unavailable")

    try:
        emb = embed_fn(turn)
        pitch = pitch_fn(turn) if pitch_fn else 0.0
        energy = energy_fn(turn) if energy_fn else 0.0
    except Exception as e:
        logger.warning("Live embedding failed: %s", e)
        return _unknown("live_match_failed")

    if emb is None or len(emb) == 0:
        return _unknown("live_match_failed")

    try:
        from asr_mcp.speaker.matcher import find_best_match

        best_name, best_dist, _best_conf, distances = find_best_match(
            list(emb), pitch, energy, voiceprints,
        )
    except Exception as e:
        logger.warning("Live voiceprint match failed: %s", e)
        return _unknown("live_match_failed")

    if not best_name or best_dist == float("inf"):
        return _unknown("no_voiceprint_match")

    # Confidence as the matcher reports it, plus the margin over the runner-up.
    best = distances.get(best_name) or {}
    conf = float(best.get("confidence", 0.0) or 0.0)
    others = sorted(
        (float(d.get("combined", 9.9)) for n, d in distances.items() if n != best_name)
    )
    margin = (others[0] - float(best_dist)) if others else float("inf")

    logger.info(
        "Live match channel=%s: %s dist=%.3f conf=%.2f margin=%s",
        channel, best_name, best_dist, conf,
        "inf" if margin == float("inf") else f"{margin:.3f}",
    )

    if conf < float(cfg["min_match_confidence"]):
        return _unknown("live_match_weak", conf=conf, margin=margin,
                        match_dist=best_dist)
    if margin < float(cfg["min_match_margin"]):
        return _unknown("live_match_ambiguous", conf=conf, margin=margin,
                        match_dist=best_dist)

    return _identified(best_name, conf, margin, best_dist)
