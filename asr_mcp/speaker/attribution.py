"""Post-hoc speaker attribution for ASR spans (pure, dependency-free).

The ASR backends decode the *whole* file in their own windows and return
timestamped items on the file timeline.  Each item is then attributed to a
diarization turn after the fact, because cutting the audio at speaker
boundaries degrades recognition.

Midpoint attribution alone is unsafe: a span that straddles a turn boundary
belongs mostly to the *other* speaker, yet the midpoint rule hands it to one of
them with a confident-looking label.  :func:`attribute_span` therefore returns
an explicit ``(speaker, confidence, source, reason)`` tuple and reports
``speaker=None`` whenever the identity is not established — per the policy in
:mod:`asr_mcp.speaker.uncertainty` (AGENTS.md lesson 30), the *identity* is
suppressed, never the text.

Kept free of torch/onnxruntime imports so it can be unit tested in isolation.
"""

from __future__ import annotations

import bisect
import logging

from asr_mcp.speaker.uncertainty import (
    SOURCE_UNKNOWN,
    confidence_source_for,
    has_speaker_identity,
    policy_enabled,
    setting,
)

logger = logging.getLogger("asr_mcp.speaker.attribution")


def turn_index_for_time(t, turns, starts):
    """Index of the turn owning time *t* (clamped to the first/last turn)."""
    if not turns:
        return None
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return 0
    if i >= len(turns):
        return len(turns) - 1
    return i


def speaker_for_span(start, end, turns, starts):
    """Diarization speaker for a time span: midpoint lookup in sorted turns."""
    if not turns:
        return None
    mid = (start + end) / 2.0
    i = turn_index_for_time(mid, turns, starts)
    turn = turns[i]
    if float(turn["start"]) <= mid <= float(turn["end"]):
        return turn.get("speaker")
    if i + 1 < len(turns):
        nxt = turns[i + 1]
        if mid - float(turn["end"]) <= float(nxt["start"]) - mid:
            return turn.get("speaker")
        return nxt.get("speaker")
    if i > 0:
        return turns[i - 1].get("speaker")
    return turn.get("speaker")


def turn_overlap(start, end, turn):
    """Seconds of [start, end) falling inside *turn*."""
    if turn is None:
        return 0.0
    return max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))


def attribute_span(start, end, turns, starts):
    """Speaker attribution for one ASR span, with explicit uncertainty.

    Pure function (no I/O) so it is unit-testable in isolation.

    Returns ``(speaker, confidence, source, reason)``.  ``speaker`` is None
    when the identity cannot be established:

    * there are no turns at all (``no_turns``);
    * the span's midpoint turn carries an uncertain label or was itself
      suppressed (``turn_uncertain`` / ``turn_label_uncertain``);
    * a **short** span (shorter than ``max_boundary_cross_span_sec``) is a
      *boundary crossing* — it overlaps the owning turn by less than
      ``max_boundary_cross_sec`` **and** less than ``(1 -
      max_boundary_cross_ratio)`` of its duration, so most of its audio belongs
      to somebody else.  This is the "Yes."-straddling-the-cut case: midpoint
      attribution cannot split it, so the identity is dropped, not guessed;
    * a **long** span is a *multi-speaker span*: it barely fits inside the
      owning turn, so no single turn owns it.  A long span is never called a
      "boundary crossing" (that test degenerates — any span longer than a few
      turns is always mostly outside one of them), but it is also not attributed
      unless the turn actually holds at least ``min_speaker_confidence`` of it
      (``low_span_turn_overlap``).
    """
    if not turns:
        return None, 0.0, SOURCE_UNKNOWN, "no_turns"

    idx = turn_index_for_time((start + end) / 2.0, turns, starts)
    turn = turns[idx]
    name = turn.get("speaker")

    dur = max(0.0, end - start)
    inside = turn_overlap(start, end, turn)
    if not policy_enabled():
        # Escape hatch (uncertainty.enabled = false): plain midpoint attribution,
        # no suppression at all — the pre-policy behaviour.
        return (
            name, 1.0, confidence_source_for(name), None,
        ) if has_speaker_identity(name) else (
            None, 0.0, SOURCE_UNKNOWN, "turn_label_uncertain",
        )

    max_cross = float(setting("max_boundary_cross_sec", 0.3))
    max_ratio = float(setting("max_boundary_cross_ratio", 0.34))
    max_span = float(setting("max_boundary_cross_span_sec", 3.0))
    min_share = float(setting("min_speaker_confidence", 0.35))
    share = (inside / dur) if dur > 0 else 1.0

    if dur > 0:
        cross_sec = dur - inside
        if dur <= max_span:
            if cross_sec > max_cross and (cross_sec / dur) > max_ratio:
                logger.info(
                    "Boundary crossing at %.2f-%.2fs: %.0f%% outside turn %d (%s) "
                    "-> speaker suppressed",
                    start, end, 100.0 * cross_sec / dur, idx, name,
                )
                return None, 0.0, SOURCE_UNKNOWN, "boundary_crossing"
        elif share < min_share:
            # Coarse backends emit spans much longer than a turn (one window =
            # one segment).  The turn owns too little of the span to name it.
            logger.info(
                "Multi-speaker span at %.2f-%.2fs: turn %d (%s) holds only "
                "%.0f%% of it -> speaker suppressed",
                start, end, idx, name, 100.0 * share,
            )
            return None, 0.0, SOURCE_UNKNOWN, "low_span_turn_overlap"

    if turn.get("uncertain"):
        return None, 0.0, SOURCE_UNKNOWN, turn.get("attribution_reason") or "turn_uncertain"
    if not has_speaker_identity(name):
        # UNKNOWN / OVERLAP / blank — the turn was suppressed upstream.
        return None, 0.0, SOURCE_UNKNOWN, "turn_label_uncertain"

    conf = turn.get("speaker_confidence")
    margin = turn.get("speaker_margin")
    # Overlap ratio of the span with its turn: 1.0 = fully inside.
    conf_f = float(conf) if isinstance(conf, (int, float)) else (inside / dur if dur > 0 else 1.0)
    if margin is not None and isinstance(margin, (int, float)):
        conf_f = min(conf_f, 1.0)
    return name, conf_f, confidence_source_for(name), None


def attribute_items(items, turns, starts):
    """Group consecutive ASR items into per-speaker runs.

    ``items`` are dicts with ``start``/``end``/``text`` on the file timeline.
    Adjacent items merge into one run only when they resolve to the **same
    non-None speaker**.  Uncertain items are never merged: two unknown spans may
    come from different turns (and different speakers), and gluing them would
    produce one huge UNKNOWN block that hides the real segmentation.  Each
    uncertain item therefore keeps its own run, tagged with the turn it landed
    in so neighbouring unknowns are still distinguishable.
    """
    runs: list[dict] = []
    for it in items:
        s = float(it.get("start") or 0.0)
        e = float(it.get("end") or s)
        spk, conf, source, reason = attribute_span(s, e, turns, starts)
        idx = turn_index_for_time((s + e) / 2.0, turns, starts) if turns else None
        if spk is None:
            key = (None, reason, idx)
        else:
            key = (spk, reason, None)
        if runs and runs[-1]["key"] == key:
            runs[-1]["items"].append(it)
        else:
            runs.append({
                "key": key, "speaker": spk, "confidence": conf,
                "source": source, "reason": reason, "items": [it],
            })
    return runs
