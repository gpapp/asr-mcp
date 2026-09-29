"""Attribution tests (file path).

The logic lives in :mod:`asr_mcp.speaker.attribution`, which is pure python, so
these run without torch/onnxruntime.  Under test: which turn owns an ASR span,
and whether the span crosses a diarization boundary badly enough that the
identity must be suppressed instead of guessed.
"""

import pytest

from asr_mcp.speaker import attribution as mod


def _turns(*pairs):
    """[(start, end, speaker, **extra), ...] -> (turns, starts)"""
    turns = []
    for start, end, speaker, *extra in pairs:
        t = {"start": start, "end": end, "speaker": speaker}
        for kv in extra:
            t.update(kv)
        turns.append(t)
    return turns, [t["start"] for t in turns]


TWO_SPEAKERS = _turns(
    (0.0, 10.0, "Speaker 1"),
    (10.0, 20.0, "Speaker 2"),
)


# --- turn lookup ----------------------------------------------------------


def test_turn_index_clamps():
    turns, starts = TWO_SPEAKERS
    assert mod.turn_index_for_time(-5.0, turns, starts) == 0
    assert mod.turn_index_for_time(5.0, turns, starts) == 0
    assert mod.turn_index_for_time(15.0, turns, starts) == 1
    assert mod.turn_index_for_time(999.0, turns, starts) == 1
    assert mod.turn_index_for_time(1.0, [], []) is None


def test_speaker_for_span_uses_midpoint():
    turns, starts = TWO_SPEAKERS
    assert mod.speaker_for_span(1.0, 2.0, turns, starts) == "Speaker 1"
    assert mod.speaker_for_span(18.0, 19.0, turns, starts) == "Speaker 2"
    assert mod.speaker_for_span(1.0, 2.0, [], []) is None


def test_speaker_for_span_nearest_when_outside_turn():
    # Gap between turns: a midpoint inside the gap goes to the nearer turn.
    turns, starts = _turns(
        (0.0, 4.0, "Speaker 1"),
        (6.0, 10.0, "Speaker 2"),
    )
    assert mod.speaker_for_span(4.5, 4.5, turns, starts) == "Speaker 1"
    assert mod.speaker_for_span(5.5, 5.5, turns, starts) == "Speaker 2"


def test_turn_overlap():
    turns, _ = TWO_SPEAKERS
    assert mod.turn_overlap(0.0, 5.0, turns[0]) == 5.0
    assert mod.turn_overlap(8.0, 12.0, turns[0]) == 2.0
    assert mod.turn_overlap(8.0, 12.0, turns[1]) == 2.0
    assert mod.turn_overlap(8.0, 12.0, None) == 0.0


# --- attribution ----------------------------------------------------------


def test_span_inside_turn_is_attributed():
    turns, starts = TWO_SPEAKERS
    spk, conf, source, reason = mod.attribute_span(2.0, 4.0, turns, starts)
    assert spk == "Speaker 1"
    assert reason is None
    assert source == "diarization_cluster"
    assert 0.0 < conf <= 1.0


def test_named_voiceprint_source():
    turns, starts = _turns((0.0, 10.0, "Bob", {"speaker_confidence": 0.9, "speaker_margin": 0.4}))
    spk, conf, source, reason = mod.attribute_span(1.0, 5.0, turns, starts)
    assert (spk, source, reason) == ("Bob", "known_voiceprint", None)
    assert conf == pytest.approx(0.9)


def test_confidence_defaults_to_overlap_ratio():
    turns, starts = TWO_SPEAKERS
    # 0.9s span, 0.6s inside the turn -> confidence 0.6/0.9.
    spk, conf, _, _ = mod.attribute_span(9.7, 10.6, turns, starts)
    assert spk == "Speaker 2"
    assert conf == pytest.approx(0.6 / 0.9)


def test_no_turns_is_unknown():
    spk, conf, source, reason = mod.attribute_span(1.0, 2.0, [], [])
    assert spk is None
    assert source == "unknown"
    assert reason == "no_turns"
    assert conf == 0.0


def test_uncertain_turn_propagates_reason():
    turns, starts = _turns((0.0, 10.0, None, {
        "uncertain": True, "attribution_reason": "ghost_speaker"}))
    spk, conf, source, reason = mod.attribute_span(2.0, 4.0, turns, starts)
    assert spk is None
    assert reason == "ghost_speaker"
    assert source == "unknown"
    assert conf == 0.0


def test_uncertain_turn_without_reason():
    turns, starts = _turns((0.0, 10.0, "Speaker 3", {"uncertain": True}))
    spk, _, _, reason = mod.attribute_span(2.0, 4.0, turns, starts)
    assert spk is None
    assert reason == "turn_uncertain"


def test_uncertain_label_is_unknown():
    turns, starts = _turns((0.0, 10.0, "UNKNOWN"))
    spk, _, source, reason = mod.attribute_span(2.0, 4.0, turns, starts)
    assert spk is None
    assert reason == "turn_label_uncertain"
    assert source == "unknown"


def test_boundary_crossing_is_suppressed():
    # A short answer starting just before the turn boundary: most of its audio
    # belongs to the other speaker, so midpoint attribution would be wrong.
    turns, starts = TWO_SPEAKERS
    spk, conf, source, reason = mod.attribute_span(9.2, 10.6, turns, starts)
    assert spk is None
    assert reason == "boundary_crossing"
    assert source == "unknown"
    assert conf == 0.0


def test_span_just_inside_turn_is_kept():
    turns, starts = TWO_SPEAKERS
    # 0.2s bleed is at/below max_boundary_cross_sec -> still Speaker 2.
    spk, _, _, reason = mod.attribute_span(9.8, 10.6, turns, starts)
    assert spk == "Speaker 2"
    assert reason is None


def test_long_span_crossing_is_kept():
    # Only *mostly* outside counts as a crossing: a 6s span with 0.4s outside
    # the turn is still Speaker 2's audio.
    turns, starts = TWO_SPEAKERS
    spk, _, _, reason = mod.attribute_span(9.6, 15.6, turns, starts)
    assert spk == "Speaker 2"
    assert reason is None


def test_tiny_bleed_is_not_a_crossing():
    # 0.2s outside is below max_boundary_cross_sec, so it stays attributed.
    turns, starts = TWO_SPEAKERS
    spk, _, _, reason = mod.attribute_span(9.8, 11.0, turns, starts)
    assert spk == "Speaker 2"
    assert reason is None


def test_zero_length_span_uses_midpoint():
    turns, starts = TWO_SPEAKERS
    spk, conf, _, reason = mod.attribute_span(5.0, 5.0, turns, starts)
    assert spk == "Speaker 1"
    assert reason is None
    assert conf > 0.0


# --- long spans (coarse backends) -----------------------------------------


def test_long_span_inside_one_turn_is_attributed():
    turns, starts = _turns((0.0, 30.0, "Speaker 1"), (30.0, 60.0, "Speaker 2"))
    spk, conf, source, reason = mod.attribute_span(0.0, 29.0, turns, starts)
    assert spk == "Speaker 1"
    assert reason is None
    assert source == "diarization_cluster"
    assert conf > 0.9


def test_long_span_spanning_two_turns_is_suppressed():
    """A 30s span that a turn holds only a sliver of cannot be named."""
    turns, starts = _turns(
        (0.0, 4.0, "Speaker 1"),
        (4.0, 8.0, "Speaker 2"),
        (8.0, 12.0, "Speaker 1"),
    )
    spk, conf, source, reason = mod.attribute_span(0.0, 30.0, turns, starts)
    assert spk is None
    assert reason == "low_span_turn_overlap"
    assert source == "unknown"
    assert conf == 0.0


def test_long_span_reason_is_not_boundary_crossing():
    # 30s span over a 20s turn: inside share is 0.67 (>= min_speaker_confidence),
    # so it IS attributed even though 33% is outside.
    turns, starts = _turns((5.0, 25.0, "Speaker 1"))
    spk, conf, _, reason = mod.attribute_span(0.0, 30.0, turns, starts)
    assert spk == "Speaker 1"
    assert reason is None
    assert conf == pytest.approx(20.0 / 30.0)


def test_short_span_still_uses_boundary_crossing():
    # Under max_boundary_cross_span_sec the boundary rule is unchanged.
    turns, starts = TWO_SPEAKERS
    spk, _, _, reason = mod.attribute_span(9.2, 10.6, turns, starts)
    assert spk is None
    assert reason == "boundary_crossing"


# --- run grouping ---------------------------------------------------------


def _items(*spans):
    return [{"start": s, "end": e, "text": f"t{s}"} for s, e in spans]


def test_runs_merge_same_speaker_items():
    turns, starts = _turns((0.0, 30.0, "Speaker 1"))
    runs = mod.attribute_items(_items((1.0, 5.0), (5.0, 9.0), (9.0, 12.0)), turns, starts)
    assert len(runs) == 1
    assert runs[0]["speaker"] == "Speaker 1"
    assert len(runs[0]["items"]) == 3


def test_runs_never_merge_uncertain_items():
    """Two UNKNOWN spans stay separate even with the same reason."""
    turns, starts = _turns(
        (0.0, 4.0, None),
        (4.0, 8.0, None),
    )
    runs = mod.attribute_items(_items((0.0, 3.0), (4.5, 7.5)), turns, starts)
    assert len(runs) == 2
    assert all(r["speaker"] is None for r in runs)


def test_runs_split_on_speaker_change():
    turns, starts = _turns((0.0, 10.0, "Speaker 1"), (10.0, 20.0, "Speaker 2"))
    runs = mod.attribute_items(_items((1.0, 4.0), (11.0, 14.0)), turns, starts)
    assert [r["speaker"] for r in runs] == ["Speaker 1", "Speaker 2"]


def test_runs_of_empty_items_is_empty():
    assert mod.attribute_items([], TWO_SPEAKERS[0], TWO_SPEAKERS[1]) == []


# --- policy escape hatch --------------------------------------------------


def test_disabled_policy_uses_plain_midpoint(monkeypatch):
    """uncertainty.enabled = false must restore pre-policy attribution."""
    # attribution.py imports the helper by value, so patch it in its namespace.
    monkeypatch.setattr(mod, "policy_enabled", lambda: False)
    turns, starts = TWO_SPEAKERS
    # A span that the policy suppresses is attributed anyway when disabled.
    spk, conf, source, reason = mod.attribute_span(9.2, 10.6, turns, starts)
    assert spk == "Speaker 1"
    assert reason is None
    assert conf == 1.0
    assert source == "diarization_cluster"
    # An uncertain turn still has no identity to report.
    turns2, starts2 = _turns((0.0, 10.0, "UNKNOWN"))
    assert mod.attribute_span(1.0, 2.0, turns2, starts2)[2] == "unknown"
