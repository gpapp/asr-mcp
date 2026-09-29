"""Result-merging tests (asr_router).

``asr_router`` pulls in the whole web + audio stack (torch via
``asr_mcp.speaker.vad``), so these are skipped in a lightweight environment.
"""

import pytest

pytest.importorskip("fastapi", reason="asr_router needs the web dependencies")
pytest.importorskip("pydantic", reason="asr_router needs the web dependencies")
pytest.importorskip("torch", reason="asr_router pulls in the audio stack")

from asr_mcp.api import asr_router  # noqa: E402


def _result(start, end, text, speaker, **kw):
    return asr_router.TranscribeResult(
        start=start, end=end, text=text, speaker=speaker,
        segments=[{"start": start, "end": end, "text": text}], **kw)


def test_merge_results_keeps_unknown_separate():
    results = [
        _result(0.0, 1.0, "a", "Speaker 1"),
        _result(1.2, 2.0, "b", None, uncertain=True,
                attribution_reason="boundary_crossing"),
        _result(2.2, 3.0, "c", "Speaker 1"),
    ]
    out = asr_router._merge_consecutive_same_speaker_results(results)
    # Merging is sequential over the *previous* result, so the UNKNOWN run also
    # splits the two Speaker 1 runs — they must not be glued across a gap whose
    # owner is unknown.
    assert len(out) == 3
    assert [r.text for r in out] == ["a", "b", "c"]
    assert out[1].speaker is None
    assert out[1].uncertain is True
    assert out[1].attribution_reason == "boundary_crossing"


def test_merge_results_joins_adjacent_runs():
    results = [
        _result(0.0, 1.0, "a", "Speaker 1"),
        _result(1.0, 2.0, "b", "Speaker 1"),
    ]
    out = asr_router._merge_consecutive_same_speaker_results(results)
    assert len(out) == 1
    assert out[0].text == "a b"
    assert out[0].end == 2.0
    assert len(out[0].segments) == 2


def test_merge_results_respects_default_gap():
    # Default result_merge_gap_sec is 1.0s, so a 2s pause splits the run even
    # though both sides are the same speaker (paragraph breaks are produced by
    # _apply_paragraph_breaks, not by a wide merge bridge).
    results = [
        _result(0.0, 1.0, "Yes.", "Speaker 1"),
        _result(3.0, 4.0, "Next.", "Speaker 1"),
    ]
    out = asr_router._merge_consecutive_same_speaker_results(results)
    assert len(out) == 2


def test_merge_results_paragraph_break_after_sentence():
    # With a wider explicit gap, a >= 2s pause after a sentence end becomes a
    # paragraph break inside one result.
    results = [
        _result(0.0, 1.0, "Yes.", "Speaker 1"),
        _result(3.0, 4.0, "Next.", "Speaker 1"),
    ]
    out = asr_router._merge_consecutive_same_speaker_results(results, max_gap_sec=3.0)
    assert len(out) == 1
    assert out[0].text == "Yes.\n\nNext."


def test_merge_results_respects_gap_limit():
    results = [
        _result(0.0, 1.0, "a", "Speaker 1"),
        _result(5.0, 6.0, "b", "Speaker 1"),
    ]
    out = asr_router._merge_consecutive_same_speaker_results(results)
    assert len(out) == 2


def test_merge_into_turns_never_merges_unknown():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": None},
        {"start": 1.1, "end": 2.0, "speaker": None},
        {"start": 5.0, "end": 6.0, "speaker": None},
    ]
    turns = asr_router._merge_into_turns(segments)
    assert len(turns) == 3
    assert all(t["speaker"] is None for t in turns)


def test_merge_into_turns_merges_same_speaker_within_gap():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
        {"start": 1.5, "end": 2.0, "speaker": "Speaker 1"},
    ]
    turns = asr_router._merge_into_turns(segments)
    assert len(turns) == 1
    assert turns[0]["end"] == 2.0
    assert len(turns[0]["segments"]) == 2


def test_turns_carry_the_identity_evidence():
    """A rename's confidence must survive turn building.

    `attribute_span` reports the turn's `speaker_confidence` when present and
    otherwise falls back to the geometric overlap ratio (1.0 for a span fully
    inside its turn) — so dropping it turns a 0.55-confidence voiceprint
    match into an apparently certain identity at the API.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Gergely Papp",
         "speaker_confidence": 0.55, "speaker_source": "known_voiceprint",
         "speaker_margin": 0.11, "speaker_match_dist": 0.225},
        {"start": 1.2, "end": 2.0, "speaker": "Gergely Papp"},
    ]
    turns = asr_router._merge_into_turns(segments)
    assert len(turns) == 1
    assert turns[0]["speaker_confidence"] == 0.55
    assert turns[0]["speaker_source"] == "known_voiceprint"
    assert turns[0]["speaker_match_dist"] == 0.225


def test_split_long_turn_keeps_the_identity_evidence():
    segments = [
        {"start": float(i * 40), "end": float(i * 40 + 39), "speaker": "Speaker 1",
         "speaker_confidence": 0.8, "speaker_source": "known_voiceprint"}
        for i in range(6)
    ]
    turn = asr_router._merge_into_turns(segments)[0]
    parts = asr_router._split_long_turn(turn)
    assert len(parts) > 1
    for p in parts:
        assert p["speaker_confidence"] == 0.8
        assert p["speaker_source"] == "known_voiceprint"
