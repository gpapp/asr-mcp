"""Diarization cleanup must SUPPRESS uncertain identities, not remap them.

Regression tests for the false-attribution bug: a short/ghost/minority cluster
used to be force-fitted to the temporally nearest dominant speaker, producing
confident-looking but wrong labels.  Now the identity is dropped
(``speaker=None`` + ``uncertain``) while the interval and text survive.
"""

import pytest


def _seg(start, end, speaker, **kw):
    d = {"start": start, "end": end, "speaker": speaker}
    d.update(kw)
    return d


# --- minority speakers ----------------------------------------------------


def test_minority_suppressed_not_remapped(segment_ops):
    segs = [
        _seg(0.0, 10.0, "Speaker 1"),
        _seg(12.0, 15.0, "Speaker 2", text="Yes?"),
    ]
    out = segment_ops.absorb_minority_speakers(segs, suppress=True)
    short = out[-1]
    assert short["speaker"] is None
    assert short["uncertain"] is True
    assert short["attribution_reason"] == "minority_speaker"
    assert short["text"] == "Yes?"
    assert short["start"] == 12.0 and short["end"] == 15.0
    assert out[0]["speaker"] == "Speaker 1"


def test_minority_legacy_remap_still_available(segment_ops):
    segs = [
        _seg(0.0, 10.0, "Speaker 1"),
        _seg(11.0, 13.0, "Speaker 2", text="Yes?"),
    ]
    out = segment_ops.absorb_minority_speakers(segs, suppress=False)
    assert out[-1]["speaker"] == "Speaker 1"


def test_minority_protected_voiceprint_kept(segment_ops):
    segs = [
        _seg(0.0, 10.0, "Speaker 1"),
        _seg(11.0, 13.0, "Bob", text="short but known"),
        _seg(14.0, 15.0, "Speaker 2", text="tiny"),
    ]
    out = segment_ops.absorb_minority_speakers(
        segs, protected_speakers={"Bob"}, suppress=True)
    by_start = {s["start"]: s for s in out}
    assert by_start[11.0]["speaker"] == "Bob"
    assert by_start[14.0]["speaker"] is None


def test_minority_ignores_already_suppressed(segment_ops):
    segs = [
        _seg(0.0, 10.0, "Speaker 1"),
        _seg(11.0, 13.0, None, uncertain=True, attribution_reason="boundary_crossing"),
    ]
    out = segment_ops.absorb_minority_speakers(segs, suppress=True)
    assert out[-1]["attribution_reason"] == "boundary_crossing"
    assert out[-1]["speaker"] is None


def test_minority_all_main_untouched(segment_ops):
    segs = [_seg(0.0, 10.0, "Speaker 1"), _seg(11.0, 25.0, "Speaker 2")]
    out = segment_ops.absorb_minority_speakers(segs, suppress=True)
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2"]


# --- ghost speakers -------------------------------------------------------


def test_ghost_suppressed_not_nearest(segment_ops):
    # The ghost sits much closer in time to Speaker 3 — the old code picked it.
    segs = [
        _seg(0.0, 20.0, "Speaker 1"),
        _seg(20.0, 21.0, "Speaker 2", text="mhm"),
        _seg(21.5, 60.0, "Speaker 3"),
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    ghost = [s for s in out if s["start"] == 20.0][0]
    assert ghost["speaker"] is None
    assert ghost["uncertain"] is True
    assert ghost["attribution_reason"] == "ghost_speaker"
    assert ghost["text"] == "mhm"


def test_ghost_legacy_nearest_reassignment(segment_ops):
    segs = [
        _seg(0.0, 20.0, "Speaker 1"),
        _seg(20.0, 21.0, "Speaker 2"),
        _seg(22.6, 60.0, "Speaker 3"),
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=False)
    ghost = out[0]
    # Legacy behaviour: pure temporal proximity invents an identity and the
    # ghost's interval is then collapsed into that speaker.
    assert ghost["speaker"] == "Speaker 1"
    assert not ghost.get("uncertain")
    assert ghost["end"] == 21.0


def test_ghost_keeps_matched_alternative(segment_ops):
    segs = [
        _seg(0.0, 20.0, "Speaker 1"),
        _seg(21.0, 25.0, "Speaker 2", alternatives=[{"speaker": "Bob", "confidence": 0.8}]),
        _seg(30.0, 60.0, "Speaker 3"),
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    ghost = [s for s in out if s["start"] == 21.0][0]
    # Positive evidence (a real voiceprint match) still wins over suppression.
    assert ghost["speaker"] == "Bob"
    assert ghost["speaker_source"] == "known_voiceprint"
    assert not ghost.get("uncertain")


def test_ghost_removed_from_profiles(segment_ops):
    profiles = {"Speaker 1": {"total_speech_sec": 20.0}, "Speaker 2": {"total_speech_sec": 1.0}}
    segs = [_seg(0.0, 20.0, "Speaker 1"), _seg(21.0, 22.0, "Speaker 2")]
    segment_ops.eliminate_ghost_speakers(
        segs, profiles=profiles, ghost_threshold_sec=10.0, suppress=True)
    assert "Speaker 2" not in profiles
    assert "Speaker 1" in profiles


def test_ghost_preserves_existing_suppression_reason(segment_ops):
    segs = [
        _seg(0.0, 20.0, "Speaker 1"),
        _seg(21.0, 22.0, None, uncertain=True, attribution_reason="boundary_crossing"),
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    assert out[-1]["attribution_reason"] == "boundary_crossing"


def test_ghost_none_ghost_returns_unchanged(segment_ops):
    segs = [_seg(0.0, 20.0, "Speaker 1"), _seg(21.0, 40.0, "Speaker 2")]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2"]


# --- collapsing -----------------------------------------------------------


def test_collapse_confident_only_merges_named(segment_ops):
    segs = [
        _seg(0.0, 1.0, "Speaker 1"),
        _seg(1.2, 2.0, "Speaker 1"),
        _seg(2.2, 3.0, "Speaker 2"),
    ]
    out = segment_ops._collapse_confident_only(segs)
    assert len(out) == 2
    assert out[0]["end"] == 2.0


def test_collapse_confident_only_keeps_suppressed_separate(segment_ops):
    # Two adjacent UNKNOWN spans are NOT necessarily the same person.
    segs = [
        _seg(0.0, 1.0, None, uncertain=True, attribution_reason="ghost_speaker"),
        _seg(1.1, 2.0, None, uncertain=True, attribution_reason="ghost_speaker"),
    ]
    out = segment_ops._collapse_confident_only(segs)
    assert len(out) == 2
    assert all(s["speaker"] is None for s in out)


def test_collapse_confident_only_respects_gap(segment_ops):
    segs = [_seg(0.0, 1.0, "Speaker 1"), _seg(9.0, 10.0, "Speaker 1")]
    assert len(segment_ops._collapse_confident_only(segs)) == 2


def test_collapse_confident_only_joins_text(segment_ops):
    segs = [
        _seg(0.0, 1.0, "Speaker 1", text="hello"),
        _seg(1.0, 2.0, "Speaker 1", text="world"),
    ]
    out = segment_ops._collapse_confident_only(segs)
    assert out[0]["text"] == "hello world"


# --- ghost share guard ----------------------------------------------------


def test_ghost_but_large_share_is_kept(segment_ops, monkeypatch):
    """A 9.7s speaker is a ghost by duration but holds 26% of the audio.

    On a 60s clip the absolute 10s rule is a knife edge, so a substantial
    share must survive — otherwise the whole file turns into UNKNOWN.
    """
    monkeypatch.setattr(segment_ops, "setting",
                        lambda key, default=None: {"ghost_max_share": 0.25}.get(key, default))
    segs = [
        _seg(0.0, 2.0, "Speaker 1"),     # 2.0s  (5%)  -> ghost
        _seg(2.0, 11.7, "Speaker 2"),   # 9.7s  (26%) -> kept
        _seg(12.0, 38.0, "Speaker 3"),  # 26.0s (69%) -> not a ghost anyway
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    # Speaker 1 really is a 2s blip -> suppressed; Speaker 2 survives on share.
    assert [s["speaker"] for s in out] == [None, "Speaker 2", "Speaker 3"]
    assert out[0]["attribution_reason"] == "ghost_speaker"
    assert not any(s.get("uncertain") for s in out[1:])


def test_small_share_ghost_is_still_suppressed(segment_ops, monkeypatch):
    monkeypatch.setattr(segment_ops, "setting",
                        lambda key, default=None: {"ghost_max_share": 0.25}.get(key, default))
    segs = [
        _seg(0.0, 2.0, "Speaker 1"),
        _seg(3.0, 60.0, "Speaker 2"),
    ]
    out = segment_ops.eliminate_ghost_speakers(segs, ghost_threshold_sec=10.0, suppress=True)
    assert out[0]["speaker"] is None
    assert out[0]["attribution_reason"] == "ghost_speaker"
    assert out[1]["speaker"] == "Speaker 2"
