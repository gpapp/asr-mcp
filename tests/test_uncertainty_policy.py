"""Unit tests for the uncertain-speaker policy (AGENTS.md lesson 30).

The policy suppresses the speaker *identity*, never the spoken content.
"""

import pytest


def _seg(**kw):
    base = {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "hello"}
    base.update(kw)
    return base


# --- label classification -------------------------------------------------


def test_is_generic_speaker(uncertainty):
    assert uncertainty.is_generic_speaker("Speaker 3")
    assert uncertainty.is_generic_speaker("SPEAKER_00")
    assert uncertainty.is_generic_speaker("speaker 12")
    assert not uncertainty.is_generic_speaker("Gergely Papp")
    assert not uncertainty.is_generic_speaker("Speakerette")
    assert not uncertainty.is_generic_speaker(None)


def test_is_uncertain_label(uncertainty):
    # Generic cluster labels are never a confident identity.
    assert uncertainty.is_uncertain_label(None)
    assert uncertainty.is_uncertain_label("")
    assert uncertainty.is_uncertain_label("UNKNOWN")
    assert uncertainty.is_uncertain_label("Speaker 2")
    assert uncertainty.is_uncertain_label("OVERLAP")
    assert not uncertainty.is_uncertain_label("Bob")


def test_is_collectible_label(uncertainty):
    assert uncertainty.is_collectible_label("Bob")
    assert not uncertainty.is_collectible_label(None)
    assert not uncertainty.is_collectible_label("Speaker 2")


# --- suppression ----------------------------------------------------------


def test_suppress_keeps_text_and_interval(uncertainty):
    out = uncertainty.suppress_uncertain_speaker(
        _seg(start=1.0, end=1.4, text="Yes"), reason="boundary_crossing")
    assert out["speaker"] is None
    assert out["uncertain"] is True
    assert out["speaker_confidence"] == 0.0
    assert out["speaker_source"] == "unknown"
    assert out["attribution_reason"] == "boundary_crossing"
    # The spoken content is never dropped by default.
    assert out["text"] == "Yes"
    assert out["start"] == 1.0 and out["end"] == 1.4
    assert out["original_speaker"] == "Speaker 1"


def test_suppress_does_not_mutate_input(uncertainty):
    seg = _seg(start=2.0, end=2.2, text="No")
    out = uncertainty.suppress_uncertain_speaker(seg, reason="ghost_speaker")
    assert seg["speaker"] == "Speaker 1"
    assert out is not seg


def test_suppress_can_discard_text(uncertainty, monkeypatch):
    monkeypatch.setattr(uncertainty, "retain_uncertain_text", lambda: False)
    out = uncertainty.suppress_uncertain_speaker(_seg(text="maybe"), reason="x")
    assert out["text"] == ""


# --- confidence gate ------------------------------------------------------


def test_confidence_threshold(uncertainty):
    seg = _seg(speaker_confidence=0.2, speaker_margin=0.5)
    assert not uncertainty.is_confident_attribution(seg)
    seg["speaker_confidence"] = 0.9
    assert uncertainty.is_confident_attribution(seg)


def test_margin_threshold(uncertainty):
    seg = _seg(speaker_confidence=0.9, speaker_margin=0.0)
    assert not uncertainty.is_confident_attribution(seg)
    seg["speaker_margin"] = 0.2
    assert uncertainty.is_confident_attribution(seg)


def test_min_attribution_duration(uncertainty):
    # A very short confident-looking match is still not enough.
    seg = _seg(start=5.0, end=5.2, speaker_confidence=0.95, speaker_margin=0.4)
    assert not uncertainty.is_confident_attribution(seg)
    seg["end"] = 5.6
    assert uncertainty.is_confident_attribution(seg)


def test_uncertain_flag_overrides(uncertainty):
    seg = _seg(speaker_confidence=0.99, speaker_margin=0.9, uncertain=True)
    assert not uncertainty.is_confident_attribution(seg)


def test_unknown_label_never_confident(uncertainty):
    assert not uncertainty.is_confident_attribution(_seg(speaker=None, uncertain=True))
    assert not uncertainty.is_confident_attribution(
        _seg(speaker=None, speaker_confidence=0.99, speaker_margin=0.9))


def test_policy_disabled_keeps_legacy(uncertainty):
    # Turning the policy off restores the previous "trust the label" behaviour.
    seg = _seg(speaker_confidence=0.01, speaker_margin=0.0)
    assert not uncertainty.is_confident_attribution(seg)
    assert uncertainty.is_confident_attribution(seg, {"enabled": False})


# --- sources --------------------------------------------------------------


def test_confidence_source_for(uncertainty):
    assert uncertainty.confidence_source_for(None) == "unknown"
    assert uncertainty.confidence_source_for("Speaker 2") == "diarization_cluster"
    assert uncertainty.confidence_source_for("Bob") == "known_voiceprint"


# --- auto-collection eligibility -----------------------------------------


def test_auto_collect_rejects_unknown(uncertainty):
    assert not uncertainty.eligible_for_auto_collect(_seg(speaker=None, uncertain=True))
    assert not uncertainty.eligible_for_auto_collect(_seg(speaker="UNKNOWN"))
    assert not uncertainty.eligible_for_auto_collect(_seg(speaker="OVERLAP"))


def test_auto_collect_rejects_generic_labels(uncertainty):
    # Generic "Speaker N" clusters never feed a voiceprint (lesson 29).
    assert not uncertainty.eligible_for_auto_collect(
        _seg(speaker="Speaker 2", speaker_confidence=0.9, speaker_margin=0.5))


def test_auto_collect_rejects_low_confidence(uncertainty):
    seg = _seg(speaker="Bob", start=0.0, end=3.0,
               speaker_confidence=0.1, speaker_margin=0.5)
    assert not uncertainty.eligible_for_auto_collect(seg)
    seg["speaker_confidence"] = 0.8
    assert uncertainty.eligible_for_auto_collect(seg)


def test_auto_collect_rejects_short(uncertainty):
    seg = _seg(speaker="Bob", start=0.0, end=0.3,
               speaker_confidence=0.9, speaker_margin=0.3)
    assert not uncertainty.eligible_for_auto_collect(seg)


def test_auto_collect_accepts_named_confident(uncertainty):
    seg = _seg(speaker="Bob", start=0.0, end=4.0,
               speaker_confidence=0.8, speaker_margin=0.3)
    assert uncertainty.eligible_for_auto_collect(seg)


# --- config plumbing ------------------------------------------------------


def test_setting_reads_config(uncertainty):
    assert uncertainty.setting("min_speaker_confidence") == 0.35
    assert uncertainty.setting("retain_uncertain_text") is True
    assert uncertainty.setting("no_such_key", "fallback") == "fallback"


# --- apply_identity ------------------------------------------------------


def test_apply_identity_stamps_evidence(uncertainty):
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "uncertain": True,
         "attribution_reason": "stale"},
        {"start": 1.0, "end": 2.0, "speaker": "Speaker 2"},
    ]
    n = uncertainty.apply_identity(
        segs, "Speaker 1", "Gergely Papp",
        confidence=0.55, margin=0.11, match_dist=0.225,
    )
    assert n == 1
    hit = segs[0]
    assert hit["speaker"] == "Gergely Papp"
    assert hit["speaker_confidence"] == 0.55
    assert hit["speaker_source"] == "known_voiceprint"
    assert hit["speaker_margin"] == 0.11
    assert hit["speaker_match_dist"] == 0.225
    assert hit["uncertain"] is False
    assert "attribution_reason" not in hit
    assert segs[1]["speaker"] == "Speaker 2", "must not touch other clusters"


def test_apply_identity_uses_diarization_source_for_generic_names(uncertainty):
    segs = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 3"}]
    uncertainty.apply_identity(segs, "Speaker 3", "Speaker 1")
    assert segs[0]["speaker_source"] == "diarization_cluster"
    assert "speaker_confidence" not in segs[0], "no evidence -> no claim"


# --------------------------------------------------------------------------
# naming_blocked_reason: a collapsed cluster may not be named at all
# --------------------------------------------------------------------------
#
# Measured on ZO249 (3439s, 2 hosts + inserted clips) forced to
# num_speakers=2: 1376 windows -> 2 clusters, c0=84.8% / c1=15.2%. BOTH were
# renamed to the same known speaker, and the WRONG cluster scored the better
# match (combined 0.105 vs 0.237) -- so neither a confidence threshold nor a
# one-to-one claim check catches it. Only refusing to name can.

def test_a_balanced_cluster_allows_naming(uncertainty):
    assert uncertainty.naming_blocked_reason([600, 300, 200]) is None


def test_a_collapsed_cluster_blocks_naming(uncertainty):
    reason = uncertainty.naming_blocked_reason([1167, 209])
    assert reason is not None
    assert "cluster_collapse" in reason
    assert "85" in reason


def test_a_single_cluster_is_never_blocked(uncertainty):
    # One cluster is the correct result for a genuinely single-speaker
    # recording, so there is no evidence of collapse to act on.
    assert uncertainty.naming_blocked_reason([1000]) is None
    assert uncertainty.naming_blocked_reason([]) is None
    assert uncertainty.naming_blocked_reason(None) is None


def test_the_naming_block_threshold_is_configurable(uncertainty):
    # ZO249's default-path balance (56.3% largest) must not be blocked...
    assert uncertainty.naming_blocked_reason([775, 601]) is None
    # ...but it is blocked once the operator tightens the limit.
    assert uncertainty.naming_blocked_reason([775, 601], {"max_share_for_naming": 0.5})


def test_zero_sized_clusters_are_ignored(uncertainty):
    assert uncertainty.naming_blocked_reason([10, 0, 0]) is None
