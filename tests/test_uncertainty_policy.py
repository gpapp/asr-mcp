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


# --- the naming / learning band -------------------------------------------


def _confidence_of_distance(cfg, distance):
    """The exact conversion speaker/matcher.py applies.

    Reimplemented rather than imported because the band is about the CONTRACT
    between two modules that both read thresholds.json. If matcher.py ever
    changes its formula, this helper is the one place that has to change with
    it, and the assertions below will say which side moved.
    """
    conf_max_dist = cfg["normalization"]["confidence_max_distance"]
    return max(0.0, 1.0 - distance / conf_max_dist)


def _cfg():
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    return json.loads((root / "asr_mcp" / "config" / "thresholds.json").read_text())


def test_naming_accepts_a_wider_band_than_learning_uses(uncertainty):
    """The two gates are in different units; the gap between them is real.

    Naming compares a DISTANCE (matching.accept_threshold), learning compares a
    CONFIDENCE (uncertainty.min_speaker_confidence), and matcher.py maps one to
    the other with confidence_max_distance. Today that leaves a 0.025 band where
    a span is shown under the speaker's NAME and its audio is refused for
    learning. The refusal is reported (reason low_confidence, with the measured
    distance), so it is a visible decision rather than a silent loss -- but it
    must be visible BY DESIGN, which is what this test is for.
    """
    cfg = _cfg()
    accept = float(cfg["matching"]["accept_threshold"])
    min_conf = float(cfg["uncertainty"]["min_speaker_confidence"])

    named_edge = _confidence_of_distance(cfg, accept)
    assert named_edge < min_conf, (
        "naming and learning now use the same threshold: the band closed, so "
        "this test and the note in thresholds.json need updating deliberately")

    # Inside the band: named by the matcher, refused by the learning gate.
    band_distance = (accept + (1.0 - min_conf) * cfg["normalization"]
                    ["confidence_max_distance"]) / 2
    in_band = _seg(speaker="Test Person One", start=0.0, end=4.0,
                   speaker_confidence=_confidence_of_distance(cfg, band_distance),
                   speaker_margin=0.3)
    assert in_band["speaker_confidence"] < min_conf
    assert uncertainty.learning_refusal_reason(in_band) == "low_confidence"

    # Comfortably inside the naming threshold: learned, no refusal at all.
    solid = _seg(speaker="Test Person One", start=0.0, end=4.0,
                 speaker_confidence=_confidence_of_distance(cfg, 0.2),
                 speaker_margin=0.3)
    assert uncertainty.learning_refusal_reason(solid) is None


def test_a_generic_cluster_is_learned_however_weak_its_label(uncertainty):
    """A nameless cluster has no match confidence at all -- and that is fine.

    Absence of evidence is not evidence of doubt: learning unnamed people is the
    whole point, and requiring a match confidence for a cluster that was never
    matched to anybody would refuse every new speaker.
    """
    assert uncertainty.learning_refusal_reason(
        _seg(speaker="Speaker 4", start=0.0, end=4.0)) is None
    # ...but the same cluster with an explicit low confidence IS refused.
    assert uncertainty.learning_refusal_reason(
        _seg(speaker="Speaker 4", start=0.0, end=4.0,
             speaker_confidence=0.1)) == "low_confidence"


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


# --- refusal reasons ------------------------------------------------------


def test_the_reason_helpers_agree_with_the_booleans(uncertainty):
    """One gate, one implementation.

    A caller that explains a refusal and a caller that enforces it must not be
    able to disagree; that is the only reason the reason-returning functions
    exist instead of a second, parallel pile of ifs.
    """
    cases = [
        _seg(speaker="Bob", start=0.0, end=4.0, speaker_confidence=0.8,
             speaker_margin=0.3),
        _seg(speaker="Bob", start=0.0, end=4.0, speaker_confidence=0.1,
             speaker_margin=0.5),
        _seg(speaker="Bob", start=0.0, end=0.2, speaker_confidence=0.9,
             speaker_margin=0.4),
        _seg(speaker="Bob", start=0.0, end=4.0, uncertain=True),
        _seg(speaker="Speaker 2", start=0.0, end=4.0),
        _seg(speaker="UNKNOWN", start=0.0, end=4.0),
        _seg(speaker="OVERLAP", start=0.0, end=4.0),
        _seg(speaker=None, start=0.0, end=4.0),
        _seg(speaker="Bob", start=0.0, end=4.0, speaker_confidence=0.9,
             speaker_margin=0.01),
    ]
    for seg in cases:
        reason = uncertainty.auto_collect_refusal_reason(seg)
        assert (reason is None) == uncertainty.eligible_for_auto_collect(seg), seg
        learn = uncertainty.learning_refusal_reason(seg)
        assert (learn is None) == uncertainty.eligible_for_learning(seg), seg


def test_each_refusal_is_named(uncertainty):
    """A refusal that cannot be named cannot be acted on."""
    def reason(**kw):
        return uncertainty.auto_collect_refusal_reason(_seg(**kw))

    assert reason(speaker="Bob", end=4.0, speaker_confidence=0.8,
                  speaker_margin=0.4) is None
    assert reason(speaker=None) == uncertainty.REASON_UNKNOWN
    assert reason(speaker="") == uncertainty.REASON_UNKNOWN
    assert reason(speaker="UNKNOWN") == uncertainty.REASON_UNKNOWN
    assert reason(speaker="OVERLAP") == uncertainty.REASON_OVERLAP
    assert reason(speaker="Speaker 2") == uncertainty.REASON_GENERIC
    assert reason(speaker="Bob", end=4.0, uncertain=True) == \
        uncertainty.REASON_SUPPRESSED
    assert reason(speaker="Bob", end=4.0, speaker_confidence=0.1,
                  speaker_margin=0.5) == uncertainty.REASON_LOW_CONFIDENCE
    assert reason(speaker="Bob", end=4.0, speaker_confidence=0.9,
                  speaker_margin=0.01) == uncertainty.REASON_LOW_MARGIN
    assert reason(speaker="Bob", end=0.2, speaker_confidence=0.9,
                  speaker_margin=0.4) == uncertainty.REASON_TOO_SHORT


def test_a_learning_refusal_never_says_generic(uncertainty):
    """"generic_label" is the answer to the wrong question for learning.

    A generic cluster is refused for collection by definition, so reporting that
    as the reason a span failed to be LEARNED would hide the gate that
    actually stopped it.
    """
    generic = _seg(speaker="Speaker 4", start=0.0, end=9.0)
    assert uncertainty.auto_collect_refusal_reason(generic) == \
        uncertainty.REASON_GENERIC
    assert uncertainty.learning_refusal_reason(generic) is None
    assert uncertainty.learning_refusal_reason(
        _seg(speaker="Speaker 4", end=9.0, uncertain=True)) == \
        uncertainty.REASON_SUPPRESSED
    assert uncertainty.learning_refusal_reason(
        _seg(speaker="Speaker 4", end=9.0, speaker_confidence=0.05)) == \
        uncertainty.REASON_LOW_CONFIDENCE
    assert uncertainty.learning_refusal_reason(
        _seg(speaker=None)) == uncertainty.REASON_NO_IDENTITY


def test_learning_ignores_duration_and_margin(uncertainty):
    """A short span or a narrow margin is not evidence of a wrong speaker.

    Both gates are about attribution quality, and both stay shut for learning
    because the alternative is learning from audio the pipeline distrusted.
    """
    short = _seg(speaker="Speaker 4", start=0.0, end=0.2)
    narrow = _seg(speaker="Speaker 4", start=0.0, end=9.0, speaker_margin=0.0)
    # Named: the duration and margin gates are reached and refuse.
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Bob", start=0.0, end=0.2)) == uncertainty.REASON_TOO_SHORT
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Bob", start=0.0, end=9.0, speaker_margin=0.0)) == \
        uncertainty.REASON_LOW_MARGIN
    # Generic: collection is refused on the label before either is consulted,
    # and learning does not consult them at all.
    assert uncertainty.auto_collect_refusal_reason(short) == \
        uncertainty.REASON_GENERIC
    assert uncertainty.learning_refusal_reason(short) is None
    assert uncertainty.learning_refusal_reason(narrow) is None


def test_a_disabled_policy_lifts_the_suppression_gates(uncertainty):
    """`enabled: false` is the documented rollback (lesson 30).

    It lifts the gates that *suppress* an identity — the explicit `uncertain`
    flag, the margin and the duration. The confidence floor is separate and
    stays: `auto_collect_requires_confidence` is its own switch, and the
    rollback is not a licence to store audio nobody vouched for.
    """
    off = {"enabled": False}
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Bob", end=4.0, uncertain=True), off) is None
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Bob", end=0.2, speaker_confidence=0.9,
             speaker_margin=0.4), off) is None
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Bob", end=4.0, speaker_confidence=0.01,
             speaker_margin=0.4), off) == uncertainty.REASON_LOW_CONFIDENCE
    # ...and a label that carries no speaker at all is still no speaker.
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Speaker 2"), off) == uncertainty.REASON_GENERIC



def test_a_missing_confidence_is_not_doubt(uncertainty):
    """A generic cluster label carries no confidence; absence ≠ refusal.

    This is why `learning_refusal_reason` checks an explicit low value only --
    the alternative would refuse every cluster the diarizer produced.
    """
    assert uncertainty.auto_collect_refusal_reason(
        _seg(speaker="Speaker 9", end=9.0)) == uncertainty.REASON_GENERIC
    assert uncertainty.learning_refusal_reason(
        _seg(speaker="Speaker 9", end=9.0)) is None
