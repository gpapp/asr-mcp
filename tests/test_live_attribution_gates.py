"""Regression for the live voiceprint-match gates, measured end-to-end.

## Why this file was rewritten

The first version pinned the gates to a two-population table whose NEGATIVE side
was corrupted audio. An end-to-end run through the real `handle_ws_stream` (real
embedding hooks, real gates, real `pack_turn` frames) then produced a genuine
false positive: a **5s turn of an unregistered speaker was confidently named as a
registered one at conf 0.539**. A corrupted-audio negative can never detect this
-- it only ever measured the extreme garbage band (combined 0.78-0.87), not the
band a different-but-similar colleague occupies.

The gates were raised from 0.15 to 0.60 to reject it. That is a deliberate,
asymmetric trade: the live path now withholds names it used to assert. The text
is never withheld (that is the uncertainty policy), and the authoritative
attribution remains the client's shutdown re-diarization, which has minutes of
context rather than one turn.

## The measurements

Genuine: 2-9s excerpts of a REGISTERED speaker, real ECAPA against the real
35-voiceprint menu. Range conf 0.196-0.65, margin 0.141-0.417.

False positive: a 5s turn of an UNREGISTERED speaker from a different recording,
conf 0.539, through the real handler.

Overlapping 6s/9s turns of the registered speaker scored conf 0.22 / 0.19, so a
0.60 bar rejects them too. That is accepted: false-positive and false-negative are
not symmetric here, and a wrong name is the expensive error.
"""

import pytest

from asr_mcp.streaming import attribution


def conf_of(combined):
    # asr_mcp.speaker.matcher.compute_distance
    return max(0.0, 1 - combined / 0.5)


# (combined_distance, margin) observed in the measurement run, REGISTERED speaker.
GENUINE = [
    (0.286, 0.373), (0.225, 0.410), (0.290, 0.355), (0.232, 0.369),
    (0.231, 0.400), (0.314, 0.313), (0.188, 0.325), (0.174, 0.417),
    (0.217, 0.228), (0.267, 0.326), (0.316, 0.371), (0.402, 0.280),
    (0.318, 0.141),
]

# The end-to-end run, through the real handler, on an UNREGISTERED speaker.
# combined 0.2305 <=> conf 0.539.
REAL_FALSE_POSITIVE = (0.2305, None)

# What the float32-as-int16 bug fed the embedder: pure noise, not a person.
CORRUPTED_AUDIO = [
    (0.783, 0.043), (0.757, 0.036), (0.869, 0.043), (0.861, 0.003),
    (0.793, 0.062), (0.863, 0.033), (0.846, 0.012), (0.833, 0.006),
    (0.868, 0.002),
]


@pytest.fixture(scope="module")
def cfg():
    return attribution.config()


def _admitted(combined, margin, cfg):
    return (conf_of(combined) >= cfg["min_match_confidence"]
            and (margin is None or margin >= cfg["min_match_margin"]))


def test_a_real_unregistered_speaker_is_not_named(cfg):
    """The regression that motivated raising the gate.

    This is the strongest evidence we have against a permissive confidence
    floor: a real person who is NOT in the menu scored conf 0.539.
    """
    combined, margin = REAL_FALSE_POSITIVE
    assert not _admitted(combined, margin, cfg), (
        f"the measured false positive (conf {conf_of(combined):.3f}) is admitted by "
        f"conf>={cfg['min_match_confidence']} -- a stranger would be named"
    )


def test_corrupted_audio_is_never_named(cfg):
    admitted = [(c, m) for c, m in CORRUPTED_AUDIO if _admitted(c, m, cfg)]
    assert admitted == []


def test_the_gate_sits_above_the_measured_false_positive(cfg):
    assert cfg["min_match_confidence"] > conf_of(REAL_FALSE_POSITIVE[0])


def test_the_strongest_genuine_match_is_still_admitted(cfg):
    """A gate that rejects everything is not a fix. The best genuine evidence
    (conf 0.65) must still produce a name, or the feature is simply off."""
    strongest = min(GENUINE, key=lambda t: t[0])
    assert _admitted(*strongest, cfg=cfg), (
        f"the strongest genuine match (conf {conf_of(strongest[0]):.2f}) is rejected by "
        f"conf>={cfg['min_match_confidence']}"
    )


def test_the_weak_genuine_matches_are_deliberately_withheld(cfg):
    """Document the accepted cost, so nobody 'fixes' it by lowering the gate.

    These are real matches we now decline to name. That is the intended
    asymmetry: the transcript keeps the words, only the identity is withheld.
    """
    withheld = [c for c, m in GENUINE if not _admitted(c, m, cfg)]
    assert withheld, "expected some genuine matches to be withheld"
    # ...but not all of them, which would mean the gate is simply off.
    assert len(withheld) < len(GENUINE)


def test_thresholds_json_agrees_with_the_module_defaults():
    from asr_mcp.config import get_config
    shipped = get_config()["live_attribution"]
    assert shipped["min_match_confidence"] == attribution._DEFAULTS["min_match_confidence"]
    assert shipped["min_match_margin"] == attribution._DEFAULTS["min_match_margin"]
