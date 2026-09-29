"""Calibration regression for the live voiceprint-match gates.

The first shipped pair was ``min_match_confidence=0.60`` /
``min_match_margin=0.05``, chosen on the assumption that "live turns are
short, so the bar must be higher than offline".  Measured against the real
35-voiceprint menu on real 2-6s excerpts of the *correct* speakers, that pair
was wrong: the genuine confidence population tops out at 0.65, so 0.60
rejected 11 of 13 genuine matches while the margin gate (0.05) admitted
non-matches with a margin as low as 0.002.

These tests pin the gates to the measured populations so the numbers cannot
drift back without someone looking at the measurements.

Populations (asr_mcp/streaming/attribution.py + docs/lessons/live-client-design.md):
  genuine  combined 0.174-0.402  -> conf 0.20-0.65  margin 0.141-0.417
  nonmatch combined 0.780-0.870  -> conf 0.00       margin 0.002-0.062
"""

import pytest

from asr_mcp.streaming import attribution

# (combined_distance, margin) pairs observed in the measurement run.
GENUINE = [
    (0.286, 0.373), (0.225, 0.410), (0.290, 0.355), (0.232, 0.369),
    (0.231, 0.400), (0.314, 0.313), (0.188, 0.325), (0.174, 0.417),
    (0.217, 0.228), (0.267, 0.326), (0.316, 0.371), (0.402, 0.280),
    (0.318, 0.141),
]
# What the float32-as-int16 bug fed the embedder: pure noise.
NONMATCH = [
    (0.783, 0.043), (0.757, 0.036), (0.869, 0.043), (0.861, 0.003),
    (0.793, 0.062), (0.863, 0.033), (0.846, 0.012), (0.833, 0.006),
    (0.868, 0.002),
]


def conf_of(combined):
    # asr_mcp.speaker.matcher.compute_distance
    return max(0.0, 1 - combined / 0.5)


@pytest.fixture(scope="module")
def cfg():
    return attribution.config()


def test_shipped_gates_admit_every_measured_genuine_match(cfg):
    rejected = [
        (c, m) for c, m in GENUINE
        if conf_of(c) < cfg["min_match_confidence"] or m < cfg["min_match_margin"]
    ]
    assert rejected == [], (
        f"live gates reject {len(rejected)}/{len(GENUINE)} measured genuine matches: "
        f"{rejected} against conf>={cfg['min_match_confidence']} "
        f"margin>={cfg['min_match_margin']}"
    )


def test_shipped_gates_reject_every_measured_non_match(cfg):
    admitted = [
        (c, m) for c, m in NONMATCH
        if conf_of(c) >= cfg["min_match_confidence"] and m >= cfg["min_match_margin"]
    ]
    assert admitted == [], (
        f"live gates admit {len(admitted)}/{len(NONMATCH)} measured non-matches: {admitted}"
    )


def test_confidence_floor_sits_below_the_genuine_ceiling(cfg):
    """0.60 shipped *above* the whole genuine population -- the actual bug."""
    ceiling = max(conf_of(c) for c, _ in GENUINE)
    assert cfg["min_match_confidence"] < ceiling, (
        f"min_match_confidence={cfg['min_match_confidence']} is at or above the "
        f"measured genuine confidence ceiling {ceiling:.2f}"
    )


def test_margin_is_the_discriminating_gate(cfg):
    """The two populations are separated by margin, not by confidence.

    A gate swap -- margin permissive and confidence strict -- is exactly the
    configuration that shipped and it does not classify either population.
    """
    genuine_margin_min = min(m for _, m in GENUINE)
    nonmatch_margin_max = max(m for _, m in NONMATCH)
    assert cfg["min_match_margin"] > nonmatch_margin_max
    assert cfg["min_match_margin"] < genuine_margin_min
    # And confidence alone could never do it: a genuine match (conf 0.20) sits
    # below a non-match-threshold score only because non-matches happen to be
    # far away. Assert the confidence populations merely touch at the floor.
    assert max(conf_of(c) for c, _ in NONMATCH) == 0.0


def test_thresholds_json_agrees_with_the_module_defaults():
    from asr_mcp.config import get_config
    shipped = get_config()["live_attribution"]
    assert shipped["min_match_confidence"] == attribution._DEFAULTS["min_match_confidence"]
    assert shipped["min_match_margin"] == attribution._DEFAULTS["min_match_margin"]
