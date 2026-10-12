"""The learning summary: what the browser is told about a run.

A colleague who does not become a voiceprint is currently indistinguishable
from one who was never a candidate — the only trace is a counter in the server
log. `_learning_summary` is the small, renderable half of the diagnostics the
service already produces, and these tests pin the two things that matter about
it: it must stay small, and it must not invent a reason.
"""

import sys
import types

from conftest import REPO_ROOT  # noqa: F401  (path setup, see _asr_router)


def _asr_router():
    if "asr_mcp.api" not in sys.modules:
        pkg = types.ModuleType("asr_mcp.api")
        pkg.__path__ = [str(REPO_ROOT / "asr_mcp/api")]
        sys.modules["asr_mcp.api"] = pkg
    import asr_mcp.api.asr_router as mod
    return mod


def _learned(**kw):
    base = {
        "collected": [{"id": 1}, {"id": 2}],
        "pending_created": [{"name": "Pending A", "cluster": "Speaker 3",
                             "speech_sec": 42.0}],
        "pending_extended": [],
        "learn_skipped": [],
        "diagnostics": {},
    }
    base.update(kw)
    return base


def test_a_clean_run_reports_what_it_stored():
    ar = _asr_router()
    s = ar._learning_summary(_learned())
    assert s["snippets_added"] == 2
    assert [p["name"] for p in s["profiles"]] == ["Pending A"]
    assert s["profiles"][0]["existing"] is False
    assert s["refused_total"] == 0
    assert s["not_collected"] == []


def test_a_missing_learned_block_is_not_a_crash():
    """`_auto_collect` returns None on some paths; the event must still emit."""
    ar = _asr_router()
    s = ar._learning_summary({})
    assert s == {"snippets_added": 0, "profiles": [], "not_learned": [],
                 "not_collected": [], "refused_total": 0,
                 "refused_reasons": {}, "refusal_text": ar.LEARNING_REFUSAL_TEXT}
    assert ar._learning_summary(None)["snippets_added"] == 0


def test_a_capped_voiceprint_is_named_in_the_summary():
    """The one case that is a decision, not an accident.

    A speaker sitting at AUTO_COLLECT_MAX_TOTAL_SEC is not being learned
    because the operator's cap says so. Silence there is how a profile quietly
    stops improving.
    """
    ar = _asr_router()
    s = ar._learning_summary(_learned(collected=[], pending_created=[], collected_ok=None) | {
        "diagnostics": {
            "collect": [
                {"speaker": "Test Person One", "skipped": "at_snippet_cap"},
                {"speaker": "Test Person Two", "skipped": "not_registered"},
                {"speaker": "Test Person Three", "skipped": None,
                 "snippets": 3},
            ],
        },
    })
    assert s["not_collected"] == [
        {"speaker": "Test Person One", "reason": "at_snippet_cap"}]
    assert s["snippets_added"] == 0


def test_a_refused_near_miss_travels_as_a_histogram():
    """The tally must be complete even though the records are capped.

    The service caps the per-span records so one run cannot produce a useless
    payload; the counts come from its own histogram, which is not capped.
    """
    ar = _asr_router()
    s = ar._learning_summary(_learned(collected=[]) | {
        "diagnostics": {
            "refused_total": 240,
            "refused": [{"reason": "low_margin"}] * 25,
            "refused_reasons": {"low_margin": 200, "overlap_label": 40},
        },
    })
    assert s["refused_total"] == 240
    assert s["refused_reasons"] == {"low_margin": 200, "overlap_label": 40}
    # The text table must cover every reason the service can emit, or the UI
    # falls back to a raw enum name.
    from asr_mcp.speaker import uncertainty
    for reason in ("no_identity", "unknown_label", "overlap_label",
                   "generic_label", "explicitly_uncertain", "low_confidence",
                   "low_margin", "too_short"):
        assert reason in s["refusal_text"]


def test_a_cluster_that_could_not_be_learned_is_reported():
    ar = _asr_router()
    s = ar._learning_summary(_learned(pending_created=[]) | {
        "learn_skipped": [
            {"cluster": "Speaker 8", "reason": "unverified:impure_group"},
            {"cluster": "Speaker 9", "reason": "too_few_spans"},
        ],
    })
    assert [n["cluster"] for n in s["not_learned"]] == ["Speaker 8", "Speaker 9"]


def test_the_summary_is_bounded():
    """A forty-speaker conference must not become a wall of SSE frame."""
    ar = _asr_router()
    n = ar.LEARNING_SUMMARY_MAX_ITEMS + 5
    s = ar._learning_summary(_learned(
        pending_created=[],
        learn_skipped=[{"cluster": f"Speaker {i}", "reason": "x"}
                       for i in range(n)],
    ) | {"diagnostics": {"collect": [
        {"speaker": f"Person {i}", "skipped": "at_snippet_cap"}
        for i in range(n)]}})
    assert len(s["not_learned"]) == ar.LEARNING_SUMMARY_MAX_ITEMS
    assert len(s["not_collected"]) == ar.LEARNING_SUMMARY_MAX_ITEMS
