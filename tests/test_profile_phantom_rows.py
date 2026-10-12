"""A profile row must never outlive the segments it describes.

Bug, found on a 70-minute 2-speaker recording (``2026-09-30 17-30-04.mp4``,
4199.8s). The client's ``SPEAKER VOICE PROFILES`` banner listed six speakers,
two of which owned no text at all::

    Gergely Papp: pitch=149Hz  energy=0.0409  speech=531s
    Speaker 5:    pitch=150Hz  energy=0.0412  speech=502s   <- never in the text
    Speaker 6:    pitch=150Hz  energy=0.0270  speech=11s    <- never in the text

The banner is rendered from ``result["profiles"]`` while the transcript body is
attributed onto the diarization turns, so the two come from different
structures and nothing reconciled them.

The leak: every merge path renames ``source``'s segments to ``target`` and then
calls ``merge_profiles(profiles, target, source)`` — which folded ``source``'s
statistics into ``target`` but left the ``source`` KEY in place. The row then
described a speaker whose every segment had been renamed away. The two
mechanical callers are ``clustering.py`` step 6 (post-match merge) and the
second pass, plus step 7.

The invariant is enforced in one place — the pipeline's return value, which
drops any profile whose name no segment carries and counts ``OVERLAP`` and
``UNKNOWN`` as label classes rather than people.
"""

import sys
import types

import numpy as np
import pytest

from conftest import REPO_ROOT, load_module, load_uncertainty

SR = 16000

# Two clusters of the SAME person (cos dist ~0.013, far under
# ``unknown_merge_threshold`` 0.25), both a good match for one voiceprint.
# This is the 2-people-one-file case that produced the phantom rows.
EMB_A = [0.900, 0.436, 0.0]
EMB_B = [0.880, 0.470, 0.05]


@pytest.fixture(scope="module")
def clustering():
    """Import clustering.py without torch or sklearn."""
    if "sklearn" not in sys.modules:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            sk = types.ModuleType("sklearn")
            cl = types.ModuleType("sklearn.cluster")

            class _Fake:  # pragma: no cover - never exercised by these tests
                def __init__(self, *a, **kw):
                    raise RuntimeError("sklearn stub")

            cl.AgglomerativeClustering = _Fake
            sk.cluster = cl
            sys.modules["sklearn"] = sk
            sys.modules["sklearn.cluster"] = cl

    load_uncertainty()
    sys.modules.setdefault("asr_mcp.diarization", types.ModuleType("asr_mcp.diarization"))
    load_module("asr_mcp.diarization.segment_ops", "asr_mcp/diarization/segment_ops.py")
    load_module("asr_mcp.speaker.matcher", "asr_mcp/speaker/matcher.py", package="asr_mcp.speaker")
    return load_module("asr_mcp.diarization.clustering", "asr_mcp/diarization/clustering.py")


@pytest.fixture
def segment_ops():
    load_uncertainty()
    return load_module("_t_profiles_segment_ops", "asr_mcp/diarization/segment_ops.py")


@pytest.fixture(autouse=True)
def _fake_embedding():
    """Stub extract_embedding; the constant audio level identifies the cluster."""
    before = sys.modules.get("asr_mcp.speaker.embedding")
    mod = types.ModuleType("asr_mcp.speaker.embedding")

    def extract_embedding(audio, sample_rate, embedding_session=None):
        return np.asarray(
            {1.0: EMB_A, 2.0: EMB_B}.get(round(float(np.mean(audio)), 3), [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )

    mod.extract_embedding = extract_embedding
    sys.modules["asr_mcp.speaker.embedding"] = mod
    yield
    after = sys.modules.get("asr_mcp.speaker.embedding")
    if after is not before:
        if before is None:
            sys.modules.pop("asr_mcp.speaker.embedding", None)
        else:
            sys.modules["asr_mcp.speaker.embedding"] = before


def _second_pass(clustering, profiles):
    audio = np.concatenate([
        np.full(int(2.0 * SR), 1.0, dtype=np.float32),
        np.full(int(1.0 * SR), 0.0, dtype=np.float32),
        np.full(int(2.0 * SR), 2.0, dtype=np.float32),
    ])
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "a"},
        {"start": 3.0, "end": 5.0, "speaker": "Speaker 2", "text": "b"},
    ]
    known = {"Gergely Papp": {
        "embedding": [1.0, 0.0, 0.0], "pitch_hz": 0.0, "total_speech_sec": 100.0,
    }}
    cfg = {"second_pass": {
        "enabled": True, "accept_threshold": 0.38,
        "unknown_merge_threshold": 0.25, "min_speaker_duration_sec": 1.0,
    }}
    out, out_profiles = clustering.collapse_unknown_speakers_second_pass(
        segments, audio, SR, known, profiles, state=object(), cfg=cfg,
        allow_renaming=True, blocked_reason=None,
    )
    return out, out_profiles


def _profiles():
    """Durations chosen to match the reported banner (502s and 11s orphans)."""
    return {
        "Gergely Papp": {"pitch_hz": 149.0, "pitch_std": 78.0,
                         "energy_rms": 0.041, "total_speech_sec": 29.0},
        "Speaker 1": {"pitch_hz": 150.0, "pitch_std": 79.0,
                      "energy_rms": 0.041, "total_speech_sec": 502.0},
        "Speaker 2": {"pitch_hz": 150.0, "pitch_std": 75.0,
                      "energy_rms": 0.027, "total_speech_sec": 11.0},
    }


# --- merge_profiles: the direct cause ------------------------------------


def test_merge_profiles_drops_the_source_key(segment_ops):
    profiles = {
        "Gergely Papp": {"pitch_hz": 149.0, "total_speech_sec": 29.0},
        "Speaker 5": {"pitch_hz": 150.0, "total_speech_sec": 502.0},
    }
    segment_ops.merge_profiles(profiles, "Gergely Papp", "Speaker 5")
    assert "Speaker 5" not in profiles
    assert profiles["Gergely Papp"]["total_speech_sec"] == 531.0


def test_merge_profiles_keeps_duration_accounting(segment_ops):
    """Dropping the key must not lose its seconds."""
    profiles = {
        "Gergely Papp": {"pitch_hz": 149.0, "energy_rms": 0.041,
                         "total_speech_sec": 29.0},
        "Speaker 5": {"pitch_hz": 150.0, "energy_rms": 0.041,
                      "total_speech_sec": 502.0},
    }
    segment_ops.merge_profiles(profiles, "Gergely Papp", "Speaker 5")
    assert sum(p["total_speech_sec"] for p in profiles.values()) == 531.0


def test_merge_profiles_is_a_noop_for_an_unknown_source(segment_ops):
    profiles = {"Speaker 1": {"total_speech_sec": 5.0}}
    segment_ops.merge_profiles(profiles, "Gergely Papp", "Speaker 99")
    assert profiles == {"Speaker 1": {"total_speech_sec": 5.0}}


def test_merge_profiles_adopts_the_row_when_the_target_is_absent(segment_ops):
    """A renamed-away cluster with no existing target row still has to go.

    This is the exact reported state — ``Gergely Papp`` 531s sitting alongside
    ``Speaker 5`` 502s, un-summed, because the early return on a missing target
    key left the source row untouched. The row must follow its segments onto
    the target name, carrying the seconds with it.
    """
    profiles = {"Speaker 5": {"pitch_hz": 150.0, "total_speech_sec": 502.0}}
    segment_ops.merge_profiles(profiles, "Gergely Papp", "Speaker 5")
    assert profiles == {"Gergely Papp": {"pitch_hz": 150.0,
                                         "total_speech_sec": 502.0}}


def test_merge_profiles_never_leaves_the_source_behind(segment_ops):
    """Whatever the target state, the source key must not survive."""
    for profiles in (
        {"Gergely Papp": {"total_speech_sec": 531.0},
         "Speaker 5": {"total_speech_sec": 502.0}},
        {"Speaker 5": {"total_speech_sec": 502.0}},
    ):
        segment_ops.merge_profiles(profiles, "Gergely Papp", "Speaker 5")
        assert "Speaker 5" not in profiles
        assert "Gergely Papp" in profiles


# --- the second pass, end to end ------------------------------------------


def test_second_pass_leaves_no_orphan_profile_row(clustering):
    """The regression: two clusters of one person, renamed to one name.

    Pre-fix ``profiles`` still held ``Speaker 1`` (502s) and ``Speaker 2``
    (11s) — rows describing a speaker that owns no segment, which is exactly
    what printed in the banner and never in the transcript.
    """
    out, out_profiles = _second_pass(clustering, _profiles())
    labels = {s["speaker"] for s in out}
    assert labels == {"Gergely Papp"}
    assert set(out_profiles) == labels
    assert "Speaker 1" not in out_profiles
    assert "Speaker 2" not in out_profiles


def test_second_pass_preserves_the_merged_totals(clustering):
    """The dropped rows' seconds must reappear on the surviving name."""
    _out, out_profiles = _second_pass(clustering, _profiles())
    assert out_profiles["Gergely Papp"]["total_speech_sec"] == 542.0


def test_second_pass_keeps_a_profile_for_every_survivor(clustering):
    """The invariant in the other direction: no segment without a row."""
    out, out_profiles = _second_pass(clustering, _profiles())
    for label in {s["speaker"] for s in out if s["speaker"]}:
        assert label in out_profiles


# --- the pipeline's return gate -------------------------------------------


def test_total_speakers_excludes_overlap_and_unknown(segment_ops):
    """``Speakers: N`` must count people, not label classes.

    ``OVERLAP`` is not a person (``uncertainty.has_speaker_identity`` says so,
    and the transcript never prints it as a speaker), yet counting it reported
    one speaker more than the transcript shows — the banner listing N rows
    against a transcript with N-1 labels.
    """
    speakers, profiles = segment_ops.speaker_summary(
        [
            {"speaker": "Gergely Papp"},
            {"speaker": "Test Person Four"},
            {"speaker": "OVERLAP"},
            {"speaker": None},
        ],
        {},
    )
    assert speakers == {"Gergely Papp", "Test Person Four"}
    assert profiles == {}


def test_pipeline_gate_drops_orphans_and_keeps_the_rest(segment_ops):
    segments = [
        {"speaker": "Gergely Papp"},
        {"speaker": "Gergely Papp"},
        {"speaker": "Speaker 10"},
        {"speaker": "OVERLAP"},
        {"speaker": None},
    ]
    profiles = {
        "Gergely Papp": {"total_speech_sec": 531.0},
        "Speaker 10": {"total_speech_sec": 17.0},
        "Speaker 5": {"total_speech_sec": 502.0},
        "Speaker 6": {"total_speech_sec": 11.0},
    }
    speakers, out_profiles = segment_ops.speaker_summary(segments, profiles)
    assert set(out_profiles) == {"Gergely Papp", "Speaker 10"}
    assert speakers == {"Gergely Papp", "Speaker 10"}


def test_pipeline_gate_is_a_noop_when_they_already_agree(segment_ops):
    profiles = {"Speaker 1": {"total_speech_sec": 5.0}}
    speakers, out_profiles = segment_ops.speaker_summary(
        [{"speaker": "Speaker 1"}], profiles)
    assert out_profiles == profiles
    assert speakers == {"Speaker 1"}