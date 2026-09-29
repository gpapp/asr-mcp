"""Regression tests for the one-to-one voiceprint claim in the second pass.

Bug (found on a 52-minute Hungarian podcast): two acoustically DIFFERENT
unknown clusters were both renamed to the same known person, because each
cluster independently picked its ``find_best_match`` winner with nothing
stopping a second cluster from claiming an already-taken voiceprint.  The
result was 99.3% of a multi-voice recording attributed to one name — exactly
the confident-looking false attribution the uncertainty policy forbids.

``collapse_unknown_speakers_second_pass`` now tracks ``claimed_by`` and leaves
a cluster as ``Speaker N`` when its best match is already held by a cluster it
is NOT consistent with (cosine distance >= ``unknown_merge_threshold``).
"""

import sys
import types

import numpy as np
import pytest

from conftest import REPO_ROOT, load_module, load_uncertainty

SR = 16000


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


def _install_fake_embedding(emb_by_level):
    """Stub asr_mcp.speaker.embedding.extract_embedding (torch-free).

    The audio is built from constant blocks, so the mean level of a speaker's
    concatenated audio identifies which embedding to return.
    """
    mod = types.ModuleType("asr_mcp.speaker.embedding")

    def extract_embedding(audio, sample_rate, embedding_session=None):
        level = round(float(np.mean(audio)), 3)
        vec = emb_by_level.get(level)
        if vec is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32)

    mod.extract_embedding = extract_embedding
    sys.modules["asr_mcp.speaker.embedding"] = mod
    return mod


@pytest.fixture(autouse=True)
def _restore_embedding_module():
    """Undo the sys.modules stub after every test in this module.

    Without this the fake leaks into every later test that imports
    asr_mcp.speaker.embedding (e.g. the live handler's embedding hooks fail to
    import and silently degrade all speaker turns to UNKNOWN).
    """
    before = sys.modules.get("asr_mcp.speaker.embedding")
    yield
    after = sys.modules.get("asr_mcp.speaker.embedding")
    if after is not before:
        if before is None:
            sys.modules.pop("asr_mcp.speaker.embedding", None)
        else:
            sys.modules["asr_mcp.speaker.embedding"] = before


def _run(clustering, emb_a, emb_b, *, spk_a, spk_b, level_a=1.0, level_b=2.0):
    """Run the second pass with two unknown clusters and ONE known voiceprint."""
    _install_fake_embedding({level_a: emb_a, level_b: emb_b})

    audio = np.concatenate([
        np.full(int(2.0 * SR), level_a, dtype=np.float32),
        np.full(int(1.0 * SR), 0.0, dtype=np.float32),
        np.full(int(2.0 * SR), level_b, dtype=np.float32),
    ])
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": spk_a, "text": "a"},
        {"start": 3.0, "end": 5.0, "speaker": spk_b, "text": "b"},
    ]
    known = {
        "Gergely Papp": {
            "embedding": [1.0, 0.0, 0.0],
            "pitch_hz": 0.0,
            "total_speech_sec": 100.0,
        }
    }
    profiles = {
        spk_a: {"pitch_hz": 0.0, "energy_rms": 0.1, "total_speech_sec": 2.0},
        spk_b: {"pitch_hz": 0.0, "energy_rms": 0.1, "total_speech_sec": 2.0},
    }
    cfg = {"second_pass": {"enabled": True, "accept_threshold": 0.38,
                           "unknown_merge_threshold": 0.25,
                           "min_speaker_duration_sec": 1.0}}
    out, out_profiles = clustering.collapse_unknown_speakers_second_pass(
        segments, audio, SR, known, profiles, state=object(), cfg=cfg
    )
    return out, out_profiles


# A is 0.25 from the reference (combined 0.225, under accept_threshold 0.38),
# B is 0.333 (combined 0.30, also accepted) — but they are 0.50 from EACH
# OTHER, far beyond unknown_merge_threshold 0.25.  Pre-fix both were renamed.
EMB_FAR_A = [0.75, 0.66, 0.0]
EMB_FAR_B = [0.667, 0.0, 0.745]
# Same-speaker split: both close to the reference (cos 0.90 / 0.88) and only
# ~0.02 from EACH OTHER, well under unknown_merge_threshold 0.25.
EMB_SAME_A = [0.900, 0.436, 0.0]
EMB_SAME_B = [0.880, 0.470, 0.05]
# The podcast case: two clusters indistinguishable from EACH OTHER (0.00
# apart) yet both ~0.30 combined away from the reference — the best of a
# 35-name menu, presented by the old code as a certain identity.
EMB_WEAK_A = [0.700, 0.714, 0.0]
EMB_WEAK_B = [0.695, 0.719, 0.0]


def test_distant_clusters_do_not_share_a_voiceprint(clustering):
    """The regression: two different voices must not both become one name."""
    out, _ = _run(clustering, EMB_FAR_A, EMB_FAR_B,
                  spk_a="Speaker 1", spk_b="Speaker 2")
    speakers = {s["speaker"] for s in out}
    assert speakers == {"Gergely Papp", "Speaker 2"}


def test_better_fitting_cluster_wins_the_claim(clustering):
    """The closest match claims the name; the weaker one is left unresolved."""
    segments, _ = _run(clustering, EMB_FAR_A, EMB_FAR_B,
                       spk_a="Speaker 1", spk_b="Speaker 2")
    by_text = {s["text"]: s["speaker"] for s in segments}
    # Speaker 1's combined distance (0.225) is the smaller, so it claims the
    # name; Speaker 2 is the one left looking for another answer.
    assert by_text["a"] == "Gergely Papp"
    assert by_text["b"] == "Speaker 2"


def test_same_person_split_across_clusters_may_share_a_voiceprint(clustering):
    """Consistent clusters (cos dist < merge threshold) may both be renamed."""
    out, _ = _run(clustering, EMB_SAME_A, EMB_SAME_B,
                  spk_a="Speaker 1", spk_b="Speaker 2")
    assert {s["speaker"] for s in out} == {"Gergely Papp"}


def test_no_unknown_speakers_is_a_noop(clustering):
    _install_fake_embedding({1.0: EMB_FAR_A})
    segs = [{"start": 0.0, "end": 2.0, "speaker": "Gergely Papp", "text": "x"}]
    out, _ = clustering.collapse_unknown_speakers_second_pass(
        segs, np.zeros(int(2.0 * SR), dtype=np.float32), SR, {}, {},
        state=object(), cfg={"second_pass": {"enabled": True}},
    )
    assert [s["speaker"] for s in out] == ["Gergely Papp"]


def test_disabled_second_pass_returns_segments_untouched(clustering):
    _install_fake_embedding({1.0: EMB_FAR_A, 2.0: EMB_FAR_B})
    audio = np.concatenate([
        np.full(int(2.0 * SR), 1.0, dtype=np.float32),
        np.full(int(3.0 * SR), 2.0, dtype=np.float32),
    ])
    segs = [
        {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "a"},
        {"start": 2.0, "end": 5.0, "speaker": "Speaker 2", "text": "b"},
    ]
    out, _ = clustering.collapse_unknown_speakers_second_pass(
        segs, audio, SR, {"Gergely Papp": {"embedding": [1.0, 0.0, 0.0]}}, {},
        state=object(), cfg={"second_pass": {"enabled": False}},
    )
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2"]


# --- Identification gate -------------------------------------------------
#
# The podcast bug that the one-to-one constraint could not catch: the two
# clusters were only 0.07 apart, so sharing a name was "allowed" — but both
# were 0.30+ combined away from the reference, i.e. the best of a 35-name
# menu.  That is not an identification, and the second pass must refuse it.


def test_weak_voiceprint_match_is_not_renamed(clustering):
    """The best of a 35-voiceprint menu is not an identification."""
    out, _ = _run(clustering, EMB_WEAK_A, EMB_WEAK_B,
                  spk_a="Speaker 1", spk_b="Speaker 2")
    # Both stay generic (and are folded together by the duplicate merge) —
    # no invented name.
    speakers = {s["speaker"] for s in out}
    assert "Gergely Papp" not in speakers
    assert all(s.startswith("Speaker ") for s in speakers)


def test_renamed_segments_carry_the_match_confidence(clustering):
    """An identity assertion must travel with the evidence for it.

    Without this the attribution layer falls back to the geometric overlap
    ratio (1.0) and reports a 0.55-confident rename as certain.
    """
    out, _ = _run(clustering, EMB_SAME_A, EMB_SAME_B,
                  spk_a="Speaker 1", spk_b="Speaker 2")
    named = [s for s in out if s["speaker"] == "Gergely Papp"]
    assert named
    for seg in named:
        assert isinstance(seg["speaker_confidence"], (int, float))
        assert seg["speaker_source"] == "known_voiceprint"
        assert seg["uncertain"] is False
        assert seg["speaker_match_dist"] < 0.25


def test_identity_gate_is_bypassed_when_the_policy_is_disabled(clustering, monkeypatch):
    """uncertainty.enabled=false is the documented full-legacy rollback."""
    unc = sys.modules["asr_mcp.speaker.uncertainty"]
    monkeypatch.setattr(unc, "policy_enabled", lambda: False)
    out, _ = _run(clustering, EMB_WEAK_A, EMB_WEAK_B,
                  spk_a="Speaker 1", spk_b="Speaker 2")
    # Same two clusters, same distances — without the policy the weak match
    # is applied, which is exactly the pre-policy behaviour.
    assert {s["speaker"] for s in out} == {"Gergely Papp"}
