"""Tests for the consolidation / outlier-review pass.

``score_outliers`` is pure, so the detection criterion is tested with vectors
built by hand -- no models, no audio. The service methods are then tested
against a REAL SQLite database with the audio stack stubbed, because the things
that must not regress are all persistence: the mark surviving with its evidence,
the profile being fitted from the kept snippets only, and the bulk delete
refining each speaker exactly once.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("numpy")

SVC = None


def _load_service_module():
    """Load ``voiceprint/service.py`` BY PATH with a stubbed audio stack.

    Importing ``asr_mcp.voiceprint`` at collection time puts the real package
    (and the real service) into ``sys.modules`` before the other modules
    install their stubs, which breaks their stubbed loaders in a full-suite
    run. Same rule as ``tests/test_voiceprint_rebuild.py``.
    """
    import numpy as np

    sf = types.ModuleType("soundfile")
    sf.write = lambda *a, **k: None
    sf.read = lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000)
    sf.info = lambda *a, **k: types.SimpleNamespace(samplerate=16000, frames=16000)
    sys.modules["soundfile"] = sf

    torch = types.ModuleType("torch")
    torch.cat = lambda xs, dim=-1: np.concatenate(list(xs), axis=-1)
    torch.from_numpy = np.asarray
    torch.Tensor = np.ndarray
    torch.float32 = np.float32
    torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: s, __exit__=lambda s, *a: False)
    sys.modules["torch"] = torch

    speaker = types.ModuleType("asr_mcp.speaker.embedding")
    # EMBEDDER maps a snippet's marker -> the angle its "audio" stands for, so
    # a test can hand one speaker a tight cluster plus one obvious intruder.
    # Keying by marker (not by call order) keeps the fixture independent of the
    # order the DB happens to return rows in.
    EMBEDDER = {"by_marker": {}, "default_angle": 0.0,
                "zero_marker": None, "calls": 0}

    def _embed(wave, sr, session):
        import numpy as np
        EMBEDDER["calls"] += 1
        try:
            marker = int(np.asarray(wave).ravel()[0])
        except Exception:
            marker = -1
        if EMBEDDER["zero_marker"] is not None and marker == EMBEDDER["zero_marker"]:
            return np.zeros(192, dtype=np.float32)
        return np.asarray(_vec(EMBEDDER["by_marker"].get(
            marker, EMBEDDER["default_angle"])), dtype=np.float32)

    speaker.extract_embedding = _embed
    speaker.compute_pitch = lambda wave, sr: (150.0, 10.0)
    speaker.compute_energy = lambda wave: 0.01
    speaker.batch_embed = lambda *a, **k: []
    speaker.batch_embed_files = lambda *a, **k: []
    sys.modules["asr_mcp.speaker.embedding"] = speaker

    import asr_mcp.voiceprint.utils as vutils

    # The stub "audio" is 4.0s long (so the loaded duration agrees with the
    # declared rows) and carries the snippet's id as its sample value, which is
    # how a test decides what embedding each file gets without relying on the
    # order the DB happens to return rows in.
    def _fake_load(path):
        import numpy as np
        marker = 0
        try:
            raw = Path(str(path)).read_bytes()
            if raw.startswith(b"ID:"):
                marker = int(raw[3:].split(b",")[0])
        except OSError:
            marker = 0
        return np.full(64000, marker, dtype="float32"), 16000

    vutils.load_audio = _fake_load
    vutils.load_audio_segment = _fake_load

    path = REPO_ROOT / "asr_mcp" / "voiceprint" / "service.py"
    spec = importlib.util.spec_from_file_location("vp_consolidation_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TEST_EMBEDDER = EMBEDDER
    return mod


def _service(tmp_path, emb_session=object(), cfg=None):
    from asr_mcp.db.manager import DatabaseManager, VoiceprintDB, SnippetDB
    global SVC
    if SVC is None:
        SVC = _load_service_module()
    SVC.TEST_EMBEDDER.update({"by_marker": {}, "default_angle": 0.0,
                              "zero_marker": None, "calls": 0})
    # Always reassigned: the module object is shared between tests, so a cfg
    # from one test would otherwise leak into the next.
    base = {"outlier_floor": 0.35, "outlier_mad_k": 6.0,
            "outlier_max_fraction": 0.2, "outlier_min_keep": 3,
            "outlier_mad_rounds": 2}
    base.update(cfg or {})
    SVC._consolidation_cfg = lambda: dict(base)
    db = DatabaseManager(str(tmp_path / "t.db"))
    svc = SVC.VoiceprintService(tmp_path, db)
    svc.set_voices_dir(tmp_path / "voices")
    svc._db = VoiceprintDB(db)
    svc._snippets = SnippetDB(db)
    svc._emb_session = lambda: emb_session
    return svc


def _vec(angle_deg: float):
    """A 2-D embedding at ``angle_deg``, embedded in the 192 dims the DB wants.

    Snippets sharing an angle are identical; 40 degrees apart is a cosine
    distance of ~0.12, 90 degrees is 1.0.
    """
    import numpy as np
    v = np.zeros(192, dtype=np.float32)
    v[0] = np.cos(np.radians(angle_deg))
    v[1] = np.sin(np.radians(angle_deg))
    return v


_MARKER = {"n": 0}


def _snippet(svc, speaker, name, dur=4.0, angle=0.0, user_id="u1",
             source="rec.wav", write_file=True):
    """A snippet row plus the audio it claims.

    The stub audio carries a marker value, and the marker is mapped to
    ``angle`` in ``TEST_EMBEDDER`` -- so what this snippet "says" is decided
    here rather than by the order the database happens to return rows in.
    """
    _MARKER["n"] += 1
    marker = _MARKER["n"]
    SVC.TEST_EMBEDDER["by_marker"][marker] = angle
    path = Path(svc.voices_dir) / user_id / speaker / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_file:
        path.write_bytes(b"ID:%d" % marker)
    svc._snippets.add(user_id=user_id, speaker_name=speaker,
                      file_path=str(path), duration_sec=dur,
                      source_audio=source, start_sec=0.0, end_sec=dur)
    return path


# ── score_outliers, pure ───────────────────────────────────────────────

def _score(angles, cfg=None):
    SVC = _load_service_module()
    full = {"outlier_floor": 0.35, "outlier_mad_k": 6.0,
            "outlier_max_fraction": 0.2, "outlier_min_keep": 3,
            "outlier_mad_rounds": 2}
    full.update(cfg or {})
    return SVC, SVC.score_outliers([_vec(a) for a in angles], full)


def test_a_tight_profile_flags_nothing():
    _, out = _score([0, 1, 2, 3, 4])
    assert out["flagged_indices"] == []
    assert out["kept_indices"] == [0, 1, 2, 3, 4]


def test_an_obvious_intruder_is_flagged_with_its_evidence():
    _, out = _score([0, 1, 2, 3, 90])
    assert out["flagged_indices"] == [4]
    assert out["distances"][4] > 0.35
    # The accusation carries the reason AND the numbers behind it: a bare
    # boolean would be an accusation the user cannot argue with.
    assert "beyond the floor" in out["reasons"][4]
    assert "0.35" in out["reasons"][4]


def test_the_intruder_is_excluded_from_the_kept_set():
    _, out = _score([0, 1, 2, 3, 90])
    assert 4 not in out["kept_indices"]
    assert len(out["kept_vectors"]) == 4


def test_a_spread_that_is_still_one_person_is_not_cut():
    # Every snippet is 40 degrees from the next: a wide but continuous speaker.
    # The absolute floor is what stops a merely-noisy profile from being cut
    # into pieces.
    _, out = _score([0, 40, 80, 120, 160])
    assert out["flagged_indices"] == []


def test_no_more_than_max_fraction_is_ever_flagged():
    # A 7-snippet majority at 0 degrees and three obvious intruders at 90/120/150.
    # The cap is 20% of 10 = 2, so only the two furthest survive.
    _, out = _score([0] * 7 + [90, 120, 150], {"outlier_max_fraction": 0.2})
    assert len(out["flagged_indices"]) == 2
    # 150 and 120 degrees are the furthest from the 0-degree majority.
    assert set(out["flagged_indices"]) == {8, 9}


def test_the_cap_says_which_accusations_it_dropped():
    _, out = _score([0] * 7 + [90, 120, 150], {"outlier_max_fraction": 0.2})
    dropped = [i for i in out["kept_indices"] if i in out["reasons"]]
    assert dropped, "a spared accusation must still say why it was spared"
    assert all("capped" in out["reasons"][i] for i in dropped)


def test_the_cap_holds_across_rounds():
    """Round 2 must not spend a fresh budget: 20% of the profile, not 20% per
    round -- otherwise a multi-round sweep flags twice what it promises."""
    _, out = _score([0] * 7 + [90, 120, 150], {"outlier_max_fraction": 0.2})
    assert len(out["flagged_indices"]) <= 2
    assert len(out["kept_indices"]) >= 8


def test_min_keep_protects_a_small_profile():
    # Three snippets, one intruder: flagging it would leave two, below min_keep.
    _, out = _score([0, 0, 90], {"outlier_min_keep": 3})
    assert out["flagged_indices"] == []
    assert any("min_keep" in r for r in out["reasons"].values())


def test_min_keep_releases_the_mildest_accusation_first():
    # Four near-identical, one intruder: min_keep=3 allows exactly one flag.
    _, out = _score([0, 0, 0, 90, 90], {"outlier_min_keep": 3})
    assert len(out["flagged_indices"]) == 1


def test_fewer_than_three_snippets_accuse_nobody():
    _, out = _score([0, 90])
    assert out["flagged_indices"] == []
    _, out = _score([90])
    assert out["flagged_indices"] == []


def test_empty_input_is_handled():
    _, out = _score([])
    assert out["distances"] == [] and out["kept_indices"] == []


def test_zero_rounds_scores_without_accusing():
    """Report-only mode: the knob for calibrating against a real library."""
    _, out = _score([0, 1, 2, 3, 90], {"outlier_mad_rounds": 0})
    assert out["flagged_indices"] == []
    # It still measured: the round statistics are recorded.
    assert out["rounds"] and "cut" in out["rounds"][0]
    assert out["distances"][4] > 0.35


def test_every_returned_vector_is_accounted_for():
    _, out = _score([0, 1, 2, 3, 90])
    assert len(out["kept_indices"]) + len(out["flagged_indices"]) == 5


# ── consolidate_speaker, against a real database ──────────────────────

def _cluster(svc, speaker, count=5, intruder=None, user_id="u1"):
    """``count`` identical snippets, optionally one at 90 degrees."""
    angles = [0.0] * count + ([90.0] if intruder is not None else [])
    return [_snippet(svc, speaker, f"s{i}.flac", angle=angle, user_id=user_id)
            for i, angle in enumerate(angles)]


def test_a_clean_profile_is_reported_clean(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5)
    out = svc.consolidate_speaker("Ada", user_id="u1")
    assert out["status"] == "clean"
    assert out["flagged"] == []
    assert out["kept"] == 5
    assert out["profile_updated"] is False


def test_the_intruder_is_marked_and_the_profile_fitted_without_it(tmp_path):
    svc = _service(tmp_path)
    paths = _cluster(svc, "Ada", count=5, intruder=True)
    out = svc.consolidate_speaker("Ada", user_id="u1")
    assert out["status"] == "consolidated"
    assert out["flagged_count"] == 1
    assert out["flagged"][0]["file_path"] == str(paths[-1])
    assert out["kept"] == 5
    assert out["profile_updated"] is True
    vp = svc._db.get("Ada", user_id="u1")
    # Fitted from the kept snippets only -- the intruder is not in the profile.
    assert vp["sample_count"] == 5
    assert vp["total_speech_sec"] == pytest.approx(20.0, abs=0.2)
    assert out["flagged"][0]["distance"] > 0.35


def test_the_mark_carries_the_score_and_the_reason(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    flagged = svc.list_flagged_snippets(user_id="u1")
    assert len(flagged) == 1
    row = flagged[0]
    assert row["review_flag"] is True
    assert row["review_score"] > 0.35
    assert "beyond the floor" in row["review_reason"]


def test_the_mean_distance_before_and_after_are_reported(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc._db.save(name="Ada", user_id="u1", embedding=_vec(0.0),
                 pitch_hz=0.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=0.0, sample_count=0, mfcc={})
    out = svc.consolidate_speaker("Ada", user_id="u1")
    assert out["mean_distance_before"] is not None
    assert out["mean_distance_after"] is not None
    # "the profile got closer to the audio it claims to describe"
    assert out["mean_distance_after"] <= out["mean_distance_before"]


def test_a_previous_accusation_is_cleared_when_the_snippet_is_kept(tmp_path):
    """A stale mark must not outlive the run that refuted it."""
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    assert len(svc.list_flagged_snippets(user_id="u1")) == 1
    # The intruder turns out to be fine after all: every snippet now sits at 0
    # degrees, so the accusation no longer holds.
    for marker in list(SVC.TEST_EMBEDDER["by_marker"]):
        SVC.TEST_EMBEDDER["by_marker"][marker] = 0.0
    assert svc.consolidate_speaker("Ada", user_id="u1")["status"] == "clean"
    assert svc.list_flagged_snippets(user_id="u1") == []


def test_keeping_a_snippet_clears_its_mark_and_keeps_the_audio(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    flagged = svc.list_flagged_snippets(user_id="u1")[0]
    out = svc.keep_flagged_snippet(flagged["id"], user_id="u1")
    assert out["status"] == "kept" and out["changed"] is True
    assert svc.list_flagged_snippets(user_id="u1") == []
    # Keeping is not deleting: the snippet and its file are still there.
    assert len(svc.list_snippets("Ada", user_id="u1")) == 6
    assert Path(flagged["file_path"]).exists()


def test_keeping_an_unmarked_snippet_is_a_no_op(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=3)
    snip = svc.list_snippets("Ada", user_id="u1")[0]
    out = svc.keep_flagged_snippet(snip["id"], user_id="u1")
    assert out["changed"] is False


def test_keeping_one_mark_twice_reports_no_change(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    fid = svc.list_flagged_snippets(user_id="u1")[0]["id"]
    svc.keep_flagged_snippet(fid, user_id="u1")
    assert svc.keep_flagged_snippet(fid, user_id="u1")["changed"] is False


def test_keeping_an_unknown_snippet_reports_an_error(tmp_path):
    svc = _service(tmp_path)
    assert "error" in svc.keep_flagged_snippet(9999, user_id="u1")


def test_clear_every_mark_of_one_speaker(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    _cluster(svc, "Bob", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    svc.consolidate_speaker("Bob", user_id="u1")
    assert len(svc.list_flagged_snippets(user_id="u1")) == 2
    out = svc.keep_flagged_speaker("Ada", user_id="u1")
    assert out["cleared"] == 1
    left = svc.list_flagged_snippets(user_id="u1")
    assert [f["speaker_name"] for f in left] == ["Bob"]


def test_flagged_listing_can_be_restricted_to_one_speaker(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    _cluster(svc, "Bob", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    svc.consolidate_speaker("Bob", user_id="u1")
    only = svc.list_flagged_snippets(user_id="u1", speaker_name="Ada")
    assert len(only) == 1 and only[0]["speaker_name"] == "Ada"


def test_without_an_embedding_session_nothing_is_scored_or_written(tmp_path):
    svc = _service(tmp_path, emb_session=None)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc._db.save(name="Ada", user_id="u1", embedding=_vec(0.0),
                 pitch_hz=5.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=99.0, sample_count=99, mfcc={})
    out = svc.consolidate_speaker("Ada", user_id="u1")
    # 'skipped' must be distinguishable from 'clean': nothing was measured.
    assert out["status"] == "skipped"
    assert svc.list_flagged_snippets(user_id="u1") == []
    vp = svc._db.get("Ada", user_id="u1")
    assert vp["pitch_hz"] == 5.0 and vp["sample_count"] == 99


def test_the_session_vanishing_mid_pass_is_reported_not_silently_clean(tmp_path):
    """The TTL monitor unloads when no job is running; _auto_refine would then
    return silently and the report would claim success."""
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        # Let the guards pass, then vanish before the embeddings run.
        return object() if state["n"] <= 1 else None

    svc._emb_session = flaky
    out = svc.consolidate_speaker("Ada", user_id="u1")
    assert out["status"] == "skipped"
    assert svc.list_flagged_snippets(user_id="u1") == []


def _marker_of(path) -> int:
    return int(Path(str(path)).read_bytes()[3:])


def test_a_zero_embedding_is_not_accused_of_being_another_person(tmp_path):
    """A zero vector means 'no usable window', not 'maximally different'."""
    svc = _service(tmp_path)
    paths = _cluster(svc, "Ada", count=5)
    # The sixth file has an angle that WOULD be flagged, but the embedder
    # returns zeros for it: there is no usable audio window, which is an
    # absence of evidence, not evidence of another speaker.
    SVC.TEST_EMBEDDER["by_marker"][_marker_of(paths[-1])] = 90.0
    SVC.TEST_EMBEDDER["zero_marker"] = _marker_of(paths[-1])
    out = svc.consolidate_speaker("Ada", user_id="u1")
    assert out["flagged_count"] == 0
    assert out["unusable_snippets"] == 1


def test_a_speaker_with_no_snippets_is_skipped(tmp_path):
    svc = _service(tmp_path)
    out = svc.consolidate_speaker("Nobody", user_id="u1")
    assert out["status"] == "skipped"
    assert out["flagged"] == []


def test_consolidation_never_crosses_users(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True, user_id="u1")
    _cluster(svc, "Ada", count=5, intruder=True, user_id="u2")
    svc.consolidate_speaker("Ada", user_id="u1")
    assert [f["user_id"] for f in svc.list_flagged_snippets(user_id="u1")] == ["u1"]
    assert svc.list_flagged_snippets(user_id="u2") == []


def test_a_pending_profile_stays_pending_through_consolidation(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Learned", count=5, intruder=True)
    svc.consolidate_speaker("Learned", user_id="u1")
    svc._db.set_pending("Learned", True, user_id="u1")
    svc.consolidate_speaker("Learned", user_id="u1")
    # The fit saves with pending=None meaning PRESERVE (lesson 39).
    assert svc._db.get("Learned", user_id="u1")["pending"] is True


# ── consolidate_library ───────────────────────────────────────────────

def test_the_sweep_covers_every_speaker_and_totals_the_flags(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    _cluster(svc, "Bob", count=5)
    _cluster(svc, "Cyd", count=5, intruder=True)
    out = svc.consolidate_library(user_id="u1")
    assert out["speakers"] == 3
    assert out["flagged_count"] == 2
    assert sorted(r["speaker_name"] for r in out["results"]) == ["Ada", "Bob", "Cyd"]


def test_the_sweep_reports_progress_per_speaker(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=3)
    _cluster(svc, "Bob", count=3)
    events = []
    svc.consolidate_library(user_id="u1",
                            progress_callback=events.append)
    stages = [e["stage"] for e in events]
    assert any("Ada" in s for s in stages)
    assert any("Bob" in s for s in stages)
    assert all("progress" in e for e in events)


def test_one_broken_speaker_does_not_stop_the_sweep(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=3)
    _cluster(svc, "Bob", count=3)

    original = svc.consolidate_speaker
    svc.consolidate_speaker = lambda name, user_id=svc and "u1", **kw: (
        (_ for _ in ()).throw(RuntimeError("boom")) if name == "Ada"
        else original(name, user_id=user_id, **kw))
    out = svc.consolidate_library(user_id="u1")
    svc.consolidate_speaker = original
    assert out["speakers"] == 2
    statuses = {r["speaker_name"]: r["status"] for r in out["results"]}
    assert statuses["Ada"] == "failed"
    assert statuses["Bob"] in ("clean", "consolidated")


def test_the_sweep_is_user_scoped(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=3, user_id="u1")
    _cluster(svc, "Zoe", count=3, user_id="u2")
    out = svc.consolidate_library(user_id="u1")
    assert [r["speaker_name"] for r in out["results"]] == ["Ada"]


# ── remove_flagged_snippets ───────────────────────────────────────────

def test_the_first_call_is_a_dry_run_and_changes_nothing(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    out = svc.remove_flagged_snippets(user_id="u1", dry_run=True)
    assert out["dry_run"] is True
    assert out["removed_count"] == 1
    assert out["refined"] == []
    # The default must be dry_run: a delete endpoint that acts on the first
    # call cannot be previewed.
    default = svc.remove_flagged_snippets(user_id="u1")
    assert default["dry_run"] is True
    assert len(svc.list_snippets("Ada", user_id="u1")) == 6


def test_dry_run_defaults_to_true_from_the_signature(tmp_path):
    import inspect
    svc = _service(tmp_path)
    sig = inspect.signature(svc.remove_flagged_snippets)
    assert sig.parameters["dry_run"].default is True


def test_applying_deletes_the_row_and_the_audio_and_refines_once(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    flagged = svc.list_flagged_snippets(user_id="u1")[0]
    refines = {"n": 0}
    original = svc._auto_refine
    svc._auto_refine = lambda *a, **k: (refines.__setitem__("n", refines["n"] + 1),
                                       original(*a, **k))[1]

    out = svc.remove_flagged_snippets(user_id="u1", dry_run=False)
    svc._auto_refine = original
    assert out["removed_count"] == 1
    assert out["refined"] == ["Ada"]
    # Once, not once per deleted snippet.
    assert refines["n"] == 1
    assert not Path(flagged["file_path"]).exists()
    assert len(svc.list_snippets("Ada", user_id="u1")) == 5
    assert svc.list_flagged_snippets(user_id="u1") == []


def test_the_profile_survives_the_delete_and_covers_the_survivors(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    svc.remove_flagged_snippets(user_id="u1", dry_run=False)
    vp = svc._db.get("Ada", user_id="u1")
    assert vp is not None
    assert vp["sample_count"] == 5
    assert vp["total_speech_sec"] == pytest.approx(20.0, abs=0.2)


def test_removing_the_last_snippet_drops_the_profile(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    # Flag everything (forced through the mark, not the detector).
    for sn in svc.list_snippets("Ada", user_id="u1"):
        svc._snippets.set_review(sn["id"], user_id="u1", flagged=True,
                                 score=0.9, reason="forced")
    out = svc.remove_flagged_snippets(user_id="u1", dry_run=False)
    assert out["removed_count"] == 6
    assert svc._db.get("Ada", user_id="u1") is None
    assert svc.list_snippets("Ada", user_id="u1") == []


def test_removal_can_be_restricted_to_one_speaker(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    _cluster(svc, "Bob", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    svc.consolidate_speaker("Bob", user_id="u1")
    out = svc.remove_flagged_snippets(user_id="u1", speaker_name="Ada",
                                      dry_run=False)
    assert {e["speaker_name"] for e in out["removed"]} == {"Ada"}
    assert [f["speaker_name"] for f in svc.list_flagged_snippets(user_id="u1")] == ["Bob"]


def test_removal_never_crosses_users(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True, user_id="u1")
    _cluster(svc, "Ada", count=5, intruder=True, user_id="u2")
    svc.consolidate_speaker("Ada", user_id="u1")
    svc.consolidate_speaker("Ada", user_id="u2")
    svc.remove_flagged_snippets(user_id="u1", dry_run=False)
    assert len(svc.list_snippets("Ada", user_id="u1")) == 5
    assert len(svc.list_snippets("Ada", user_id="u2")) == 6


def test_removing_nothing_is_ok(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=3)
    out = svc.remove_flagged_snippets(user_id="u1", dry_run=False)
    assert out["removed_count"] == 0 and out["ok"] is True
    assert len(svc.list_snippets("Ada", user_id="u1")) == 3


def test_the_plan_lists_the_evidence_it_would_act_on(tmp_path):
    svc = _service(tmp_path)
    _cluster(svc, "Ada", count=5, intruder=True)
    svc.consolidate_speaker("Ada", user_id="u1")
    plan = svc.remove_flagged_snippets(user_id="u1")
    entry = plan["removed"][0]
    for key in ("snippet_id", "speaker_name", "file_path", "duration_sec",
                "distance", "reason"):
        assert key in entry, f"the plan omits {key}"
    assert entry["deleted"] is False


# ── the wiring, read from the source ──────────────────────────────────

def _service_source() -> str:
    return (REPO_ROOT / "asr_mcp" / "voiceprint" / "service.py").read_text(
        encoding="utf-8")


def test_the_embedding_happens_once_for_the_whole_pass():
    """Two passes would cost ~25 min of GPU per 219 snippets."""
    src = _service_source()
    body = src[src.index("def consolidate_speaker"):src.index("def consolidate_library")]
    assert body.count("_embed_snippets(") == 1
    # The shared helper is what _auto_refine uses too.
    start = src.index("def _auto_refine")
    refine = src[start:src.index("def _cleanup_speaker_dir", start)]
    assert "_embed_snippets(" in refine
    assert "extract_embedding(" not in refine, (
        "_auto_refine embeds directly again -- the two passes can now drift")


def test_the_router_registers_a_job_so_the_models_stay_loaded():
    router = (REPO_ROOT / "asr_mcp" / "api" / "voiceprint_router.py").read_text(
        encoding="utf-8")
    body = router[router.index("def consolidate_stream"):]
    assert 'job_state.start_job(mode="consolidate"' in body
    assert "ensure_finished(job)" in body
    # Lesson 25: 'done' finishes the job, so progress events must not use it.
    assert '{"stage": "done"' in body
    assert "run_in_executor" in body


def test_the_bulk_delete_does_not_go_through_delete_snippet():
    """delete_snippet re-embeds on every call: 16 flags would mean 16 embeds."""
    src = _service_source()
    body = src[src.index("def remove_flagged_snippets"):src.index("def delete_snippet")]
    assert "self.delete_snippet(" not in body
    assert "_delete_snippet_row(" in body
    assert body.count("self._auto_refine(") == 1