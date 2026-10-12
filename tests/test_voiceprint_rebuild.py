"""Tests for the automatic voiceprint rebuild driven by the audit.

Three layers, deliberately separated:

* ``rebuild_targets`` is pure, so the finding -> speaker mapping is tested
  without a database;
* the queue file is tested against a temp path with a fake service, so the
  attempt/attempts/done bookkeeping is pinned without models;
* ``rebuild_profile_from_scratch`` is tested against a REAL SQLite database
  with the audio stack stubbed, because the thing that must not regress is
  ``pending`` and ``purity`` surviving a row delete.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_rebuild():
    """Load ``voiceprint/rebuild.py`` BY PATH, not via the package.

    ``asr_mcp.voiceprint.__init__`` imports ``VoiceprintService``, which needs
    soundfile and torch. Importing it here at collection time would put the
    real package and the real service module into ``sys.modules`` before
    ``tests/test_learning_boundaries.py`` installs its stubs, and that module's
    stubbed loader would then load the REAL service against a dict-backed fake
    database -- 33 failures that only appear in a full-suite run. Same reason
    ``tests/test_voiceprint_audit.py`` loads by path.
    """
    path = REPO_ROOT / "asr_mcp" / "voiceprint" / "rebuild.py"
    spec = importlib.util.spec_from_file_location("vp_rebuild", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rebuild = _load_rebuild()


def _emb(value: float = 0.1):
    """VoiceprintDB.save serialises with .tolist(), so it needs an array."""
    import numpy as np
    return np.full(192, value, dtype=np.float32)


# ── rebuild_targets ───────────────────────────────────────────────────

def _report(**kw):
    base = {
        "user_id": "u1",
        "near_duplicate_profiles": [],
        "name_collisions": [],
        "profiles_without_snippets": [],
        "snippets_without_profile": [],
        "missing_snippet_files": [],
        "orphan_snippet_files": [],
        "duplicate_snippet_content": [],
        "stale_speech_totals": [],
    }
    base.update(kw)
    return base


def test_targets_cover_the_three_repairable_findings():
    targets = rebuild.rebuild_targets(_report(
        snippets_without_profile=["Ada"],
        profiles_without_snippets=["Bob"],
        missing_snippet_files=[{"speaker": "Cyd", "snippet_id": 3}],
    ))
    assert [t["speaker"] for t in targets] == ["Ada", "Bob", "Cyd"]
    assert targets[0]["reasons"] == ["snippets_without_profile"]
    assert targets[2]["reasons"] == ["missing_snippet_files"]


def test_targets_ignore_findings_that_are_the_users_call():
    report = _report(
        near_duplicate_profiles=[{"a": "A", "b": "B"}],
        name_collisions=[{"normalised": "ada", "names": ["Ada", "ada"]}],
        duplicate_snippet_content=[{"speaker": "Ada", "duplicate_id": 2}],
        orphan_snippet_files=[{"speaker": "Ada", "file": "x.flac"}],
        stale_speech_totals=[{"speaker": "Ada", "total_speech_sec": 99.0}],
    )
    # Merging two profiles and deleting a duplicate clip are decisions, not
    # rebuilds -- nothing here may put a speaker on the queue.
    assert rebuild.rebuild_targets(report) == []


def test_targets_merge_reasons_for_one_speaker():
    targets = rebuild.rebuild_targets(_report(
        snippets_without_profile=["Ada"],
        missing_snippet_files=[{"speaker": "Ada"}],
    ))
    assert targets == [{"speaker": "Ada",
                        "reasons": ["snippets_without_profile",
                                    "missing_snippet_files"]}]


def test_targets_ignore_entries_without_a_speaker():
    assert rebuild.rebuild_targets(_report(
        missing_snippet_files=[{"speaker": ""}, {"no": "name"}, "Ada"])) == [
        {"speaker": "Ada", "reasons": ["missing_snippet_files"]}]


# ── queue file ────────────────────────────────────────────────────────

class _FakeService:
    def __init__(self, results=None, raises=None):
        self.results = results or {}
        self.raises = raises or {}
        self.calls = []

    def rebuild_profile_from_scratch(self, speaker, user_id="default"):
        self.calls.append((user_id, speaker))
        if (user_id, speaker) in self.raises:
            raise RuntimeError(self.raises[(user_id, speaker)])
        return self.results.get((user_id, speaker), {"status": "rebuilt"})


def test_enqueue_writes_only_the_flagged_speakers(tmp_path):
    path = rebuild.queue_path(tmp_path)
    added = rebuild.enqueue_from_reports(path, [_report(
        user_id="u1", snippets_without_profile=["Ada"],
        profiles_without_snippets=["Bob"])])
    assert added == 2
    entries = rebuild.pending_entries(path)
    assert {(e["user_id"], e["speaker"]) for e in entries} == {
        ("u1", "Ada"), ("u1", "Bob")}


def test_enqueue_is_idempotent_and_merges_reasons(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    added = rebuild.enqueue_from_reports(path, [
        _report(snippets_without_profile=["Ada"],
                missing_snippet_files=[{"speaker": "Ada"}])])
    assert added == 0
    entries = rebuild.pending_entries(path)
    assert len(entries) == 1
    assert entries[0]["reasons"] == ["missing_snippet_files",
                                      "snippets_without_profile"]
    assert entries[0]["attempts"] == 0


def test_enqueue_keeps_users_apart(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(user_id="u1",
                                               snippets_without_profile=["Ada"])])
    rebuild.enqueue_from_reports(path, [_report(user_id="u2",
                                               snippets_without_profile=["Ada"])])
    assert len(rebuild.pending_entries(path)) == 2


def test_corrupt_queue_file_is_tolerated(tmp_path):
    path = rebuild.queue_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    # A truncated file must not stop the server: warn and start fresh.
    assert rebuild.pending_entries(path) == []
    assert rebuild.enqueue_from_reports(
        path, [_report(snippets_without_profile=["Ada"])]) == 1


def test_a_done_finding_is_not_re_queued(tmp_path):
    path = rebuild.queue_path(tmp_path)
    report = [_report(snippets_without_profile=["Ada"])]
    rebuild.enqueue_from_reports(path, report)
    rebuild.run_pending(_FakeService(), path)
    assert rebuild.pending_entries(path) == []
    # Same finding on the next startup must not refill the queue -- the file
    # is still missing, so the audit will report it forever.
    assert rebuild.enqueue_from_reports(path, report) == 0
    assert rebuild.pending_entries(path) == []


def test_a_changed_finding_is_re_queued(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    rebuild.run_pending(_FakeService(), path)
    added = rebuild.enqueue_from_reports(
        path, [_report(snippets_without_profile=["Ada", "Bob"])])
    assert added == 1
    assert {e["speaker"] for e in rebuild.pending_entries(path)} == {"Bob"}


# ── run_pending ───────────────────────────────────────────────────────

def test_run_pending_only_touches_flagged_speakers(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [
        _report(user_id="u1", snippets_without_profile=["Ada"]),
        _report(user_id="u2", profiles_without_snippets=["Bob"])])
    service = _FakeService()
    summary = rebuild.run_pending(service, path)
    assert sorted(service.calls) == [("u1", "Ada"), ("u2", "Bob")]
    assert summary["rebuilt"] == 2
    assert rebuild.pending_entries(path) == []


def test_a_skipped_result_is_not_an_attempt(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    service = _FakeService(results={("u1", "Ada"): {"status": "skipped"}})
    summary = rebuild.run_pending(service, path)
    assert summary["skipped"] == 1
    assert summary["rebuilt"] == 0
    entries = rebuild.pending_entries(path)
    assert len(entries) == 1
    # The models were not ready, so nothing was tried and nothing is recorded.
    assert entries[0]["attempts"] == 0


def test_a_failure_is_retried_then_given_up_on(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    service = _FakeService(results={("u1", "Ada"): {"status": "failed"}})
    for attempt in range(1, rebuild.MAX_ATTEMPTS):
        rebuild.run_pending(service, path)
        assert len(rebuild.pending_entries(path)) == 1
    rebuild.run_pending(service, path)
    # A permanently broken speaker must not wedge the queue behind it.
    assert rebuild.pending_entries(path) == []
    assert len(service.calls) == rebuild.MAX_ATTEMPTS


def test_a_raising_rebuild_is_retried_then_given_up_on(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    service = _FakeService(raises={("u1", "Ada"): "cuda exploded"})
    for _ in range(rebuild.MAX_ATTEMPTS - 1):
        summary = rebuild.run_pending(service, path)
        assert summary["failed"] == 1
    rebuild.run_pending(service, path)
    assert rebuild.pending_entries(path) == []


def test_a_dropped_profile_counts_as_done(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [
        _report(profiles_without_snippets=["Ghost"])])
    service = _FakeService(results={("u1", "Ghost"): {"status": "dropped"}})
    summary = rebuild.run_pending(service, path)
    assert summary["dropped"] == 1
    assert rebuild.pending_entries(path) == []


def test_one_failure_does_not_block_the_other_entries(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [
        _report(snippets_without_profile=["Ada", "Bob"])])
    service = _FakeService(results={("u1", "Ada"): {"status": "failed"}})
    rebuild.run_pending(service, path)
    assert [e["speaker"] for e in rebuild.pending_entries(path)] == ["Ada"]


def test_run_pending_on_an_empty_queue_does_not_write(tmp_path):
    path = rebuild.queue_path(tmp_path)
    summary = rebuild.run_pending(_FakeService(), path)
    assert summary["attempted"] == 0
    assert not path.exists()


def test_run_pending_never_raises_on_a_broken_service(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])

    class _Exploding:
        def rebuild_profile_from_scratch(self, *a, **k):
            raise RuntimeError("boom")

    summary = rebuild.run_pending(_Exploding(), path)
    assert summary["failed"] == 1
    assert len(rebuild.pending_entries(path)) == 1


def test_the_queue_file_is_valid_json_after_a_write(tmp_path):
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [_report(snippets_without_profile=["Ada"])])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["entries"][0]["speaker"] == "Ada"
    assert isinstance(data["entries"][0]["first_seen"], float)
    # The temp file used for the atomic write must not survive it.
    assert list(path.parent.glob("*.tmp")) == []


# ── rebuild_profile_from_scratch, against a real database ────────────

def _service(tmp_path, emb_session):
    """A VoiceprintService with a real DB and a stubbed audio stack.

    The ML stack is stubbed in ``sys.modules`` and the service module is loaded
    BY PATH, so the tests run on a host with no torch/soundfile while the
    module under test is still the real one.
    """
    import importlib.util
    import numpy as np

    sf = types.ModuleType("soundfile")
    sf.write = lambda *a, **k: None
    sf.read = lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000)
    sf.info = lambda *a, **k: types.SimpleNamespace(samplerate=16000, frames=16000)
    sys.modules["soundfile"] = sf

    torch = types.ModuleType("torch")
    torch.cat = lambda xs, dim=-1: np.concatenate(list(xs), axis=-1)
    torch.from_numpy = np.asarray
    # utils.py and service.py use these in evaluated annotations and isinstance
    # checks at import time, so a stub without them fails the import.
    torch.Tensor = np.ndarray
    torch.float32 = np.float32
    torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: s, __exit__=lambda s, *a: False)
    sys.modules["torch"] = torch

    speaker = types.ModuleType("asr_mcp.speaker.embedding")
    speaker.extract_embedding = lambda wave, sr, session: np.full(192, 0.5,
                                                                  dtype=np.float32)
    speaker.compute_pitch = lambda wave, sr: (150.0, 10.0)
    speaker.compute_energy = lambda wave: 0.01
    speaker.batch_embed = lambda *a, **k: []
    # service.py imports this by name too; a stub module missing an attribute
    # fails the import, not the call, so it has to be here.
    speaker.batch_embed_files = lambda *a, **k: []
    sys.modules["asr_mcp.speaker.embedding"] = speaker

    from asr_mcp.db.manager import DatabaseManager, VoiceprintDB, SnippetDB
    import asr_mcp.voiceprint.utils as vutils
    # 4.0s of audio, so the loaded duration agrees with the snippet rows the
    # tests declare -- _auto_refine derives total_speech_sec from the audio.
    vutils.load_audio = lambda path: (np.zeros(64000, dtype="float32"), 16000)
    vutils.load_audio_segment = vutils.load_audio

    db = DatabaseManager(str(tmp_path / "t.db"))
    import asr_mcp.voiceprint.service as _real
    spec = importlib.util.spec_from_file_location(
        "vp_service_under_test", Path(_real.__file__))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    svc = mod.VoiceprintService(tmp_path, db)
    svc.set_voices_dir(tmp_path / "voices")
    svc._db = VoiceprintDB(db)
    svc._snippets = SnippetDB(db)
    svc._emb_session = lambda: emb_session
    return svc


def _snippet(svc, speaker, name="a.flac", dur=4.0, source="rec.wav",
             user_id="u1", write_file=True):
    """Add a snippet row, and by default the audio it claims.

    The file has to EXIST: the rebuild purges every row whose audio does not
    resolve, so a fixture pointing at a non-existent path is asserting the
    purge path by accident. ``write_file=False`` is how a dead row is made.
    """
    if write_file:
        path = Path(svc.voices_dir) / user_id / speaker / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFstub")
    else:
        path = Path(svc.voices_dir) / user_id / speaker / name
    svc._snippets.add(user_id=user_id, speaker_name=speaker,
                      file_path=str(path), duration_sec=dur,
                      source_audio=source, start_sec=0.0, end_sec=dur)
    return path


def test_rebuild_recreates_the_profile_from_the_survivors(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada")
    svc._db.save(name="Ada", user_id="u1",
                 embedding=_emb(), pitch_hz=1.0, pitch_std=0.0,
                 energy_rms=0.0, total_speech_sec=999.0, sample_count=1,
                 mfcc={})
    # A stale row: the numbers do not describe the snippets at all.
    out = svc.rebuild_profile_from_scratch("Ada", user_id="u1")
    assert out["status"] == "rebuilt"
    assert out["snippet_count"] == 1
    vp = svc._db.get("Ada", user_id="u1")
    assert vp is not None
    # Rebuilt, not patched: the stale totals are gone.
    assert vp["total_speech_sec"] == pytest.approx(4.0, abs=0.2)
    assert vp["sample_count"] == 1
    # The snippets -- the source of truth -- are untouched.
    assert len(svc._snippets.list_by_speaker("Ada", user_id="u1")) == 1


def test_a_pending_profile_stays_pending(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Learned")
    svc._db.save(name="Learned", user_id="u1", embedding=_emb(),
                 pitch_hz=0.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=0.0, sample_count=0, mfcc={}, pending=True)
    out = svc.rebuild_profile_from_scratch("Learned", user_id="u1")
    assert out["status"] == "rebuilt"
    assert out["pending"] is True
    # Deleting the row destroys `pending`, and _auto_refine saves with
    # pending=None meaning PRESERVE -- so without the re-apply the profile
    # would silently become matchable (lesson 39).
    assert svc._db.get("Learned", user_id="u1")["pending"] is True
    # ...and it must not appear among the profiles matching may use.
    assert "Learned" not in svc._db.list_all(user_id="u1",
                                            include_pending=False)


def test_the_purity_verdict_survives(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada")
    verdict = {"score": 0.42, "verdict": "mixed"}
    svc._db.save(name="Ada", user_id="u1", embedding=_emb(),
                 pitch_hz=0.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=0.0, sample_count=0, mfcc={},
                 purity=verdict)
    assert svc.rebuild_profile_from_scratch("Ada", user_id="u1")["status"] == "rebuilt"
    assert svc._db.get("Ada", user_id="u1")["purity"] == verdict


def test_a_profile_with_no_snippets_is_dropped(tmp_path):
    svc = _service(tmp_path, object())
    svc._db.save(name="Ghost", user_id="u1", embedding=_emb(),
                 pitch_hz=0.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=1.0, sample_count=1, mfcc={})
    out = svc.rebuild_profile_from_scratch("Ghost", user_id="u1")
    # There is nothing to rebuild from; keeping the row would keep the finding.
    assert out["status"] == "dropped"
    assert svc._db.get("Ghost", user_id="u1") is None


def test_without_an_embedding_session_nothing_is_touched(tmp_path):
    svc = _service(tmp_path, None)
    _snippet(svc, "Ada")
    svc._db.save(name="Ada", user_id="u1", embedding=_emb(),
                 pitch_hz=7.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=3.0, sample_count=1, mfcc={})
    out = svc.rebuild_profile_from_scratch("Ada", user_id="u1")
    # 'skipped' is not an attempt -- the queue must keep the entry.
    assert out["status"] == "skipped"
    vp = svc._db.get("Ada", user_id="u1")
    assert vp is not None and vp["pitch_hz"] == 7.0


def test_a_rebuild_never_crosses_users(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada", source="mine.wav")
    svc._db.save(name="Ada", user_id="u2", embedding=_emb(0.2),
                 pitch_hz=3.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=1.0, sample_count=1, mfcc={})
    # u2 has the profile but no snippets; u1 has snippets but no profile.
    assert svc.rebuild_profile_from_scratch("Ada", user_id="u2")["status"] == "dropped"
    assert svc.rebuild_profile_from_scratch("Ada", user_id="u1")["status"] == "rebuilt"
    assert svc._db.get("Ada", user_id="u2") is None
    assert svc._db.get("Ada", user_id="u1") is not None


def test_the_queue_and_the_service_agree_end_to_end(tmp_path):
    """An audit finding routed through the queue rebuilds the right speaker."""
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada")
    svc._db.save(name="Ada", user_id="u1", embedding=_emb(),
                 pitch_hz=1.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=999.0, sample_count=9, mfcc={})
    path = rebuild.queue_path(tmp_path)
    rebuild.enqueue_from_reports(path, [
        _report(user_id="u1", snippets_without_profile=["Ada"])])
    summary = rebuild.run_pending(svc, path)
    assert summary["rebuilt"] == 1
    assert rebuild.pending_entries(path) == []
    assert svc._db.get("Ada", user_id="u1")["total_speech_sec"] < 100.0


# ── the purge: clean the records, THEN rebuild from disk ──────────────

def test_dead_rows_are_deleted_and_the_profile_built_from_disk(tmp_path):
    """The order is the whole repair: clean up, then rebuild what survived."""
    svc = _service(tmp_path, object())
    live = _snippet(svc, "Ada", name="live.flac", dur=4.0)
    dead = _snippet(svc, "Ada", name="gone.wav", dur=30.0, write_file=False)
    assert not Path(dead).exists()
    svc._db.save(name="Ada", user_id="u1", embedding=_emb(),
                 pitch_hz=1.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=999.0, sample_count=9, mfcc={})

    out = svc.rebuild_profile_from_scratch("Ada", user_id="u1")

    assert out["status"] == "rebuilt"
    assert out["dead_snippets"] == 1
    assert out["seconds"] == pytest.approx(30.0)
    # The dead row is gone; the live row and its audio are untouched.
    ids = [r["id"] for r in svc._snippets.list_by_speaker("Ada", user_id="u1")]
    assert len(ids) == 1
    assert Path(live).is_file()
    # And the profile describes only the 4s that survived, not the 34s of rows.
    assert out["snippet_count"] == 1
    assert svc._db.get("Ada", user_id="u1")["total_speech_sec"] == pytest.approx(
        4.0, abs=0.2)


def test_a_speaker_whose_every_snippet_is_dead_loses_its_profile(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ghost", name="a.wav", write_file=False)
    svc._db.save(name="Ghost", user_id="u1", embedding=_emb(),
                 pitch_hz=1.0, pitch_std=0.0, energy_rms=0.0,
                 total_speech_sec=10.0, sample_count=1, mfcc={})
    out = svc.rebuild_profile_from_scratch("Ghost", user_id="u1")
    assert out["status"] == "dropped"
    assert out["dead_snippets"] == 1
    # A profile whose only audio is gone cannot be rebuilt, and keeping it
    # would keep the finding.
    assert svc._db.get("Ghost", user_id="u1") is None
    assert svc._snippets.list_by_speaker("Ghost", user_id="u1") == []


def test_the_purge_never_unlinks_audio_that_exists(tmp_path):
    svc = _service(tmp_path, object())
    keep = _snippet(svc, "Ada", name="keep.flac")
    _snippet(svc, "Ada", name="gone.wav", write_file=False)
    svc.rebuild_profile_from_scratch("Ada", user_id="u1")
    # Deleting a dead row deletes a ROW. Its file is already gone; a file that
    # resolves must never be unlinked by a cleanup pass.
    assert Path(keep).is_file()


def test_the_purge_is_scoped_to_one_speaker_and_one_user(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada", name="a.flac")
    _snippet(svc, "Bo", name="b.wav", write_file=False)
    _snippet(svc, "Ada", name="other.wav", source="x.wav", user_id="u2",
             write_file=False)
    svc.rebuild_profile_from_scratch("Ada", user_id="u1")
    assert svc._snippets.list_by_speaker("Ada", user_id="u1") != []
    # Bo's dead row and u2's dead row are not Ada's to delete.
    assert len(svc._snippets.list_by_speaker("Bo", user_id="u1")) == 1
    assert len(svc._snippets.list_by_speaker("Ada", user_id="u2")) == 1


def test_purge_dead_false_leaves_the_rows_alone(tmp_path):
    svc = _service(tmp_path, object())
    _snippet(svc, "Ada", name="a.flac")
    _snippet(svc, "Ada", name="gone.wav", write_file=False)
    out = svc.rebuild_profile_from_scratch("Ada", user_id="u1", purge_dead=False)
    assert out["dead_snippets"] == 0
    assert len(svc._snippets.list_by_speaker("Ada", user_id="u1")) == 2


def test_the_purge_happens_even_when_the_models_are_not_loaded(tmp_path):
    """Cleanup needs no GPU, so a missing session must not skip it."""
    svc = _service(tmp_path, None)
    _snippet(svc, "Ada", name="a.flac")
    _snippet(svc, "Ada", name="gone.wav", write_file=False)
    out = svc.rebuild_profile_from_scratch("Ada", user_id="u1")
    assert out["status"] == "skipped"
    assert out["dead_snippets"] == 1
    assert len(svc._snippets.list_by_speaker("Ada", user_id="u1")) == 1
