"""Repairing what the startup audit finds: ``cleanup_integrity``.

The audit (``tests/test_voiceprint_audit.py``) is read-only on purpose, and this
file pins the half that acts. Two things make it worth testing separately from
the audit itself:

1. **The default does nothing.** ``dry_run`` is true, so the first call returns
   a plan. An endpoint that deletes audio and cannot be called safely is an
   endpoint nobody calls twice.
2. **It only touches what it says it will.** Six of the audit's eight findings
   have no single correct answer -- only the user knows whether two similar
   profiles are one person or two. Those must come back in ``unfixed`` and be
   left alone, including the tempting ones.

``service.py`` is loaded by path with ``torch``/``soundfile`` stubbed when they
are absent: ``cleanup_integrity`` needs numpy, the DB and the filesystem, none
of which are the audio stack. Where the real packages exist (in the container)
nothing is stubbed and the real module is used.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="DB layer needed")

from asr_mcp.db.manager import DatabaseManager, VoiceprintDB  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Only stubbed if the real import fails, so the container runs the real module.
_STUBBED = ("torch", "soundfile", "torchaudio")


def _stub_missing() -> list:
    """Install stand-ins for the absent audio stack. Returns what was injected.

    Only used to get ``service.py`` past its module-scope ``import torch`` /
    ``soundfile``; none of the code under test calls into them.
    """
    injected = []
    for name in _STUBBED:
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            # Any attribute access yields a throwaway class: enough for the
            # `import torch`-at-module-scope pattern, never actually called.
            module.__getattr__ = lambda attr: type(attr, (), {})
            module.__path__ = []
            sys.modules[name] = module
            injected.append(name)
    return injected


def _load_service():
    path = REPO_ROOT / "asr_mcp" / "voiceprint" / "service.py"
    spec = importlib.util.spec_from_file_location("vp_service_cleanup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_schemas():
    path = REPO_ROOT / "asr_mcp" / "api" / "schemas.py"
    spec = importlib.util.spec_from_file_location("_schemas_cleanup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IntegrityCleanupRequest = _load_schemas().IntegrityCleanupRequest


def service_module():
    """Load service.py on FIRST USE, not at import time.

    pytest collects every test file before running any test, so a module-level
    load would execute the real ``asr_mcp.voiceprint`` package during
    collection, with the torch stubs already gone. See ``_absent_audio_stack``
    for what that costs the rest of the suite.
    """
    global _service_mod
    if _service_mod is None:
        _service_mod = _load_service()
    return _service_mod


_service_mod = None


@pytest.fixture(scope="module", autouse=True)
def _absent_audio_stack():
    """Stand in for what this interpreter does not have, for the whole module.

    Two things, both undone on teardown:

    1. ``torch`` / ``soundfile`` / ``torchaudio``. ``service.py`` imports them at
       module scope; nothing under test calls into them. A fake left behind
       would make a LATER file's ``importorskip("torch")`` stop skipping, and its
       database tests would then run against a stub.
    2. The ``asr_mcp.*`` package objects themselves. The real
       ``asr_mcp/voiceprint/__init__.py`` re-imports ``service``, so letting it
       run registers the canonical ``asr_mcp.voiceprint.service`` in
       ``sys.modules`` -- and ``conftest.load_module`` hands that cached copy to
       whoever asks next. tests/test_learning_boundaries.py asks, expecting a
       module built against its own dict-backed ``asr_mcp.db.manager``; getting
       the real one instead fails 22 of its tests, all the way across the suite,
       with ``'NoneType' object has no attribute 'get_session'``. A package stub
       with a ``__path__`` still resolves real submodules underneath.
    """
    injected = _stub_missing()
    packages: dict[str, types.ModuleType] = {}
    for name in ("asr_mcp", "asr_mcp.db", "asr_mcp.speaker", "asr_mcp.voiceprint"):
        if name in sys.modules:
            continue
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(REPO_ROOT / name.replace(".", "/"))]
        sys.modules[name] = pkg
        packages[name] = pkg
    yield
    for name in injected:
        sys.modules.pop(name, None)
    for name in packages:
        sys.modules.pop(name, None)


def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def store(tmp_path):
    """A service over a real temp SQLite DB and a real voices tree.

    ``refined`` records every ``_auto_refine`` call, because the count is part
    of the contract: one per affected speaker, not one per deleted snippet.
    """
    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    service = service_module().VoiceprintService(tmp_path, mgr)
    voices = tmp_path / "voices"
    service.set_voices_dir(voices)
    service.refined = []
    service._auto_refine = lambda name, user_id=None, **kw: (
        service.refined.append(name) or None)
    try:
        yield service, voices
    finally:
        mgr.close()


def _add_pair(service, voices, *, speaker="Test Person One", payload=b"same bytes",
              stale_total=None, user_id="u"):
    """Two byte-identical snippets for one speaker, optionally with a bad total."""
    for i, name in enumerate(("first.flac", "second.flac")):
        _write(voices / user_id / speaker / name, payload)
    service._db.save(speaker, _unit(1), user_id=user_id,
                     total_speech_sec=stale_total if stale_total else 10.0,
                     pitch_hz=150.0, energy_rms=0.04)
    ids = [
        service._snippets.add(speaker_name=speaker,
                              file_path=str(voices / user_id / speaker / name),
                              duration_sec=10.0, user_id=user_id)
        for name in ("first.flac", "second.flac")
    ]
    return speaker, ids


# ── the default is inert ───────────────────────────────────────────────────

def test_the_default_call_changes_nothing(store):
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices, stale_total=9999.0)

    result = service.cleanup_integrity("u")

    assert result["dry_run"] is True
    assert len(service.refined) == 0
    assert service._snippets.get(dup_id, user_id="u") is not None
    assert (voices / "u" / speaker / "second.flac").is_file()
    assert service._db.get(speaker, user_id="u")["total_speech_sec"] == 9999.0


def test_the_plan_names_what_would_happen(store):
    """A dry run that does not say what it would do is not a dry run."""
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices, stale_total=9999.0)

    plan = service.cleanup_integrity("u")

    assert [e["deleted_snippet_id"] for e in plan["duplicate_snippets"]] == [dup_id]
    assert [e["kept_snippet_id"] for e in plan["duplicate_snippets"]] == [keep_id]
    assert plan["duplicate_snippets"][0]["speaker"] == speaker
    # "now" is what the profile WOULD describe: the duplicate is still on disk in
    # a dry run, but it is going, so it must not be counted. One surviving
    # 10 s snippet, not the 20 s the audit snapshot saw.
    assert plan["stale_totals"] == [{"speaker": speaker, "was": 9999.0, "now": 10.0}]


def test_the_planned_total_does_not_count_the_snippets_being_removed(store):
    """Both switches in one call: the total must reflect the SURVIVING rows.

    The audit runs once, before anything is deleted, so its figure still counts
    the duplicate copies. Writing that number back would leave the profile stale
    by exactly the audio the same call removed -- which is how a "cleanup" ends
    up reporting a speaker it just made more wrong.
    """
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices, stale_total=9999.0)

    service.cleanup_integrity("u", dry_run=False)

    row = service._db.get(speaker, user_id="u")
    surviving = [r["id"] for r in service._snippets.list_by_speaker(speaker,
                                                                   user_id="u")]
    assert surviving == [keep_id]
    assert row["total_speech_sec"] == pytest.approx(10.0)



def test_the_request_schema_defaults_to_a_dry_run():
    """The safe default has to live in the schema, not only in the handler.

    A handler that defaults the flag itself can be defeated by a caller that
    passes the other two switches; the model is the one place both meet.
    """
    assert IntegrityCleanupRequest().dry_run is True
    assert IntegrityCleanupRequest().duplicates is True
    assert IntegrityCleanupRequest().stale_totals is True


# ── applying ───────────────────────────────────────────────────────────────

def test_applying_removes_the_duplicate_row_and_its_file(store):
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices)
    # The two names sort either way round depending on insertion order, so
    # assert on ids: the keeper is the LOWER id, independent of row order.
    keeper, doomed = min(keep_id, dup_id), max(keep_id, dup_id)
    doomed_file = service._snippets.get(doomed, user_id="u")["file_path"]

    result = service.cleanup_integrity("u", dry_run=False)

    assert service._snippets.get(doomed, user_id="u") is None
    assert not Path(doomed_file).exists()
    assert service._snippets.get(keeper, user_id="u") is not None
    assert result["duplicate_snippets"][0]["deleted"] is True
    assert result["duplicate_snippets"][0]["deleted_snippet_id"] == doomed


def test_applying_refines_each_affected_speaker_exactly_once(store):
    """Not once per deleted snippet.

    ``_auto_refine`` re-embeds every snippet of the speaker. A profile with
    eight redundant copies would otherwise be re-embedded eight times, and the
    intermediate refines would each fold a different (wrong) corpus.
    """
    service, voices = store
    payload = b"identical audio"
    speaker = "Test Person One"
    for name in ("a.flac", "b.flac", "c.flac", "d.flac"):
        _write(voices / "u" / speaker / name, payload)
    service._db.save(speaker, _unit(1), user_id="u", total_speech_sec=40.0)
    for name in ("a.flac", "b.flac", "c.flac", "d.flac"):
        service._snippets.add(speaker_name=speaker,
                              file_path=str(voices / "u" / speaker / name),
                              duration_sec=10.0, user_id="u")

    service.cleanup_integrity("u", dry_run=False)

    assert service.refined == [speaker]  # once, not three times


def test_the_stale_total_is_corrected_without_touching_the_embedding(store):
    """Fixing a number must not become re-registering the speaker."""
    service, voices = store
    speaker, _ = _add_pair(service, voices, stale_total=9999.0)
    before = service._db.get(speaker, user_id="u")
    before_embedding = before["embedding"].copy()
    before_pitch, before_energy = before["pitch_hz"], before["energy_rms"]

    service.cleanup_integrity("u", dry_run=False, duplicates=False)

    after = service._db.get(speaker, user_id="u")
    # Both 10 s snippets are still stored, so 20 s is the honest sum: with the
    # duplicate switch off, removing nothing means nothing is excluded.
    assert after["total_speech_sec"] == pytest.approx(20.0)
    assert np.array_equal(after["embedding"], before_embedding)
    assert after["pitch_hz"] == before_pitch
    assert after["energy_rms"] == before_energy


def test_correcting_a_total_leaves_a_pending_profile_pending(store):
    """A pending profile must not become matchable as a side effect.

    Lesson 39: pending is what keeps an unnamed profile out of matching. The
    refine path preserves the flag (``save(pending=None)``); this write must be
    as careful, because it never goes near ``save()``.
    """
    service, voices = store
    speaker, _ = _add_pair(service, voices, stale_total=9999.0)
    service._db.save(speaker, _unit(1), user_id="u", total_speech_sec=9999.0,
                     pending=True)

    service.cleanup_integrity("u", dry_run=False, duplicates=False)

    assert service._db.get(speaker, user_id="u")["pending"] is True


def test_the_switches_are_independent(store):
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices, stale_total=9999.0)

    result = service.cleanup_integrity("u", dry_run=False, stale_totals=False)

    assert service._snippets.get(dup_id, user_id="u") is None
    assert service._db.get(speaker, user_id="u")["total_speech_sec"] == 9999.0
    assert result["stale_totals"] == []


def test_a_container_path_is_unlinked_through_the_voices_dir(store):
    """The stored path is absolute inside the container and absent outside it.

    Unlinking ``/app/voices/...`` verbatim would delete the row, leave the file
    and turn the duplicate into an ORPHAN -- which the next ``rescan`` would
    adopt as a new snippet. The audit's resolver is what makes this safe.
    """
    service, voices = store
    speaker, (keep_id, dup_id) = _add_pair(service, voices)
    real = voices / "u" / speaker / "second.flac"
    service._snippets.update_file_path(
        dup_id, "/app/voices/u/%s/second.flac" % speaker, user_id="u")

    result = service.cleanup_integrity("u", dry_run=False)

    assert result["duplicate_snippets"][0]["deleted"] is True
    assert not real.exists()


def test_a_row_for_another_speaker_is_never_deleted(store):
    """A stale report must not be able to redirect a deletion.

    The audit runs a moment before the cleanup, and the speaker name travels
    with each finding precisely so the delete can be checked against it.
    """
    service, voices = store
    other = "Test Person Two"
    speaker, (keep_id, dup_id) = _add_pair(service, voices)
    assert service._delete_snippet_row(dup_id, other, user_id="u") is False
    assert service._snippets.get(dup_id, user_id="u") is not None
    assert (voices / "u" / speaker / "second.flac").is_file()


def test_a_row_that_is_already_gone_is_reported_not_raised(store):
    service, voices = store
    _speaker, (_keep, dup_id) = _add_pair(service, voices)
    service._snippets.delete(dup_id, user_id="u")

    result = service.cleanup_integrity("u", dry_run=False)

    assert result["duplicate_snippets"] == []  # the audit no longer sees it


# ── what it refuses to fix ─────────────────────────────────────────────────

def test_findings_without_one_right_answer_come_back_unfixed(store):
    """The point of the narrow scope.

    A near-duplicate profile pair is a registered profile and a second one that
    looks like the same person. Merging them is a guess, and guessing here
    would attribute one colleague's speech to another.
    """
    service, voices = store
    _write(voices / "u" / "Test Person One" / "a.flac", b"a")
    service._db.save("Test Person One", _unit(1), user_id="u", total_speech_sec=10.0)
    service._db.save("Test Person One (old)", _unit(1) + 1e-4, user_id="u",
                     total_speech_sec=10.0)
    service._snippets.add(speaker_name="Test Person One",
                          file_path=str(voices / "u" / "Test Person One" / "a.flac"),
                          duration_sec=10.0, user_id="u")

    result = service.cleanup_integrity("u", dry_run=False)

    assert result["unfixed"]["near_duplicate_profiles"], "should be reported"
    # ...and neither profile was touched.
    assert set(service._db.list_all(user_id="u")) == {"Test Person One",
                                                      "Test Person One (old)"}
    assert service.refined == []


def test_a_profile_without_snippets_is_not_deleted(store):
    """It still matches; deleting it would silently change who is known.

    The finding is the important part -- an embedding nothing can rebuild is
    drift -- but removing it is the user's call.
    """
    service, voices = store
    service._db.save("Ghost", _unit(2), user_id="u", total_speech_sec=30.0)

    result = service.cleanup_integrity("u", dry_run=False)

    assert result["unfixed"]["profiles_without_snippets"] == ["Ghost"]
    assert service._db.get("Ghost", user_id="u") is not None


def test_a_healthy_store_reports_an_empty_plan(store):
    service, voices = store
    speaker = "Test Person One"
    _write(voices / "u" / speaker / "only.flac", b"only")
    service._db.save(speaker, _unit(1), user_id="u", total_speech_sec=10.0)
    service._snippets.add(speaker_name=speaker,
                          file_path=str(voices / "u" / speaker / "only.flac"),
                          duration_sec=10.0, user_id="u")

    result = service.cleanup_integrity("u", dry_run=False)

    assert result["duplicate_snippets"] == []
    assert result["stale_totals"] == []
    assert result["refined"] == []
    assert not any(result["unfixed"].values())


# ── the DB helper on its own ───────────────────────────────────────────────

def test_update_speech_total_is_a_single_column_write(db_manager):
    vp_db = VoiceprintDB(db_manager)
    embedding = _unit(3)
    vp_db.save("Test Person One", embedding, user_id="u", total_speech_sec=9999.0,
               pitch_hz=150.0, energy_rms=0.04, sample_count=7, pending=False)
    before = vp_db.get("Test Person One", user_id="u")

    assert vp_db.update_speech_total("Test Person One", 12.5, user_id="u") is True

    after = vp_db.get("Test Person One", user_id="u")
    assert after["total_speech_sec"] == pytest.approx(12.5)
    assert np.array_equal(after["embedding"], before["embedding"])
    assert after["sample_count"] == before["sample_count"] == 7
    assert after["pitch_hz"] == before["pitch_hz"]
    assert after["pending"] is False


def test_update_speech_total_reports_a_missing_profile(db_manager):
    assert VoiceprintDB(db_manager).update_speech_total(
        "Nobody", 1.0, user_id="u") is False


@pytest.fixture
def db_manager(tmp_path):
    mgr = DatabaseManager(str(tmp_path / "asr-standalone.db"))
    yield mgr
    mgr.close()
