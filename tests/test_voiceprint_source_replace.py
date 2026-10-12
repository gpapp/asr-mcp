"""Re-running a file replaces that file's snippets instead of stacking on them.

The overfitting hazard: ``add_snippet`` names its file after the source stem,
the span and the duration, so re-processing one recording ADDS another copy of
the same room, the same channel and the same speaker take. The profile then
looks well-represented and is actually fitted to a single recording. So a
from-file add removes what that file already contributed to THIS speaker, and
the response says how much went away.

Scope is deliberately narrow, and the tests below are mostly about what must
NOT be touched:

* another file's snippets survive -- the corpus stays multi-recording;
* another speaker's snippet from the same file survives -- one recording may
  legitimately feed several profiles;
* ``replace_source=False`` still accumulates (the rollback);
* the auto-collect / learn path (``add_snippet_from_segment``) is unchanged: it
  collects MANY spans out of one recording on purpose, per lesson 29.

``service.py`` is loaded by path with the audio stack stubbed, the same way
``test_voiceprint_integrity_cleanup.py`` does it.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="DB layer needed")

from asr_mcp.db.manager import DatabaseManager  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000

_STUBBED = ("torch", "soundfile", "torchaudio")
_service_mod = None


def _stub_missing() -> list:
    injected = []
    for name in _STUBBED:
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__getattr__ = lambda attr: type(attr, (), {})
            module.__path__ = []
            sys.modules[name] = module
            injected.append(name)
    return injected


def service_module():
    """Load ``service.py`` by path, once, on first use (see the sibling test)."""
    global _service_mod
    if _service_mod is None:
        path = REPO_ROOT / "asr_mcp" / "voiceprint" / "service.py"
        spec = importlib.util.spec_from_file_location("vp_service_source_replace", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _service_mod = mod
    return _service_mod


@pytest.fixture(scope="module", autouse=True)
def _absent_audio_stack():
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


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A service over a real temp DB and voices tree, with FLAC writing faked.

    ``soundfile.write`` is stubbed to write real bytes: the ordering rule below
    (delete BEFORE writing) is only observable if the file really lands on the
    path the next add is about to reuse.
    """
    mod = service_module()

    def _fake_write(path, data, sample_rate, format=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(np.asarray(data, dtype=np.float32).tobytes())
        return None

    monkeypatch.setattr(mod.sf, "write", _fake_write, raising=False)

    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    service = mod.VoiceprintService(tmp_path, mgr)
    service.set_voices_dir(tmp_path / "voices")
    try:
        yield service, tmp_path / "voices"
    finally:
        mgr.close()


def _audio(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, size=int(seconds * SAMPLE_RATE)).astype(np.float32)


def _add(service, speaker, source, seconds=5.0, seed=0, **kw):
    return service.add_snippet(
        speaker_name=speaker,
        audio_data=_audio(seconds, seed),
        user_id=kw.pop("user_id", "u"),
        sample_rate=SAMPLE_RATE,
        source_audio=source,
        **kw,
    )


def _rows(service, speaker, user_id="u"):
    return service._snippets.list_by_speaker(speaker, user_id=user_id)


def _sources(service, speaker, user_id="u"):
    return sorted(r["source_audio"] for r in _rows(service, speaker, user_id))


# ── the replacement itself ─────────────────────────────────────────────────

def test_rerunning_a_file_removes_its_previous_snippets(store):
    service, voices = store
    _add(service, "Ana", "meeting.wav", seconds=5.0)
    _add(service, "Ana", "meeting.wav", seconds=4.0, start_sec=30.0)
    assert len(_rows(service, "Ana")) == 2

    result = _add(service, "Ana", "meeting.wav", seconds=6.0, replace_source=True)

    assert result["replaced_snippets"] == 2
    assert len(_rows(service, "Ana")) == 1
    # Every file the earlier runs left behind is gone, not just the rows:
    # an orphan .flac is what the integrity audit reports as a duplicate.
    assert sorted(p.name for p in (voices / "u" / "Ana").iterdir()) == [
        Path(result["file_path"]).name]


def test_the_new_snippet_survives_its_own_replacement_of_the_same_path(store):
    """The filename is derived from stem+span+duration, so a re-run lands on
    the SAME path. Deleting after the write would unlink the file it just
    wrote, leaving a row pointing at nothing."""
    service, voices = store
    first = _add(service, "Ana", "meeting.wav", seconds=5.0, replace_source=True)
    second = _add(service, "Ana", "meeting.wav", seconds=5.0, replace_source=True)

    assert second["file_path"] == first["file_path"]
    assert Path(second["file_path"]).is_file()
    assert [r["id"] for r in _rows(service, "Ana")] == [second["id"]]


def test_the_response_reports_the_replacement_even_when_nothing_went(store):
    service, _ = store
    result = _add(service, "Ana", "meeting.wav", replace_source=True)
    assert result["replaced_snippets"] == 0


# ── what must survive ──────────────────────────────────────────────────────

def test_snippets_from_OTHER_files_are_kept(store):
    service, _ = store
    _add(service, "Ana", "meeting.wav", seconds=5.0)
    _add(service, "Ana", "interview.wav", seconds=5.0)

    _add(service, "Ana", "meeting.wav", seconds=6.0, replace_source=True)

    # The other recording is untouched, and exactly ONE meeting.wav remains --
    # the new one, not the old one plus this.
    assert _sources(service, "Ana") == ["interview.wav", "meeting.wav"]
    assert len(_rows(service, "Ana")) == 2


def test_another_speakers_snippet_from_the_same_file_is_kept(store):
    """One recording can feed several profiles. Re-running it for one speaker
    must not strip the audio the others learned from."""
    service, _ = store
    _add(service, "Ana", "meeting.wav", seconds=5.0)
    _add(service, "Bo", "meeting.wav", seconds=5.0)

    _add(service, "Ana", "meeting.wav", seconds=6.0, replace_source=True)

    assert len(_rows(service, "Ana")) == 1
    assert len(_rows(service, "Bo")) == 1


def test_another_users_snippet_from_the_same_file_is_kept(store):
    service, _ = store
    _add(service, "Ana", "meeting.wav", seconds=5.0, user_id="u")
    _add(service, "Ana", "meeting.wav", seconds=5.0, user_id="other")

    _add(service, "Ana", "meeting.wav", seconds=6.0, replace_source=True, user_id="u")

    assert len(_rows(service, "Ana", user_id="u")) == 1
    assert len(_rows(service, "Ana", user_id="other")) == 1


def test_snippets_with_no_recorded_source_are_never_replaced(store):
    """A NULL source belongs to no file (rescan-adopted snippets), so it can
    never be the file being re-run."""
    service, _ = store
    _add(service, "Ana", None, seconds=5.0)
    assert _rows(service, "Ana")[0]["source_audio"] is None

    result = _add(service, "Ana", "meeting.wav", seconds=5.0, replace_source=True)

    assert result["replaced_snippets"] == 0
    assert len(_rows(service, "Ana")) == 2


def test_the_default_does_not_replace(store):
    """``replace_source`` is opt-in at the service layer; the rollback is a
    single flag away, and the accumulate behaviour is the historical one."""
    service, _ = store
    _add(service, "Ana", "meeting.wav", seconds=5.0)

    result = _add(service, "Ana", "meeting.wav", seconds=6.0)

    assert result["replaced_snippets"] == 0
    assert len(_rows(service, "Ana")) == 2


def test_the_learn_path_still_accumulates_spans_from_one_file(store, monkeypatch):
    """``add_snippet_from_segment`` collects MANY spans out of one recording on
    purpose (lesson 29: gating is on the cumulative total). It must never be
    routed through the replace path -- doing so would delete the spans it just
    added, one per call."""
    mod = service_module()
    service, _ = store

    class _Waveform:
        def __init__(self, n):
            self._n = n

        def numpy(self):
            return np.zeros((1, self._n), dtype=np.float32)

        def squeeze(self):
            return self

    monkeypatch.setattr(
        mod, "load_audio_segment",
        lambda path, start, end: (_Waveform(int((end - start) * SAMPLE_RATE)),
                                  SAMPLE_RATE))

    ids = [
        service.add_snippet_from_segment(
            "Pending 1", "/audio/meeting.wav", s, s + 5.0, user_id="u",
            source_audio="meeting.wav",
        )["id"]
        for s in (0.0, 10.0, 20.0)
    ]

    assert [r["id"] for r in _rows(service, "Pending 1")] == ids


# ── the DB query it rests on ───────────────────────────────────────────────

def test_source_rows_match_by_filename_hash_not_by_the_stored_string(store):
    """The identity of a recording is the hash ``add_snippet`` names its files
    after -- md5 of the source STEM. A reprocess changes the directory, often
    the extension, and always the spans; none of that may break the match."""
    service, _ = store
    _add(service, "Ana", "meeting.wav", seconds=5.0)
    _add(service, "Ana", "/data/2026/meeting.mp4", seconds=5.0, start_sec=60.0)
    _add(service, "Ana", "interview.wav", seconds=5.0)
    _add(service, "Bo", "meeting.wav", seconds=5.0)

    rows = service._source_snippet_rows("Ana", "meeting.wav", user_id="u")

    # Both spellings of the same recording, neither span, and nothing else.
    assert len(rows) == 2
    assert {r["speaker_name"] for r in rows} == {"Ana"}
    assert {r["source_audio"] for r in rows} == {"meeting.wav", "/data/2026/meeting.mp4"}


def test_source_rows_of_an_empty_origin_match_nothing(store):
    """A NULL source belongs to no file, so it can never be the file re-run."""
    service, _ = store
    _add(service, "Ana", None, seconds=5.0)
    _add(service, "Ana", "meeting.wav", seconds=5.0)

    assert service._source_snippet_rows("Ana", None, user_id="u") == []
    assert service._source_snippet_rows("Ana", "", user_id="u") == []
    assert len(service._source_snippet_rows("Ana", "meeting.wav", user_id="u")) == 1


def test_a_stem_collision_is_the_storage_layout_not_a_new_bug(store):
    """Two DIFFERENT recordings sharing a stem hash together -- and they
    already collided when their snippets were named from that same hash, so the
    replacement rule inherits the layout's identity, it does not add one."""
    service, _ = store
    _add(service, "Ana", "/a/meeting.wav", seconds=5.0)
    _add(service, "Ana", "/b/meeting.wav", seconds=5.0)

    rows = service._source_snippet_rows("Ana", "/c/meeting.wav", user_id="u")

    assert len(rows) == 2


# ── the reprocess path: collect, then drop the previous take ───────────────

def _seed(service, speaker, source, spans, seconds=5.0, user_id="u"):
    for i, (start, end) in enumerate(spans):
        service.add_snippet(
            speaker_name=speaker, audio_data=_audio(end - start, i),
            user_id=user_id, sample_rate=SAMPLE_RATE, source_audio=source,
            start_sec=start, end_sec=end,
        )


def test_replace_drops_the_old_spans_and_keeps_the_new_ones(store):
    service, _ = store
    _seed(service, "Ana", "meeting.wav", [(0.0, 5.0), (10.0, 15.0)])

    fresh = _add(service, "Ana", "meeting.wav", seconds=6.0, start_sec=2.0)
    report = service.replace_source_contribution(
        "Ana", "meeting.wav", user_id="u",
        fresh_rows=[fresh], fresh_paths={fresh["file_path"]})

    assert report == {"removed": 2, "seconds": 10.0}
    assert [r["id"] for r in _rows(service, "Ana")] == [fresh["id"]]


def test_replace_keeps_clips_from_other_recordings(store):
    service, _ = store
    _seed(service, "Ana", "meeting.wav", [(0.0, 5.0)])
    _seed(service, "Ana", "interview.wav", [(0.0, 5.0), (10.0, 15.0)])

    fresh = _add(service, "Ana", "meeting.wav", seconds=6.0, start_sec=2.0)
    service.replace_source_contribution(
        "Ana", "meeting.wav", user_id="u",
        fresh_rows=[fresh], fresh_paths={fresh["file_path"]})

    assert _sources(service, "Ana") == [
        "interview.wav", "interview.wav", "meeting.wav"]
    assert len(_rows(service, "Ana")) == 3


def test_replace_is_scoped_to_one_speaker(store):
    service, _ = store
    _seed(service, "Ana", "meeting.wav", [(0.0, 5.0)])
    _seed(service, "Bo", "meeting.wav", [(0.0, 5.0)])

    fresh = _add(service, "Ana", "meeting.wav", seconds=6.0, start_sec=2.0)
    service.replace_source_contribution(
        "Ana", "meeting.wav", user_id="u",
        fresh_rows=[fresh], fresh_paths={fresh["file_path"]})

    assert len(_rows(service, "Ana")) == 1
    assert len(_rows(service, "Bo")) == 1


def test_replace_leaves_the_new_snippets_file_on_disk(store):
    """A reprocess whose span and duration round to the same filename writes
    OVER the old file. Deleting the old row then has to drop the row without
    unlinking what the surviving row reads."""
    service, voices = store
    old = _add(service, "Ana", "meeting.wav", seconds=5.0)
    fresh = _add(service, "Ana", "meeting.wav", seconds=5.0)
    assert fresh["file_path"] == old["file_path"]

    report = service.replace_source_contribution(
        "Ana", "meeting.wav", user_id="u",
        fresh_rows=[fresh], fresh_paths={fresh["file_path"]})

    assert report["removed"] == 1
    assert Path(fresh["file_path"]).is_file()
    assert [r["id"] for r in _rows(service, "Ana")] == [fresh["id"]]
