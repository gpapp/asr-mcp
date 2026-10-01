"""Phase P2: learn from the boundaries the transcript actually uses.

``tests/test_speaker_learning.py`` covers the pending-profile *contract* against
a real SQLite database, but it needs torch + sqlalchemy, so it only runs in the
server image.  Everything in this module is about the arithmetic of the learning
path -- snippet spans, per-snippet caps, the purity gate, the extend-by-voice
lookup, one decode per request -- and none of it needs a database or an audio
decoder, so it runs everywhere.

The loader below stubs ``torch`` (a thin numpy shim), ``soundfile`` (an
in-memory file that is also written to disk so ``stat()`` works) and
``asr_mcp.db.manager`` (dict-backed ``VoiceprintDB`` / ``SnippetDB`` with the
same preserve-on-``None`` semantics as the SQLAlchemy versions), then loads
``asr_mcp/voiceprint/service.py`` by path the way ``tests/conftest.py`` does for
the light-weight modules.  The policy module is the REAL one.
"""

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT, load_module

# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class _Tensor:
    """The handful of torch.Tensor methods the voiceprint code actually calls."""

    def __init__(self, array):
        self.a = np.asarray(array)

    @property
    def shape(self):
        return self.a.shape

    def numpy(self):
        return self.a

    def unsqueeze(self, axis=0):
        return _Tensor(np.expand_dims(self.a, axis))

    def squeeze(self, *axes):
        return _Tensor(np.squeeze(self.a, *axes) if axes else np.squeeze(self.a))

    def astype(self, dtype):
        return _Tensor(self.a.astype(dtype))

    def float(self):
        return self

    def reshape(self, *shape):
        return _Tensor(self.a.reshape(*shape))

    def tolist(self):
        return self.a.tolist()

    def __getitem__(self, key):
        return _Tensor(self.a[key])

    def __len__(self):
        return len(self.a)


def _as_array(obj):
    return obj.a if isinstance(obj, _Tensor) else np.asarray(obj)


def _install_stubs():
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_p2_stub", False):
        return

    torch = types.ModuleType("torch")
    torch._p2_stub = True
    torch.Tensor = _Tensor
    torch.float32 = np.float32
    torch.from_numpy = _Tensor
    torch.tensor = _Tensor
    torch.cat = lambda seq, dim=0: _Tensor(
        np.concatenate([np.atleast_2d(_as_array(x)) for x in seq], axis=dim))
    torch.zeros_like = lambda a: np.zeros_like(_as_array(a))
    sys.modules["torch"] = torch

    files: dict[str, tuple] = {}
    sf = types.ModuleType("soundfile")
    sf._p2_stub = True
    sf.FLAC = "FLAC"
    sf.WAV = "WAV"

    def _write(path, data, samplerate, format=None, subtype=None):
        arr = _as_array(data).astype(np.float32)
        files[str(path)] = (arr, int(samplerate))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"RIFF-stub")

    def _read(path, dtype="float32"):
        arr, sr = files.get(str(path), (np.zeros(16000, dtype=np.float32), 16000))
        return arr.astype(dtype or np.float32), sr

    sf.write = _write
    sf.read = _read
    sys.modules["soundfile"] = sf

    # ---- asr_mcp.db.manager (dict-backed, same contract) -------------------
    dbm = types.ModuleType("asr_mcp.db.manager")
    dbm.DEFAULT_USER = "default"

    class _Row(dict):
        pass

    class VoiceprintDB:
        def __init__(self, db_manager):
            self.rows: dict[str, dict] = {}

        def _copy(self, row):
            out = dict(row)
            out["embedding"] = np.asarray(row["embedding"], dtype=np.float32)
            return out

        def get(self, name, user_id="default"):
            row = self.rows.get(name)
            return self._copy(row) if row else None

        def list_all(self, user_id="default", include_pending=True):
            return {
                n: self._copy(r) for n, r in sorted(self.rows.items())
                if include_pending or not r.get("pending")
            }

        def save(self, name, embedding, user_id="default", pending=None,
                 purity=None, total_speech_sec=0.0, sample_count=0, **kw):
            row = self.rows.get(name)
            if row is None:
                row = {"name": name, "pending": False, "purity": {},
                       "total_speech_sec": 0.0, "sample_count": 0}
                self.rows[name] = row
            row["embedding"] = np.asarray(embedding, dtype=np.float32)
            row["total_speech_sec"] = total_speech_sec
            row["sample_count"] = sample_count
            # pending / purity = None PRESERVE -- the contract the SQL version
            # documents, and the one the learner depends on.
            if pending is not None:
                row["pending"] = bool(pending)
            if purity is not None:
                row["purity"] = dict(purity)
            return True

        def set_pending(self, name, pending, user_id="default"):
            if name not in self.rows:
                return False
            self.rows[name]["pending"] = bool(pending)
            return True

        def delete(self, name, user_id="default"):
            return self.rows.pop(name, None) is not None

        def rename(self, old, new, user_id="default"):
            if old not in self.rows or new in self.rows:
                return False
            self.rows[new] = self.rows.pop(old)
            self.rows[new]["name"] = new
            return True

        def count(self, user_id="default"):
            return len(self.rows)

        def search(self, embedding, user_id="default", top_k=5):
            return []

    class SnippetDB:
        def __init__(self, db_manager):
            self.rows: list[dict] = []

        def add(self, speaker_name, file_path, duration_sec, user_id="default",
                source_audio=None, start_sec=None, end_sec=None, **kw):
            row = {
                "id": len(self.rows) + 1, "speaker_name": speaker_name,
                "file_path": str(file_path), "duration_sec": float(duration_sec),
                "source_audio": source_audio, "start_sec": start_sec,
                "end_sec": end_sec,
            }
            self.rows.append(row)
            return row["id"]

        def list_by_speaker(self, speaker_name, user_id="default"):
            return [dict(r) for r in self.rows
                    if r["speaker_name"] == speaker_name]

        def all_speakers(self, user_id="default"):
            out: dict[str, dict] = {}
            for r in self.rows:
                entry = out.setdefault(r["speaker_name"],
                                       {"count": 0, "total_duration": 0.0})
                entry["count"] += 1
                entry["total_duration"] += r["duration_sec"]
            return out

        def count(self, speaker_name, user_id="default"):
            return len(self.list_by_speaker(speaker_name))

        def get(self, snippet_id, user_id="default"):
            for r in self.rows:
                if r["id"] == snippet_id:
                    return dict(r)
            return None

        def delete(self, snippet_id, user_id="default"):
            before = len(self.rows)
            self.rows = [r for r in self.rows if r["id"] != snippet_id]
            return len(self.rows) != before

        def find_duplicate(self, source_audio, start_sec, user_id="default"):
            for r in self.rows:
                if r["source_audio"] == source_audio and r["start_sec"] == start_sec:
                    return dict(r)
            return None

        def rename_speaker(self, old, new, user_id="default"):
            n = 0
            for r in self.rows:
                if r["speaker_name"] == old:
                    r["speaker_name"] = new
                    n += 1
            return n

        def update_file_path(self, snippet_id, file_path, user_id="default"):
            for r in self.rows:
                if r["id"] == snippet_id:
                    r["file_path"] = file_path
                    return True
            return False

    class DatabaseManager:
        def __init__(self, *a, **kw):
            pass

    class TranscriptDB:
        def __init__(self, *a, **kw):
            pass

        def rename_speaker(self, *a, **kw):
            return 0

    dbm.VoiceprintDB = VoiceprintDB
    dbm.SnippetDB = SnippetDB
    dbm.DatabaseManager = DatabaseManager
    dbm.TranscriptDB = TranscriptDB
    sys.modules["asr_mcp.db.manager"] = dbm

    # ---- asr_mcp.speaker.embedding (never called: every test stubs it) -----
    emb = types.ModuleType("asr_mcp.speaker.embedding")
    emb.extract_embedding = lambda *a, **kw: None
    emb.batch_embed_files = lambda *a, **kw: []
    emb.compute_pitch = lambda *a, **kw: (0.0, 0.0)
    emb.compute_energy = lambda *a, **kw: 0.0
    sys.modules["asr_mcp.speaker.embedding"] = emb

    for name in ("asr_mcp", "asr_mcp.db", "asr_mcp.speaker",
                 "asr_mcp.voiceprint"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(REPO_ROOT / name.replace(".", "/"))]
            sys.modules[name] = mod


def _service():
    _install_stubs()
    load_module("asr_mcp.speaker.uncertainty", "asr_mcp/speaker/uncertainty.py",
                package="asr_mcp.speaker")
    load_module("asr_mcp.voiceprint.utils", "asr_mcp/voiceprint/utils.py",
                package="asr_mcp.voiceprint")
    return load_module("asr_mcp.voiceprint.service",
                       "asr_mcp/voiceprint/service.py",
                       package="asr_mcp.voiceprint")


@pytest.fixture(scope="module", autouse=True)
def _isolate_stubs():
    """Undo every ``sys.modules`` entry this module injected.

    ``tests/test_speaker_learning.py`` needs the REAL torch and a real
    SQLAlchemy database; if a fake ``torch`` were left in ``sys.modules`` its
    ``importorskip("torch")`` would pass and every one of its DB tests would
    then run against the dict-backed stub.  Restoring the module table exactly
    is the only safe way to share one interpreter with a stubbed loader.
    """
    before = dict(sys.modules)
    yield
    for name in [n for n in sys.modules if n not in before]:
        del sys.modules[name]
    for name, mod in before.items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod


@pytest.fixture(scope="module")
def svc(_isolate_stubs):
    return _service()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def _voice(sign: float, sep: float, seed: int, jitter: float = 0.0) -> np.ndarray:
    """A unit vector at cosine DISTANCE ``sep`` from the shared axis."""
    s = float(np.sqrt(sep / 2.0))
    v = np.zeros(8, dtype=np.float32)
    v[0] = np.sqrt(max(0.0, 1.0 - s * s))
    v[2] = sign * s
    if jitter:
        r = np.random.default_rng(seed).normal(0, 1, 8).astype(np.float32)
        v = v + jitter * (r / np.linalg.norm(r))
    return (v / np.linalg.norm(v)).astype(np.float32)


def _segments(speaker, spans):
    return [{"speaker": speaker, "start": float(a), "end": float(b)}
            for a, b in spans]


@pytest.fixture
def make_service(svc, tmp_path):
    """A VoiceprintService over the dict-backed DB and a single counted decode."""
    utils = sys.modules["asr_mcp.voiceprint.utils"]

    # 300s of file: the span layouts below reach t=400s, and a slice past the
    # end of the audio is empty, which the snippet duration floor then rejects.
    def _make(embed, duration=300.0, register=None):
        class S:
            data_dir = tmp_path
            voices_dir = tmp_path / "voices"
        S.voices_dir.mkdir(exist_ok=True)

        n = int(duration * 16000)
        audio = np.linspace(0.0, 1.0, n, dtype=np.float32)

        sf = sys.modules["soundfile"]

        # Decodes of the SOURCE file.  Snippet reads go through the same
        # function but hit ".flac", which is what _auto_refine opens; only the
        # source decode is the one step 16 is about.
        decodes: list[str] = []

        def fake_load_audio(path, target_sr=16000):
            if not str(path).endswith(".flac"):
                decodes.append(str(path))
            # A path the learner has already WRITTEN is a snippet: return the
            # samples that were stored, so _auto_refine really does embed the
            # snippet it is about to weigh.
            try:
                stored, stored_sr = sf.read(str(path))
            except Exception:
                stored = None
            if stored is not None and str(path).endswith(".flac"):
                return _Tensor(stored)[None, :], stored_sr
            return _Tensor(audio)[None, :], 16000

        # Two lookups of load_audio: ``service`` for _auto_refine's snippets
        # and ``utils`` for the request-wide AudioSegmentSource.
        svc.load_audio = fake_load_audio
        utils.load_audio = fake_load_audio

        def fake_embed(waveform, sr, *a, **kw):
            # The waveform is a slice of one ramp, so its first sample says
            # where in the file this snippet came from.  The voice functions map
            # that offset back to a voice, which makes "one span, one voice"
            # explicit in the test rather than implicit in a stub.
            start = float(_as_array(waveform).reshape(-1)[0]) * duration
            return embed(start)

        svc.extract_embedding = fake_embed
        svc.batch_embed_files = lambda wf, *a, **kw: [fake_embed(w, 16000) for w in wf]
        svc.compute_pitch = lambda w, sr: (120.0, 5.0)
        svc.compute_energy = lambda w: 0.05

        vps = svc.VoiceprintService(S.data_dir, None)
        vps._db = svc.VoiceprintDB(None)
        vps._snippets = svc.SnippetDB(None)
        vps._db_manager = None
        vps.set_voices_dir(S.voices_dir)
        vps.set_embedding_session(object())
        # Every actual decode is recorded, so a test can assert the file was
        # decoded ONCE -- not merely that the cached loader was constructed.
        vps.decodes = decodes
        for name, vec in (register or {}).items():
            vps._db.save(name, vec, user_id="u", total_speech_sec=30.0)
        return vps

    return _make


def _one_voice(jitter=0.05):
    """Every span is the same person: one vector plus a per-span jitter.

    The jitter is deterministic in the span's offset, so the within-profile
    spread is reproducible instead of zero -- a profile whose own spans agree
    EXACTLY is not a case worth testing the purity statistic on.
    """
    def _embed(t):
        return _voice(1.0, 0.0, seed=int(t) + 1, jitter=jitter)
    return _embed


def _two_voices(split_at=100.0, sep=0.42, jitter=0.02):
    """Spans before ``split_at`` are one voice, the rest another."""
    def _embed(t):
        if t < split_at:
            return _voice(1.0, sep, seed=int(t) + 1, jitter=jitter)
        return _voice(-1.0, sep, seed=int(t) + 1, jitter=jitter)
    return _embed


def _sections_for(spans, pad=0.3, gap=0.0):
    """VAD sections that sit strictly inside each span (a pause before each)."""
    out = []
    for a, b in spans:
        out.append({"start": float(a) + pad, "end": float(b)})
    return out


# --------------------------------------------------------------------------
# step 11 -- snippet edges land on VAD section edges
# --------------------------------------------------------------------------

def test_snippet_edges_are_trimmed_onto_vad_sections(svc):
    trim = svc.trim_span_to_vad
    sections = [
        {"start": 3.0, "end": 6.0},
        {"start": 8.0, "end": 12.0},
    ]
    # The span starts 2.5s before the first speech and ends 1.0s after the last.
    assert trim(0.5, 13.0, sections) == (3.0, 12.0)
    # Exactly on the sections: unchanged.
    assert trim(3.0, 12.0, sections) == (3.0, 12.0)
    # A section straddling an edge is NOT followed -- padding outward would
    # pull the neighbouring speaker's speech in (turns abut by design).
    straddling = [{"start": 9.0, "end": 14.0}]
    assert trim(10.0, 20.0, straddling) == (10.0, 14.0)
    # No usable section -> span untouched.
    assert trim(0.0, 5.0, [{"start": 40.0, "end": 44.0}]) == (0.0, 5.0)
    assert trim(0.0, 5.0, None) == (0.0, 5.0)


def test_learned_snippets_record_the_trimmed_start_and_end(make_service):
    """start_sec / end_sec are asserted here for the first time in the repo."""
    s = make_service(_one_voice())
    spans = [(0.0, 20.0), (25.0, 45.0), (50.0, 70.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
        vad_sections=_sections_for(spans, pad=1.5),
    )
    assert len(out["pending_created"]) == 1
    snips = s.list_snippets(out["pending_created"][0]["name"], user_id="u")
    assert len(snips) == 3
    for sn, (a, b) in zip(sorted(snips, key=lambda x: x["start_sec"]), spans):
        assert sn["start_sec"] == pytest.approx(a + 1.5), sn
        assert sn["end_sec"] == pytest.approx(b), sn


def test_snippet_audio_is_exactly_the_recorded_span(make_service):
    """The FLAC must hold [start_sec, end_sec] and nothing else."""
    s = make_service(_one_voice())
    spans = [(0.0, 20.0), (25.0, 45.0), (50.0, 70.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
        vad_sections=_sections_for(spans, pad=1.5),
    )
    snips = sorted(s.list_snippets(out["pending_created"][0]["name"], user_id="u"),
                   key=lambda x: x["start_sec"])
    for sn in snips:
        stored = sys.modules["soundfile"].read(sn["file_path"])[0]
        assert abs(len(stored) - (sn["end_sec"] - sn["start_sec"]) * 16000) <= 2, (
            len(stored), sn["start_sec"], sn["end_sec"])


# --------------------------------------------------------------------------
# step 16 -- one decode per request
# --------------------------------------------------------------------------

def test_one_load_audio_per_request_not_one_per_segment(make_service):
    s = make_service(_one_voice(), duration=300.0)
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(12)]
    assert spans[-1][1] < 300.0
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    name = out["pending_created"][0]["name"]
    assert s._snippets.count(name, user_id="u") == 12
    # Every snippet was cut, from ONE decode: the learner, the cohesion pass and
    # the segment loader all share the request's AudioSegmentSource.
    assert len(s.decodes) == 1, s.decodes
    assert out["audio_loads"] == 1, out["audio_loads"]


def test_audio_segment_source_decodes_once_and_slices(svc):
    utils = sys.modules["asr_mcp.voiceprint.utils"]
    source = svc.AudioSegmentSource("meeting.wav")
    calls = {"n": 0}
    assert source.loads == 0

    def fake_load(path, target_sr=16000):
        calls["n"] += 1
        return _Tensor(np.arange(3 * 16000, dtype=np.float32))[None, :], 16000

    utils.load_audio = fake_load
    for a, b in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
        chunk, sr = source.segment(a, b)
        assert chunk.shape[-1] == 16000, chunk.shape
        assert sr == 16000
        # The slice really is the requested second of the file.
        assert chunk.numpy().reshape(-1)[0] == pytest.approx(a * 16000)
    assert calls["n"] == 1, calls
    assert source.loads == 1


# --------------------------------------------------------------------------
# step 13 -- label collisions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cluster,other,expect_extend", [
    ("Speaker 1", "Pending 2026-01-01 10:00 Speaker_10 zo249", False),
    ("Speaker 10", "Pending 2026-01-01 10:00 Speaker_1 zo249", False),
    ("Speaker 1", "Pending 2026-01-01 10:00 Speaker_1 zo249", True),
])
def test_speaker_1_is_not_a_substring_of_speaker_10(cluster, other, expect_extend):
    svc = _service()
    assert svc._tokens_contain(svc._name_tokens(other),
                               svc._name_tokens(cluster)) is expect_extend


def test_label_matching_is_token_exact_not_substring(svc):
    assert svc._name_tokens("Pending x Speaker_1 y") == [
        "pending", "x", "speaker", "1", "y"]
    assert svc._tokens_contain(svc._name_tokens("Speaker_10"),
                               svc._name_tokens("Speaker_1")) is False
    assert svc._tokens_contain(svc._name_tokens("Speaker_1"),
                               svc._name_tokens("Speaker_1")) is True
    assert svc._tokens_contain(svc._name_tokens("a Speaker_1 b"),
                               svc._name_tokens("Speaker_1")) is True


def test_a_second_run_of_the_same_file_does_not_overwrite_speaker_10(make_service):
    """Cluster 'Speaker 1' must not extend 'Speaker 10' in the same recording."""
    s = make_service(_one_voice())
    spans10 = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    spans1 = [(70.0, 88.0), (90.0, 108.0), (110.0, 128.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav",
        segments=_segments("Speaker 10", spans10) + _segments("Speaker 1", spans1),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    names = sorted(p["name"] for p in out["pending_created"])
    assert len(names) == 2, out["pending_created"]
    for name in names:
        assert s._snippets.count(name, user_id="u") == 3


def test_collision_suffix_checks_snippet_directories_not_just_the_db(make_service):
    """A profile with snippets but NO row must still get a '#2' name."""
    svc = _service()
    s = make_service(_one_voice())
    base = svc.pending_profile_name("meeting.wav", "Speaker 5")
    # Exactly the state of a profile learned while the embedding session was
    # unavailable: a snippet directory, no voiceprint row.
    (s.voices_dir / "u" / base).mkdir(parents=True)
    spans = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    created = [p["name"] for p in out["pending_created"]]
    assert created and all(name != base for name in created), created
    assert any(name.startswith(base + " #") for name in created), created


# --------------------------------------------------------------------------
# step 12 -- extend by VOICE, so a second recording is not a second profile
# --------------------------------------------------------------------------

def test_the_same_colleague_in_a_second_recording_extends_the_first_profile(
        make_service):
    s = make_service(_one_voice(jitter=0.03))
    spans_a = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    first = s.auto_collect_from_diarization(
        audio_path="first.wav", segments=_segments("Speaker 5", spans_a),
        user_id="u", source_id="first.wav", learn_new=True,
    )
    assert len(first["pending_created"]) == 1
    name = first["pending_created"][0]["name"]

    # A DIFFERENT recording, and a DIFFERENT cluster label, but the same voice.
    spans_b = [(10.0, 28.0), (30.0, 48.0), (50.0, 68.0)]
    second = s.auto_collect_from_diarization(
        audio_path="second.wav", segments=_segments("Speaker 2", spans_b),
        user_id="u", source_id="second.wav", learn_new=True,
    )
    assert second["pending_created"] == [], second
    assert [p["name"] for p in second["pending_extended"]] == [name]
    assert s._snippets.count(name, user_id="u") == 6
    assert len(s.pending_profiles(user_id="u")) == 1


def test_a_different_person_in_a_second_recording_mints_a_second_profile(
        make_service):
    s = make_service(_two_voices(split_at=100.0, sep=0.9))
    spans_a = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    s.auto_collect_from_diarization(
        audio_path="first.wav", segments=_segments("Speaker 5", spans_a),
        user_id="u", source_id="first.wav", learn_new=True,
    )
    # Same recording, so the cluster label keeps them apart.
    spans_b = [(110.0, 128.0), (130.0, 148.0), (150.0, 168.0)]
    out = s.auto_collect_from_diarization(
        audio_path="second.wav", segments=_segments("Speaker 2", spans_b),
        user_id="u", source_id="second.wav", learn_new=True,
    )
    # Different voice AND a different recording -> a new profile, not a merge.
    assert len(out["pending_created"]) == 1, out
    assert len(s.pending_profiles(user_id="u")) == 2


# --------------------------------------------------------------------------
# steps 14/15 -- caps, purity, fail closed
# --------------------------------------------------------------------------

def test_per_snippet_cap_splits_a_long_span(make_service, monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc, "_learning_cfg", lambda: {
        "split_clusters": True, "fail_closed": True,
        "max_intra_cluster_dist": 0.32, "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32, "extend_max_dist": 0.32,
        "max_snippet_sec": 10.0, "max_profile_total_sec": 0.0,
        "max_refine_weight_ratio": 4.0,
    })
    s = make_service(_one_voice())
    spans = [(0.0, 35.0), (40.0, 60.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    name = out["pending_created"][0]["name"]
    snips = sorted(s.list_snippets(name, user_id="u"), key=lambda x: x["start_sec"])
    # 35s at a 10s cap = 4 snippets, plus the 20s span = 2.
    assert len(snips) == 6, [(x["start_sec"], x["end_sec"]) for x in snips]
    assert max(x["duration_sec"] for x in snips) <= 10.0 + 1e-6


def test_per_profile_total_cap_stops_the_learn(make_service, monkeypatch):
    svc = _service()
    monkeypatch.setattr(svc, "_learning_cfg", lambda: {
        "split_clusters": True, "fail_closed": True,
        "max_intra_cluster_dist": 0.32, "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32, "extend_max_dist": 0.32,
        "max_snippet_sec": 0.0, "max_profile_total_sec": 30.0,
        "max_refine_weight_ratio": 4.0,
    })
    s = make_service(_one_voice())
    spans = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0), (60.0, 78.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    name = out["pending_created"][0]["name"]
    snips = s.list_snippets(name, user_id="u")
    total = sum(x["duration_sec"] for x in snips)
    assert total <= 30.0 + 1e-6, total
    assert len(snips) == 2


def _refine_weights(svc, durations):
    """The weighting _auto_refine applies, reproduced from its own formula."""
    arr = np.asarray(durations, dtype=np.float64)
    ratio = float(svc._learning_cfg()["max_refine_weight_ratio"])
    capped = np.minimum(arr, ratio * float(np.median(arr)))
    return capped / capped.sum()


def test_auto_refine_caps_the_length_weight_of_one_long_snippet(svc):
    """One 200s snippet must not carry 95% of a profile of 5s snippets."""
    uncapped = np.array([5.0, 5.0, 5.0, 5.0, 200.0])
    uncapped = uncapped / uncapped.sum()
    assert uncapped.max() > 0.9          # the bug being fixed
    assert _refine_weights(svc, [5.0, 5.0, 5.0, 5.0, 200.0]).max() <= 0.5 + 1e-9
    # And it is a NO-OP when no snippet is an outlier, so every existing
    # profile of comparable snippets keeps its exact embedding.
    uniform = np.array([6.0, 6.0, 6.0, 6.0])
    assert np.allclose(_refine_weights(svc, uniform), uniform / uniform.sum())


def test_one_long_snippet_cannot_dominate_a_learned_profile(make_service, svc,
                                                             monkeypatch):
    """End to end: the learned embedding is the CAPPED weighted mean.

    The per-snippet cap is switched off here so the ONLY thing limiting the
    60s snippet's influence is ``max_refine_weight_ratio`` -- otherwise the two
    mechanisms overlap and neither test says anything.
    """
    cfg = dict(svc._learning_cfg())
    cfg["max_snippet_sec"] = 0.0
    monkeypatch.setattr(svc, "_learning_cfg", lambda: cfg)
    embed = _one_voice(jitter=0.30)
    s = make_service(embed, duration=300.0)
    spans = [(float(i * 40), float(i * 40 + 5)) for i in range(4)] + [(200.0, 260.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    name = out["pending_created"][0]["name"]
    snips = sorted(s.list_snippets(name, user_id="u"), key=lambda x: x["start_sec"])
    assert len(snips) == 5, [(x["start_sec"], x["end_sec"]) for x in snips]
    assert [round(x["duration_sec"], 1) for x in snips] == [5.0, 5.0, 5.0, 5.0, 60.0]

    durs = np.array([x["duration_sec"] for x in snips])
    vecs = [np.asarray(embed(x["start_sec"]), dtype=np.float64) for x in snips]

    def blend(weights):
        out_ = sum(v * w for v, w in zip(vecs, weights))
        return out_ / np.linalg.norm(out_)

    capped = np.minimum(durs, 4.0 * float(np.median(durs)))
    expected = blend(capped / capped.sum())
    stored = np.asarray(s._db.get(name, user_id="u")["embedding"], dtype=np.float64)
    stored = stored / np.linalg.norm(stored)
    assert float(np.dot(expected, stored)) > 0.9999, (expected, stored)

    # The bug being fixed: uncapped, that 60s snippet carries 60/80 = 75% of the
    # weight and the "profile" is one recording.  The assertion above is what
    # bites -- drop the cap from _auto_refine and the stored vector becomes
    # blend(durs/durs.sum()) instead, which no longer matches ``expected``.
    raw = durs / durs.sum()
    assert raw.max() > 0.7


def test_purity_is_retained_and_exposed(make_service):
    s = make_service(_one_voice(jitter=0.05))
    spans = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    prof = out["pending_created"][0]
    assert prof["purity"]["intra_max"] is not None
    assert prof["purity"]["intra_max"] < 0.32
    assert prof["purity"]["groups"] == 1
    # Exposed on the pending card and on the speaker list.
    card = [p for p in s.pending_profiles(user_id="u")][0]
    assert card["purity"]["intra_max"] == prof["purity"]["intra_max"]
    row = s._db.get(prof["name"], user_id="u")
    assert row["purity"] == prof["purity"]
    listed = [x for x in s.list_speakers(user_id="u") if x["name"] == prof["name"]][0]
    assert listed["purity"] == prof["purity"]


def test_purity_survives_a_later_plain_refine(make_service):
    """A refine that did not learn must not erase the measurement."""
    s = make_service(_one_voice(jitter=0.05))
    spans = [(0.0, 18.0), (20.0, 38.0), (40.0, 58.0)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    name = out["pending_created"][0]["name"]
    before = s._db.get(name, user_id="u")["purity"]
    s._auto_refine(name, user_id="u")
    assert s._db.get(name, user_id="u")["purity"] == before


def test_a_blend_is_learned_as_two_profiles_each_with_its_own_purity(make_service):
    pytest.importorskip("sklearn")
    s = make_service(_two_voices(split_at=100.0, sep=0.42, jitter=0.02))
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)] + \
            [(200.0 + i * 20, 200.0 + i * 20 + 18) for i in range(3)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 6", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    created = out["pending_created"]
    assert len(created) == 2, created
    for prof in created:
        assert prof["snippets"] == 3
        p = prof["purity"]
        assert p["groups"] == 2
        assert p["intra_max"] is not None and p["intra_max"] < 0.32
        assert p["inter_min"] is not None and p["inter_min"] > 0.32
        assert p["purity_ratio"] < 1.0


@pytest.mark.parametrize("break_it,reason", [
    ("no_session", "unverified:no_embedding_session"),
    ("one_vector", "unverified:too_few_embeddings"),
    ("no_sklearn", "unverified:no_sklearn"),
    ("clustering_raises", "unverified:clustering_failed"),
])
def test_an_unverifiable_cluster_learns_nothing(make_service, monkeypatch,
                                                break_it, reason):
    """Fail closed: 'cannot tell' must never become 'learn a blend'."""
    pytest.importorskip("sklearn")
    svc = _service()
    embed = _one_voice()
    if break_it == "no_session":
        monkeypatch.setattr(svc.VoiceprintService, "_emb_session", lambda self: None)
    elif break_it == "one_vector":
        # Only the FIRST span embeds; the rest raise inside _embed_span.
        def embed(t):
            if t > 0.0:
                raise RuntimeError("embed blew up")
            return _voice(1.0, 0.0, seed=1, jitter=0.02)
    elif break_it == "no_sklearn":
        monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    elif break_it == "clustering_raises":
        import sklearn.cluster as sc
        monkeypatch.setattr(
            sc, "AgglomerativeClustering",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("nope")),
        )

    s = make_service(embed)
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)] + \
            [(200.0 + i * 20, 200.0 + i * 20 + 18) for i in range(3)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 6", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    assert out["pending_created"] == [], out
    assert out["pending_extended"] == []
    assert [m["reason"] for m in out["learn_skipped"]] == [reason]


def test_a_split_whose_groups_sit_closer_than_the_bar_is_not_learned(
        make_service, monkeypatch):
    """The groups must be FURTHER apart than the threshold that made them.

    Reachable with clean data only by chaining, so the bar is raised here to a
    value the two voices cannot clear -- which is exactly the condition the
    guard exists for: a division produced by the clustering, not by evidence.
    """
    pytest.importorskip("sklearn")
    svc = _service()
    cfg = dict(svc._learning_cfg())
    cfg["min_inter_dist"] = 0.9
    monkeypatch.setattr(svc, "_learning_cfg", lambda: cfg)
    s = make_service(_two_voices(split_at=100.0, sep=0.42, jitter=0.0))
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)] + \
            [(200.0 + i * 20, 200.0 + i * 20 + 18) for i in range(3)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 6", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    assert out["pending_created"] == [], out
    assert [m["reason"] for m in out["learn_skipped"]] == [
        "unverified:groups_too_close"]


def test_a_group_whose_own_spans_disagree_is_not_learned(make_service, svc):
    """max_profile_intra_dist: a group that is not one person is dropped.

    Asserted on the gate directly: the arithmetic that produces an impure
    group is the clustering's, and a group the clustering itself would have
    separated cannot be fed to it here without also changing the split.
    """
    s = make_service(_one_voice(jitter=0.05))
    spans = _segments("Speaker 5", [(float(i * 20), float(i * 20 + 18))
                                    for i in range(3)])
    out = s._learn_unknown_speaker(
        cluster="Speaker 5", segs=spans, audio_path="meeting.wav",
        user_id="u", source_id="meeting.wav",
        source=svc.AudioSegmentSource("meeting.wav"),
        vectors=[_unit(1), _unit(2), _unit(3)],
        intra_max=0.9, inter_min=0.4, groups=2,
    )
    assert out == {"skipped": "impure_group"}, out
    assert s.pending_profiles(user_id="u") == []
    assert s._snippets.rows == []


def test_fail_closed_can_be_rolled_back_to_fail_open(make_service, monkeypatch):
    """learning.fail_closed=false restores the old 'learn it anyway' behaviour."""
    svc = _service()
    monkeypatch.setattr(svc.VoiceprintService, "_emb_session", lambda self: None)
    monkeypatch.setattr(svc, "_learning_cfg", lambda: {
        "split_clusters": True, "fail_closed": False,
        "max_intra_cluster_dist": 0.32, "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32, "extend_max_dist": 0.32,
        "max_snippet_sec": 0.0, "max_profile_total_sec": 0.0,
        "max_refine_weight_ratio": 4.0,
    })
    s = make_service(_one_voice())
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 5", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    assert len(out["pending_created"]) == 1
    assert out["learn_skipped"] == []


def test_split_clusters_off_still_learns_one_profile(make_service, monkeypatch):
    pytest.importorskip("sklearn")
    svc = _service()
    monkeypatch.setattr(svc, "_learning_cfg", lambda: {
        "split_clusters": False, "fail_closed": True,
        "max_intra_cluster_dist": 0.32, "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32, "extend_max_dist": 0.32,
        "max_snippet_sec": 0.0, "max_profile_total_sec": 0.0,
        "max_refine_weight_ratio": 4.0,
    })
    s = make_service(_two_voices(split_at=100.0, sep=0.42, jitter=0.02))
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)] + \
            [(200.0 + i * 20, 200.0 + i * 20 + 18) for i in range(3)]
    out = s.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=_segments("Speaker 6", spans),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    assert len(out["pending_created"]) == 1
    assert out["pending_created"][0]["snippets"] == 6
    assert out["pending_created"][0]["cluster"] == "Speaker 6"


# --------------------------------------------------------------------------
# learn_new=False must be untouched
# --------------------------------------------------------------------------

def test_learn_new_false_is_byte_identical(make_service):
    s2 = make_service(_one_voice(), register={"Gergely Papp": _unit(3)})
    spans = [(float(i * 20), float(i * 20 + 18)) for i in range(3)]
    segs = (_segments("Gergely Papp", spans)
            + _segments("Speaker 5", [(100.0 + i * 20, 118.0 + i * 20)
                                      for i in range(3)]))
    out = s2.auto_collect_from_diarization(
        audio_path="meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav",
    )
    assert out["pending_created"] == []
    assert out["pending_extended"] == []
    assert out["learn_skipped"] == []
    # Collect-only: the registered profile grew, the generic cluster did not.
    assert out["collected"], out
    assert sorted({c["speaker_name"] for c in out["collected"]}) == ["Gergely Papp"]
    assert "Speaker 5" not in s2.list_speakers(user_id="u")


# --------------------------------------------------------------------------
# steps 10/17 -- the router learns from the TURNS, after they exist
# --------------------------------------------------------------------------

def _router_source() -> str:
    return (REPO_ROOT / "asr_mcp" / "api" / "asr_router.py").read_text()


def _body(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    return src[start:src.index(end_marker, start)]


def test_run_transcribe_prepares_turns_before_it_learns():
    """A source-order guard, not a behavioural one.

    ``run_transcribe``'s body is a closure inside the endpoint, so it cannot be
    called directly without a whole upload. What must not regress is the
    ORDER and WHAT IS PASSED: the learner has to run after ``_prepare_turns``
    and has to receive the turns -- the boundary set that
    ``display_segments`` is built from and ``attribute_items`` runs on -- rather
    than the raw diarization segments it used to cut on.
    """
    src = _router_source()
    body = _body(src, "    async def run_transcribe():",
                 "    async def event_stream():")
    i_turns = body.index("_prepare_turns(")
    i_learn = body.index("await _auto_collect(")
    assert i_turns < i_learn, "auto-collect ran before the turns existed"
    assert "segments=learn_spans" in body
    assert "learn_spans = turns or list(segments)" in body
    # A word-boundary match, not a substring: ``raw_vad_sections=...`` contains
    # ``vad_sections=...`` and a plain `in` check cannot tell them apart.
    assert re.search(r"(?<![\w])vad_sections=raw_vad_sections", body)
    # And the learner is given the very spans the client is shown.
    assert "for t in turns" in body


def test_attribution_also_learns_from_the_turns():
    src = _router_source()
    body = _body(src, "async def _attribute_items_against_audio(",
                 "async def attribution_endpoint(")
    assert body.index("_prepare_turns(") < body.index("await _auto_collect(")
    assert "learn_spans = turns or list(segments)" in body
    assert re.search(r"(?<![\w])vad_sections=raw_vad_sections", body)


def test_the_pending_profile_event_still_carries_the_learned_names():
    """``asr-client/transcribe_client.py`` reads evt["pending_profiles"] on
    ``diarization_complete``; moving the learner must not move or drop it."""
    src = _router_source()
    assert '"stage": "diarization_complete"' in src
    assert '"pending_profiles": pending_profiles' in src
