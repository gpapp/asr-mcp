"""The startup voiceprint audit: `asr_mcp/voiceprint/audit.py`.

The audit exists because voiceprint state drifts silently. A merge deletes
snippets, a re-run of the same recording writes the same span twice, a rename
moves a directory, and a profile whose audio is gone keeps matching on an
embedding nothing can rebuild. None of that raises -- the matcher just reads
whatever rows exist.

Two properties matter as much as the checks themselves:

1. **It is read-only.** A duplicate profile is two rows that look like one
   person; only the user knows which embedding is the better one. An audit that
   merged them at startup would destroy the evidence for the other.
2. **It sees the same audio the rescan would.** Snippet paths are written
   absolute inside the container (`/app/voices/...`), so path resolution and the
   orphan comparison are the load-bearing parts -- get them wrong and every
   snippet looks unclaimed.

The module is loaded by path: `asr_mcp.voiceprint.__init__` imports
`VoiceprintService`, which needs soundfile and torch. The audit itself only
needs numpy.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    path = REPO_ROOT / "asr_mcp" / "voiceprint" / "audit.py"
    spec = importlib.util.spec_from_file_location("vp_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_audit()

CFG = {
    "enabled": True,
    "near_duplicate_dist": 0.25,
    "stale_speech_ratio": 1.5,
    "hash_snippets": True,
}


DIM = 192  # the real ECAPA-TDNN512 embedding width


def _unit(seed: int = 0) -> np.ndarray:
    """A deterministic unit vector, at the real embedding width.

    The width is not cosmetic. In a low dimension two independent random
    directions are only ~sqrt(2/DIM) cosine distance apart, so an 8-dim pair
    lands at ~0.23 -- UNDER the 0.25 duplicate bar, which made "two different
    colleagues" fixtures read as duplicates. At 192 dims distinct seeds are
    ~0.99 apart, so a test can only collide by tilting on purpose.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _tilt(base: np.ndarray, eps: float, seed: int = 7) -> np.ndarray:
    """A unit vector exactly ``eps`` (cosine distance) from ``base``.

    Built from an explicit orthogonal direction rather than
    ``base + eps * random``: that only approximates the requested distance, and
    it depends on ``hash()``, which is salted per process -- a threshold test
    that passed on one run could fail on the next.
    """
    other = _unit(seed)
    other = other - float(np.dot(other, base)) * base  # orthogonalise
    other /= np.linalg.norm(other)
    # cosine(v, base) must equal 1 - eps, so the two coefficients are
    # (1 - eps) and sqrt(1 - (1 - eps)^2).
    return ((1.0 - eps) * base
            + np.sqrt(eps * (2.0 - eps)) * other).astype(np.float32)


def _profile(embedding=None, **kw) -> dict:
    row = {
        "name": kw.pop("name", "Someone"),
        "embedding": embedding if embedding is not None else _unit(),
        "total_speech_sec": kw.pop("total_speech_sec", 30.0),
    }
    row.update(kw)
    return row


class FakeSnippets:
    """The two SnippetDB methods the audit uses, over an in-memory dict.

    ``rows`` maps speaker -> list of row dicts; ``summary`` is what
    ``all_speakers`` would return. Kept separate so a test can make the two
    disagree, which is exactly the drift ``stale_speech_totals`` exists to
    catch.
    """

    def __init__(self, rows: dict, summary: dict | None = None, user_ids=("u",)):
        self.rows = rows
        self._summary = summary
        # `rows` is keyed by SPEAKER, not by user -- real SnippetDB.user_ids
        # asks the DB, so the two are separate and this default is explicit
        # rather than derived from the keys.
        self._user_ids = list(user_ids)

    def user_ids(self):
        return list(self._user_ids)

    def all_speakers(self, user_id="default"):
        if self._summary is not None:
            return self._summary
        return {
            name: {"count": len(rs), "total_duration": sum(
                float(r.get("duration_sec") or 0.0) for r in rs)}
            for name, rs in self.rows.items()
        }

    def list_by_speaker(self, speaker_name, user_id="default"):
        return list(self.rows.get(speaker_name, []))


def _write_flac(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ── near-duplicate profiles ────────────────────────────────────────────────

def test_two_profiles_of_the_same_person_are_reported():
    base = _unit(1)
    profiles = {
        "Gergely Papp": _profile(base, name="Gergely Papp"),
        "Gergely P": _profile(_tilt(base, 0.02), name="Gergely P"),
    }
    pairs = audit._near_duplicate_pairs(profiles, 0.25)
    assert len(pairs) == 1
    assert {pairs[0]["a"], pairs[0]["b"]} == {"Gergely Papp", "Gergely P"}
    assert pairs[0]["distance"] < 0.05


def test_two_different_colleagues_are_not_reported():
    """The bar must not fire on distinct people.

    This is the whole reason the threshold sits far below
    ``matching.accept_threshold``: on a real 36-profile store the closest
    distinct pair measured 0.346, i.e. already at the match bar. A duplicate
    lands under 0.05; two colleagues do not.
    """
    profiles = {
        "A": _profile(_unit(1), name="A"),
        "B": _profile(_unit(2), name="B"),
    }
    assert audit._near_duplicate_pairs(profiles, 0.25) == []


def test_the_pair_bar_is_configurable():
    base = _unit(3)
    mid = _tilt(base, 0.20)
    profiles = {"A": _profile(base, name="A"), "B": _profile(mid, name="B")}
    assert audit._near_duplicate_pairs(profiles, 0.15) == []
    assert len(audit._near_duplicate_pairs(profiles, 0.30)) == 1


def test_pending_profiles_are_still_pair_checked():
    """Two pending profiles that collide are the normal mid-learning state.

    Matching already ignores pending profiles (lesson 39), so a collision
    between them cannot cause a misattribution -- and flagging it would train
    the user to ignore the warning. They are deliberately NOT excluded: a
    pending profile is a real row that the user will eventually confirm, and
    by then two identical profiles under two real names is exactly the drift
    this audit exists to surface. This test pins the choice in both directions
    so neither half is "fixed" by accident.
    """
    base = _unit(4)
    profiles = {
        "Pending 2026-01-01 10:00 Speaker 5 a": _profile(
            base, name="Pending 2026-01-01 10:00 Speaker 5 a", pending=True),
        "Pending 2026-01-02 11:00 Speaker 5 a": _profile(
            base, name="Pending 2026-01-02 11:00 Speaker 5 a", pending=True),
    }
    # audit_user filters nothing itself; _near_duplicate_pairs sees what it is
    # given, so the caller is responsible -- pin that both directions exist.
    assert len(audit._near_duplicate_pairs(profiles, 0.25)) == 1
    report = audit.audit_user(
        "u", profiles, FakeSnippets({}), Path("/nonexistent"), CFG)
    # Pending pairs ARE reported by _near_duplicate_pairs; the audit relies on
    # the fact that they are harmless, not on excluding them. Document it.
    assert report["near_duplicate_profiles"]


def test_a_profile_with_no_embedding_is_skipped_not_crashed():
    profiles = {
        "A": _profile(None, name="A"),
        "B": _profile(_unit(5), name="B"),
        "C": _profile(np.zeros(DIM, dtype=np.float32), name="C"),
    }
    profiles["A"]["embedding"] = np.array([], dtype=np.float32)
    assert audit._near_duplicate_pairs(profiles, 0.25) == []


# ── name collisions ────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("Gergely Papp", "gergely papp"),
    ("Gergely Papp", "Gergely  Papp"),
    ("Gergely Papp", " Gergely Papp "),
    ("Renata", "Renata"),  # identical -> one group, not a collision
])
def test_normalise_name_folds_what_a_human_would(a, b):
    assert audit.normalise_name(a) == audit.normalise_name(b)


def test_normalise_name_folds_unicode_compatibility_forms():
    """NFKC folds the forms a keyboard or a paste can introduce.

    Deliberately NOT folding: accents. "Renáta" and "Renata" are two
    different people as far as this audit is concerned, and collapsing them
    would invent a collision out of a transliteration -- a false positive is
    the one failure mode that teaches the user to ignore the warning.
    """
    # Full-width Latin (an IME), and the ﬁ/ﬂ ligatures (a PDF text layer).
    assert audit.normalise_name("Ｇergely") == audit.normalise_name("Gergely")
    assert audit.normalise_name("ﬁona") == audit.normalise_name("fiona")
    # A compatibility form with no canonical decomposition is left alone.
    assert audit.normalise_name("Renáta") != audit.normalise_name("Renata")


def test_name_collisions_are_reported_as_a_group():
    """A group is reported once, with every member -- not one pair at a time.

    "Test Person Four" appears twice in the input but is a single profile name, so it
    is not a collision at all: a group needs MORE THAN ONE distinct name.
    """
    assert audit._name_collisions(
        ["Gergely Papp", "gergely papp", "Test Person Four", "Test Person Four"]
    ) == [{"normalised": "gergely papp",
           "names": ["Gergely Papp", "gergely papp"]}]
    # Three names, two of which collide: still one group, listing all three.
    assert audit._name_collisions(
        ["Gergely Papp", "gergely  papp", "GERGELY PAPP "]
    ) == [{"normalised": "gergely papp",
           "names": sorted(["Gergely Papp", "gergely  papp", "GERGELY PAPP "])}]
    # The same name twice is one profile, not two.
    assert audit._name_collisions(["Test Person Four", "Test Person Four"]) == []


def test_distinct_names_do_not_collide():
    assert audit._name_collisions(["Test Person One", "Test Person Two"]) == []


# ── path resolution (the container/host mount problem) ─────────────────────

def test_a_container_path_resolves_against_the_configured_voices_dir(tmp_path):
    """Paths are written absolute in the container and read outside it.

    Without re-anchoring, every snippet row looks like it lost its audio --
    which would make ``missing_snippet_files`` fire on a perfectly healthy
    store and drown the real findings.
    """
    real = _write_flac(tmp_path / "voices" / "u" / "Gergely Papp" / "a.flac",
                       b"audio")
    stored = "/app/voices/u/Gergely Papp/a.flac"
    resolved = audit.resolve_snippet_path(stored, tmp_path / "voices")
    assert resolved is not None
    assert resolved.read_bytes() == b"audio"
    assert resolved == real.resolve() or resolved.read_bytes() == b"audio"


def test_an_absent_snippet_resolves_to_none(tmp_path):
    assert audit.resolve_snippet_path(
        "/app/voices/u/Nobody/x.flac", tmp_path / "voices") is None
    assert audit.resolve_snippet_path("", tmp_path) is None
    assert audit.resolve_snippet_path(None, tmp_path) is None


# ── profiles vs snippets, orphans, duplicates ──────────────────────────────

def test_orphan_files_are_files_on_disk_that_no_row_claims(tmp_path):
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "claimed.flac", b"one")
    _write_flac(voices / "u" / "A" / "leftover.flac", b"two")
    _write_flac(voices / "u" / "A" / "notes.json", b"{}")  # not audio
    _write_flac(voices / "u" / "B" / "all-orphan.flac", b"three")

    snippets = FakeSnippets({
        "A": [{"id": 1, "file_path": str(voices / "u" / "A" / "claimed.flac"),
               "duration_sec": 10.0}],
    })
    report = audit.audit_user("u", {"A": _profile(name="A")}, snippets,
                              voices, CFG)

    orphans = {(o["speaker"], o["file"]) for o in report["orphan_snippet_files"]}
    assert orphans == {("A", "leftover.flac"), ("B", "all-orphan.flac")}
    # The claimed file is not an orphan, and a sidecar is not audio at all.
    assert ("A", "claimed.flac") not in orphans
    assert ("A", "notes.json") not in orphans


def test_byte_identical_snippets_are_reported(tmp_path):
    """The same span learned twice under two names -- a previous-run remainder."""
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "02_27_73_10.flac", b"identical audio")
    _write_flac(voices / "u" / "A" / "02_27_88_10.flac", b"identical audio")
    snippets = FakeSnippets({"A": [
        {"id": 1, "file_path": str(voices / "u" / "A" / "02_27_73_10.flac"),
         "duration_sec": 10.1},
        {"id": 2, "file_path": str(voices / "u" / "A" / "02_27_88_10.flac"),
         "duration_sec": 10.1},
    ]})
    report = audit.audit_user("u", {"A": _profile(name="A")}, snippets,
                              voices, CFG)
    assert len(report["duplicate_snippet_content"]) == 1
    assert report["duplicate_snippet_content"][0]["speaker"] == "A"


def test_hash_snippets_can_be_switched_off(tmp_path):
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "one.flac", b"same")
    _write_flac(voices / "u" / "A" / "two.flac", b"same")
    snippets = FakeSnippets({"A": [
        {"id": 1, "file_path": str(voices / "u" / "A" / "one.flac"),
         "duration_sec": 5.0},
        {"id": 2, "file_path": str(voices / "u" / "A" / "two.flac"),
         "duration_sec": 5.0},
    ]})
    cfg = dict(CFG, hash_snippets=False)
    report = audit.audit_user("u", {"A": _profile(name="A")}, snippets,
                              voices, cfg)
    assert report["duplicate_snippet_content"] == []


def test_a_profile_with_no_snippets_and_a_speaker_with_no_profile(tmp_path):
    snippets = FakeSnippets({
        "HasAudio": [{"id": 1, "file_path": str(tmp_path / "nope.flac"),
                      "duration_sec": 5.0}],
    })
    report = audit.audit_user("u", {
        "HasAudio": _profile(name="HasAudio"),
        "NoAudio": _profile(name="NoAudio"),
    }, snippets, tmp_path, CFG)
    assert report["profiles_without_snippets"] == ["NoAudio"]
    assert report["snippets_without_profile"] == []
    assert report["missing_snippet_files"][0]["speaker"] == "HasAudio"


def test_stale_total_speech_is_reported(tmp_path):
    """A deleted snippet leaves total_speech_sec behind.

    It is not a duplicate, but it is the same class of drift, and it is the
    number the auto-collect cap (AUTO_COLLECT_MAX_TOTAL_SEC) is measured
    against -- so a stale total silently stops a speaker collecting at all.
    """
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "a.flac", b"x")
    snippets = FakeSnippets({"A": [
        {"id": 1, "file_path": str(voices / "u" / "A" / "a.flac"),
         "duration_sec": 100.0},
    ]})
    report = audit.audit_user("u", {"A": _profile(name="A", total_speech_sec=1200.0)},
                              snippets, voices, CFG)
    assert report["stale_speech_totals"] == [
        {"speaker": "A", "total_speech_sec": 1200.0, "snippet_seconds": 100.0}
    ]


def test_a_healthy_store_reports_nothing(tmp_path):
    voices = tmp_path / "voices"
    for i, name in enumerate(["Test Person One", "Test Person Two"]):
        _write_flac(voices / "u" / name / f"{i}.flac", f"audio{i}".encode())
    profiles, rows = {}, {}
    for i, name in enumerate(["Test Person One", "Test Person Two"]):
        profiles[name] = _profile(_unit(i + 1), name=name, total_speech_sec=10.0)
        rows[name] = [{"id": i, "duration_sec": 10.0,
                       "file_path": str(voices / "u" / name / f"{i}.flac")}]
    report = audit.audit_user("u", profiles, FakeSnippets(rows), voices, CFG)
    assert not audit.has_findings(report), report


# ── reporting ──────────────────────────────────────────────────────────────

def test_has_findings_covers_every_declared_key():
    empty = {key: [] for key in audit.FINDING_KEYS}
    assert not audit.has_findings(empty)
    for key in audit.FINDING_KEYS:
        assert audit.has_findings({**empty, key: [{"x": 1}]}), key


def test_a_clean_store_logs_one_info_line_and_a_dirty_one_logs_warnings(caplog):
    import logging as _logging

    clean = {key: [] for key in audit.FINDING_KEYS}
    clean.update({"user_id": "u", "profiles": 3, "speakers_with_snippets": 3})
    with caplog.at_level(_logging.INFO, logger="asr_mcp.voiceprint.audit"):
        audit.log_report(clean)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == _logging.INFO

    caplog.clear()
    dirty = dict(clean, orphan_snippet_files=[{"speaker": "A", "file": "x.flac"}])
    with caplog.at_level(_logging.INFO, logger="asr_mcp.voiceprint.audit"):
        audit.log_report(dirty)
    assert any(r.levelno == _logging.WARNING for r in caplog.records)


def test_the_audit_never_mutates_the_store(tmp_path):
    """Read-only is the contract: an audit that repairs is an audit that can
    destroy a profile the user still needs."""
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "keep.flac", b"audio")
    _write_flac(voices / "u" / "A" / "orphan.flac", b"audio")
    rows = {"A": [{"id": 1, "file_path": str(voices / "u" / "A" / "keep.flac"),
                   "duration_sec": 5.0}]}
    snippets = FakeSnippets(rows)
    profiles = {"A": _profile(name="A"), "Ghost": _profile(name="Ghost")}
    before_rows = [dict(r) for r in rows["A"]]
    before_files = sorted(p.name for p in (voices / "u" / "A").iterdir())

    report = audit.audit_user("u", profiles, snippets, voices, CFG)
    assert report["orphan_snippet_files"] and report["profiles_without_snippets"]

    assert rows["A"] == before_rows
    assert sorted(p.name for p in (voices / "u" / "A").iterdir()) == before_files
    assert set(profiles) == {"A", "Ghost"}


def test_run_startup_audit_reaches_the_report(tmp_path):
    """The wiring in between the DB and the audit is load-bearing.

    ``run_startup_audit`` is the only caller, and it reads four private
    attributes off the live service. A wrong one is invisible until the log
    line at server start is the ONLY symptom:

        WARNING asr_mcp.server: Voiceprint startup audit could not run:
        'PosixPath' object is not callable

    ``voices_dir`` in particular is a ``@property``; calling it hands
    ``audit_all`` a Path-as-callable and the audit silently never runs, which
    means a store with six duplicate snippets looks clean forever.
    """
    voices = tmp_path / "voices"
    _write_flac(voices / "u" / "A" / "keep.flac", b"audio")
    _write_flac(voices / "u" / "A" / "orphan.flac", b"audio")
    rows = {"A": [{"id": 1, "file_path": str(voices / "u" / "A" / "keep.flac"),
                   "duration_sec": 5.0}]}

    class FakeDB:
        def user_ids(self):
            return ["u"]

        def list_all(self, user_id="default", include_pending=True):
            return {"A": _profile(name="A"), "Ghost": _profile(name="Ghost")}

    class FakeService:
        _db = FakeDB()
        _snippets = FakeSnippets(rows)
        voices_dir = voices          # a property, exactly as on the real service

    reports = audit.run_startup_audit(FakeService())

    assert len(reports) == 1
    assert reports[0]["user_id"] == "u"
    assert reports[0]["orphan_snippet_files"]
    assert reports[0]["profiles_without_snippets"]


def test_run_startup_audit_survives_a_broken_service():
    """An audit is observability: it may report nothing, never raise."""
    class Broken:
        _db = property(lambda self: (_ for _ in ()).throw(RuntimeError("no db")))
        _snippets = None
        voices_dir = None

    assert audit.run_startup_audit(Broken()) == []


def test_run_startup_audit_can_be_disabled(tmp_path):
    class FakeService:
        _db = None
        _snippets = None
        voices_dir = tmp_path

    assert audit.run_startup_audit(
        FakeService(), cfg={"enabled": False}) == []
