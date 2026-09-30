"""Auto-learning an unknown speaker as a PENDING profile.

The problem this solves: ``auto_collect_from_diarization`` only ever ADDED
snippets to an already-registered voiceprint (it hard-skipped anything without
a row in the DB), so an unregistered colleague could never be learned -- a
chicken-and-egg. The fix creates a *pending* profile for an unidentified
``Speaker N`` cluster.

The whole safety argument rests on ONE invariant, and most of these tests exist
to pin it:

    a pending profile accumulates snippets but is EXCLUDED from matching

so that learning someone can never turn into a confident misattribution --
which is precisely what the uncertainty policy forbids.

These tests are pure: the DB layer is exercised against a real temporary
SQLite file (so the migration and the ``pending`` column are actually tested),
but embeddings, audio and the embedding session are stubbed out.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="DB layer needed")
pytest.importorskip("torch", reason="voiceprint service imports torch")

from asr_mcp.db.manager import DatabaseManager, VoiceprintDB  # noqa: E402


def _unit(seed: int = 0) -> np.ndarray:
    """A deterministic, L2-normalised embedding vector.

    The values themselves are irrelevant here -- these tests are about which
    rows the matching queries return, not about what a voice sounds like.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=8).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def db(tmp_path):
    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    yield mgr
    mgr.close()


# --------------------------------------------------------------------------
# The invariant: pending never matches
# --------------------------------------------------------------------------

def test_pending_profile_is_excluded_from_matching(db):
    vp_db = VoiceprintDB(db)
    vp_db.save("Gergely Papp", _unit(), user_id="u", total_speech_sec=30)
    vp_db.save("Pending 2026-01-01 10:00 (a)", _unit(), user_id="u",
               total_speech_sec=30, pending=True)

    assert set(vp_db.list_all(user_id="u")) == {
        "Gergely Papp", "Pending 2026-01-01 10:00 (a)"}
    matchable = vp_db.list_all(user_id="u", include_pending=False)
    assert set(matchable) == {"Gergely Papp"}
    # And the search path the identify endpoint uses must agree.
    assert all(n != "Pending 2026-01-01 10:00 (a)"
               for n, _, _ in vp_db.search(_unit(), user_id="u"))


def test_load_known_speakers_excludes_pending(db, monkeypatch):
    """The single funnel every matching path uses."""
    from asr_mcp.api import asr_router

    vp_db = VoiceprintDB(db)
    vp_db.save("Gergely Papp", _unit(), user_id="u", total_speech_sec=30)
    vp_db.save("Pending 2026-01-01 10:00 (a)", _unit(), user_id="u",
               total_speech_sec=30, pending=True)

    class S:
        db_path = db._db_path
    loaded = asr_router._load_known_speakers(S(), "u")
    assert set(loaded) == {"Gergely Papp"}


def test_confirming_a_pending_profile_makes_it_matchable(db, tmp_path):
    vp_db = VoiceprintDB(db)
    vp_db.save("Pending 2026-01-01 10:00 (a)", _unit(), user_id="u",
               total_speech_sec=30, pending=True)
    assert set(vp_db.list_all(user_id="u", include_pending=False)) == set()

    vp_db.rename("Pending 2026-01-01 10:00 (a)", "Ismael", user_id="u")
    vp_db.set_pending("Ismael", False, user_id="u")
    assert set(vp_db.list_all(user_id="u", include_pending=False)) == {"Ismael"}


def test_save_does_not_silently_promote_on_rebuild(db):
    """A refine/re-register must not clear ``pending``.

    ``VoiceprintDB.save`` defaults ``pending=False`` for convenience, so every
    rebuild has to pass the flag through or an unnamed speaker becomes
    matchable the first time their profile is recomputed.
    """
    vp_db = VoiceprintDB(db)
    vp_db.save("Pending x", _unit(), user_id="u", pending=True)
    vp_db.save("Pending x", _unit(), user_id="u")          # naive rebuild
    assert vp_db.get("Pending x", user_id="u")["pending"] is True
    vp_db.save("Pending x", _unit(), user_id="u", pending=True)   # correct
    assert vp_db.get("Pending x", user_id="u")["pending"] is True


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------

def test_pending_column_is_added_to_an_existing_database(tmp_path):
    """A DB created before the column existed must gain it, defaulting to 0."""
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    # The real pre-migration schema: every column except ``pending``.
    con.execute("CREATE TABLE voiceprints ("
                "user_id VARCHAR(255) NOT NULL, name VARCHAR(255) NOT NULL, "
                "embedding BLOB NOT NULL, mfcc BLOB, pitch_hz FLOAT, "
                "pitch_std FLOAT, energy_rms FLOAT, spectral_centroid FLOAT, "
                "spectral_rolloff FLOAT, total_speech_sec FLOAT, "
                "sample_count INTEGER, created_at DATETIME, "
                "updated_at DATETIME, PRIMARY KEY (user_id, name))")
    con.execute("INSERT INTO voiceprints (user_id, name, embedding) "
                "VALUES ('u', 'Old Speaker', X'00000000')")
    con.commit()
    con.close()

    mgr = DatabaseManager(str(path))          # runs init_db's migration
    try:
        row = VoiceprintDB(mgr).get("Old Speaker", user_id="u")
        assert row is not None
        assert row["pending"] is False, "pre-existing profiles must not become pending"
    finally:
        mgr.close()


# --------------------------------------------------------------------------
# The learning gate
# --------------------------------------------------------------------------

def test_learning_accepts_a_generic_label_but_collect_does_not():
    """The one deliberate difference between the two gates."""
    from asr_mcp.speaker import uncertainty as u

    generic = {"speaker": "Speaker 5", "start": 0.0, "end": 9.0}
    assert u.eligible_for_learning(generic) is True
    assert u.eligible_for_auto_collect(generic) is False


@pytest.mark.parametrize("segment", [
    {"speaker": None, "start": 0, "end": 9},
    {"speaker": "", "start": 0, "end": 9},
    {"speaker": "UNKNOWN", "start": 0, "end": 9},
    {"speaker": "OVERLAP", "start": 0, "end": 9},
    {"speaker": "Speaker 5", "uncertain": True, "start": 0, "end": 9},
    {"speaker": "Speaker 5", "speaker_confidence": 0.05, "start": 0, "end": 9},
])
def test_learning_rejects_everything_the_policy_distrusts(segment):
    from asr_mcp.speaker import uncertainty as u
    assert u.eligible_for_learning(segment) is False

# --------------------------------------------------------------------------
# The learning path end to end (audio + embedding stubbed)
# --------------------------------------------------------------------------

@pytest.fixture
def service(db, tmp_path, monkeypatch):
    """VoiceprintService with real SQLite/FLAC but no GPU and no real audio."""
    from asr_mcp.voiceprint import service as svc

    class S:
        data_dir = tmp_path
        voices_dir = tmp_path / "voices"
    S.voices_dir.mkdir()

    # 60s of deterministic audio; the slice is what matters, not the content.
    wav = tmp_path / "meeting.wav"
    t = np.arange(60 * 16000) / 16000
    import soundfile as sf
    sf.write(str(wav), (np.sin(2 * np.pi * 220 * t) * 0.2).astype(np.float32),
             16000, subtype="PCM_16")

    monkeypatch.setattr(svc, "load_audio_segment",
                        lambda p, a, b: (__import__("torch").from_numpy(
                            np.sin(2 * np.pi * 220 * np.arange(
                                int((b - a) * 16000)) / 16000
                            )[np.newaxis, :] * 0.2).float(), 16000))

    # The REAL _auto_refine runs, so the pending flag actually survives a
    # profile rebuild (a stubbed refine would hide exactly that bug). Only the
    # embedding itself is faked.
    # Signature: batch_embed_files(waveforms, sample_rates, durations,
    # embedding_session, block_sec) -> list[np.ndarray | None]
    def fake_batch_embed_files(waveforms, sample_rates, durations, *a, **kw):
        return [_unit(i) for i, _ in enumerate(waveforms)]

    monkeypatch.setattr(svc, "batch_embed_files", fake_batch_embed_files)

    def fake_load_audio(path):
        # 3s of "audio" so it clears MIN_SNIPPET_DURATION (1.5s).
        import torch
        n = 3 * 16000
        return torch.from_numpy(
            np.sin(2 * np.pi * 220 * np.arange(n) / 16000).astype(np.float32)
        )[None, :], 16000

    monkeypatch.setattr(svc, "load_audio", fake_load_audio)
    monkeypatch.setattr(svc, "extract_embedding",
                        lambda wf, sr, *a, **kw: _unit(len(wf.shape) * 7 + 1))
    monkeypatch.setattr(svc, "compute_pitch", lambda w, sr: (120.0, 5.0))
    monkeypatch.setattr(svc, "compute_energy", lambda w: 0.05)

    vps = svc.VoiceprintService(S.data_dir, db)
    vps.set_voices_dir(S.voices_dir)
    vps.set_embedding_session(object())
    return vps


def _segs(speaker, n=3, dur=6.0, start=0.0):
    return [{"speaker": speaker, "start": start + i * 20.0,
             "end": start + i * 20.0 + dur} for i in range(n)]


def test_an_unknown_speaker_becomes_a_pending_profile(service, db):
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs("Speaker 5"),
        user_id="u", source_id="meeting.wav", learn_new=True,
    )
    assert out["collected"] == [], "a generic cluster must never be collected"
    assert len(out["pending_created"]) == 1
    prof = out["pending_created"][0]
    assert prof["cluster"] == "Speaker 5"
    assert prof["name"].startswith("Pending ")
    assert prof["existing"] is False

    vp_db = VoiceprintDB(db)
    matchable = vp_db.list_all(user_id="u", include_pending=False)
    assert matchable == {}, "a learned speaker must not be matchable yet"
    # It must be VISIBLE as pending even though _auto_refine is stubbed out and
    # no voiceprint row was written -- otherwise the snippets are unreachable.
    listed = service.pending_profiles(user_id="u")
    assert [p["name"] for p in listed] == [prof["name"]]
    assert listed[0]["snippet_count"] == 3
    assert [s["name"] for s in service.list_speakers(user_id="u")
            if s["pending"]] == [prof["name"]]


def test_learning_reports_nothing_for_a_cough(service):
    """Under the duration/segment floor nothing is created."""
    segs = [{"speaker": "Speaker 3", "start": 0.0, "end": 4.0}]
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav", learn_new=True)
    assert out["pending_created"] == []
    assert out["pending_extended"] == []


def test_learning_is_off_unless_asked(service, db):
    """The default stays exactly as it was: collect-only."""
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs("Speaker 5"),
        user_id="u", source_id="meeting.wav")
    assert out["pending_created"] == []
    assert out["pending_extended"] == []
    assert VoiceprintDB(db).list_all(user_id="u") == {}


def test_the_same_recording_extends_one_pending_profile(service, db):
    segs = _segs("Speaker 5", n=3, start=0.0) + _segs("Speaker 6", n=3, start=90.0)
    first = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav", learn_new=True)
    assert len(first["pending_created"]) == 2
    # A second pass over the same recording must not mint two more profiles.
    second = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav", learn_new=True)
    assert second["pending_created"] == []
    assert len(second["pending_extended"]) == 2
    assert len(VoiceprintDB(db).list_all(user_id="u")) == 2


def test_extending_a_profile_does_not_duplicate_snippets(service, db):
    """Re-processing the same recording must be idempotent.

    Counting PROFILES is not enough: an earlier version passed the profile-count
    assertion while adding the same three snippet files a second time, so the
    profile filled with duplicates and the embedding was computed over
    doubly-counted audio.
    """
    segs = _segs("Speaker 5", n=3, start=0.0)
    service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav", learn_new=True)
    first = service.list_snippets(service.pending_profiles(user_id="u")[0]["name"],
                                  user_id="u")
    service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=segs, user_id="u",
        source_id="meeting.wav", learn_new=True)
    second = service.list_snippets(service.pending_profiles(user_id="u")[0]["name"],
                                   user_id="u")
    assert len(first) == 3
    assert len(second) == len(first), "the second pass duplicated snippets"
    assert {x["file_path"] for x in first} == {x["file_path"] for x in second}


def test_renaming_a_profile_repoints_snippet_paths_at_disk(service, db, tmp_path):
    """The directory moves, so the DB rows must move with it.

    Copying file_path verbatim is the obvious implementation and it is wrong:
    the row keeps pointing at a directory that no longer exists, and every later
    load of that snippet fails with "No such file or directory".
    """
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs("Speaker 5"),
        user_id="u", source_id="meeting.wav", learn_new=True)
    name = out["pending_created"][0]["name"]
    assert service.confirm_pending(name, "Ismael", user_id="u")["ok"] is True

    snips = service.list_snippets("Ismael", user_id="u")
    assert snips, "confirming a profile lost its snippets"
    for sn in snips:
        assert Path(sn["file_path"]).is_file(), (
            f"snippet row points at a path that does not exist: {sn['file_path']}")
        assert name not in sn["file_path"], (
            "the row still carries the old pending directory")
        assert sn["speaker_name"] == "Ismael"


def test_confirm_pending_renames_and_unlocks(service, db):
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs("Speaker 5"),
        user_id="u", source_id="meeting.wav", learn_new=True)
    name = out["pending_created"][0]["name"]
    assert service.confirm_pending(name, "  Ismael  ", user_id="u")["ok"] is True
    vp_db = VoiceprintDB(db)
    assert "Ismael" in vp_db.list_all(user_id="u", include_pending=False)
    assert name not in vp_db.list_all(user_id="u")
    assert service.pending_profiles(user_id="u") == []


def test_confirm_pending_refuses_to_overwrite_someone(service, db):
    vp_db = VoiceprintDB(db)
    vp_db.save("Gergely Papp", _unit(1), user_id="u", total_speech_sec=30)
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs("Speaker 5"),
        user_id="u", source_id="meeting.wav", learn_new=True)
    name = out["pending_created"][0]["name"]
    res = service.confirm_pending(name, "Gergely Papp", user_id="u")
    assert "already exists" in res["error"]
    assert vp_db.get(name, user_id="u")["pending"] is True


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_confirm_pending_requires_a_name(service, db, bad):
    assert "name is required" in service.confirm_pending("Pending x", bad, user_id="u")["error"]


# --------------------------------------------------------------------------
# "Is this learner someone I already have?" -- candidates + merge
# --------------------------------------------------------------------------
#
# confirm_pending REFUSES a name that is already taken, which is safe but is the
# wrong answer when the learner is a person who is already registered: renaming
# gives one person two profiles and the uncertainty policy then splits their
# speech between two names forever. The correct resolution is a merge, and
# until now there was no way to reach one from the UI.

def _learn(service, cluster="Speaker 5", source="meeting.wav", n=3, user="u"):
    out = service.auto_collect_from_diarization(
        audio_path="/tmp/meeting.wav", segments=_segs(cluster, n=n),
        user_id=user, source_id=source, learn_new=True,
    )
    return (out["pending_created"] + out["pending_extended"])[0]["name"]


def test_candidates_rank_the_nearest_registered_speaker_first(service, db):
    from asr_mcp.db.manager import VoiceprintDB
    vp_db = VoiceprintDB(db)
    ref_a, ref_b = _unit(11), _unit(22)
    vp_db.save("Alice", ref_a, user_id="u", total_speech_sec=30)
    vp_db.save("Bob", ref_b, user_id="u", total_speech_sec=30)

    pending = _learn(service)
    # Point the learned profile squarely at Alice.
    vp_db.save(pending, ref_a, user_id="u", total_speech_sec=20, pending=True)

    cands = service.pending_candidates(pending, user_id="u")
    assert [c["name"] for c in cands][:2] == ["Alice", "Bob"]
    assert cands[0]["distance"] < cands[1]["distance"]
    assert cands[0]["likely"] is True
    # Never a pending profile as a candidate: an unnamed profile has no
    # identity to lend.
    assert all(not c["name"].startswith("Pending ") for c in cands)


def test_candidates_are_empty_without_an_embedding_or_registered_speakers(service, db):
    from asr_mcp.db.manager import VoiceprintDB
    pending = _learn(service)
    vp_db = VoiceprintDB(db)
    # No registered speaker at all -> nothing to compare against.
    assert service.pending_candidates(pending, user_id="u") == []
    # Now a registered speaker, but the pending profile has no embedding
    # (learned while the embedding session was unavailable). get() hands back a
    # copy, so the row has to be blanked with SQL.
    vp_db.save("Alice", _unit(11), user_id="u", total_speech_sec=30)
    import sqlalchemy as sa
    with db.get_session() as s:
        s.execute(sa.text("UPDATE voiceprints SET embedding = x'' WHERE name = :n"),
                  {"n": pending})
        s.commit()
    assert service.pending_candidates(pending, user_id="u") == []


def test_merging_a_pending_profile_into_an_existing_speaker(service, db):
    from asr_mcp.db.manager import VoiceprintDB
    vp_db = VoiceprintDB(db)
    vp_db.save("Ismael", _unit(11), user_id="u", total_speech_sec=30)
    pending = _learn(service, n=3)
    snips_before = len(service.list_snippets("Ismael", user_id="u"))

    out = service.merge_pending_into(pending, "Ismael", user_id="u")
    assert out.get("error") is None, out
    assert out["status"] == "merged_into_existing"
    assert out["snippets_moved"] == 3

    # The pending profile is gone from every view...
    assert pending not in vp_db.list_all(user_id="u", include_pending=True)
    assert all(p["name"] != pending for p in service.pending_profiles(user_id="u"))
    # ...and its snippets now belong to the real speaker, on disk.
    moved = service.list_snippets("Ismael", user_id="u")
    assert len(moved) == snips_before + 3
    assert all(Path(s["file_path"]).exists() for s in moved)
    # Ismael is still one ordinary, matchable profile.
    assert "Ismael" in vp_db.list_all(user_id="u", include_pending=False)


def test_merge_refuses_the_awkward_cases(service, db):
    from asr_mcp.db.manager import VoiceprintDB
    vp_db = VoiceprintDB(db)
    vp_db.save("Ismael", _unit(11), user_id="u", total_speech_sec=30)
    pending = _learn(service)

    assert "into itself" in service.merge_pending_into(pending, pending, user_id="u")["error"]
    assert "No speaker named" in service.merge_pending_into(pending, "Nobody", user_id="u")["error"]
    assert "No pending profile" in service.merge_pending_into("Pending nope", "Ismael",
                                                              user_id="u")["error"]
    # A NAMED profile is not a pending profile: this must not be able to move
    # a colleague's audio around. Needs a distinct target or the into-itself
    # guard fires first.
    vp_db.save("Bob", _unit(22), user_id="u", total_speech_sec=30)
    assert "not a pending profile" in service.merge_pending_into("Ismael", "Bob",
                                                                 user_id="u")["error"]
    # Still pending after all those refusals, and still intact.
    assert any(p["name"] == pending for p in service.pending_profiles(user_id="u"))
    assert len(service.list_snippets(pending, user_id="u")) == 3
