"""Startup consistency audit of the voiceprint store.

Voiceprints accumulate state that nothing reconciles on its own: a merge
deletes a profile's snippets, a rename moves a directory, a re-run of the same
recording writes a second snippet, and a profile whose snippets all vanished
keeps its embedding forever. None of that raises -- the matcher just reads
whatever rows exist, so the damage surfaces as a speaker who wins votes they
should not, or as one person holding two profiles and never being recognised.

This module finds those conditions and REPORTS them. It is deliberately
read-only: nothing here deletes a profile or a snippet. A duplicate profile is
two rows that look like one person, and only the user knows which of the two
embeddings is the better one -- auto-merging at startup would quietly pick one
and destroy the evidence for the other. The audit's job is to make the list
visible in the log before a run, not to act on it.

The checks, and why each one matters:

``near_duplicate_profiles``
    Two profiles whose embeddings are closer than ``near_duplicate_dist``. The
    matcher resolves a cluster by nearest neighbour over these same vectors, so
    a pair this close cannot be told apart: whoever wins a marginal vote is
    decided by noise. Note this is a *different, tighter* question from
    "could these two win the same utterance" -- that bar is
    ``matching.accept_threshold`` (0.35), and two DIFFERENT colleagues routinely
    sit near it. A real double registration of one person lands far closer.

``name_collisions``
    Two rows whose names normalise to the same string (case, surrounding or
    repeated whitespace, Unicode compatibility forms). They are distinct keys,
    so the matcher treats them as two speakers, but a human reading the UI sees
    one name twice and cannot tell which profile is which.

``profiles_without_snippets`` / ``snippets_without_profile``
    A profile with no snippets cannot be refined -- ``_auto_refine`` returns at
    its second guard -- yet it still matches, on an embedding that can no longer
    be rebuilt. A speaker with snippets but no profile is the mirror image: the
    audio is there and nothing was ever learned from it.

``missing_snippet_files`` / ``orphan_snippet_files``
    A row pointing at audio that is gone, and audio on disk that no row claims.
    Both are remainders of an interrupted or hand-edited run. The orphan side
    matters most: ``rescan_voices_dir`` adopts unregistered audio, so a leftover
    file silently becomes a snippet and is folded into the next refine.

``duplicate_snippet_content``
    Byte-identical audio stored twice under one speaker. ``add_snippet`` names
    files after a hash of the source span, so this means the same span was
    learned twice under different names -- the exact "remainder from a previous
    run" this audit exists to surface. Each finding carries both snippet ids and
    says which copy is the keeper (the lower id), because
    ``VoiceprintService.cleanup_integrity`` acts on this list and a plan whose
    "delete this one" flips between runs is not a plan.

``stale_speech_totals``
    ``voiceprints.total_speech_sec`` disagreeing with the sum of the speaker's
    snippet durations. Not a duplicate, but the same class of drift: a deleted
    snippet leaves the total behind, and the number is what the auto-collect cap
    is measured against.

Snippet paths are stored ABSOLUTE at write time (``/app/voices/...`` inside the
container), so a path that does not resolve is retried relative to the
configured voices directory before it is called missing. Without that, every row
looks absent whenever the audit runs outside the container that wrote it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("asr_mcp.voiceprint.audit")

#: Mirrors the extension set ``rescan_voices_dir`` adopts, so the audit counts
#: the same files a rescan would.
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mkv", ".mp4",
              ".webm", ".opus"}

DEFAULT_CFG = {
    "enabled": True,
    # Cosine distance below which two profiles are treated as the same person.
    # Deliberately far below matching.accept_threshold (0.35): two different
    # colleagues sit near that bar by nature, while one person registered twice
    # lands well under this.
    "near_duplicate_dist": 0.25,
    # total_speech_sec / sum(snippet durations) beyond which the total is stale.
    "stale_speech_ratio": 1.5,
    # Content hashing reads every snippet. Cheap on a few hundred files, but it
    # is the only check that touches file bytes, so it can be turned off.
    "hash_snippets": True,
}

_WS = re.compile(r"\s+")


def _cfg() -> dict:
    """``integrity`` section of thresholds.json, with the shipped defaults."""
    out = dict(DEFAULT_CFG)
    try:
        from asr_mcp.config import get_config

        section = (get_config() or {}).get("integrity") or {}
        if isinstance(section, dict):
            for key, value in section.items():
                if key in out:
                    out[key] = value
    except Exception as e:  # config unavailable -- defaults are authoritative
        logger.debug("Integrity thresholds unavailable, using defaults: %s", e)
    return out


def normalise_name(name: str) -> str:
    """The name a human would consider the same as ``name``.

    NFKC folds compatibility forms (full-width Latin, ligatures) onto their
    canonical spellings, ``casefold`` is stronger than ``lower`` for scripts
    like German ß, and whitespace runs collapse so a trailing space or a double
    space cannot make two profiles look distinct.
    """
    text = unicodedata.normalize("NFKC", name or "")
    return _WS.sub(" ", text).strip().casefold()


def cosine_distance(a, b) -> float:
    """1 - cosine similarity; 0.0 means the same direction."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 1e-8 or nb <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def resolve_snippet_path(file_path: str, voices_dir: Path) -> Optional[Path]:
    """Locate a stored snippet path, tolerating a different mount prefix.

    Paths are written absolute inside the container (``/app/voices/...``). When
    the audit runs against the same layout that resolves directly; otherwise the
    ``<user>/<speaker>/<file>`` tail is re-anchored on ``voices_dir``. Returns
    ``None`` only when the audio really is not there.
    """
    if not file_path:
        return None
    direct = Path(file_path)
    if direct.is_file():
        return _canonical(direct)

    # Re-anchor on the voices dir, trying successively shorter tails so a stored
    # path with more or fewer leading components still lands.
    try:
        parts = direct.parts
    except Exception:
        return None
    voices = Path(voices_dir)
    for start in range(len(parts)):
        candidate = voices.joinpath(*parts[start:])
        if candidate.is_file():
            return _canonical(candidate)
    return None


def _canonical(path: Path) -> Path:
    """Absolute, symlink-resolved form, so two spellings compare equal.

    Orphan detection compares a resolved snippet path against the directory walk
    it produced. One of those arrives as ``/app/voices/...`` and the other as a
    path under the configured voices dir; without this they are different Path
    objects and every snippet looks like an orphan.
    """
    try:
        return Path(os.path.realpath(str(path)))
    except OSError:
        return path


def _content_digest(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.debug("Could not hash %s: %s", path, e)
        return None


def _near_duplicate_pairs(profiles: dict, threshold: float) -> list[dict]:
    """Profile pairs closer than ``threshold``, closest first.

    Pending profiles are excluded. A pending profile is an unnamed blend that
    matching already ignores (lesson 39), so two of them colliding is not a
    matching failure -- and it is the normal state while several recordings are
    being learned from in parallel.
    """
    items = []
    for name, vp in profiles.items():
        emb = vp.get("embedding")
        if emb is None or len(emb) == 0:
            continue
        norm = float(np.linalg.norm(np.asarray(emb, dtype=np.float64)))
        if not np.isfinite(norm) or norm <= 1e-8:
            continue
        items.append((name, np.asarray(emb, dtype=np.float64) / norm))
    items.sort(key=lambda kv: kv[0])

    pairs = []
    for i in range(len(items)):
        name_i, vec_i = items[i]
        for j in range(i + 1, len(items)):
            name_j, vec_j = items[j]
            dist = float(1.0 - np.dot(vec_i, vec_j))
            if dist <= threshold:
                pairs.append({
                    "a": name_i, "b": name_j, "distance": round(dist, 4),
                })
    pairs.sort(key=lambda p: p["distance"])
    return pairs


def _name_collisions(names) -> list[dict]:
    """Distinct names that normalise to the same string.

    Two spellings of one name are the duplicate this is looking for; the SAME
    name twice is not. The caller passes the keys of a ``name -> profile``
    dict, which cannot repeat, but a set here costs nothing and means a group
    can never be two copies of one row wearing a collision's clothes.
    """
    groups: dict[str, set] = {}
    for name in names:
        groups.setdefault(normalise_name(name), set()).add(name)
    return [
        {"normalised": key, "names": sorted(members)}
        for key, members in sorted(groups.items())
        if len(members) > 1
    ]


def audit_user(user_id: str, voiceprints, snippet_db, voices_dir: Path,
               cfg: Optional[dict] = None) -> dict:
    """Audit one user's store.

    ``voiceprints`` is name -> profile dict (as ``VoiceprintDB.list_all``
    returns). ``snippet_db`` is the ``SnippetDB`` itself, not a summary dict:
    the file-level checks need each row's stored path, and ``all_speakers``
    only carries counts. Everything else here is pure, so a test can pass a
    stand-in with the two methods used.
    """
    snippets = snippet_db.all_speakers(user_id=user_id)
    cfg = cfg or _cfg()
    report = {
        "user_id": user_id,
        "profiles": len(voiceprints),
        "speakers_with_snippets": len(snippets),
        "near_duplicate_profiles": _near_duplicate_pairs(
            voiceprints, float(cfg.get("near_duplicate_dist", 0.25))
        ),
        "name_collisions": _name_collisions(voiceprints),
        "profiles_without_snippets": [],
        "snippets_without_profile": [],
        "missing_snippet_files": [],
        "orphan_snippet_files": [],
        "duplicate_snippet_content": [],
        "stale_speech_totals": [],
    }

    hash_snippets = bool(cfg.get("hash_snippets", True))

    for name in sorted(voiceprints):
        if name not in snippets:
            report["profiles_without_snippets"].append(name)
    for name in sorted(snippets):
        if name not in voiceprints:
            report["snippets_without_profile"].append(name)

    # Per-speaker file walk. `all_speakers` gives counts only, so the rows are
    # re-read here; this is the only place the audit touches the snippets table.
    claimed: set[Path] = set()
    ratio = float(cfg.get("stale_speech_ratio", 1.5))
    for name in sorted(snippets):
        rows = snippet_db.list_by_speaker(name, user_id=user_id)
        digests: dict[str, dict] = {}
        total = 0.0
        for row in rows:
            total += float(row.get("duration_sec") or 0.0)
            resolved = resolve_snippet_path(row.get("file_path"), voices_dir)
            if resolved is None:
                if len(report["missing_snippet_files"]) < 50:
                    report["missing_snippet_files"].append({
                        "speaker": name, "snippet_id": row.get("id"),
                        "file_path": row.get("file_path"),
                    })
                continue
            claimed.add(resolved)
            if not hash_snippets:
                continue
            digest = _content_digest(resolved)
            if digest is None:
                continue
            seen = digests.get(digest)
            if seen is None:
                digests[digest] = {"id": row.get("id"), "file": resolved.name}
            elif len(report["duplicate_snippet_content"]) < 50:
                # The keeper is the LOWER snippet id, so "which copy survives"
                # does not depend on the order rows happen to come back in --
                # this finding is acted on by cleanup_integrity(), and a plan
                # that flips between two runs is not a plan.
                keeper, dup = sorted((seen, {"id": row.get("id"),
                                             "file": resolved.name}),
                                     key=lambda r: (r["id"] is None, r["id"]))
                report["duplicate_snippet_content"].append({
                    "speaker": name,
                    "first_id": keeper["id"], "first": keeper["file"],
                    "duplicate_id": dup["id"], "duplicate": dup["file"],
                })

        expected = float(snippets.get(name, {}).get("total_duration") or 0.0)
        if expected > 0 and abs(total - expected) > max(1.0, expected * 0.05):
            logger.debug("Snippet duration sum for %s/%s: %.1fs vs stored %.1fs",
                         user_id, name, total, expected)

        vp = voiceprints.get(name)
        if vp:
            stored = float(vp.get("total_speech_sec") or 0.0)
            if stored > 0 and total > 0 and stored > total * ratio:
                report["stale_speech_totals"].append({
                    "speaker": name,
                    "total_speech_sec": round(stored, 1),
                    "snippet_seconds": round(total, 1),
                })

    # Audio on disk that no row claims. A rescan adopts these silently, so an
    # orphan left by an interrupted run becomes a snippet on the next pass.
    user_dir = Path(voices_dir) / user_id
    if user_dir.is_dir():
        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for entry in sorted(speaker_dir.iterdir()):
                if not entry.is_file() or entry.suffix.lower() not in AUDIO_EXTS:
                    continue
                if _canonical(entry) in claimed:
                    continue
                if len(report["orphan_snippet_files"]) < 50:
                    report["orphan_snippet_files"].append({
                        "speaker": speaker_dir.name, "file": entry.name,
                    })

    return report


def audit_all(voiceprint_db, snippet_db, voices_dir: Path,
              user_ids=None, cfg: Optional[dict] = None) -> list[dict]:
    """Audit every user that has voiceprints or snippets."""
    cfg = cfg or _cfg()
    if user_ids is None:
        user_ids = sorted(
            set(voiceprint_db.user_ids()) | set(snippet_db.user_ids())
        )
    reports = []
    for user_id in user_ids:
        try:
            reports.append(audit_user(
                user_id,
                voiceprint_db.list_all(user_id=user_id, include_pending=True),
                snippet_db,
                voices_dir, cfg,
            ))
        except Exception as e:
            logger.warning("Voiceprint audit failed for %r: %s", user_id, e)
    return reports


#: Keys whose presence means the audit found something worth a human's time.
#: Ordered most-serious first, which is the order they are logged in.
FINDING_KEYS = (
    "near_duplicate_profiles",
    "name_collisions",
    "profiles_without_snippets",
    "snippets_without_profile",
    "duplicate_snippet_content",
    "missing_snippet_files",
    "orphan_snippet_files",
    "stale_speech_totals",
)


def has_findings(report: dict) -> bool:
    return any(report.get(key) for key in FINDING_KEYS)


def log_report(report: dict, logger_: Optional[logging.Logger] = None) -> None:
    """Log one audit result: one line when clean, one line per finding when not.

    A single INFO line on a healthy store is what makes a WARNING line mean
    something on the next startup.
    """
    log = logger_ or logger
    user = report.get("user_id")
    if not has_findings(report):
        log.info("Voiceprint audit clean for %r: %d profile(s), %d with snippets",
                 user, report.get("profiles", 0),
                 report.get("speakers_with_snippets", 0))
        return
    for key in FINDING_KEYS:
        items = report.get(key) or []
        if not items:
            continue
        log.warning("Voiceprint audit [%s] %s: %d", user, key, len(items))
        for item in items[:10]:
            log.warning("    %s", item)
        if len(items) > 10:
            log.warning("    ... and %d more", len(items) - 10)


def run_startup_audit(voiceprint_service, logger_: Optional[logging.Logger] = None,
                      cfg: Optional[dict] = None):
    """Audit the live service's store. Never raises.

    Called from the FastAPI lifespan. A failure here must not stop the server:
    an unreadable voices directory is a reason to warn, not to refuse to
    transcribe. ``cfg`` defaults to thresholds.json; the parameter exists so
    the tests can drive it without editing the shipped config.
    """
    log = logger_ or logger
    cfg = cfg or _cfg()
    if not cfg.get("enabled", True):
        log.info("Voiceprint startup audit disabled")
        return []
    try:
        reports = audit_all(
            voiceprint_service._db,
            voiceprint_service._snippets,
            # voices_dir is a @property, not a method: calling it hands audit_all
            # a TypeError's worth of PosixPath-as-callable.
            voiceprint_service.voices_dir,
            cfg=cfg,
        )
    except Exception as e:
        log.warning("Voiceprint startup audit could not run: %s", e)
        return []
    for report in reports:
        log_report(report, log)
    return reports
