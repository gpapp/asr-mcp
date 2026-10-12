"""Rebuild a voiceprint profile from scratch when the audit says it is broken.

The audit (``voiceprint/audit.py``) is read-only by design: it reports, a
human decides. Two of its findings are different in kind, though. A speaker
whose snippets have no profile row, or whose row was built from audio that no
longer exists, is not a judgement call -- there is simply no usable profile
left, and the fix is mechanical: drop the DERIVED row and recompute it from the
snippet audio, which is the source of truth and is never touched here.

This module is that repair, and it is deliberately narrow:

* the three findings a rebuild can act on are targets --
  ``snippets_without_profile`` (rebuild), ``profiles_without_snippets`` (drop,
  there is nothing to rebuild from) and ``missing_snippet_files`` (delete the
  dead rows, then rebuild from the audio that survives). Each rebuild purges
  first, so a store cannot accumulate rows that point at files nobody has;
* near-duplicate profiles, name collisions and duplicate snippet CONTENT are
  NOT touched. Deciding that two profiles are the same person is the user's
  call, and the duplicate-content repair belongs to ``cleanup_integrity``,
  which deletes the higher-id copy for a byte-identical pair;
* nothing is lost by acting. A profile row is a derived artefact: the snippets
  and their FLAC files stay exactly as they are, so the action is idempotent
  and reversible by re-refining.

Why a FILE QUEUE instead of doing the work inside the audit. The audit runs in
the FastAPI lifespan, before any request, and models are deliberately not
loaded at start (see AGENTS.md, Model Lifecycle). Rebuilding needs the ECAPA
embedding session, so the work is queued to disk and a background task drains
it once the models are up and no job is running. The queue is the reason the
repair survives a restart that happens before the models ever loaded.

Why a DONE list. A rebuild can still fail to clear a finding -- the snippets
may be unembeddable, the model may be unavailable, a profile may be rebuilt
into something the audit dislikes again. The audit runs on every startup, so
without a record of what has already been attempted the queue would refill
forever and the server would re-embed the same speaker on every boot. A
fingerprint of (user, speaker, reasons) is therefore remembered, and a repeat
of the exact same finding is not re-queued. A *changed* finding -- different
reasons, a different speaker -- is a new fingerprint and is acted on again.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("asr_mcp.voiceprint.rebuild")

#: Filename of the queue, relative to ``settings.data_dir``.
QUEUE_FILENAME = "voiceprint_rebuild_queue.json"

#: How many times one entry may be tried before it is dropped. A speaker whose
#: embedding keeps failing must not wedge the queue behind it forever.
MAX_ATTEMPTS = 3

#: Cap on remembered "already attempted" fingerprints (see :func:`_prune_done`).
MAX_DONE = 200

#: Audit findings a rebuild can act on -> the report key that carries the
#: speaker name. ``missing_snippet_files`` is a list of dicts with a ``speaker``
#: field; the other two are lists of names.
#:
#: ``missing_snippet_files`` earns its place only because the rebuild CLEANS UP
#: before it rebuilds (``rebuild_profile_from_scratch(purge_dead=True)``): it
#: deletes the rows whose audio is gone and then recomputes the profile from
#: what is on disk. An earlier version of this module rebuilt on this finding
#: without the purge and re-embedded 219 snippets -- 25 minutes of GPU -- for a
#: store where 73 rows pointed at ``.wav`` files a migration had removed. The
#: finding survived the rebuild, the queue refilled, and the store stayed dirty.
#: The order is the whole point: clean the records, then rebuild from disk.
TARGET_FINDINGS = (
    "snippets_without_profile",
    "profiles_without_snippets",
    "missing_snippet_files",
)


def _speaker_of(item) -> Optional[str]:
    if isinstance(item, str):
        return item or None
    if isinstance(item, dict):
        name = item.get("speaker")
        return name or None
    return None


def rebuild_targets(report: dict) -> list[dict]:
    """Speakers in one audit report that a from-scratch rebuild would help.

    Pure, so the mapping is testable without a database and without models.
    Returns ``[{"speaker": name, "reasons": [...]}, ...]`` sorted by name, with
    the reasons in :data:`TARGET_FINDINGS` order.
    """
    reasons: dict[str, list[str]] = {}
    for key in TARGET_FINDINGS:
        for item in report.get(key) or ():
            name = _speaker_of(item)
            if not name:
                continue
            reasons.setdefault(name, []).append(key)
    return [{"speaker": name, "reasons": reasons[name]}
            for name in sorted(reasons)]


def _fingerprint(user_id: str, speaker: str, reasons: Iterable[str]) -> str:
    return "{}|{}|{}".format(user_id, speaker, ",".join(sorted(reasons)))


def _read(path: Path) -> dict:
    """Load the queue file, tolerating absence and corruption.

    A half-written or hand-edited file must not stop the server: an unreadable
    queue is a warning and a fresh start, never an exception.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"entries": [], "done": []}
    except (OSError, ValueError) as e:
        logger.warning("Rebuild queue %s unreadable (%s) -- starting a fresh one",
                       path, e)
        return {"entries": [], "done": []}
    if not isinstance(data, dict):
        logger.warning("Rebuild queue %s is not an object -- starting a fresh one", path)
        return {"entries": [], "done": []}
    entries = [e for e in (data.get("entries") or []) if isinstance(e, dict)]
    done = [d for d in (data.get("done") or []) if isinstance(d, str)]
    return {"entries": entries, "done": done}


def _write(path: Path, data: dict) -> None:
    """Atomic write: tmp file in the same directory, then ``os.replace``.

    The queue is read at start-up and written from a background task, so a
    crash mid-write must not leave a truncated file that loses the queue.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def queue_path(data_dir) -> Path:
    return Path(data_dir) / QUEUE_FILENAME


def enqueue_from_reports(path, reports: Iterable[dict],
                         logger_: Optional[logging.Logger] = None) -> int:
    """Add every rebuild target in ``reports`` to the queue. Returns the count.

    Entries are keyed by ``(user_id, speaker)``: re-queuing a speaker already
    present merges the new reasons into the existing entry and keeps its
    ``attempts`` and ``first_seen``, so a second startup does not reset a
    speaker's progress nor queue it twice.
    """
    log = logger_ or logger
    data = _read(Path(path))
    entries = data["entries"]
    done = set(data["done"])
    known = {(e.get("user_id"), e.get("speaker")): e for e in entries}

    added = 0
    changed = False
    for report in reports or ():
        user_id = report.get("user_id")
        if not user_id:
            continue
        for target in rebuild_targets(report):
            speaker = target["speaker"]
            fp = _fingerprint(user_id, speaker, target["reasons"])
            if fp in done:
                log.debug("Rebuild already attempted for %s/%s (%s) -- not re-queued",
                          user_id, speaker, ", ".join(target["reasons"]))
                continue
            existing = known.get((user_id, speaker))
            if existing is None:
                entries.append({
                    "user_id": user_id,
                    "speaker": speaker,
                    "reasons": list(target["reasons"]),
                    "first_seen": time.time(),
                    "attempts": 0,
                })
                known[(user_id, speaker)] = entries[-1]
                added += 1
                changed = True
                log.warning(
                    "Voiceprint audit: queueing %s/%s for a from-scratch rebuild "
                    "(%s). Snippets and their audio are NOT touched -- only the "
                    "derived profile row is dropped and recomputed.",
                    user_id, speaker, ", ".join(target["reasons"]))
            else:
                # The entry exists but the finding set grew. Persist the merge:
                # a later run_pending reads reasons from the file, not memory.
                merged = sorted(set(existing.get("reasons") or ())
                                | set(target["reasons"]))
                if merged != sorted(existing.get("reasons") or ()):
                    existing["reasons"] = merged
                    changed = True

    if changed:
        _write(Path(path), data)
    return added


def pending_entries(path) -> list[dict]:
    """Entries still waiting for an attempt."""
    return list(_read(Path(path))["entries"])


def _prune_done(data: dict) -> None:
    """Bound the done list without losing the markers that matter.

    A marker is what stops a finding the rebuild cannot fix from re-queuing on
    every startup, so it must outlive its entry -- entries are removed as soon
    as they are handled, which would empty the list if "pruned" meant "kept
    only while queued". It is a history, not a cache: trim the OLDEST beyond
    :data:`MAX_DONE`, which is generous next to a three-attempt cap.
    """
    if len(data["done"]) > MAX_DONE:
        data["done"] = data["done"][-MAX_DONE:]


def run_pending(service, path, logger_: Optional[logging.Logger] = None) -> dict:
    """Try every queued rebuild. Never raises.

    An entry is removed from the queue when the service reports it is done --
    ``rebuilt`` or ``dropped`` -- or when it has burned ``MAX_ATTEMPTS``. A
    ``skipped`` result (no embedding session yet) is NOT an attempt: the entry
    stays, because nothing was tried.
    """
    log = logger_ or logger
    path = Path(path)
    data = _read(path)
    summary = {"attempted": 0, "rebuilt": 0, "dropped": 0,
               "skipped": 0, "failed": 0, "exhausted": 0}
    if not data["entries"]:
        return summary

    remaining = []
    for entry in data["entries"]:
        user_id = entry.get("user_id")
        speaker = entry.get("speaker")
        reasons = sorted(entry.get("reasons") or ())
        try:
            result = service.rebuild_profile_from_scratch(
                speaker, user_id=user_id) or {}
        except Exception as e:
            summary["failed"] += 1
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            log.warning("Rebuild of %s/%s raised: %s", user_id, speaker, e)
            if entry["attempts"] >= MAX_ATTEMPTS:
                summary["exhausted"] += 1
                log.error("Rebuild of %s/%s failed %d times -- dropping it from "
                          "the queue; run POST /api/voiceprint/integrity/cleanup "
                          "or a manual refine for this speaker",
                          user_id, speaker, entry["attempts"])
                data["done"].append(_fingerprint(user_id, speaker, reasons))
            else:
                remaining.append(entry)
            continue

        status = result.get("status")
        summary["attempted"] += 1
        if status in ("rebuilt", "dropped"):
            summary["rebuilt" if status == "rebuilt" else "dropped"] += 1
            data["done"].append(_fingerprint(user_id, speaker, reasons))
            log.warning("Voiceprint %s: %s/%s (%s)", status, user_id, speaker,
                        ", ".join(reasons))
        elif status == "skipped":
            # Not a failure and not an attempt -- the models were not ready.
            summary["skipped"] += 1
            remaining.append(entry)
        else:
            summary["failed"] += 1
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            log.warning("Rebuild of %s/%s did not complete: %s", user_id, speaker,
                        result.get("error") or result)
            if entry["attempts"] >= MAX_ATTEMPTS:
                summary["exhausted"] += 1
                log.error("Rebuild of %s/%s failed %d times -- dropping it from "
                          "the queue", user_id, speaker, entry["attempts"])
                data["done"].append(_fingerprint(user_id, speaker, reasons))
            else:
                remaining.append(entry)

    data["entries"] = remaining
    _prune_done(data)
    _write(path, data)
    if summary["attempted"] or summary["failed"]:
        log.info("Voiceprint rebuild queue: %s", summary)
    return summary
