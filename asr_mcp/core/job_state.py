"""Single active upload job registry so GUIs can reattach mid-run.

The original POST /transcribe/upload (or /diarize/upload) keeps its own
response queue; this registry mirrors every event so a later
GET /active/stream (e.g. after a page reload) can replay history and
follow the same job live.  All mutators run on the event loop with no
awaits between snapshot and subscribe, so attach is race-free.
"""
import asyncio
import time
import uuid
from typing import List, Optional, Tuple


class ActiveJob:
    def __init__(self, mode: str, filename: str, user_id: str):
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.filename = filename
        self.user_id = user_id
        self.started_at = time.time()
        self.status = "running"
        self.events: List[dict] = []
        self.subscribers: List[asyncio.Queue] = []

    def meta(self) -> dict:
        last = self.events[-1] if self.events else {}
        return {
            "id": self.id,
            "mode": self.mode,
            "filename": self.filename,
            "started_at": self.started_at,
            "status": self.status,
            "stage": last.get("stage"),
            "progress": last.get("progress"),
            "phase": last.get("phase"),
        }


_active: Optional[ActiveJob] = None


def start_job(mode: str, filename: str, user_id: str) -> ActiveJob:
    global _active
    job = ActiveJob(mode=mode, filename=filename, user_id=user_id)
    _active = job
    return job


def get_running() -> Optional[ActiveJob]:
    return _active


def get_for_user(user_id: str) -> Optional[ActiveJob]:
    job = _active
    if job is None or job.user_id != user_id:
        return None
    return job


def publish(evt) -> None:
    job = _active
    if job is None:
        return
    job.events.append(evt)
    for q in list(job.subscribers):
        q.put_nowait(evt)
    stage = evt.get("stage")
    if stage in ("done", "error"):
        finish(job, "done" if stage == "done" else "error")


def attach(job: ActiveJob) -> Tuple[List[dict], Optional[asyncio.Queue]]:
    """Snapshot events and subscribe atomically (no awaits).

    If the job already finished, returns its full event history and no
    queue so the stream can replay and close.
    """
    if _active is not job:
        return list(job.events), None
    snap = list(job.events)
    q: asyncio.Queue = asyncio.Queue()
    job.subscribers.append(q)
    return snap, q


def finish(job: ActiveJob, status: str) -> None:
    global _active
    if job.status != "running":
        return
    job.status = status
    for q in list(job.subscribers):
        q.put_nowait(None)
    job.subscribers.clear()
    if _active is job:
        _active = None


def ensure_finished(job: Optional[ActiveJob] = None) -> None:
    """Close a job left running (producer crashed before done/error).

    Pass the job created by this request so a newer job started in the
    meantime is never closed by a late finally-block.
    """
    target = job if job is not None else _active
    if target is not None and target.status == "running":
        finish(target, "error")
