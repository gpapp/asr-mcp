import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from asr_mcp.api.schemas import (
    SpeakerRenameRequest, SpeakerMergeRequest,
    SpeakerSnippetInfo, VoiceprintSpeakerInfo,
    VoiceprintSpeakerListResponse, VoiceprintSnippetListResponse,
    RescanResponse,
    PendingProfileConfirm, PendingProfileConfirmResponse,
    PendingCandidate, PendingCandidatesResponse, PendingMergeRequest,
    PendingMergeResponse, IntegrityCleanupRequest,
    FlaggedSnippetInfo, FlaggedSnippetListResponse, FlaggedRemoveRequest,
)
from asr_mcp.api.security import get_current_user, validate_upload_filename
from asr_mcp.config.settings import Settings, get_settings
from asr_mcp.core import job_state

logger = logging.getLogger("asr_mcp.api.voiceprint_router")
router = APIRouter(prefix="/voiceprint", tags=["Voiceprint"])


def _get_service(settings: Settings, ensure_gpu: bool = False):
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.voiceprint.service import VoiceprintService
    from asr_mcp.core.model_state import state

    if ensure_gpu:
        state.ensure_diarize_ready()
    db = DatabaseManager(settings.db_path)
    service = VoiceprintService(settings.data_dir, db)
    service.set_voices_dir(settings.voices_dir)
    return service


@router.get("/speakers", response_model=VoiceprintSpeakerListResponse)
async def list_speakers(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings)
    speakers = service.list_speakers(user_id=user_id)
    items = [VoiceprintSpeakerInfo(**s) for s in speakers]
    return VoiceprintSpeakerListResponse(speakers=items, count=len(items))


@router.get("/pending", response_model=VoiceprintSpeakerListResponse)
async def list_pending_profiles(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Auto-learned speakers still waiting for a name.

    These profiles collect snippets and are excluded from matching. Renaming one
    (POST /pending/{name}/confirm) is what makes it matchable.
    """
    service = _get_service(settings)
    profiles = service.pending_profiles(user_id=user_id)
    items = [VoiceprintSpeakerInfo(**p) for p in profiles]
    return VoiceprintSpeakerListResponse(speakers=items, count=len(items))


@router.post("/pending/{speaker_name}/confirm",
             response_model=PendingProfileConfirmResponse)
async def confirm_pending_profile(
    speaker_name: str,
    req: PendingProfileConfirm,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Name an auto-learned profile, making it a real, matchable voiceprint.

    The only route that promotes a pending profile. Refuses a name that is
    already taken so confirming cannot overwrite an existing profile.
    """
    service = _get_service(settings, ensure_gpu=True)
    result = service.confirm_pending(speaker_name, req.new_name, user_id=user_id)
    if not result.get("ok"):
        return PendingProfileConfirmResponse(**result)
    return PendingProfileConfirmResponse(ok=True, name=result["name"],
                                         renamed_from=result["renamed_from"])


@router.get("/pending/{speaker_name}/candidates",
             response_model=PendingCandidatesResponse)
async def pending_profile_candidates(
    speaker_name: str,
    limit: int = 5,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Which registered speakers does this pending profile resemble?

    A learner is very often someone who is already registered. The UI uses
    this to offer a one-click "merge into this person" instead of forcing a
    rename that would create a second profile for one person. Purely advisory
    -- nothing is ever merged without an explicit POST.
    """
    service = _get_service(settings)
    return PendingCandidatesResponse(
        pending=speaker_name,
        candidates=[PendingCandidate(**c)
                    for c in service.pending_candidates(speaker_name, user_id=user_id,
                                                         limit=limit)],
    )


@router.post("/pending/{speaker_name}/merge", response_model=PendingMergeResponse)
async def merge_pending_profile(
    speaker_name: str,
    req: PendingMergeRequest,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Fold a pending profile into an EXISTING speaker and re-refine them.

    The resolution when the learner turns out to be someone already
    registered. Snippets move, the pending profile ceases to exist, and the
    target profile is rebuilt over the larger corpus.
    """
    service = _get_service(settings, ensure_gpu=True)
    result = service.merge_pending_into(speaker_name, req.into, user_id=user_id)
    if result.get("error"):
        return PendingMergeResponse(error=result["error"])
    return PendingMergeResponse(ok=True, status=result.get("status"),
                                name=result.get("into"),
                                renamed_from=result.get("from"),
                                snippets_moved=result.get("snippets_moved", 0))


@router.get("/speakers/{speaker_name}/snippets", response_model=VoiceprintSnippetListResponse)
async def list_snippets(
    speaker_name: str,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings)
    snippets = service.list_snippets(speaker_name, user_id=user_id)
    items = [SpeakerSnippetInfo(**s) for s in snippets]
    return VoiceprintSnippetListResponse(speaker_name=speaker_name, snippets=items, count=len(items))


@router.post("/speakers/{speaker_name}/rename")
async def rename_speaker(
    speaker_name: str,
    req: SpeakerRenameRequest,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings, ensure_gpu=True)
    return service.rename_speaker(speaker_name, req.new_name, user_id=user_id)


@router.post("/speakers/merge")
async def merge_speakers(
    req: SpeakerMergeRequest,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings, ensure_gpu=True)
    return service.merge_speakers(req.primary, req.secondary, user_id=user_id)


@router.post("/speakers/{speaker_name}/upload")
async def upload_snippet(
    speaker_name: str,
    file: UploadFile = File(...),
    start_sec: float = Query(None),
    end_sec: float = Query(None),
    replace_source: bool = Query(
        True,
        description=(
            "Replace this speaker's snippets already recorded from this same "
            "file instead of adding to them (one file, one contribution)."
        ),
    ),
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return {"error": "File too large (max 200MB)"}

    validate_upload_filename(file.filename)

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    try:
        from asr_mcp.voiceprint.utils import load_audio
        waveform, sr = load_audio(str(tmp_path))
        audio_data = waveform.numpy().squeeze()

        if start_sec is not None or end_sec is not None:
            total_sec = len(audio_data) / sr
            s = max(0.0, float(start_sec or 0.0))
            e = min(total_sec, float(end_sec) if end_sec is not None else total_sec)
            if e <= s:
                return {"error": f"Invalid segment: end ({e:.1f}s) must be after start ({s:.1f}s)"}
            audio_data = audio_data[int(s * sr):int(e * sr)]

        service = _get_service(settings, ensure_gpu=True)
        result = service.add_snippet(
            speaker_name=speaker_name,
            audio_data=audio_data,
            user_id=user_id,
            sample_rate=sr,
            source_audio=file.filename,
            start_sec=start_sec,
            end_sec=end_sec,
            replace_source=replace_source,
        )
        if result.get("replaced_snippets"):
            # Say it out loud: the response is the only place the caller learns
            # that audio it asked to keep is gone.
            logger.info(
                "%s/%s: replaced %d earlier snippet(s) from %s",
                user_id, speaker_name, result["replaced_snippets"], file.filename)
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            tmp_path.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass


@router.post("/speakers/{speaker_name}/refine")
async def refine_speaker(
    speaker_name: str,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings, ensure_gpu=True)
    return service.refine_speaker(speaker_name, user_id=user_id)


@router.get("/integrity")
async def voiceprint_integrity(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """The same audit the server logs at startup, for the signed-in user.

    Read-only. The startup log is the same numbers, but this is scoped to one
    user and returns them as JSON, so the Voiceprints tab can show them without
    anyone reading a container log.
    """
    return _get_service(settings).integrity_report(user_id=user_id)


@router.post("/integrity/cleanup")
async def voiceprint_integrity_cleanup(
    req: IntegrityCleanupRequest,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Repair the audit findings that have one correct answer.

    Destructive, and explicit twice over: the only automatic repair is the
    profile rebuild (which only ever drops a DERIVED row), and ``dry_run``
    defaults to true so the first call here returns the plan and touches
    nothing. Pass ``{"dry_run": false}`` to act on it.

    ``dead_snippets`` deletes rows whose audio is not on disk -- the wav->flac
    migration leftovers, all with ``source_audio IS NULL`` and therefore
    invisible to the "one file, one contribution" replacement. No file that
    resolves is ever unlinked; a dead row's file is already gone.

    ``trim_snippets`` is the one step that is not an audit finding: it enforces
    the snippet policy (top ``AUTO_COLLECT_MAX_PER_SOURCE`` per file, none under
    ``AUTO_COLLECT_MIN_DURATION``) over the whole library, which a reprocess
    only ever applies to the file it re-runs. Ranked by stored duration because
    a row keeps no attribution confidence, and rows with no recorded source are
    left alone.

    ``ensure_gpu=True`` is load-bearing. Removing snippets leaves the profile
    describing audio that is gone, so the speakers it touched are re-refined
    afterwards -- and ``_auto_refine`` returns silently when no embedding
    session is loaded. Without this a "refined" line in the response could be a
    lie. Only VAD + embedding load here; the ASR backend stays unloaded.
    """
    service = _get_service(settings, ensure_gpu=True)
    return service.cleanup_integrity(
        user_id=user_id,
        duplicates=req.duplicates,
        dead_snippets=req.dead_snippets,
        stale_totals=req.stale_totals,
        trim_snippets=req.trim_snippets,
        dry_run=req.dry_run,
    )


@router.delete("/snippets/{snippet_id}")
async def delete_snippet(
    snippet_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings)
    return service.delete_snippet(snippet_id, user_id=user_id)


@router.get("/snippets/{snippet_id}/audio")
async def get_snippet_audio(
    snippet_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings)
    sn = service._snippets.get(snippet_id, user_id=user_id)
    if not sn:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Snippet not found")
    file_path = Path(sn["file_path"])
    if not file_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/flac" if file_path.suffix == ".flac" else "audio/wav"
    return FileResponse(str(file_path), media_type=media_type)


@router.delete("/speakers/{speaker_name}")
async def delete_speaker(
    speaker_name: str,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings)
    return service.delete_speaker_bulk(speaker_name, user_id=user_id)


@router.post("/rescan", response_model=RescanResponse)
async def rescan_voices(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    service = _get_service(settings, ensure_gpu=True)
    return service.rescan_voices_dir(user_id=user_id)


def _safe_json(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    return obj


async def _sse_put(queue: asyncio.Queue, evt) -> None:
    await queue.put(evt)
    job_state.publish(evt)
    await asyncio.sleep(0.01)


@router.get("/flagged", response_model=FlaggedSnippetListResponse)
async def list_flagged_snippets(
    speaker_name: str = Query(None, description="Restrict to one speaker"),
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Snippets the outlier pass marked for review, worst score first.

    Read-only. A mark is an accusation with evidence attached, never a
    deletion -- the audio is still there and still part of nothing but the
    report until POST /flagged/remove is called with dry_run=false.
    """
    rows = _get_service(settings).list_flagged_snippets(
        user_id=user_id, speaker_name=speaker_name)
    items = [FlaggedSnippetInfo(
        id=r["id"], speaker_name=r["speaker_name"], file_path=r["file_path"],
        duration_sec=r.get("duration_sec") or 0.0,
        source_audio=r.get("source_audio"), start_sec=r.get("start_sec"),
        end_sec=r.get("end_sec"), review_score=r.get("review_score"),
        review_reason=r.get("review_reason"),
    ) for r in rows]
    return FlaggedSnippetListResponse(snippets=items, count=len(items))


@router.post("/flagged/{snippet_id}/keep")
async def keep_flagged_snippet(
    snippet_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Clear one snippet's review mark: this audio IS the speaker.

    The snippet is not deleted and the profile is not touched; re-run
    consolidation to fold it back into the fitted profile.
    """
    return _get_service(settings).keep_flagged_snippet(snippet_id, user_id=user_id)


@router.post("/flagged/remove")
async def remove_flagged_snippets(
    req: FlaggedRemoveRequest,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Delete every snippet marked for review, then re-refine each speaker once.

    ``dry_run`` defaults to true, so the first call returns the plan. Pass
    ``{"dry_run": false, "speaker_name": "..."}`` to act -- optionally for one
    speaker only.
    """
    return _get_service(settings, ensure_gpu=True).remove_flagged_snippets(
        user_id=user_id, speaker_name=req.speaker_name, dry_run=req.dry_run)


@router.post("/consolidate/stream")
async def consolidate_stream(
    speaker_name: str = Query(None, description="One speaker; empty = the whole library"),
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    """Recompute profiles from all snippets and flag the outliers.

    A library-wide pass embeds every snippet the user owns, which is minutes of
    GPU, so it streams progress like the rescan does. Registering the job also
    keeps the TTL monitor from unloading the embedding session underneath the
    pass -- an unloaded session makes the refine silently do nothing while the
    report still claims success.
    """
    busy = job_state.get_for_user(user_id)
    if busy is not None:
        from fastapi.responses import JSONResponse
        content = {"detail": "You already have a job running", "job": busy.meta()}
        return JSONResponse(status_code=409, content=content)

    service = _get_service(settings, ensure_gpu=True)
    scope = speaker_name or "all speakers"
    queue: asyncio.Queue = asyncio.Queue()
    job = job_state.start_job(mode="consolidate", filename=scope, user_id=user_id)
    loop = asyncio.get_running_loop()

    def _threadsafe_progress(evt: dict):
        def _do():
            queue.put_nowait(evt)
            job_state.publish(evt)
        loop.call_soon_threadsafe(_do)

    async def run_consolidate():
        try:
            if speaker_name:
                result = service.consolidate_speaker(
                    speaker_name, user_id=user_id, progress_callback=_threadsafe_progress)
                result = {"speakers": 1, "flagged_count": result.get("flagged_count", 0),
                          "results": [result]}
            else:
                result = await loop.run_in_executor(
                    None,
                    lambda: service.consolidate_library(
                        user_id=user_id, progress_callback=_threadsafe_progress),
                )
            await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": result})
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            job_state.ensure_finished(job)
            await queue.put(None)

    asyncio.create_task(run_consolidate())

    async def event_stream():
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield f"data: {json.dumps(_safe_json(evt))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/rescan/stream")
async def rescan_voices_stream(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    busy = job_state.get_for_user(user_id)
    if busy is not None:
        from fastapi.responses import JSONResponse
        content = {"detail": "You already have a job running", "job": busy.meta()}
        return JSONResponse(status_code=409, content=content)

    service = _get_service(settings, ensure_gpu=True)

    queue: asyncio.Queue = asyncio.Queue()
    job = job_state.start_job(mode="rescan", filename="voiceprint_directory", user_id=user_id)
    loop = asyncio.get_running_loop()

    def _threadsafe_progress(evt: dict):
        def _do():
            queue.put_nowait(evt)
            job_state.publish(evt)
        loop.call_soon_threadsafe(_do)

    async def run_rescan():
        try:
            result = await loop.run_in_executor(
                None,
                lambda: service.rescan_voices_dir(user_id=user_id, progress_callback=_threadsafe_progress),
            )
            await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": result})
        except Exception as e:
            logger.error("Rescan failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            job_state.ensure_finished(job)
            await queue.put(None)

    asyncio.create_task(run_rescan())

    async def event_stream():
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield f"data: {json.dumps(_safe_json(evt))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
