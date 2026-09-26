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
        )
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


@router.post("/rescan/stream")
async def rescan_voices_stream(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    busy = job_state.get_running()
    if busy is not None:
        from fastapi.responses import JSONResponse
        content = {"detail": "Another job is already running"}
        if busy.user_id == user_id:
            content["job"] = busy.meta()
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
