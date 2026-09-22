import hashlib
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response

from asr_mcp.api.security import get_current_user
from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.transcript_router")
router = APIRouter(prefix="/transcripts", tags=["Transcripts"])


def _get_transcript_db(settings: Settings):
    from asr_mcp.db.manager import DatabaseManager, TranscriptDB
    db = DatabaseManager(settings.db_path)
    return TranscriptDB(db)


@router.get("")
async def list_transcripts(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    tdb = _get_transcript_db(settings)
    items = tdb.list_all(user_id=user_id)
    return {"transcripts": items, "count": len(items)}


@router.get("/{transcript_id}")
async def get_transcript(
    transcript_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    tdb = _get_transcript_db(settings)
    item = tdb.get(transcript_id, user_id=user_id)
    if not item:
        return JSONResponse(status_code=404, detail="Transcription not found")
    return item


@router.get("/{transcript_id}/download")
async def download_transcript(
    transcript_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    tdb = _get_transcript_db(settings)
    item = tdb.get(transcript_id, user_id=user_id)
    if not item:
        return JSONResponse(status_code=404, detail="Transcription not found")

    result = item.get("result", {})
    segments = result.get("segments", [])
    results_list = result.get("results", [])

    lines = []
    lines.append(f"Audio: {item['audio_filename']}")
    lines.append(f"Date: {item['created_at']}")
    lines.append(f"Speakers: {item.get('total_speakers', 0)}")
    lines.append(f"Duration: {item.get('audio_duration_sec', 0):.1f}s")
    lines.append("")

    if results_list:
        for r in results_list:
            text = r.get("text", "")
            speaker = r.get("speaker", "")
            start = r.get("start")
            end = r.get("end")
            if speaker and start is not None and end is not None:
                body = text.replace("\n", "\n    ")
                lines.append(f"[{speaker}] {start:.1f}s - {end:.1f}s: {body}")
            else:
                lines.append(text)
    elif segments:
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lines.append(f"[{speaker}] {start:.1f}s - {end:.1f}s: {text}")

    content = "\n".join(lines)

    stem = item["audio_filename"].rsplit(".", 1)[0]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    filename = f"{safe_name}.txt"

    return Response(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{transcript_id}")
async def delete_transcript(
    transcript_id: int,
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    tdb = _get_transcript_db(settings)
    deleted = tdb.delete(transcript_id, user_id=user_id)
    if not deleted:
        return JSONResponse(status_code=404, detail="Transcription not found")
    return {"status": "deleted", "id": transcript_id}
