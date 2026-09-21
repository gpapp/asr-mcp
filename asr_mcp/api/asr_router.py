import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from asr_mcp.api.schemas import (
    DiarizeRequest, DiarizeResponse, DiarizeResult,
    TranscribeResponse, TranscribeResult,
)
from asr_mcp.api.security import verify_api_key, get_current_user
from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.asr_router")
router = APIRouter(prefix="/asr", tags=["ASR"])


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize_endpoint(
    req: DiarizeRequest,
    settings: Settings = Depends(get_settings),
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.diarization.pipeline import Diarizer

    state.ensure_ready()
    diarizer = Diarizer(state, settings)
    result = await diarizer.run(
        audio_path=req.wav_path,
        num_speakers=req.num_speakers,
        diarization_threshold=req.diarization_threshold,
        vad_threshold=req.vad_threshold,
        known_speakers=req.known_speakers,
    )

    if "error" in result:
        return DiarizeResponse(segments=[], total_time_sec=0, error=result["error"])

    segments = result.get("segments", [])

    try:
        from asr_mcp.db.manager import DatabaseManager
        from asr_mcp.voiceprint.service import VoiceprintService
        db = DatabaseManager(settings.db_path)
        vp_service = VoiceprintService(settings.data_dir, db)
        vp_service.set_voices_dir(settings.voices_dir)
        vp_service.set_embedding_session(state.embedding_session)
        collected = vp_service.auto_collect_from_diarization(
            audio_path=req.wav_path, segments=segments, user_id=user_id,
        )
        if collected:
            logger.info("Auto-collected %d snippets for user %s", len(collected), user_id)
    except Exception as e:
        logger.warning("Auto-collect failed: %s", e)

    return DiarizeResponse(
        segments=[DiarizeResult(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in segments],
        total_time_sec=result.get("total_time_sec", 0),
        total_speakers=result.get("total_speakers", 0),
        audio_duration_sec=result.get("audio_duration_sec", 0),
    )


@router.post("/diarize/upload")
async def diarize_upload(
    file: UploadFile = File(...),
    num_speakers: int = None,
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, detail="File too large (max 200MB)")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.core.model_state import state
    from asr_mcp.config.settings import get_settings
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import convert_to_wav

    settings = get_settings()

    wav_path = convert_to_wav(str(tmp_path), tmp_dir)
    try:
        diarizer = Diarizer(state, settings)
        result = await diarizer.run(
            audio_path=str(wav_path),
            num_speakers=num_speakers,
        )

        segments = result.get("segments", [])
        try:
            from asr_mcp.db.manager import DatabaseManager
            from asr_mcp.voiceprint.service import VoiceprintService
            db = DatabaseManager(settings.db_path)
            vp_service = VoiceprintService(settings.data_dir, db)
            vp_service.set_voices_dir(settings.voices_dir)
            vp_service.set_embedding_session(state.embedding_session)
            vp_service.auto_collect_from_diarization(
                audio_path=str(wav_path), segments=segments, user_id=user_id,
            )
        except Exception as e:
            logger.warning("Auto-collect failed: %s", e)

        return result
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/transcribe")
async def transcribe_endpoint(
    req: DiarizeRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync, _compute_mel_spectrogram_fast
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    state.ensure_ready()
    if not state.is_ready:
        return TranscribeResponse(
            results=[TranscribeResult(error="Models not loaded. CUDA GPU required.")],
            total_time_sec=0,
        )

    diarizer = Diarizer(state, settings)
    diarization = await diarizer.run(audio_path=req.wav_path, num_speakers=req.num_speakers)

    waveform, sr = load_audio(req.wav_path)
    audio_np = waveform.numpy().squeeze().astype(np.float32)

    segments = diarization.get("segments", [])
    if not segments:
        mel = _compute_mel_spectrogram_fast(audio_np)
        result = transcribe_audio_sync(mel_spectrogram=mel)
        return TranscribeResponse(
            results=[TranscribeResult(**result)],
            total_time_sec=result.get("inference_time_sec", 0),
        )

    results = []
    for seg in segments:
        start_sample = int(seg["start"] * 16000)
        end_sample = int(seg["end"] * 16000)
        chunk = audio_np[start_sample:end_sample]
        if len(chunk) < 1600:
            continue
        result = transcribe_audio_sync(audio=chunk)
        result["text"] = f"[{seg.get('speaker', 'UNKNOWN')}] {result.get('text', '')}"
        results.append(TranscribeResult(**result))

    total_time = sum(r.inference_time_sec for r in results)
    return TranscribeResponse(results=results, total_time_sec=total_time)


@router.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    num_speakers: int = None,
    settings: Settings = Depends(get_settings),
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"detail": "File too large (max 200MB)"})

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.voiceprint.utils import convert_to_wav
    wav_path = convert_to_wav(str(tmp_path), tmp_dir)

    try:
        from asr_mcp.core.model_state import state
        from asr_mcp.core.transcriber import transcribe_audio_sync, _compute_mel_spectrogram_fast
        from asr_mcp.diarization.pipeline import Diarizer
        from asr_mcp.voiceprint.utils import load_audio
        import numpy as np

        if not state.is_ready:
            state.ensure_ready()
        if not state.is_ready:
            return JSONResponse(status_code=503, content={"detail": "Models not loaded. CUDA GPU required."})

        diarizer = Diarizer(state, settings)
        diarization = await diarizer.run(audio_path=str(wav_path), num_speakers=num_speakers)

        waveform, sr = load_audio(str(wav_path))
        audio_np = waveform.numpy().squeeze().astype(np.float32)

        segments = diarization.get("segments", [])

        try:
            from asr_mcp.db.manager import DatabaseManager
            from asr_mcp.voiceprint.service import VoiceprintService
            db = DatabaseManager(settings.db_path)
            vp_service = VoiceprintService(settings.data_dir, db)
            vp_service.set_voices_dir(settings.voices_dir)
            vp_service.set_embedding_session(state.embedding_session)
            vp_service.auto_collect_from_diarization(
                audio_path=str(wav_path), segments=segments, user_id=user_id,
            )
        except Exception as e:
            logger.warning("Auto-collect failed: %s", e)

        if not segments:
            mel = _compute_mel_spectrogram_fast(audio_np)
            result = transcribe_audio_sync(mel_spectrogram=mel)
            return TranscribeResponse(
                results=[TranscribeResult(**result)],
                total_time_sec=result.get("inference_time_sec", 0),
            )

        results = []
        for seg in segments:
            start_sample = int(seg["start"] * 16000)
            end_sample = int(seg["end"] * 16000)
            chunk = audio_np[start_sample:end_sample]
            if len(chunk) < 1600:
                continue
            result = transcribe_audio_sync(audio=chunk)
            result["text"] = f"[{seg.get('speaker', 'UNKNOWN')}] {result.get('text', '')}"
            results.append(TranscribeResult(**result))

        total_time = sum(r.inference_time_sec for r in results)
        return TranscribeResponse(results=results, total_time_sec=total_time)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/stream")
async def stream_placeholder():
    from starlette.responses import JSONResponse as StarletteJSONResponse
    return StarletteJSONResponse(
        status_code=501,
        content={"detail": "Streaming is available via WebSocket at /api/asr/ws/stream"},
    )


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    from asr_mcp.streaming.handler import handle_ws_stream
    await handle_ws_stream(websocket)
