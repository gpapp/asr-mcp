import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from asr_mcp.api.schemas import (
    DiarizeRequest, DiarizeResponse, DiarizeResult,
    TranscribeResponse, TranscribeResult,
)
from asr_mcp.api.security import verify_api_key, get_current_user
from asr_mcp.api.auth import get_session_user
from asr_mcp.config.settings import Settings, get_settings
from asr_mcp.speaker.vad import split_at_energy_dips

logger = logging.getLogger("asr_mcp.api.asr_router")
router = APIRouter(prefix="/asr", tags=["ASR"])


def _load_known_speakers(settings, user_id: str) -> dict:
    """Load all stored voiceprints from DB for speaker matching."""
    try:
        from asr_mcp.db.manager import DatabaseManager, VoiceprintDB
        db = DatabaseManager(settings.db_path)
        vp_db = VoiceprintDB(db)
        all_vps = vp_db.list_all(user_id=user_id)
        if all_vps:
            logger.info("Loaded %d known voiceprints for user %s", len(all_vps), user_id)
        return all_vps
    except Exception as e:
        logger.warning("Failed to load known voiceprints: %s", e)
        return {}


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
    known_speakers = req.known_speakers or _load_known_speakers(settings, user_id)
    result = await diarizer.run(
        audio_path=req.wav_path,
        num_speakers=req.num_speakers,
        diarization_threshold=req.diarization_threshold,
        vad_threshold=req.vad_threshold,
        known_speakers=known_speakers,
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
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import convert_to_wav

    settings = get_settings()

    try:
        wav_path = convert_to_wav(str(tmp_path), tmp_dir)
    except Exception as e:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"detail": f"Audio conversion failed: {e}"})

    queue = asyncio.Queue()

    async def run_diarize():
        try:
            diarizer = Diarizer(state, settings)
            known_speakers = _load_known_speakers(settings, user_id)

            async def progress_cb(evt):
                await queue.put(evt)

            result = await diarizer.run(
                audio_path=str(wav_path),
                num_speakers=num_speakers,
                known_speakers=known_speakers or None,
                progress_callback=progress_cb,
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

            await queue.put({"stage": "done", "progress": 1.0, "result": result})
        except Exception as e:
            logger.error("Diarize failed: %s", e)
            await queue.put({"stage": "error", "error": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_diarize())

    async def event_stream():
        import shutil
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/transcribe")
async def transcribe_endpoint(
    req: DiarizeRequest,
    request: Request,
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
    user_id = get_session_user(request) or "default"
    known_speakers = req.known_speakers or _load_known_speakers(settings, user_id)
    diarization = await diarizer.run(
        audio_path=req.wav_path, num_speakers=req.num_speakers,
        known_speakers=known_speakers,
    )

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

        speaker_name = seg.get("speaker", "UNKNOWN")
        seg_start = seg["start"]
        seg_end = seg["end"]

        max_samples = int(30.0 * 16000)
        if len(chunk) > max_samples:
            sub_segments = split_at_energy_dips(
                [{"start": 0, "end": len(chunk)}],
                chunk, sample_rate=16000,
                min_segment_dur=5.0, dip_ratio=0.35, min_dip_dur=0.3,
                min_split_piece=3.0,
            )
            buffer = []
            buffer_dur = 0.0
            for i, sub in enumerate(sub_segments):
                sub_len = sub["end"] - sub["start"]
                if buffer:
                    prev_end = buffer[-1]["end"]
                    gap = sub["start"] - prev_end
                    if gap > int(0.5 * 16000) and buffer_dur > 0:
                        buf_audio = np.concatenate([
                            audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                        ])
                        if len(buf_audio) >= 1600:
                            result = transcribe_audio_sync(audio=buf_audio)
                            results.append(TranscribeResult(
                                text=result.get("text", ""),
                                segments=result.get("segments"),
                                start=seg_start, end=seg_end, speaker=speaker_name,
                                audio_duration_sec=result.get("audio_duration_sec", 0),
                                inference_time_sec=result.get("inference_time_sec", 0),
                                tokens_generated=result.get("tokens_generated", 0),
                            ))
                        buffer = []
                        buffer_dur = 0.0
                if buffer_dur + sub_len > max_samples and buffer:
                    buf_audio = np.concatenate([
                        audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                    ])
                    if len(buf_audio) >= 1600:
                        result = transcribe_audio_sync(audio=buf_audio)
                        results.append(TranscribeResult(
                            text=result.get("text", ""),
                            segments=result.get("segments"),
                            start=seg_start, end=seg_end, speaker=speaker_name,
                            audio_duration_sec=result.get("audio_duration_sec", 0),
                            inference_time_sec=result.get("inference_time_sec", 0),
                            tokens_generated=result.get("tokens_generated", 0),
                        ))
                    buffer = []
                    buffer_dur = 0.0
                buffer.append(sub)
                buffer_dur += sub_len
            if buffer:
                buf_audio = np.concatenate([
                    audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                ])
                if len(buf_audio) >= 1600:
                    result = transcribe_audio_sync(audio=buf_audio)
                    results.append(TranscribeResult(
                        text=result.get("text", ""),
                        segments=result.get("segments"),
                        start=seg_start, end=seg_end, speaker=speaker_name,
                        audio_duration_sec=result.get("audio_duration_sec", 0),
                        inference_time_sec=result.get("inference_time_sec", 0),
                        tokens_generated=result.get("tokens_generated", 0),
                    ))
        else:
            result = transcribe_audio_sync(audio=chunk)
            results.append(TranscribeResult(
                text=result.get("text", ""),
                segments=result.get("segments"),
                start=seg_start, end=seg_end, speaker=speaker_name,
                audio_duration_sec=result.get("audio_duration_sec", 0),
                inference_time_sec=result.get("inference_time_sec", 0),
                tokens_generated=result.get("tokens_generated", 0),
            ))

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

    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync, _compute_mel_spectrogram_fast
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    if not state.is_ready:
        state.ensure_ready()
    if not state.is_ready:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=503, content={"detail": "Models not loaded. CUDA GPU required."})

    queue = asyncio.Queue()

    async def run_transcribe():
        try:
            diarizer = Diarizer(state, settings)
            known_speakers = _load_known_speakers(settings, user_id)

            async def progress_cb(evt):
                evt.setdefault("phase", "diarization")
                await queue.put(evt)

            await queue.put({"stage": "Running diarization", "progress": 0.0, "phase": "diarization"})
            diarization = await diarizer.run(
                audio_path=str(wav_path), num_speakers=num_speakers,
                known_speakers=known_speakers or None,
                progress_callback=progress_cb,
            )

            waveform, sr = load_audio(str(wav_path))
            audio_np = waveform.numpy().squeeze().astype(np.float32)

            segments = diarization.get("segments", [])
            audio_dur = diarization.get("audio_duration_sec", 0.0)

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

            await queue.put({
                "stage": "diarization_complete",
                "progress": 1.0,
                "phase": "diarization",
                "segments": segments,
                "audio_duration_sec": audio_dur,
                "total_speakers": diarization.get("total_speakers", 0),
            })

            if not segments:
                await queue.put({"stage": "Transcribing audio", "progress": 0.0, "phase": "transcription"})
                mel = _compute_mel_spectrogram_fast(audio_np)
                result = transcribe_audio_sync(mel_spectrogram=mel)
                diarization["results"] = [TranscribeResult(**result).__dict__]
                await queue.put({"stage": "done", "progress": 1.0, "result": diarization})
                return

            await queue.put({"stage": "Transcribing audio", "progress": 0.0, "phase": "transcription"})
            results = []
            total_segs = len(segments)
            for seg_idx, seg in enumerate(segments):
                p = seg_idx / max(total_segs, 1)
                await queue.put({
                    "stage": f"Transcribing segment {seg_idx+1}/{total_segs}",
                    "progress": p,
                    "phase": "transcription",
                    "segment_index": seg_idx,
                    "total_segments": total_segs,
                    "segment_speaker": seg.get("speaker", "UNKNOWN"),
                    "segment_start": seg.get("start", 0),
                    "segment_end": seg.get("end", 0),
                })
                start_sample = int(seg["start"] * 16000)
                end_sample = int(seg["end"] * 16000)
                chunk = audio_np[start_sample:end_sample]
                if len(chunk) < 1600:
                    continue

                speaker_name = seg.get("speaker", "UNKNOWN")
                seg_start = seg["start"]
                seg_end = seg["end"]

                max_samples = int(30.0 * 16000)
                if len(chunk) > max_samples:
                    sub_segments = split_at_energy_dips(
                        [{"start": 0, "end": len(chunk)}],
                        chunk, sample_rate=16000,
                        min_segment_dur=5.0, dip_ratio=0.35, min_dip_dur=0.3,
                        min_split_piece=3.0,
                    )
                    buffer = []
                    buffer_dur = 0.0
                    for i, sub in enumerate(sub_segments):
                        sub_len = sub["end"] - sub["start"]
                        if buffer:
                            prev_end = buffer[-1]["end"]
                            gap = sub["start"] - prev_end
                            if gap > int(0.5 * 16000) and buffer_dur > 0:
                                buf_audio = np.concatenate([
                                    audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                                ])
                                if len(buf_audio) >= 1600:
                                    tr = transcribe_audio_sync(audio=buf_audio)
                                    results.append(TranscribeResult(
                                        text=tr.get("text", ""),
                                        segments=tr.get("segments"),
                                        start=seg_start, end=seg_end, speaker=speaker_name,
                                        audio_duration_sec=tr.get("audio_duration_sec", 0),
                                        inference_time_sec=tr.get("inference_time_sec", 0),
                                        tokens_generated=tr.get("tokens_generated", 0),
                                    ))
                                buffer = []
                                buffer_dur = 0.0
                        if buffer_dur + sub_len > max_samples and buffer:
                            buf_audio = np.concatenate([
                                audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                            ])
                            if len(buf_audio) >= 1600:
                                tr = transcribe_audio_sync(audio=buf_audio)
                                results.append(TranscribeResult(
                                    text=tr.get("text", ""),
                                    segments=tr.get("segments"),
                                    start=seg_start, end=seg_end, speaker=speaker_name,
                                    audio_duration_sec=tr.get("audio_duration_sec", 0),
                                    inference_time_sec=tr.get("inference_time_sec", 0),
                                    tokens_generated=tr.get("tokens_generated", 0),
                                ))
                            buffer = []
                            buffer_dur = 0.0
                        buffer.append(sub)
                        buffer_dur += sub_len
                    if buffer:
                        buf_audio = np.concatenate([
                            audio_np[start_sample + b["start"]:start_sample + b["end"]] for b in buffer
                        ])
                        if len(buf_audio) >= 1600:
                            tr = transcribe_audio_sync(audio=buf_audio)
                            results.append(TranscribeResult(
                                text=tr.get("text", ""),
                                segments=tr.get("segments"),
                                start=seg_start, end=seg_end, speaker=speaker_name,
                                audio_duration_sec=tr.get("audio_duration_sec", 0),
                                inference_time_sec=tr.get("inference_time_sec", 0),
                                tokens_generated=tr.get("tokens_generated", 0),
                            ))
                else:
                    tr = transcribe_audio_sync(audio=chunk)
                    results.append(TranscribeResult(
                        text=tr.get("text", ""),
                        segments=tr.get("segments"),
                        start=seg_start, end=seg_end, speaker=speaker_name,
                        audio_duration_sec=tr.get("audio_duration_sec", 0),
                        inference_time_sec=tr.get("inference_time_sec", 0),
                        tokens_generated=tr.get("tokens_generated", 0),
                    ))

            diarization["results"] = [r.__dict__ if hasattr(r, '__dict__') else r for r in results]
            diarization["total_time_sec"] = sum(
                (r.inference_time_sec if hasattr(r, 'inference_time_sec') else 0) for r in results
            )

            try:
                import hashlib as _hl
                file_hash = _hl.sha256(content).hexdigest()[:16]
                from asr_mcp.db.manager import DatabaseManager, TranscriptDB
                db = DatabaseManager(settings.db_path)
                tdb = TranscriptDB(db)
                tdb.save(
                    audio_filename=file.filename,
                    result=diarization,
                    user_id=user_id,
                    file_hash=file_hash,
                    total_speakers=diarization.get("total_speakers", 0),
                    audio_duration_sec=diarization.get("audio_duration_sec", 0.0),
                    processing_time_sec=diarization.get("total_time_sec", 0.0),
                )
            except Exception as e:
                logger.warning("Failed to save transcription: %s", e)

            await queue.put({"stage": "done", "progress": 1.0, "result": diarization})
        except Exception as e:
            logger.error("Transcribe failed: %s", e)
            await queue.put({"stage": "error", "error": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_transcribe())

    async def event_stream():
        import shutil
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
