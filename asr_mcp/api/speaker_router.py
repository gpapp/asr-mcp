import logging
import tempfile
from pathlib import Path

import numpy as np
import torchaudio
from fastapi import APIRouter, Depends, File, UploadFile

from asr_mcp.api.schemas import (
    SpeakerRegisterRequest, SpeakerIdentifyRequest,
    SpeakerIdentifyResponse, SpeakerInfoResponse, SpeakerListResponse,
)
from asr_mcp.api.security import verify_api_key, validate_upload_filename
from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.speaker_router")
router = APIRouter(prefix="/speaker", tags=["Speaker"])


@router.post("/register")
async def register_speaker(
    req: SpeakerRegisterRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.service import SpeakerService
    from asr_mcp.db.manager import DatabaseManager

    state.ensure_diarize_ready()
    db = DatabaseManager(settings.db_path)
    service = SpeakerService(settings.data_dir, db)
    service._db._db = db

    if req.wav_path and req.start_sec is not None and req.end_sec is not None:
        result = service.register_from_audio(
            name=req.name, wav_path=req.wav_path,
            start_sec=req.start_sec, end_sec=req.end_sec,
        )
    else:
        return {"error": "Provide wav_path with start_sec and end_sec"}

    return result


@router.post("/register/upload")
async def register_speaker_upload(
    name: str,
    file: UploadFile = File(...),
    start_sec: float = 0.0,
    end_sec: float = None,
    _: str = Depends(verify_api_key),
):
    settings = get_settings()
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.speaker.service import SpeakerService

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return {"error": "File too large"}

    validate_upload_filename(file.filename)

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.voiceprint.utils import load_audio
    waveform, sr = load_audio(str(tmp_path))

    if end_sec is None:
        end_sec = waveform.shape[-1] / 16000

    start_sample = int(start_sec * 16000)
    end_sample = int(end_sec * 16000)
    chunk = waveform[..., start_sample:end_sample]

    from asr_mcp.speaker.embedding import extract_embedding, compute_pitch, compute_energy
    from asr_mcp.core.model_state import state

    state.ensure_diarize_ready()
    embedding = extract_embedding(chunk, 16000, state.embedding_session)
    pitch_hz, pitch_std = compute_pitch(chunk, 16000)
    energy_rms = compute_energy(chunk)

    db = DatabaseManager(settings.db_path)
    service = SpeakerService(settings.data_dir, db)
    result = service.register_speaker(
        name=name, embedding=embedding,
        pitch_hz=pitch_hz, pitch_std=pitch_std,
        energy_rms=energy_rms,
        total_speech_sec=(end_sec - start_sec),
        sample_count=1,
    )

    try:
        tmp_path.unlink()
        tmp_dir.rmdir()
    except Exception:
        pass

    return result


@router.post("/identify", response_model=list[SpeakerIdentifyResponse])
async def identify_speaker(
    req: SpeakerIdentifyRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.embedding import extract_embedding
    from asr_mcp.voiceprint.utils import load_audio_segment
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.speaker.service import SpeakerService

    state.ensure_diarize_ready()
    waveform, sr = load_audio_segment(req.wav_path, req.start_sec, req.end_sec or 99999)
    embedding = extract_embedding(waveform, 16000, state.embedding_session)

    db = DatabaseManager(settings.db_path)
    service = SpeakerService(settings.data_dir, db)
    results = service.identify_speaker(embedding, top_k=req.top_k)

    return [SpeakerIdentifyResponse(**r) for r in results]


@router.get("/list", response_model=SpeakerListResponse)
async def list_speakers(
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.speaker.service import SpeakerService

    db = DatabaseManager(settings.db_path)
    service = SpeakerService(settings.data_dir, db)
    speakers = service.list_speakers()

    response_speakers = {}
    for name, info in speakers.items():
        response_speakers[name] = SpeakerInfoResponse(
            name=name,
            pitch_hz=info.get("pitch_hz", 0),
            pitch_std=info.get("pitch_std", 0),
            energy_rms=info.get("energy_rms", 0),
            total_speech_sec=info.get("total_speech_sec", 0),
            sample_count=info.get("sample_count", 0),
        )

    return SpeakerListResponse(speakers=response_speakers, count=len(response_speakers))


@router.delete("/{speaker_name}")
async def delete_speaker(
    speaker_name: str,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.speaker.service import SpeakerService

    db = DatabaseManager(settings.db_path)
    service = SpeakerService(settings.data_dir, db)
    deleted = service.remove_speaker(speaker_name)
    if deleted:
        return {"status": "deleted", "name": speaker_name}
    return {"error": f"Speaker '{speaker_name}' not found"}
