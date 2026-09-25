import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends

from asr_mcp.api.security import verify_api_key
from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.mcp_router")
router = APIRouter(prefix="/mcp", tags=["MCP"])


@router.get("/tools")
async def get_tools():
    return {
        "tools": [
            {
                "name": "diarize_audio",
                "description": "Perform speaker diarization on an audio file, identifying who spoke when",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "wav_path": {"type": "string", "description": "Path to WAV audio file"},
                        "num_speakers": {"type": "integer", "description": "Force exact speaker count (optional)"},
                        "diarization_threshold": {"type": "number", "description": "Clustering threshold (default 0.35)"},
                    },
                    "required": ["wav_path"],
                },
            },
            {
                "name": "transcribe_audio",
                "description": "Transcribe an audio file with optional speaker diarization",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "wav_path": {"type": "string", "description": "Path to WAV audio file"},
                        "num_speakers": {"type": "integer", "description": "Force exact speaker count (optional)"},
                        "language": {"type": "string", "description": "Language code (default: en)"},
                    },
                    "required": ["wav_path"],
                },
            },
            {
                "name": "identify_speaker",
                "description": "Identify a speaker from an audio segment against known voiceprints",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "wav_path": {"type": "string", "description": "Path to WAV audio file"},
                        "start_sec": {"type": "number", "description": "Start time in seconds"},
                        "end_sec": {"type": "number", "description": "End time in seconds"},
                        "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"},
                    },
                    "required": ["wav_path"],
                },
            },
            {
                "name": "register_voiceprint",
                "description": "Register a new voiceprint from an audio segment",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Speaker name"},
                        "wav_path": {"type": "string", "description": "Path to WAV audio file"},
                        "start_sec": {"type": "number", "description": "Start time in seconds"},
                        "end_sec": {"type": "number", "description": "End time in seconds"},
                    },
                    "required": ["name", "wav_path", "start_sec", "end_sec"],
                },
            },
            {
                "name": "list_speakers",
                "description": "List all registered voiceprints/speakers",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
    }


@router.get("/resources")
async def get_resources():
    return {
        "resources": [
            {
                "uri": "asr://voiceprints",
                "name": "Registered Voiceprints",
                "description": "List of all registered speaker voiceprints",
                "mimeType": "application/json",
            },
            {
                "uri": "asr://health",
                "name": "System Health",
                "description": "Current system health and model status",
                "mimeType": "application/json",
            },
        ]
    }


@router.post("/call")
async def mcp_call(
    body: dict = {},
    tool_name: Optional[str] = None,
    arguments: dict = {},
    settings: Settings = Depends(get_settings),
):
    if body:
        tool_name = body.get("tool") or body.get("tool_name") or tool_name
        arguments = body.get("arguments") or body.get("args") or arguments
    if not tool_name:
        return {"error": "tool_name is required"}
    from asr_mcp.core.model_state import state

    if tool_name in {"diarize_audio", "transcribe_audio", "identify_speaker", "register_voiceprint"}:
        state.ensure_ready()

    if tool_name == "diarize_audio":
        from asr_mcp.diarization.pipeline import Diarizer
        diarizer = Diarizer(state, settings)
        result = await diarizer.run(
            audio_path=arguments.get("wav_path"),
            num_speakers=arguments.get("num_speakers"),
            diarization_threshold=arguments.get("diarization_threshold"),
        )
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    elif tool_name == "transcribe_audio":
        from asr_mcp.core.transcriber import transcribe_audio_sync
        from asr_mcp.voiceprint.utils import load_audio
        import numpy as np

        wav_path = arguments.get("wav_path")
        waveform, sr = load_audio(wav_path)
        audio_np = waveform.numpy().squeeze().astype(np.float32)
        result = transcribe_audio_sync(audio=audio_np)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    elif tool_name == "identify_speaker":
        from asr_mcp.speaker.embedding import extract_embedding
        from asr_mcp.voiceprint.utils import load_audio_segment
        from asr_mcp.db.manager import DatabaseManager
        from asr_mcp.speaker.service import SpeakerService

        wav_path = arguments.get("wav_path")
        start = arguments.get("start_sec", 0.0)
        end = arguments.get("end_sec")
        waveform, _ = load_audio_segment(wav_path, start, end or 99999)
        embedding = extract_embedding(waveform, 16000, state.embedding_session)

        db = DatabaseManager(settings.db_path)
        service = SpeakerService(settings.data_dir, db)
        results = service.identify_speaker(embedding, top_k=arguments.get("top_k", 5))
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}

    elif tool_name == "register_voiceprint":
        from asr_mcp.voiceprint.service import VoiceprintService
        from asr_mcp.db.manager import DatabaseManager

        db = DatabaseManager(settings.db_path)
        service = VoiceprintService(settings.data_dir, db, state.embedding_session)
        result = service.register_from_audio(
            name=arguments["name"],
            wav_path=arguments["wav_path"],
            start_sec=arguments["start_sec"],
            end_sec=arguments["end_sec"],
        )
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    elif tool_name == "list_speakers":
        from asr_mcp.db.manager import DatabaseManager
        from asr_mcp.speaker.service import SpeakerService

        db = DatabaseManager(settings.db_path)
        service = SpeakerService(settings.data_dir, db)
        speakers = service.list_speakers()
        summary = {name: {"speech_sec": info.get("total_speech_sec", 0)} for name, info in speakers.items()}
        return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}

    return {"error": f"Unknown tool: {tool_name}"}


@router.get("/resource/{uri:path}")
async def read_resource(uri: str, settings: Settings = Depends(get_settings)):
    if uri == "asr://voiceprints":
        from asr_mcp.db.manager import DatabaseManager
        from asr_mcp.speaker.service import SpeakerService
        db = DatabaseManager(settings.db_path)
        service = SpeakerService(settings.data_dir, db)
        speakers = service.list_speakers()
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(speakers, indent=2, default=str)}]}

    elif uri == "asr://health":
        from asr_mcp.core.model_state import state
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps({
            "status": "healthy" if state.is_ready else "loading",
            "model_status": "ready" if state.is_ready else "not_ready",
        })}]}

    return {"error": f"Unknown resource: {uri}"}
