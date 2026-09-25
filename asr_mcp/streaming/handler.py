import asyncio
import json
import logging
import struct
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("asr_mcp.streaming.handler")

SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2  # int16


def _rms(chunk_bytes: bytes) -> float:
    samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _has_speech(audio: np.ndarray, threshold: float = 0.02) -> bool:
    if isinstance(audio, np.ndarray):
        return float(np.sqrt(np.mean(audio ** 2))) > threshold
    return False


async def handle_ws_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket stream connected")

    from asr_mcp.core.model_state import state
    from asr_mcp.config.settings import get_settings
    from asr_mcp.speaker.embedding import extract_embedding
    from asr_mcp.speaker.matcher import find_best_match
    from asr_mcp.core.transcriber import transcribe_audio_sync

    settings = get_settings()
    state.ensure_ready()
    mic_buffer = bytearray()
    speaker_buffer = bytearray()
    mic_audio_chunks = []
    speaker_audio_chunks = []

    recording_start = time.time()
    speech_active = False
    speech_start = 0.0

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 4:
                continue

            header = struct.unpack("<II", data[:8])
            channel = "mic" if header[0] == 0 else "speaker"
            chunk = data[8:]

            now = time.time() - recording_start
            rms = _rms(chunk)

            if channel == "mic":
                mic_audio_chunks.append(chunk)
            else:
                speaker_audio_chunks.append(chunk)

            is_speech = rms > 0.02

            if is_speech and not speech_active:
                speech_active = True
                speech_start = now
            elif not is_speech and speech_active:
                speech_duration = now - speech_start
                if speech_duration > 0.5:
                    await _process_utterance(
                        websocket, mic_audio_chunks, speech_start, now,
                        state, settings, recording_start,
                    )
                speech_active = False
                mic_audio_chunks = []

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass


async def _process_utterance(
    websocket: WebSocket,
    audio_chunks: list[bytes],
    start_time: float,
    end_time: float,
    state,
    settings,
    recording_start: float,
):
    if not audio_chunks:
        return

    state.touch()
    pcm_data = b"".join(audio_chunks)
    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < SAMPLE_RATE * 0.3:
        return

    try:
        result = transcribe_audio_sync(audio=audio_np)

        text = result.get("text", "")
        if not text.strip():
            return

        await websocket.send_json({
            "type": "transcript",
            "speaker": "SPEAKER",
            "text": text,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
        })

    except Exception as e:
        logger.error("Utterance processing failed: %s", e)
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
