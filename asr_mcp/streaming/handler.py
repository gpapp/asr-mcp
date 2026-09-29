"""Live (WebSocket) dual-channel transcription.

Wire format
-----------
Each binary frame is::

    struct.pack("<II", channel, sequence) + int16 LE PCM (16 kHz, mono)

* ``channel`` — ``0`` = microphone, anything else = speaker (room/channel).
* ``sequence`` — opaque monotonic counter.  No in-repo client defines its
  unit, so it is used only for diagnostics (gap/reorder detection); turn
  timestamps are always derived from the *sample count*, never from wall clock.

Design notes (see AGENTS.md lesson 31)
--------------------------------------
* The VAD/turn detector is fed the **mic channel only** — a loud speaker
  channel must not open, extend or close a mic turn.
* The receiver never runs ASR inline.  Completed turns go into a single
  bounded worker queue drained by one ASR worker thread, so packets keep
  arriving (and keep being buffered) while a previous utterance is decoded.
  Queue overflow drops the *oldest* pending turn and is reported.
* An open turn is flushed on disconnect, exactly once.
* Speaker identity follows the shared uncertainty policy
  (``asr_mcp.speaker.uncertainty``): an unattributed utterance is emitted with
  ``speaker=None`` and ``uncertain=True`` — never a nearest-speaker guess.
"""

import asyncio
import logging
import struct
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from asr_mcp.streaming.turn_detector import Turn, TurnDetector, config as detector_config

logger = logging.getLogger("asr_mcp.streaming.handler")

HEADER_SIZE = 8
CHANNEL_MIC = 0
_HEADER = struct.Struct("<II")


def _as_dict(msg: dict) -> dict:
    if msg.get("type") == "error":
        return msg
    msg.setdefault("speaker", None)
    msg.setdefault("speaker_source", "unknown")
    msg.setdefault("speaker_confidence", 0.0)
    msg.setdefault("uncertain", True)
    msg.setdefault("attribution_reason", "live_turn_unattributed")
    return msg


def _speaker_for_live_turn(turn: Turn) -> Optional[str]:
    """Live turns have no diarization evidence -> always an explicit unknown.

    Previously this returned the literal string ``"SPEAKER"`` for every
    utterance, which read as a confident identity in clients.
    """
    return None


def _turn_message(turn: Turn, text: str, inference_sec: float, tokens: int) -> dict:
    return {
        "type": "transcript",
        "speaker": _speaker_for_live_turn(turn),
        "text": text,
        "start": round(turn.start_sec, 2),
        "end": round(turn.end_sec, 2),
        "duration": round(turn.duration_sec, 2),
        "rms": round(turn.mean_rms, 5),
        "end_reason": turn.reason,
        "inference_time_sec": round(inference_sec, 3),
        "tokens_generated": tokens,
    }


async def handle_ws_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket stream connected")

    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync

    state.ensure_ready()

    cfg = detector_config()
    queue_size = int(cfg.get("queue_size", 8))
    drop_on_overflow = bool(cfg.get("drop_on_overflow", True))

    detector = TurnDetector()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    stats = {
        "packets": 0,
        "malformed": 0,
        "mic_packets": 0,
        "speaker_packets": 0,
        "queued_turns": 0,
        "dropped_turns": 0,
    }

    def _submit(turn: Turn) -> None:
        stats["queued_turns"] += 1
        try:
            queue.put_nowait(turn)
            return
        except asyncio.QueueFull:
            pass
        if not drop_on_overflow:
            stats["dropped_turns"] += 1
            logger.warning("ASR queue full (%d) — turn dropped", queue_size)
            return
        # Backpressure policy: drop the OLDEST pending turn.  The newest audio
        # is the most relevant to the live listener, and the receiver keeps
        # buffering while the decoder catches up.
        dropped = None
        try:
            dropped = queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - race with the worker
            pass
        stats["dropped_turns"] += 1
        logger.warning(
            "ASR queue full (%d) — dropped turn %.2f-%.2fs (decoder cannot keep up)",
            queue_size,
            dropped.start_sec if dropped else -1.0,
            dropped.end_sec if dropped else -1.0,
        )
        try:
            queue.put_nowait(turn)
        except asyncio.QueueFull:  # pragma: no cover - race with the worker
            stats["dropped_turns"] += 1

    def _transcribe_turn(turn: Turn) -> dict:
        state.touch()
        result = transcribe_audio_sync(
            audio=turn.audio,
            # pre_segmented: no second VAD pass, no previous-text carry-over
            # across turns (each turn is a separate speaker utterance).
            pre_segmented=True,
        )
        text = (result.get("text") or "").strip()
        if not text:
            return {
                "type": "empty",
                "start": round(turn.start_sec, 2),
                "end": round(turn.end_sec, 2),
                "duration": round(turn.duration_sec, 2),
            }
        return _as_dict(_turn_message(
            turn, text,
            result.get("inference_time_sec", 0.0),
            result.get("tokens_generated", 0),
        ))

    async def _asr_worker() -> None:
        """Single decoder: turns are transcribed strictly one at a time.

        Serialising here is what prevents turn mixing — a shared model decoded
        from two threads would interleave KV caches / arena state.
        """
        while True:
            turn = await queue.get()
            if turn is None:
                queue.task_done()
                return
            try:
                msg = await loop.run_in_executor(None, _transcribe_turn, turn)
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Utterance processing failed: %s", e)
                msg = {"type": "error", "message": str(e), "start": round(turn.start_sec, 2)}
            try:
                await websocket.send_json(msg)
            except Exception:
                logger.debug("Could not send transcript (socket closing)")
            finally:
                queue.task_done()

    worker = asyncio.create_task(_asr_worker())

    # ── Receiver loop ──────────────────────────────────────────────────
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                # Text/control frame: not audio, but it must not kill the
                # connection the way receive_bytes() assertions used to.
                stats["malformed"] += 1
                logger.warning("Dropped non-binary frame (type=%s)", msg.get("type"))
                continue
            stats["packets"] += 1

            # Validate BEFORE unpacking: struct.unpack("<II") raises on short
            # buffers, which used to kill the whole connection on a 4-byte
            # frame.
            if len(data) < HEADER_SIZE:
                stats["malformed"] += 1
                logger.warning(
                    "Dropped malformed packet: %d bytes (need >= %d)",
                    len(data), HEADER_SIZE,
                )
                continue

            channel, sequence = _HEADER.unpack_from(data, 0)
            chunk = data[HEADER_SIZE:]

            if channel != CHANNEL_MIC:
                # Speaker channel is recorded/attributed independently; it must
                # never drive the mic turn state machine.
                stats["speaker_packets"] += 1
                continue

            stats["mic_packets"] += 1
            for turn in detector.feed(chunk):
                _submit(turn)

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        worker.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
        return

    # Flush an in-flight turn so the last words are not lost, then wait for
    # the decoder to drain before closing.
    tail = detector.flush()
    if tail is not None:
        _submit(tail)
    try:
        await asyncio.wait_for(queue.join(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Timed out with %d turn(s) still pending", queue.qsize())
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:  # pragma: no cover - only after a join timeout
        logger.warning("Could not signal worker shutdown (queue still full)")
    try:
        await asyncio.wait_for(worker, timeout=5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        logger.warning("ASR worker did not stop cleanly")

    logger.info("WebSocket stream stats: %s", stats)
