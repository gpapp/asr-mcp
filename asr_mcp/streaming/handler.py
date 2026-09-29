"""Live (WebSocket) dual-channel transcription.

Wire format
-----------
Two frame shapes are accepted; see :mod:`asr_mcp.streaming.protocol` for the
byte layouts.

* **Turn frames** (preferred) — the client runs its own turn detector and sends
  one already-cut utterance with its exact ``start_sample`` on the file
  timeline.  The server does no endpointing, so live boundaries and the
  offline diarization boundaries are the same ones.
* **Legacy raw PCM** — ``struct.pack("<II", channel, sequence)`` + int16 LE PCM.
  Still supported; the server cuts turns itself with
  :class:`~asr_mcp.streaming.turn_detector.TurnDetector`.

* ``channel`` — ``0`` = microphone, anything else = speaker/loopback.  The
  endpointing detector is fed the **mic channel only**: a loud loopback must
  not open, extend or close a mic turn.
* Turn timestamps always come from **sample counts**, never wall clock, so
  back-pressure cannot shift them.

Design notes (see AGENTS.md lessons 31 and 34)
----------------------------------------------
* The receiver never runs ASR inline.  Completed turns go into a single
  bounded worker queue drained by one ASR worker, so packets keep arriving
  while a previous utterance is decoded.  Overflow drops the *oldest* pending
  turn and is reported to the client as a gap.
* An open turn is flushed on disconnect, exactly once.
* Speaker identity comes from :mod:`asr_mcp.streaming.attribution` and follows
  the shared uncertainty policy: an unattributable utterance is emitted with
  ``speaker=None`` and ``uncertain=True`` — never a nearest-speaker guess.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from asr_mcp.streaming import protocol as proto
from asr_mcp.streaming.attribution import CHANNEL_MIC, attribute_live_turn
from asr_mcp.streaming.turn_detector import Turn, TurnDetector, config as detector_config

logger = logging.getLogger("asr_mcp.streaming.handler")

HEADER_SIZE = proto.RAW_HEADER_SIZE
CHANNEL_MIC = CHANNEL_MIC  # re-export for backwards compatibility
_HEADER = proto.RAW_HEADER


def _as_dict(msg: dict) -> dict:
    if msg.get("type") == "error":
        return msg
    msg.setdefault("speaker", None)
    msg.setdefault("speaker_source", "unknown")
    msg.setdefault("speaker_confidence", 0.0)
    msg.setdefault("uncertain", True)
    msg.setdefault("attribution_reason", "live_turn_unattributed")
    return msg


def _turn_message(turn: Turn, text: str, inference_sec: float, tokens: int,
                  attribution: Optional[dict] = None) -> dict:
    msg = {
        "type": "transcript",
        "speaker": None,
        "text": text,
        "start": round(turn.start_sec, 2),
        "end": round(turn.end_sec, 2),
        "duration": round(turn.duration_sec, 2),
        "rms": round(turn.mean_rms, 5),
        "end_reason": turn.reason,
        "inference_time_sec": round(inference_sec, 3),
        "tokens_generated": tokens,
    }
    if attribution:
        msg.update(attribution)
    return _as_dict(msg)


def _turn_from_frame(header: dict, pcm: bytes) -> Turn:
    """Build a Turn from a client-cut turn frame (no detector involved)."""
    import numpy as np

    start_sample = int(header["start_sample"])
    n_samples = int(header["n_samples"])
    samples = np.frombuffer(pcm, dtype=np.int16)
    return Turn(
        start_sample=start_sample,
        end_sample=start_sample + n_samples,
        audio=pcm,
        reason="client_turn",
        peak_rms=0.0,
        mean_rms=0.0,
    )


async def handle_ws_stream(websocket: WebSocket, language: str = "auto",
                           user_id: Optional[str] = None):
    await websocket.accept()
    logger.info("WebSocket stream connected (language=%s, user=%s)", language, user_id)

    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync

    state.ensure_ready()

    cfg = detector_config()
    queue_size = int(cfg.get("queue_size", 8))
    drop_on_overflow = bool(cfg.get("drop_on_overflow", True))

    voiceprints = None
    embed_fn = pitch_fn = energy_fn = None
    try:
        from asr_mcp.api.asr_router import _load_known_speakers
        from asr_mcp.config.settings import get_settings

        voiceprints = _load_known_speakers(get_settings(), user_id or "default")
    except Exception as e:
        # Voiceprints are optional: without them speaker-channel turns are
        # reported as UNKNOWN rather than failing the session.
        logger.warning("No voiceprints available for live attribution: %s", e)
        voiceprints = None

    if voiceprints:
        try:
            import numpy as np
            import torch
            from asr_mcp.speaker.embedding import (
                compute_energy, compute_pitch, extract_embedding,
            )

            def _waveform(turn):
                samples = np.frombuffer(turn.audio, dtype=np.int16)
                return torch.from_numpy(samples.astype("float32") / 32768.0)

            def embed_fn(turn):
                # sample_rate is REQUIRED: extract_embedding takes it
                # positionally, and omitting it raises TypeError inside
                # attribute_live_turn, which reports "live_match_failed" and
                # silently downgrades every speaker turn to UNKNOWN.
                return extract_embedding(_waveform(turn), 16000)

            def pitch_fn(turn):
                return compute_pitch(_waveform(turn), 16000)[0]

            def energy_fn(turn):
                return compute_energy(_waveform(turn))

        except Exception as e:
            logger.warning("Live embedding hooks unavailable: %s", e)
            voiceprints = None

    detector = TurnDetector()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    stats = {
        "packets": 0,
        "malformed": 0,
        "mic_packets": 0,
        "speaker_packets": 0,
        "turn_frames": 0,
        "raw_frames": 0,
        "queued_turns": 0,
        "dropped_turns": 0,
    }
    # Highest audio sample seen, whichever path produced it.  Reported to the
    # client so it knows how much of its recording the server actually covered.
    stream_end_sample = {"value": 0}

    def _note_end(turn: Turn) -> None:
        stream_end_sample["value"] = max(
            stream_end_sample["value"], int(turn.end_sample)
        )

    def _submit(turn: Turn, channel: int) -> None:
        stats["queued_turns"] += 1
        _note_end(turn)
        try:
            queue.put_nowait((turn, channel))
            return
        except asyncio.QueueFull:
            pass
        if not drop_on_overflow:
            stats["dropped_turns"] += 1
            logger.warning("ASR queue full (%d) — turn dropped", queue_size)
            _notify_gap(turn, None)
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
            dropped[0].start_sec if dropped else -1.0,
            dropped[0].end_sec if dropped else -1.0,
        )
        # The client is told explicitly, so the saved .asr.json sidecar can
        # record the hole instead of silently returning a shorter transcript.
        if dropped:
            loop.create_task(_notify_gap(dropped[0], dropped[1]))
        try:
            queue.put_nowait((turn, channel))
        except asyncio.QueueFull:  # pragma: no cover - race with the worker
            stats["dropped_turns"] += 1
            _notify_gap(turn, None)

    async def _notify_gap(turn: Turn, channel: int) -> None:
        try:
            await websocket.send_json({
                "type": "dropped",
                "start": round(turn.start_sec, 2),
                "end": round(turn.end_sec, 2),
                "channel": int(channel) if channel is not None else None,
                "reason": "queue_overflow",
            })
        except Exception:
            logger.debug("Could not notify client of dropped turn")

    def _transcribe_turn(turn: Turn, channel: int) -> dict:
        state.touch()
        result = transcribe_audio_sync(
            audio=turn.audio,
            # pre_segmented: no second VAD pass, no previous-text carry-over
            # across turns (each turn is a separate speaker utterance).
            pre_segmented=True,
            language=language,
        )
        attribution = attribute_live_turn(
            turn, channel, voiceprints=voiceprints, embed_fn=embed_fn,
            pitch_fn=pitch_fn, energy_fn=energy_fn,
        )
        text = (result.get("text") or "").strip()
        if not text:
            return {
                "type": "empty",
                "start": round(turn.start_sec, 2),
                "end": round(turn.end_sec, 2),
                "duration": round(turn.duration_sec, 2),
                "channel": int(channel),
                **attribution,
            }
        msg = _turn_message(
            turn, text,
            result.get("inference_time_sec", 0.0),
            result.get("tokens_generated", 0),
            attribution=attribution,
        )
        msg["channel"] = int(channel)
        return msg

    async def _asr_worker() -> None:
        """Single decoder: turns are transcribed strictly one at a time.

        Serialising here is what prevents turn mixing — a shared model decoded
        from two threads would interleave KV caches / arena state.
        """
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            turn, channel = item
            try:
                msg = await loop.run_in_executor(
                    None, _transcribe_turn, turn, channel,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Utterance processing failed: %s", e)
                msg = {"type": "error", "message": str(e),
                       "start": round(turn.start_sec, 2)}
            try:
                await websocket.send_json(msg)
            except Exception:
                logger.debug("Could not send transcript (socket closing)")
            finally:
                queue.task_done()

    worker = asyncio.create_task(_asr_worker())

    # ── Receiver loop ──────────────────────────────────────────────────
    remote_flushed = False
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                text = msg.get("text")
                if text is not None:
                    _handle_control_text(text, stats)
                    continue
                # Not audio, but it must not kill the connection the way
                # receive_bytes() assertions used to.
                stats["malformed"] += 1
                logger.warning("Dropped non-binary frame (type=%s)", msg.get("type"))
                continue
            stats["packets"] += 1

            # ── Turn frames (client-side endpointing) ──
            if proto.is_turn_frame(data):
                stats["turn_frames"] += 1
                try:
                    header, pcm = proto.unpack_turn(data)
                except ValueError as e:
                    stats["malformed"] += 1
                    logger.warning("Malformed turn frame: %s", e)
                    continue
                if header["msg_type"] == proto.MSG_FLUSH:
                    remote_flushed = True
                    logger.info("Client requested flush after %d turn frames",
                                stats["turn_frames"])
                    break
                if header["msg_type"] != proto.MSG_TURN:
                    stats["malformed"] += 1
                    logger.warning("Unknown turn msg_type=%s", header["msg_type"])
                    continue
                channel = header["channel"]
                if channel == CHANNEL_MIC:
                    stats["mic_packets"] += 1
                else:
                    stats["speaker_packets"] += 1
                try:
                    turn = _turn_from_frame(header, pcm)
                except Exception as e:
                    stats["malformed"] += 1
                    logger.warning("Could not build turn from frame: %s", e)
                    continue
                _submit(turn, channel)
                continue

            # ── Legacy raw PCM (server-side endpointing) ──
            stats["raw_frames"] += 1
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

            channel, _sequence = _HEADER.unpack_from(data, 0)
            chunk = data[HEADER_SIZE:]

            if channel != CHANNEL_MIC:
                # Speaker channel is recorded/attributed independently; it must
                # never drive the mic turn state machine.
                stats["speaker_packets"] += 1
                continue

            stats["mic_packets"] += 1
            for turn in detector.feed(chunk):
                _submit(turn, CHANNEL_MIC)

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
    if not remote_flushed:
        tail = detector.flush()
        if tail is not None:
            _submit(tail, CHANNEL_MIC)
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

    try:
        await websocket.send_json({
            "type": "stats",
            **stats,
            "covered_sec": round(
                stream_end_sample["value"] / proto.SAMPLE_RATE, 2
            ),
        })
    except Exception:
        pass

    logger.info("WebSocket stream stats: %s", stats)


def _handle_control_text(text: str, stats: dict) -> None:
    """Handle a JSON control message from the client."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        stats["malformed"] += 1
        logger.warning("Dropped non-JSON control frame")
        return
    if not isinstance(msg, dict):
        stats["malformed"] += 1
        return
    kind = msg.get("type")
    if kind == "ping":
        logger.debug("Client ping")
    elif kind == "hello":
        logger.info("Client hello: %s", {k: v for k, v in msg.items() if k != "type"})
    else:
        stats["malformed"] += 1
        logger.warning("Unknown control message type=%r", kind)
