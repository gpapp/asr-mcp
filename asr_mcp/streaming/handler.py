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
from asr_mcp.streaming.attribution import (
    CHANNEL_MIC, attribute_live_turn, unattributed as _unknown_attribution,
)
from asr_mcp.streaming.speech_gate import (
    probe_speech,
    trim_turn_edges,
    turn_has_speech,
)
from asr_mcp.streaming.turn_detector import (
    Turn, TurnCoalescer, TurnDetector, config as detector_config,
    samples_as_float32,
)

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


def _err(exc: Exception, limit: int = 300) -> str:
    """A bounded one-line description of an exception.

    A backend that was handed the wrong type embeds the whole offending
    payload in its message, so an untruncated ``logger.error("...: %s", e)``
    wrote megabytes of hex per failed turn.
    """
    detail = str(exc)
    if len(detail) > limit:
        detail = f"{detail[:limit]}... ({len(detail)} chars total)"
    return detail


def _turn_from_frame(header: dict, pcm: bytes) -> Turn:
    """Build a Turn from a client-cut turn frame (no detector involved).

    ``header["start_sample"]`` is the client's own claim about its recording
    timeline. It is validated by :class:`~asr_mcp.streaming.protocol.TimelineGuard`
    before it gets here (and clamped when it regressed), so this copy is of an
    already-checked value -- see :func:`_check_timeline`.
    """
    import numpy as np

    start_sample = int(header["start_sample"])
    n_samples = int(header["n_samples"])
    # Turn.audio is a float32 numpy array in [-1, 1] everywhere else (see
    # turn_detector._close_turn). Passing the raw bytes through instead made
    # every single live turn die in the ASR backend with
    # "could not convert string to float: b'\\x00\\x00...'", and the exception
    # text carried the whole PCM payload into the log.
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return Turn(
        start_sample=start_sample,
        end_sample=start_sample + n_samples,
        audio=samples,
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
                # NOT np.frombuffer(turn.audio, dtype=np.int16): the server's
                # Turn.audio is already a float32 array, and reinterpreting it
                # as int16 doubles the length and feeds ECAPA the IEEE-754 bit
                # patterns -- noise.  That yielded dist 0.78-0.87 / conf 0.00
                # for every live turn (see samples_as_float32).
                return torch.from_numpy(samples_as_float32(turn.audio))

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
    # Speech gate. Silero is already loaded for the file pipeline, so this
    # costs nothing extra; if the model is not there the gate fails OPEN
    # (turn_has_speech returns True for probe=None) rather than silently
    # dropping real speech.
    def _speech_probe(turn):
        # Resolve the session at CALL time, never pin it: the idle-TTL monitor
        # can unload the VAD between two turns (lesson 26).
        import numpy as np

        session = state.vad_session
        if session is None:
            return None
        audio = turn.audio
        if isinstance(audio, (bytes, bytearray, memoryview)):
            audio = np.frombuffer(audio, dtype=np.int16)
        return probe_speech(
            audio, session,
            threshold=float(cfg.get("vad_frame_threshold", 0.5)),
        )

    # Always installed.  Deciding once, at connect, silently disabled the gate
    # for every turn of a session that connected before the VAD was resident --
    # and the model is lazy-loaded, so that is the common case, not the rare one.
    # The probe resolves the session per call and reports "unavailable" instead.
    speech_probe = _speech_probe
    if state.vad_session is None:
        logger.info("No VAD session loaded yet: live turns are transcribed "
                    "unfiltered until one is resident (streaming.min_speech_prob)")
    # Only the legacy raw-PCM path endpointing happens here; turn frames
    # arrive already cut (and already coalesced) from the client, so they must
    # NOT go through this -- a second coalescer would re-merge the client's
    # turns and the two would never agree on a boundary.
    coalescer = TurnCoalescer()
    loop = asyncio.get_running_loop()
    connected_at = loop.time()
    # The client's declared start_sample is a claim about its own recording.
    # One dropped capture block or a late-starting getDisplayMedia stream
    # shifts every later boundary on that channel silently, so the claim is
    # checked against what has already been seen and reported when it cannot be
    # true. A bad frame is flagged, never fatal.
    timeline = proto.TimelineGuard(cfg)
    notified_channels = set()
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
        "non_speech_turns": 0,
        "edge_trimmed_turns": 0,
        "edge_trimmed_sec": 0.0,
        "timeline_faults": 0,
        "drifted_channels": [],
    }
    # Highest audio sample seen, whichever path produced it.  Reported to the
    # client so it knows how much of its recording the server actually covered.
    # A coalesced turn declares a span shorter than the audio it was cut from,
    # so its audio_end_sample is the position that matters here.
    stream_end_sample = {"value": 0}

    def _note_end(turn: Turn) -> None:
        true_end = getattr(turn, "audio_end_sample", None)
        end = int(true_end if true_end is not None else turn.end_sample)
        stream_end_sample["value"] = max(stream_end_sample["value"], end)
        timeline.note_end(end)

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

    def _check_timeline(header: dict, channel: int) -> dict:
        """Validate a client-declared span; return the header to build from.

        The first fault on a channel is pushed to the client immediately, in the
        existing ``dropped`` vocabulary, so the saved sidecar records that the
        channel's clock was untrustworthy; repeats only move the counters (a
        client stuck in a loop must not be able to flood the socket).
        """
        start, fault = timeline.observe(
            channel, header["start_sample"], header["n_samples"],
            elapsed_sec=loop.time() - connected_at,
        )
        if fault is None:
            return header
        stats["timeline_faults"] = timeline.faults
        stats["drifted_channels"] = timeline.drifted_channels()
        logger.warning(
            "Client timeline fault on channel %s: %s (declared %.2fs, "
            "expected >= %.2fs, %d so far)",
            fault["channel"], fault["reason"], fault["declared_start_sec"],
            fault["expected_start_sec"], timeline.faults,
        )
        if fault["channel"] not in notified_channels:
            notified_channels.add(fault["channel"])
            loop.create_task(_notify_drift(fault))
        if start == header["start_sample"]:
            return header
        return {**header, "start_sample": int(start)}

    async def _notify_drift(fault: dict) -> None:
        try:
            await websocket.send_json({
                "type": "dropped",
                "channel": int(fault["channel"]),
                "start": fault["declared_start_sec"],
                "end": fault["declared_start_sec"],
                "reason": "timeline_drift",
                "detail": f"{fault['reason']}: {fault['note']}",
            })
        except Exception:
            logger.debug("Could not notify client of a timeline fault")

    def _transcribe_turn(turn: Turn, channel: int) -> dict:
        state.touch()
        # Edge refinement FIRST, and before the gate, so the gate scores the
        # audio that will actually be decoded rather than the pre-roll padding
        # the detector attached.  The client's boundary is an energy crossing
        # plus a fixed pre/post-roll, so it is early at the onset and late at
        # the offset by up to ~0.2 s; those padded frames widen the item span
        # that the shutdown re-attribution matches against the diarization
        # turns, which is how a 0.2 s boundary error becomes an
        # UNKNOWN (boundary_crossing) on the final transcript.  The VAD session
        # is resident for the gate one line below, so this costs no extra model.
        turn, trimmed_sec = trim_turn_edges(turn, vad_session=state.vad_session,
                                           cfg=detector_config())
        if trimmed_sec > 0:
            stats["edge_trimmed_turns"] += 1
            stats["edge_trimmed_sec"] = round(
                stats["edge_trimmed_sec"] + trimmed_sec, 3)
            logger.debug("Trimmed %.0f ms off turn %.2f-%.2fs",
                         trimmed_sec * 1000.0, turn.start_sec, turn.end_sec)
        # Speech gate. The endpointing detector cuts on energy, so a sniff or a
        # chair creak arrives here as a perfectly good 0.5s turn and Whisper
        # answers it with its standard hallucination ("Thank you.").
        # Silero already knows the difference and costs microseconds here.
        ok, score, reason = turn_has_speech(turn, probe=speech_probe,
                                            cfg=detector_config())
        if not ok:
            stats["non_speech_turns"] += 1
            logger.debug("Skipped non-speech turn %.2f-%.2fs (prob=%.2f)",
                         turn.start_sec, turn.end_sec, score)
            return {
                "type": "empty",
                "skipped": reason,
                "speech_score": round(score, 3),
                "start": round(turn.start_sec, 2),
                "end": round(turn.end_sec, 2),
                "duration": round(turn.duration_sec, 2),
                "channel": int(channel),
                **_unknown_attribution("live_no_speech"),
            }
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
        if trimmed_sec > 0:
            msg["edge_trimmed_sec"] = round(trimmed_sec, 3)
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
                logger.error(
                    "Utterance processing failed: %s: %s",
                    type(e).__name__, _err(e),
                )
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
                header = _check_timeline(header, channel)
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
                for ready in coalescer.submit(turn):
                    _submit(ready, CHANNEL_MIC)
            for ready in coalescer.poll():
                _submit(ready, CHANNEL_MIC)

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s: %s", type(e).__name__, _err(e))
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
            # Through the coalescer, NOT straight to _submit: releasing the
            # tail first and the held turn second enqueued them backwards
            # (tail at 2.08s, merged turn starting at 0.90s), so the client
            # received transcripts out of chronological order.
            for ready in coalescer.submit(tail):
                _submit(ready, CHANNEL_MIC)
    for ready in coalescer.flush():
        _submit(ready, CHANNEL_MIC)
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
            "timeline": timeline.stats(),
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
