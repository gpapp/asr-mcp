"""Streaming handler tests (WebSocket live transcription).

The handler itself needs FastAPI (for the WebSocket type) and onnxruntime (via
``asr_mcp.core.model_state``), but not torch.  Models and the ASR backend are
monkeypatched out, so no GPU/model download is required.
"""

import asyncio
import struct

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="handler needs the web dependencies")
pytest.importorskip("onnxruntime", reason="handler touches ModelState")

from asr_mcp.streaming import handler as hd  # noqa: E402

SR = 16000
AMP = 0.3


def _pcm(seconds, amp=AMP):
    return (np.sin(2 * np.pi * 220 * np.arange(int(SR * seconds)) / SR) * amp * 32767
            ).astype("<i2").tobytes()


def _silence(seconds):
    return np.zeros(int(SR * seconds), dtype="<i2").tobytes()


def _frame(channel, payload, sequence=0):
    return struct.pack("<II", channel, sequence) + payload


class FakeWebSocket:
    """Minimal stand-in for starlette's WebSocket."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self._frames:
            return self._frames.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, *a, **kw):
        self.closed = True


@pytest.fixture
def patched(monkeypatch):
    """Neutralise model loading and record the ASR calls."""
    from asr_mcp.core import model_state, transcriber

    calls = []

    def fake_transcribe(audio=None, **kw):
        calls.append({"dur": len(audio) / SR, "kwargs": kw})
        return {"text": "hello", "inference_time_sec": 0.01, "tokens_generated": 3}

    monkeypatch.setattr(model_state.state, "ensure_ready", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(transcriber, "transcribe_audio_sync", fake_transcribe)
    return calls


def _run(frames):
    ws = FakeWebSocket(frames)
    asyncio.run(hd.handle_ws_stream(ws))
    return ws


# --- packet validation -----------------------------------------------------


def test_short_frame_does_not_kill_connection(patched):
    """A 4-byte frame must be dropped, and the stream must keep working."""
    ws = _run([
        {"type": "websocket.receive", "bytes": b"\x01\x02\x03\x04"},
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _pcm(0.6), 1)},
    ])
    texts = [m for m in ws.sent if m.get("type") == "transcript"]
    assert texts, ws.sent


def test_text_frame_is_ignored(patched):
    ws = _run([
        {"type": "websocket.receive", "text": "hello"},
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert any(m.get("type") == "transcript" for m in ws.sent)


def test_speaker_channel_never_opens_a_turn(patched):
    """A loud room channel must not trigger the mic turn state machine."""
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(1, _pcm(2.0))},
        {"type": "websocket.receive", "bytes": _frame(7, _pcm(1.0), 1)},
    ])
    assert patched == []  # the ASR backend was never called
    assert [m for m in ws.sent if m.get("type") in ("transcript", "empty")] == []


# --- policy ----------------------------------------------------------------


def test_live_turn_has_no_speaker(patched):
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    msg = next(m for m in ws.sent if m.get("type") == "transcript")
    assert msg["speaker"] is None
    assert msg["speaker_source"] == "unknown"
    assert msg["speaker_confidence"] == 0.0
    assert msg["uncertain"] is True
    assert msg["attribution_reason"] == "live_turn_unattributed"
    assert msg["text"] == "hello"


def test_open_turn_is_flushed_on_disconnect(patched):
    """No hangover silence: the disconnect itself must close the turn."""
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    msg = next(m for m in ws.sent if m.get("type") == "transcript")
    assert msg["end_reason"] == "flush"
    assert msg["duration"] > 0.5


def test_turns_are_sent_with_sample_timestamps(patched):
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
        # > hangover (320ms) of silence closes the first turn.
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(0.6) + _pcm(0.6), 1)},
    ])
    msgs = [m for m in ws.sent if m.get("type") == "transcript"]
    assert len(msgs) == 2
    assert msgs[0]["start"] < msgs[1]["start"]
    assert msgs[1]["end"] > msgs[0]["end"]
    # start/end are derived from the sample counter, not wall clock.
    assert msgs[0]["start"] == pytest.approx(1.0, abs=0.2)
    for m in msgs:
        assert m["end"] - m["start"] == pytest.approx(m["duration"], abs=0.02)


def test_pre_segmented_is_passed_to_backend(patched):
    _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert patched and patched[0]["kwargs"]["pre_segmented"] is True


def test_empty_transcript_is_reported_as_empty(patched, monkeypatch):
    from asr_mcp.core import transcriber

    monkeypatch.setattr(
        transcriber, "transcribe_audio_sync",
        lambda audio=None, **kw: {"text": "   ", "inference_time_sec": 0.0},
    )
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert [m["type"] for m in ws.sent] == ["empty"]


def test_backend_error_does_not_break_the_stream(patched, monkeypatch):
    from asr_mcp.core import transcriber

    def boom(audio=None, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(transcriber, "transcribe_audio_sync", boom)
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert any(m.get("type") == "error" for m in ws.sent)
