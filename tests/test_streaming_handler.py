"""Streaming handler tests (WebSocket live transcription).

The handler itself needs FastAPI (for the WebSocket type) and onnxruntime (via
``asr_mcp.core.model_state``), but not torch.  Models and the ASR backend are
monkeypatched out, so no GPU/model download is required.
"""

import asyncio
import importlib.util
import struct

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="handler needs the web dependencies")
pytest.importorskip("onnxruntime", reason="handler touches ModelState")

from asr_mcp.streaming import attribution  # noqa: E402
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


def _turn_frame(channel, payload, start_sample=0):
    """A client-framed turn (magic-prefixed), as live_client.py sends it."""
    from asr_mcp.streaming import protocol as proto

    return proto.pack_turn(channel, start_sample, payload)


def proto_flush():
    from asr_mcp.streaming import protocol as proto

    return proto.pack_control(proto.MSG_FLUSH)


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
        # Assert the real contract, not just a length. `len(audio)` happens to
        # work on raw bytes, which let `_turn_from_frame` hand the backend the
        # whole PCM payload instead of a float32 array and every live turn die
        # with "could not convert string to float" while these tests stayed
        # green. The backend is a real array consumer, so assert as one.
        assert isinstance(audio, np.ndarray), (
            f"backend received {type(audio).__name__}, expected np.ndarray"
        )
        assert audio.dtype == np.float32, f"dtype was {audio.dtype}"
        calls.append({"dur": len(audio) / SR, "kwargs": kw, "audio": audio})
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


def test_mic_turn_is_labelled_from_the_channel_not_a_voiceprint(patched):
    """Lesson 34: the mic channel IS the identity evidence.

    The default input device is the local user, who is in their own
    voiceprints; embedding-matching it would return a confident wrong answer.
    """
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    msg = next(m for m in ws.sent if m.get("type") == "transcript")
    assert msg["speaker"] == attribution.LOCAL_SPEAKER_LABEL
    assert msg["speaker_source"] == "input_device"
    assert msg["uncertain"] is False
    assert msg["text"] == "hello"


def test_speaker_channel_turn_without_voiceprints_is_unknown(patched):
    """No voiceprints loaded -> withhold the name, keep the text.

    Uses a client-framed turn: on the legacy raw-PCM path the server only ever
    feeds its own detector with the mic channel, so a speaker-channel turn can
    only arrive as an explicit turn frame.
    """
    ws = _run([
        {"type": "websocket.receive",
         "bytes": _turn_frame(attribution.CHANNEL_SPEAKER, _pcm(0.6), 0)},
    ])
    msg = next(m for m in ws.sent if m.get("type") == "transcript")
    assert msg["speaker"] is None
    assert msg["uncertain"] is True
    assert msg["attribution_reason"] == "no_voiceprints"
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
        # A pause LONGER than streaming.merge_gap_sec (1.0s) closes the first
        # turn and keeps the second one separate. A shorter pause now merges --
        # see test_short_pause_turns_are_coalesced.
        {"type": "websocket.receive",
         "bytes": _frame(hd.CHANNEL_MIC, _silence(2.0) + _pcm(0.6), 1)},
    ])
    msgs = [m for m in ws.sent if m.get("type") == "transcript"]
    assert len(msgs) == 2
    assert msgs[0]["start"] < msgs[1]["start"]
    assert msgs[1]["end"] > msgs[0]["end"]
    # start/end are derived from the sample counter, not wall clock.
    assert msgs[0]["start"] == pytest.approx(1.0, abs=0.2)
    for m in msgs:
        assert m["end"] - m["start"] == pytest.approx(m["duration"], abs=0.02)


def test_short_pause_turns_are_coalesced(patched):
    """A 0.6s pause used to cut an utterance in two.

    Both fragments are useless on their own -- Whisper hallucinates on 0.4s of
    speech-free audio and ECAPA cannot name anyone from it -- so the server
    holds a closed turn for up to streaming.merge_gap_sec and merges the next
    one into it.
    """
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.8))},
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(0.6) + _pcm(0.8), 1)},
    ])
    msgs = [m for m in ws.sent if m.get("type") == "transcript"]
    assert len(msgs) == 1, [m["start"] for m in msgs]
    assert msgs[0]["start"] == pytest.approx(0.9, abs=0.2)


def test_coalesced_turn_reaches_the_client_in_chronological_order(patched):
    """The held turn must be released BEFORE the one that displaced it.

    Releasing the coalescer's flush tail straight to the queue (skipping
    submit) enqueued them backwards: tail at 2.08s, then the merged turn
    starting at 0.90s.
    """
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(2.0) + _pcm(0.6), 1)},
    ])
    starts = [m["start"] for m in ws.sent if m.get("type") == "transcript"]
    assert starts == sorted(starts), starts


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
    # The trailing stats frame is sent before close so the client can tell how
    # much of its recording the server actually covered.
    assert [m["type"] for m in ws.sent] == ["empty", "stats"]


def test_backend_error_does_not_break_the_stream(patched, monkeypatch):
    from asr_mcp.core import transcriber

    def boom(audio=None, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(transcriber, "transcribe_audio_sync", boom)
    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert any(m.get("type") == "error" for m in ws.sent)


def test_non_speech_turn_never_reaches_the_decoder(patched, monkeypatch):
    """The gate's whole point: no decode, no hallucinated sentence.

    Whisper answers a sniff with a fluent "Thank you." and no signal in its
    output reveals it -- measured against the real backend, no_speech_prob is
    0.0000 for noise and for speech alike, and silence decodes at avg_logprob
    -0.29 vs real speech at -0.30 (see transcribers/whisper.py). So the Silero
    gate in front of the decoder is the only thing standing between a chair
    creak and a line in the transcript.
    """
    from asr_mcp.core import model_state

    # the handler only builds a probe when a VAD session exists
    monkeypatch.setattr(model_state.state, "vad_session", object(),
                        raising=False)
    monkeypatch.setattr(hd, "probe_speech", lambda *a, **k: (0.04, 0.0),
                        raising=False)

    ws = _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert patched == [], "the decoder was called for a non-speech turn"
    kinds = [m["type"] for m in ws.sent]
    assert "transcript" not in kinds
    empty = [m for m in ws.sent if m["type"] == "empty"]
    assert empty and empty[0]["skipped"] == "no_speech_low_prob"
    stats = [m for m in ws.sent if m["type"] == "stats"][0]
    assert stats["non_speech_turns"] == 1


def test_a_missing_vad_session_lets_the_turn_through(patched, monkeypatch):
    """Fail-open: with models unloaded there is no gate, and a real speaker
    must still be transcribed rather than silently dropped."""
    from asr_mcp.core import model_state

    monkeypatch.setattr(model_state.state, "vad_session", None, raising=False)
    _run([
        {"type": "websocket.receive", "bytes": _frame(hd.CHANNEL_MIC, _silence(1.0) + _pcm(0.6))},
    ])
    assert len(patched) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch (server-only path)")
def test_live_embedding_hooks_call_the_real_functions_correctly(patched, monkeypatch):
    """Regression: embed_fn must pass sample_rate positionally.

    extract_embedding(waveform, sample_rate) takes sample_rate as a required
    positional argument. Omitting it raises TypeError, which attribute_live_turn
    catches and reports as "live_match_failed" — silently downgrading every
    speaker turn to UNKNOWN instead of naming anyone.
    """
    from asr_mcp.core import model_state
    from asr_mcp.speaker import embedding as emb_mod

    seen = {}

    def fake_extract(waveform, sample_rate, *a, **kw):
        seen["sample_rate"] = sample_rate
        seen["n"] = int(waveform.numel())
        return np.zeros(192, dtype=np.float32)

    def fake_pitch(waveform, sample_rate=16000):
        seen["pitch_sr"] = sample_rate
        return (120.0, 0.9)

    def fake_energy(waveform):
        seen["energy_n"] = int(waveform.numel())
        return 0.2

    monkeypatch.setattr(emb_mod, "extract_embedding", fake_extract)
    monkeypatch.setattr(emb_mod, "compute_pitch", fake_pitch)
    monkeypatch.setattr(emb_mod, "compute_energy", fake_energy)

    # Force the handler down the voiceprint branch and capture the built hooks.
    captured = {}
    import asr_mcp.streaming.handler as handler_mod
    real_attr = handler_mod.attribute_live_turn

    def spy(turn, channel, voiceprints=None, embed_fn=None, pitch_fn=None,
            energy_fn=None):
        captured.update(embed_fn=embed_fn, pitch_fn=pitch_fn, energy_fn=energy_fn)
        return real_attr(turn, channel, voiceprints=voiceprints,
                         embed_fn=embed_fn, pitch_fn=pitch_fn, energy_fn=energy_fn)

    monkeypatch.setattr(handler_mod, "attribute_live_turn", spy)
    # _load_known_speakers is imported inside the function body, so patch it
    # where it is defined, not on the handler module.
    from asr_mcp.api import asr_router
    monkeypatch.setattr(asr_router, "_load_known_speakers",
                        lambda *a, **k: {"Someone": {"embedding": [0.0] * 192}})

    ws = _run([
        {"type": "websocket.receive",
         "bytes": _turn_frame(attribution.CHANNEL_SPEAKER, _pcm(0.6), 0)},
    ])

    assert captured.get("embed_fn") is not None, "embedding hooks were not built"
    turn = type("T", (), {"audio": _pcm(0.6), "duration_sec": 0.6})()
    emb = captured["embed_fn"](turn)          # must not raise
    assert emb.shape == (192,)
    assert seen["sample_rate"] == 16000
    assert seen["n"] == int(0.6 * SR)          # int16 -> float32, 1 sample per frame
    assert captured["pitch_fn"](turn) == 120.0
    assert captured["energy_fn"](turn) == 0.2
    # 0.6s of audio = 9600 float samples, not 9600 bytes
    assert seen["energy_n"] == int(0.6 * SR)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch (server-only path)")
def test_cpu_embedding_session_does_not_receive_gpu_arena_options():
    """Regression: GPU_SHRINK_RUN_OPTIONS names gpu:0.

    A CPU session has no such arena, so passing the options to it raises
    INVALID_ARGUMENT "Did not find an arena based allocator ... gpu:0". That
    message is not recognised by is_gpu_oom(), so the CPU fallback did not
    catch it and every live embedding failed -> all turns UNKNOWN.
    """
    from asr_mcp.core.model_state import GPU_SHRINK_RUN_OPTIONS
    from asr_mcp.speaker import embedding as emb

    class FakeSession:
        def __init__(self, providers):
            self._providers = providers
            self.calls = []

        def get_providers(self):
            return self._providers

        def run(self, names, feed, run_options=None):
            self.calls.append(run_options)
            if run_options is not None and "gpu:0" in str(run_options):
                raise RuntimeError(
                    "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Did not find "
                    "an arena based allocator registered for device-id "
                    "combination in the memory arena shrink list: gpu:0")
            return ["ok"]

    cpu = FakeSession(["CPUExecutionProvider"])
    assert emb._run_with_cpu_fallback(cpu, {}, ["out"]) == ["ok"]
    assert cpu.calls == [None], "CPU session must not get GPU arena options"

    gpu = FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert emb._run_with_cpu_fallback(gpu, {}, ["out"]) == ["ok"]
    assert gpu.calls == [GPU_SHRINK_RUN_OPTIONS], "GPU session must still shrink"


# --- the turn frame payload contract ---------------------------------------

def test_turn_frame_audio_is_a_float32_array():
    """Regression: raw PCM bytes were passed to the ASR backend.

    `Turn.audio` is a float32 numpy array in [-1, 1] everywhere else. The
    turn-frame path passed `pcm` (bytes) through instead, so every live turn
    raised "could not convert string to float: b'...'" and the exception text
    dumped the entire payload into the log. The real-world symptom was a
    session reporting 0 transcripts with megabytes of hex in the server log.
    """
    from asr_mcp.streaming import protocol as proto

    pcm = _pcm(0.2)
    header, data = proto.unpack_turn(proto.pack_turn(0, 0, pcm))
    turn = hd._turn_from_frame(header, data)

    assert isinstance(turn.audio, np.ndarray), type(turn.audio)
    assert turn.audio.dtype == np.float32
    assert abs(float(np.abs(turn.audio).max()) - 0.3) < 0.01
    assert turn.end_sample - turn.start_sample == len(turn.audio)


def test_turn_frame_audio_matches_the_server_detector_output():
    """A client cut and a server cut must produce the same array convention."""
    from asr_mcp.streaming import protocol as proto
    from asr_mcp.streaming.turn_detector import TurnDetector

    pcm = _pcm(0.3)
    header, data = proto.unpack_turn(proto.pack_turn(0, 0, pcm))
    framed = hd._turn_from_frame(header, data)

    det = TurnDetector()
    det.feed(_silence(0.4))
    det.feed(pcm)
    detected = det.feed(_silence(1.2))[0]

    # Both paths must hand the backend the same KIND of thing: float32 in
    # [-1, 1]. They are not byte-identical because the server re-frames the
    # audio and adds pre/post-roll, so only the convention is comparable.
    for turn in (framed, detected):
        assert turn.audio.dtype == np.float32
        assert np.abs(turn.audio).max() <= 1.0
        assert np.abs(turn.audio).max() > 0.1
    assert np.isclose(np.abs(framed.audio).max(),
                      np.abs(detected.audio).max(), atol=0.01)


def test_transcribed_turn_reaches_the_client(patched):
    """End to end through the TURN FRAME path (not the legacy raw-PCM path).

    The earlier end-to-end check drove the legacy raw-PCM path, which goes
    through the server's own detector and therefore always had a proper
    array. This exercises the path the live client actually uses.
    """
    frames = [
        {"type": "websocket.receive", "bytes": _turn_frame(0, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto_flush()},
    ]
    ws = _run(frames)
    texts = [m for m in ws.sent if m.get("type") == "transcript"]
    assert texts, f"no transcript message; sent={ws.sent}"
    assert texts[0]["text"] == "hello"
    assert patched, "the backend was never called"
    assert patched[0]["audio"].dtype == np.float32
