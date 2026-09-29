"""Live-client tests: protocol agreement, endpointing, resampling, sidecar.

The client (``asr-client/live_client.py``) re-implements the wire protocol and
the turn detector because it ships as a standalone zip and must not import
server code.  Re-implementation is only safe if the two cannot drift, so the
first tests here assert agreement against the server modules directly.

Everything that needs no third-party package runs anywhere; the tests that
need numpy/soxr skip when those are missing.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = REPO_ROOT / "asr-client"

# The server image excludes asr-client/ (.dockerignore) — the client ships as a
# standalone zip, not as part of the image. These tests compare the client
# against the server, so they only make sense where both are present.
pytestmark = pytest.mark.skipif(
    not (CLIENT_DIR / "live_client.py").is_file(),
    reason="asr-client/live_client.py not present (excluded from the server image)",
)


def _load_live_client():
    """Import live_client.py by path (it lives outside the asr_mcp package)."""
    path = CLIENT_DIR / "live_client.py"
    spec = importlib.util.spec_from_file_location("live_client_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _load_live_client()


# ── Protocol must match the server byte for byte ───────────────────────────

def test_client_protocol_matches_server(live):
    from asr_mcp.streaming import protocol as proto

    assert live.TURN_MAGIC == proto.TURN_MAGIC
    assert live.PROTOCOL_VERSION == proto.PROTOCOL_VERSION
    assert live.MSG_TURN == proto.MSG_TURN
    assert live.MSG_FLUSH == proto.MSG_FLUSH
    assert live.SAMPLE_RATE == proto.SAMPLE_RATE
    assert live.CHANNEL_MIC == 0
    assert live.CHANNEL_SPEAKER == 1
    assert struct.calcsize(live.TURN_HEADER_FMT) == proto.TURN_HEADER_SIZE
    assert live.TURN_HEADER_SIZE == proto.TURN_HEADER_SIZE


def test_client_packed_turn_is_accepted_by_server(live):
    """A frame built by the client must parse with the server's unpacker."""
    from asr_mcp.streaming import protocol as proto

    np = pytest.importorskip("numpy")
    pcm = (np.arange(-500, 500, dtype=np.int16)).tobytes()  # 1000 samples
    frame = live.pack_turn(1, 16000, pcm, sequence=7)

    assert proto.is_turn_frame(frame)
    header, payload = proto.unpack_turn(frame)
    assert header["channel"] == 1
    assert header["start_sample"] == 16000
    assert header["n_samples"] == 1000
    assert header["sequence"] == 7
    assert payload == pcm


def test_client_flush_frame_is_accepted_by_server(live):
    from asr_mcp.streaming import protocol as proto

    header, payload = proto.unpack_turn(live.pack_flush())
    assert header["msg_type"] == proto.MSG_FLUSH
    assert payload == b""


def test_pack_turn_rejects_odd_payload(live):
    with pytest.raises(ValueError):
        live.pack_turn(0, 0, b"\x00")


def test_ws_url_preserves_prefix_and_scheme(live):
    assert live._ws_url("http://127.0.0.1:8087/asr-mcp") == \
        "ws://127.0.0.1:8087/asr-mcp/api/asr/ws/stream"
    assert live._ws_url("https://h.example/asr-mcp/") == \
        "wss://h.example/asr-mcp/api/asr/ws/stream"
    # No trailing slash in the base must not produce a double slash.
    assert live._ws_url("https://h.example") == \
        "wss://h.example/api/asr/ws/stream"


# ── Endpointing must match the server detector ─────────────────────────────

def test_client_detector_defaults_match_server_config(live):
    from asr_mcp.streaming.turn_detector import config as server_config

    server = server_config()
    for key, default in live.DETECTOR_DEFAULTS.items():
        assert server.get(key, default) == default, (
            f"{key}: server={server.get(key)} client default={default}"
        )


def _pcm(seconds, amp=8000, rate=16000):
    import numpy as np

    n = int(seconds * rate)
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


def _silence(seconds, rate=16000):
    import numpy as np

    return np.zeros(int(seconds * rate), dtype=np.int16).tobytes()


def test_client_detector_emits_a_turn(live):
    det = live.TurnDetector()
    turns = []
    turns += det.feed(_silence(0.5))
    turns += det.feed(_pcm(0.5))
    turns += det.feed(_silence(1.0))
    assert len(turns) == 1
    assert turns[0].end_sec > turns[0].start_sec


def test_client_detector_keeps_a_short_reply(live):
    """'Yes' must survive — a fixed high gate used to drop these."""
    det = live.TurnDetector()
    det.feed(_silence(0.5))
    turns = det.feed(_pcm(0.2)) + det.feed(_silence(1.0))
    assert len(turns) == 1


def test_client_detector_splits_two_turns(live):
    det = live.TurnDetector()
    turns = det.feed(_silence(0.4) + _pcm(0.4) + _silence(1.2)
                     + _pcm(0.4) + _silence(1.0))
    assert len(turns) == 2
    assert turns[0].end_sec <= turns[1].start_sec


def test_client_detector_flush_is_idempotent(live):
    det = live.TurnDetector()
    det.feed(_silence(0.4) + _pcm(0.4))
    first = det.flush()
    assert first is not None
    assert first.reason == "flush"
    assert det.flush() is None


def test_client_and_server_detectors_agree_on_turn_count(live):
    """The whole design depends on live and offline boundaries matching."""
    from asr_mcp.streaming.turn_detector import TurnDetector as ServerDetector

    pytest.importorskip("numpy")
    audio = (_silence(0.4) + _pcm(0.35) + _silence(1.1)
             + _pcm(0.30) + _silence(1.0))
    client_turns = live.TurnDetector().feed(audio)
    server_turns = ServerDetector().feed(audio)
    assert len(client_turns) == len(server_turns) == 2
    for c, s in zip(client_turns, server_turns):
        # Pre/post-roll padding must land on the same samples, or a live turn
        # and its offline counterpart would disagree on the file timeline.
        # The client detector was written to mirror the server's arithmetic
        # (pre-roll from confirm_idx, trailing trim = -hangover + post_roll),
        # so exact equality is required — a tolerance here would hide drift.
        assert (c.start_sample, c.end_sample) == (s.start_sample, s.end_sample)
        assert c.reason == s.reason


# ── Resampling ─────────────────────────────────────────────────────────────

def test_resampler_converts_to_16k(live):
    pytest.importorskip("soxr")
    import numpy as np

    res = live.Resampler(48000)
    chunk = (np.sin(2 * np.pi * 220 * np.arange(4800) / 48000) * 8000)
    out = res.process((chunk * 32767).astype(np.int16).tobytes())
    samples = np.frombuffer(out, dtype=np.int16)
    # 4800 samples at 48 kHz is 0.1s -> ~1600 samples at 16 kHz.
    assert 1500 < len(samples) < 1700


def test_resampler_flush_ends_the_stream(live):
    pytest.importorskip("soxr")
    res = live.Resampler(44100)
    res.process(b"\x00\x00" * 4410)
    res.process(b"\x00\x00" * 4410)
    assert isinstance(res.flush(), bytes)


# ── Session bookkeeping (no audio, no server) ──────────────────────────────

def _session(tmp_path, monkeypatch, live, **over):
    monkeypatch.setattr(live, "get_config", lambda: ("http://s:8087/asr-mcp", "tok"))
    monkeypatch.setattr(live, "get_language", lambda: "hu")
    args = type("A", (), {
        "outdir": str(tmp_path), "language": None, "no_rediag": False,
        "no_speaker": False, "device": None, "rate": None, "block_ms": 64.0,
    })()
    for k, v in over.items():
        setattr(args, k, v)
    return live.LiveSession(args)


def test_session_paths_are_timestamp_named(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    # 20260929-141203 style, derived from the session start.
    assert s.mic_path.name == f"{s.stamp}.wav"
    assert s.speaker_path.name == f"{s.stamp}-speaker.wav"
    assert s.txt_path.name == f"{s.stamp}.txt"
    assert s.sidecar_path.name == f"{s.stamp}.asr.json"


def test_on_message_records_item_and_prints(tmp_path, monkeypatch, live, capsys):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({
        "type": "transcript", "start": 1.4, "end": 3.0, "text": "hello",
        "channel": 1, "speaker": "Gergely Papp", "speaker_confidence": 0.8,
        "speaker_source": "known_voiceprint", "uncertain": False,
    })
    assert s.turn_count == 1
    assert s.items[0]["text"] == "hello"
    assert s.items[0]["speaker"] == "Gergely Papp"
    out = capsys.readouterr().out
    assert "Gergely Papp" in out and "00:00:01" in out


def test_dropped_turn_is_recorded_as_a_gap(tmp_path, monkeypatch, live, capsys):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({"type": "dropped", "start": 1.0, "end": 2.0,
                  "channel": 0, "reason": "queue_overflow"})
    assert s.drop_count == 1
    assert s.gaps[0]["reason"] == "queue_overflow"
    assert "dropped" in capsys.readouterr().err


def test_sidecar_records_gaps_and_source(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({"type": "transcript", "start": 0.0, "end": 1.0,
                  "text": "hi", "channel": 0, "speaker": "You"})
    s.on_message({"type": "dropped", "start": 1.0, "end": 2.0,
                  "channel": 0, "reason": "queue_overflow"})
    s.on_message({"type": "stats", "packets": 10, "covered_sec": 2.5})
    path = s.write_sidecar({"final_transcript": str(s.txt_path)})

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["asr_source"] == "live_stream"
    assert data["turns"] == 1
    assert len(data["gaps"]) == 1
    assert data["server_stats"]["covered_sec"] == 2.5
    assert data["items"][0]["text"] == "hi"
    assert data["recording"]["mic_wav"].endswith(".wav")


def test_final_result_prefers_rediag_output(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({"type": "transcript", "start": 0.0, "end": 1.0,
                  "text": "live", "channel": 0, "speaker": None})
    rediag = {"results": [{"start": 0.0, "end": 1.0, "text": "live",
                           "speaker": "Gergely Papp", "uncertain": False,
                           "segments": [{"start": 0.0, "end": 1.0, "text": "live"}]}]}
    result, used = s.final_result(rediag)
    assert used is True
    assert result["results"][0]["speaker"] == "Gergely Papp"


def test_final_result_falls_back_to_live(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({"type": "transcript", "start": 0.0, "end": 1.0,
                  "text": "live", "channel": 0, "speaker": "You"})
    result, used = s.final_result(None)
    assert used is False
    assert result["results"][0]["text"] == "live"


def test_rediag_error_falls_back_to_live(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    s.on_message({"type": "transcript", "start": 0.0, "end": 1.0,
                  "text": "live", "channel": 0, "speaker": "You"})
    result, used = s.final_result({"error": "Diarization failed", "results": []})
    assert used is False


def test_transcript_written_uses_shared_format(tmp_path, monkeypatch, live):
    """Live output must match transcribe_client's [HH:MM:SS] Speaker (NN%): form."""
    s = _session(tmp_path, monkeypatch, live)
    result = {
        "total_speakers": 1, "audio_duration_sec": 1.0,
        "results": [{
            "start": 0.0, "end": 1.0, "speaker": "Gergely Papp",
            "uncertain": False, "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.8}],
        }],
    }
    path = s.write_transcript(result, used_rediag=True)
    text = path.read_text(encoding="utf-8")
    assert "[00:00:00] Gergely Papp (80%): hello" in text
    assert text.startswith("# Re-diarized")
    assert "Live (diarization skipped)" not in text


def test_uncertain_speaker_kept_with_reason(tmp_path, monkeypatch, live):
    s = _session(tmp_path, monkeypatch, live)
    result = {
        "total_speakers": 0, "audio_duration_sec": 1.0,
        "results": [{
            "start": 0.0, "end": 1.0, "speaker": None, "uncertain": True,
            "attribution_reason": "live_match_weak",
            "segments": [{"start": 0.0, "end": 1.0, "text": "mumble"}],
        }],
    }
    text = s.write_transcript(result, used_rediag=False).read_text(encoding="utf-8")
    assert "UNKNOWN (live_match_weak)" in text
    assert "mumble" in text  # text retained — identity withheld, not content
    assert "Live (diarization skipped)" in text
