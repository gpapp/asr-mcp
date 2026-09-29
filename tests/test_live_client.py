"""Live-client tests: protocol agreement, endpointing, resampling, sidecar.

The client (``asr-client/live_client.py``) re-implements the wire protocol and
the turn detector because it ships as a standalone zip and must not import
server code.  Re-implementation is only safe if the two cannot drift, so the
first tests here assert agreement against the server modules directly.

Everything that needs no third-party package runs anywhere; the tests that
need numpy/soxr skip when those are missing.
"""

import importlib.util
import io
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
    # 4800 samples at 48 kHz is 0.1s -> ~1600 samples at 16 kHz. A streaming
    # resampler holds back its filter delay, so the count is short until
    # flush(); check the total, not a single chunk.
    assert 1400 < len(samples) < 1600
    total = len(samples) + len(np.frombuffer(res.flush(), dtype=np.int16))
    assert 1590 <= total <= 1620


def test_resampler_accepts_int16_chunks(live):
    """soxr's ResampleStream defaults to float32 and rejects int16 input.

    Passing the quality as the 4th positional argument lands it in the dtype
    slot and raises "data type 'HQ' not understood"; omitting the dtype then
    raises on the first chunk. Both crashed the client at startup.
    """
    pytest.importorskip("soxr")
    res = live.Resampler(48000)
    assert isinstance(res.process(b"\\x01\\x00" * 512), bytes)
    assert isinstance(res.flush(), bytes)


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
    assert s.transcript_count == 1
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
    # This test drives on_message directly, so no turn was ever sent --
    # which is exactly the split the two counters exist to expose.
    assert data["turns_sent"] == 0
    assert data["transcripts_received"] == 1
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


# ── Device selection ───────────────────────────────────────────────────────
#
# The device list below is the REAL output of `live_client.py --devices` on
# the tester's Windows machine, trimmed to the entries that matter. It is the
# fixture that exposed the original bug: `find_loopback_device` returned
# device 14 ("Microphone Array (Intel)") as the "speaker loopback", because
# sounddevice cannot expose WASAPI loopback at all and the code settled for
# the first WASAPI device with an input channel -- which is a microphone.
# Nothing would have errored; it would have opened a second mic and labelled
# it "everyone else".

# The REAL, complete output of `live_client.py --devices` on the tester's
# Windows machine (a laptop with a Plantronics Blackwire 3220 headset). It is
# the fixture that exposed the original bug: `find_loopback_device` returned
# device 14 ("Microphone Array (Intel)") as the "speaker loopback", because
# sounddevice cannot expose WASAPI loopback at all and the code settled for
# the first WASAPI device with an input channel -- which is a microphone.
# Nothing would have errored; it would have opened a second mic and labelled
# it "everyone else".
#
# Note the shape: every real device appears three times (MME / DirectSound /
# WASAPI), and MME contributes "Sound Mapper" aliases that are not devices at
# all. Matching on list order or on the first entry therefore picks the wrong
# layer.

WINDOWS_DEVICES = [
    {"index": 0, "name": "Microsoft Sound Mapper - Input", "api": "MME",
     "in": 2, "out": 0, "rate": 44100, "is_loopback": False},
    # MME truncates device names to 31 characters. This is verbatim what
    # PyAudioWPatch reports on the tester's machine, and it is the reason a
    # name comparison against the WASAPI / [Loopback] entries of the same
    # hardware used to fail: the OS default came back truncated.
    {"index": 1, "name": "Headset Microphone (2- Plantron",
     "api": "MME", "in": 2, "out": 0, "rate": 44100, "is_loopback": False},
    {"index": 2, "name": "Microphone Array (Intel(R) Smart Sound Technology "
                          "for Digital Microphones)",
     "api": "MME", "in": 2, "out": 0, "rate": 44100, "is_loopback": False},
    {"index": 3, "name": "Microsoft Sound Mapper - Output", "api": "MME",
     "in": 0, "out": 2, "rate": 44100, "is_loopback": False},
    {"index": 4, "name": "Headset Earphone (2- Plantronic",
     "api": "MME", "in": 0, "out": 2, "rate": 44100, "is_loopback": False},
    {"index": 5, "name": "Speakers (Realtek(R) Audio)", "api": "MME",
     "in": 0, "out": 2, "rate": 44100, "is_loopback": False},
    {"index": 6, "name": "Primary Sound Capture Driver",
     "api": "Windows DirectSound", "in": 2, "out": 0, "rate": 44100,
     "is_loopback": False},
    {"index": 7, "name": "Headset Microphone (2- Plantronics Blackwire 3220 Series)",
     "api": "Windows DirectSound", "in": 2, "out": 0, "rate": 44100,
     "is_loopback": False},
    {"index": 8, "name": "Microphone Array (Intel(R) Smart Sound Technology "
                          "for Digital Microphones)",
     "api": "Windows DirectSound", "in": 2, "out": 0, "rate": 44100,
     "is_loopback": False},
    {"index": 9, "name": "Primary Sound Driver", "api": "Windows DirectSound",
     "in": 0, "out": 2, "rate": 44100, "is_loopback": False},
    {"index": 10, "name": "Headset Earphone (2- Plantronics Blackwire 3220 Series)",
     "api": "Windows DirectSound", "in": 0, "out": 2, "rate": 44100,
     "is_loopback": False},
    {"index": 11, "name": "Speakers (Realtek(R) Audio)",
     "api": "Windows DirectSound", "in": 0, "out": 2, "rate": 44100,
     "is_loopback": False},
    {"index": 12, "name": "Speakers (Realtek(R) Audio)", "api": "Windows WASAPI",
     "in": 0, "out": 2, "rate": 48000, "is_loopback": False},
    {"index": 13, "name": "Headset Earphone (2- Plantronics Blackwire 3220 Series)",
     "api": "Windows WASAPI", "in": 0, "out": 2, "rate": 48000,
     "is_loopback": False},
    {"index": 14, "name": "Microphone Array (Intel(R) Smart Sound Technology "
                           "for Digital Microphones)",
     "api": "Windows WASAPI", "in": 4, "out": 0, "rate": 48000,
     "is_loopback": False},
    {"index": 15, "name": "Headset Microphone (2- Plantronics Blackwire 3220 Series)",
     "api": "Windows WASAPI", "in": 2, "out": 0, "rate": 48000,
     "is_loopback": False},
    {"index": 16, "name": "Speakers (Realtek(R) Audio) [Loopback]",
     "api": "Windows WASAPI", "in": 2, "out": 0, "rate": 48000,
     "is_loopback": True},
    {"index": 17, "name": "Headset Earphone (2- Plantronics Blackwire 3220 "
                           "Series) [Loopback]",
     "api": "Windows WASAPI", "in": 2, "out": 0, "rate": 48000,
     "is_loopback": True},
]

_HEADSET_OUT = "Headset Earphone (2- Plantronics Blackwire 3220 Series)"
_SPEAKERS_OUT = "Speakers (Realtek(R) Audio)"
# The full, untruncated WASAPI-layer name for the headset microphone.
_HEADSET_MIC = "Headset Microphone (2- Plantronics Blackwire 3220 Series)"


def test_select_devices_never_uses_a_microphone_as_the_loopback(live):
    """The regression that matters: no [Loopback] device -> no loopback."""
    devs = [d for d in WINDOWS_DEVICES if not d["is_loopback"]]
    mic, loop, reason = live.select_devices(devs)
    assert loop is None, "a microphone must never be used as the speaker channel"
    assert reason and "no [Loopback] device" in reason
    assert "force-reinstall PyAudioWPatch" in reason


def test_select_devices_picks_the_loopback_of_the_default_output(live):
    """Two loopbacks exist here; the wrong one captures silence or own echo."""
    _, loop, reason = live.select_devices(
        WINDOWS_DEVICES, default_output=_HEADSET_OUT)
    assert reason is None
    assert loop["index"] == 17, "headset output must select the headset loopback"


def test_select_devices_switches_loopback_with_the_output(live):
    """Plugging in / unplugging the headset must change the captured device."""
    _, loop, _ = live.select_devices(
        WINDOWS_DEVICES, default_output=_SPEAKERS_OUT)
    assert loop["index"] == 16


def test_select_devices_never_matches_an_mme_alias(live):
    """The MME 'Sound Mapper' aliases are not devices and have no loopback.

    Matching against the first output in list order picked the Realtek
    speakers even while the headset was the default output -- the bug this
    machine exposed.
    """
    assert "Microsoft Sound Mapper - Output" in WINDOWS_DEVICES[3]["name"]
    _, loop, _ = live.select_devices(
        WINDOWS_DEVICES, default_output="Microsoft Sound Mapper - Output")
    # No exact match exists, so it must fall back to a real loopback rather
    # than to the alias.
    assert loop["is_loopback"] is True


def test_select_devices_prefers_the_os_default_microphone(live):
    mic, _, _ = live.select_devices(
        WINDOWS_DEVICES, default_input=_HEADSET_OUT.replace("Earphone", "Microphone"))
    assert mic["index"] == 15


def test_select_devices_avoids_the_mme_alias_microphone(live):
    """Without a known default, take a real device rather than Sound Mapper."""
    mic, _, _ = live.select_devices(WINDOWS_DEVICES)
    assert mic["name"] != "Microsoft Sound Mapper - Input"
    assert mic["is_loopback"] is False


def test_select_devices_honours_an_explicit_microphone(live):
    mic, _, _ = live.select_devices(
        WINDOWS_DEVICES, mic_index=14, default_output=_HEADSET_OUT)
    assert mic["name"].startswith("Microphone Array")


def test_select_devices_rejects_an_invalid_microphone_index(live):
    mic, _, _ = live.select_devices(WINDOWS_DEVICES, mic_index=999)
    assert mic is None


def test_select_devices_reports_the_missing_playback_device_too(live):
    devs = [{"index": 0, "name": "In", "api": "WASAPI", "in": 1, "out": 0,
             "rate": 48000, "is_loopback": False}]
    mic, loop, reason = live.select_devices(devs)
    assert mic["index"] == 0 and loop is None
    assert "no playback device at all" in reason


def test_select_devices_is_a_pure_function(live):
    """No mutation, no I/O -- it is the part we can test off-Windows."""
    devs = list(WINDOWS_DEVICES)
    snapshot = [dict(d) for d in devs]
    live.select_devices(devs, default_output=_HEADSET_OUT)
    assert devs == snapshot


def test_default_device_names_survives_a_failing_getter(live):
    class _Broken:
        def get_default_wasapi_device_info(self):
            raise RuntimeError("no WASAPI on this host")

        def get_default_input_device_info(self):
            return {"name": "Mic"}

        def get_default_output_device_info(self):
            return {"name": "Speakers"}

    assert live.default_device_names(_Broken()) == {
        "input": "Mic", "output": "Speakers"}


def test_same_device_matches_an_mme_truncated_name(live):
    """MME truncates device names to 31 chars; the rest of the layers do not.

    This is the exact mismatch the tester's machine exposed: Windows reported
    the default output as "Headset Earphone (2- Plantronic", which matches no
    loopback by equality, so the real headset loopback was never selected.
    """
    assert live._same_device("Headset Earphone (2- Plantronic",
                             "Headset Earphone (2- Plantronics "
                             "Blackwire 3220 Series)")
    assert live._same_device("Headset Earphone (2- Plantronics "
                             "Blackwire 3220 Series",
                             "Headset Earphone (2- Plantronic")


def test_same_device_rejects_unrelated_and_short_names(live):
    assert not live._same_device("Speakers (Realtek(R) Audio)",
                                 "Headset Earphone (2- Plantronics)")
    assert not live._same_device("", "Headset")
    assert not live._same_device(None, None)
    # A short common prefix must not be treated as the same device.
    assert not live._same_device("Mic", "Microphone Array (Intel)")


def test_truncated_default_output_selects_the_right_loopback(live):
    """End to end with the OS default reported by the MME layer."""
    _, loop, reason = live.select_devices(
        WINDOWS_DEVICES, default_output="Headset Earphone (2- Plantronic")
    assert reason is None
    assert loop["index"] == 17


def test_truncated_default_input_upgrades_to_wasapi(live):
    """A truncated MME default must still resolve to the WASAPI mic (15)."""
    mic, _, _ = live.select_devices(
        WINDOWS_DEVICES, default_input="Headset Microphone (2- Plantron")
    assert mic["index"] == 15, "must capture at native 48 kHz, not via MME"
    assert mic["rate"] == 48000


def test_default_device_names_rejects_a_wrong_direction_answer(live):
    """get_default_wasapi_device_info() reports the INPUT, so asking it for
    the output yields a microphone name. The answer is dropped instead of
    being trusted."""
    class _P:
        def get_default_input_device_info(self):
            return {"name": _HEADSET_MIC}

        def get_default_output_device_info(self):
            return {"name": _HEADSET_MIC}

    got = live.default_device_names(_P(), WINDOWS_DEVICES)
    assert got["input"] is not None
    assert got["output"] is None, "a microphone name is not a valid output default"


def test_default_device_names_prefers_direction_correct_names(live):
    class _P:
        def get_default_input_device_info(self):
            return {"name": _HEADSET_MIC}

        def get_default_output_device_info(self):
            return {"name": "Speakers (Realtek(R) Audio)"}

    got = live.default_device_names(_P(), WINDOWS_DEVICES)
    assert got["input"] == _HEADSET_MIC
    assert got["output"] == "Speakers (Realtek(R) Audio)"


# ── Level meter: telling silence from a dead device ────────────────────────

def test_level_meter_shows_a_flat_bar_for_silence(live):
    pytest.importorskip("numpy")
    m = live.LevelMeter(interval=0, stream=io.StringIO())
    m.feed(0, b"\x00\x00" * 1600)
    m.feed(1, b"\x00\x00" * 1600)
    line = m.tick(now=1.0)
    assert line is not None
    assert "#" not in line, "silence must render as an empty bar"
    assert "-60.0 dB" in line
    assert "mic" in line and "speakers" in line


def test_level_meter_shows_a_bar_for_audio(live):
    pytest.importorskip("numpy")
    import numpy as np
    m = live.LevelMeter(interval=0, stream=io.StringIO())
    loud = (np.sin(np.arange(16000) / 40) * 20000).astype(np.int16).tobytes()
    m.feed(0, loud)
    m.feed(1, b"\x00\x00" * 100)
    line = m.tick(now=1.0)
    mic_part = line.split("speakers")[0]
    spk_part = line.split("speakers")[1]
    assert "#" in mic_part, "audio must render a bar"
    assert "#" not in spk_part, "the silent channel must stay flat"


def test_level_meter_hides_the_speaker_column_when_absent(live):
    m = live.LevelMeter(show_speaker=False, interval=0, stream=io.StringIO())
    line = m.tick(now=1.0)
    assert "speakers" not in line
    assert "mic" in line


def test_level_meter_throttles_redraws(live):
    m = live.LevelMeter(interval=0.4, stream=io.StringIO())
    assert m.tick(now=1.0) is not None
    assert m.tick(now=1.2) is None, "must not redraw inside the interval"
    assert m.tick(now=1.5) is not None


def test_level_meter_describe_is_sticky_across_a_tick(live):
    """tick() resets the per-interval peak, so describe must not use it."""
    pytest.importorskip("numpy")
    import numpy as np
    m = live.LevelMeter(interval=0, stream=io.StringIO())
    loud = (np.sin(np.arange(16000) / 40) * 20000).astype(np.int16).tobytes()
    m.feed(0, loud)
    m.feed(1, b"\x00\x00" * 100)
    m.tick(now=1.0)
    assert "audio present" in m.describe(0)
    assert "SILENT" in m.describe(1)


def test_level_meter_ignores_a_zero_length_buffer(live):
    pytest.importorskip("numpy")
    m = live.LevelMeter(interval=0, stream=io.StringIO())
    m.feed(0, b"")
    assert "SILENT" in m.describe(0)


def test_level_meter_close_is_safe_when_nothing_was_drawn(live):
    m = live.LevelMeter(stream=io.StringIO())
    m.close()  # must not raise


def test_transport_falls_back_when_legacy_kwarg_is_unsupported(live):
    """Older `websockets` builds have no `legacy` parameter.

    Passing it unconditionally raises TypeError, which would abort the
    session before any audio is sent.
    """
    calls = []

    class _Old:
        legacy = "unsupported"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def connect_new(url, **kw):
        calls.append(kw)
        return _Old()

    def connect_old(url, **kw):
        # A pre-legacy `websockets` build rejects the keyword outright.
        if "legacy" in kw:
            raise TypeError(
                "connect() got an unexpected keyword argument 'legacy'")
        calls.append(kw)
        return _Old()

    t = live.Transport("ws://x/asr-mcp", "tok", lambda m: None)
    t._connect = connect_old
    with t:
        pass
    assert "legacy" not in calls[0], "must retry without legacy"

    t2 = live.Transport("ws://x/asr-mcp", "tok", lambda m: None)
    t2._connect = connect_new
    with t2:
        pass
    assert calls[1]["legacy"] is True


# ── Capture: module vs instance attributes ─────────────────────────────────

class _FakeModule:
    """pyaudiowpatch: paInt16 / paContinue are MODULE-level constants."""
    paInt16 = 8
    paContinue = 1
    paComplete = 2


class _FakeInstance:
    """PyAudio(): has open(), and deliberately nothing else."""

    def __init__(self, log):
        self._log = log

    def open(self, **kw):
        self._log.append(kw)
        return _FakeStream(kw["stream_callback"])


class _FakeStream:
    def __init__(self, callback=None):
        self._cb = callback
        self.started = self.stopped = False

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.stopped = True

    def close_stream(self):
        pass


def test_capture_reads_format_constants_from_the_module(live):
    """Regression: reading paInt16 off the instance crashes at open time.

    The real failure was "Could not open audio device: 'PyAudio' object has
    no attribute 'paInt16'" -- the first thing the client does after the
    banner prints, so nothing worked at all.
    """
    pytest.importorskip("numpy")
    log = []
    got = []
    dev = {"index": 15, "name": "Mic", "rate": 48000, "in": 2}
    cap = live.Capture(_FakeInstance(log), _FakeModule, dev, 2, 16000, 20,
                       got.append)
    assert log, "the stream was never opened"
    assert log[0]["format"] == _FakeModule.paInt16
    assert log[0]["input_device_index"] == 15
    assert log[0]["channels"] == 2
    assert log[0]["input"] is True
    # The callback must downmix to mono and keep the audio thread alive.
    cap._stream._cb(b"\x01\x00\x02\x00\x03\x00\x04\x00", 2, None, 0)
    assert len(got) == 1 and isinstance(got[0], (bytes, bytearray))
    cap.start()
    assert cap._stream.started
    cap.stop()


def test_capture_swallows_callback_errors(live):
    """An exception in the audio callback must not kill PortAudio's thread."""
    pytest.importorskip("numpy")
    cap = live.Capture(_FakeInstance([]), _FakeModule,
                       {"index": 0, "name": "M", "rate": 48000, "in": 1},
                       1, 16000, 20, lambda pcm: 1 / 0)
    out = cap._stream._cb(b"\x00" * 8, 4, None, 0)
    assert out is not None


# ── Downmix ────────────────────────────────────────────────────────────────

def test_to_mono_averages_stereo_int16(live):
    import numpy as np
    left = np.array([1000, -2000], dtype=np.int16)
    right = np.array([3000, -1000], dtype=np.int16)
    inter = np.empty(4, dtype=np.int16)
    inter[0::2], inter[1::2] = left, right
    out = np.frombuffer(live.to_mono(inter.tobytes(), 2), dtype=np.int16)
    assert out.tolist() == [2000, -1500]


def test_to_mono_passes_a_mono_stream_through(live):
    raw = b"\x01\x00\x02\x00"
    assert live.to_mono(raw, 1) == raw


def test_to_mono_ignores_a_trailing_partial_frame(live):
    """A dropped sample must not shift every channel by one."""
    import numpy as np
    pcm = np.array([100, 200, 300, 400, 999], dtype=np.int16).tobytes()
    out = np.frombuffer(live.to_mono(pcm, 2), dtype=np.int16)
    assert out.tolist() == [150, 350]


def test_to_mono_handles_an_empty_buffer(live):
    assert live.to_mono(b"", 2) == b""
    assert live.to_mono(b"\x01", 2) == b""


def test_to_mono_clips_instead_of_wrapping(live):
    """Loud stereo must clip to int16 range, not overflow to a negative peak."""
    import numpy as np
    pcm = np.array([32767, 32767, -32768, -32768], dtype=np.int16).tobytes()
    out = np.frombuffer(live.to_mono(pcm, 2), dtype=np.int16)
    assert out.tolist() == [32767, -32768]


# ── probe_devices ──────────────────────────────────────────────────────────

class _FakePyAudio:
    def get_device_count(self):
        return 2

    def get_device_info_by_index(self, i):
        return {
            0: {"name": "Mic", "maxInputChannels": 1, "maxOutputChannels": 0,
                "defaultSampleRate": 48000.0, "hostApi": 0},
            1: {"name": "Speakers [Loopback]", "maxInputChannels": 2,
                "maxOutputChannels": 0, "defaultSampleRate": 44100.0,
                "hostApi": 0, "isLoopbackDevice": True},
        }[i]

    def get_host_api_info_by_index(self, i):
        return {"name": "Windows WASAPI"}


def test_probe_devices_normalises_the_table(live):
    devs = live.probe_devices(_FakePyAudio())
    assert devs == [
        {"index": 0, "name": "Mic", "api": "Windows WASAPI", "in": 1,
         "out": 0, "rate": 48000, "is_loopback": False},
        {"index": 1, "name": "Speakers [Loopback]", "api": "Windows WASAPI",
         "in": 2, "out": 0, "rate": 44100, "is_loopback": True},
    ]


def test_probe_devices_detects_loopback_by_name_too(live):
    """A build without the isLoopbackDevice field still names them."""
    class _NoFlag(_FakePyAudio):
        def get_device_info_by_index(self, i):
            d = dict(super().get_device_info_by_index(i))
            d.pop("isLoopbackDevice", None)
            return d

    assert live.probe_devices(_NoFlag())[1]["is_loopback"] is True


def test_probe_devices_survives_a_missing_host_api(live):
    class _BadApi(_FakePyAudio):
        def get_host_api_info_by_index(self, i):
            raise RuntimeError("gone")

    assert live.probe_devices(_BadApi())[0]["api"] == "?"


# ── Quiet microphones: a level sweep, not one hand-picked signal ──────────
#
# Regression tests for a real Windows failure: 59s of speech from a headset
# mic produced ZERO turns, because `noise_floor_min` doubled as the absolute
# detection gate (0.004 * 2.5 = 0.01 RMS = -40 dBFS) and a normal speaking
# level on that mic sits below it. Each fix below is pinned independently so
# the next one cannot silently re-break the previous.

def _speechlike(seconds, peak=0.01, seed=0, rate=16000):
    """Alternating voiced bursts with Hanning envelopes and gaps.

    A sine tone is not representative: it has a stable RMS across the whole
    turn, so it never exercises the noise floor the way speech does.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = np.zeros(n, dtype=np.float32)
    t = 0
    while t < n:
        seg = min(int(rng.integers(0.15 * rate, 0.6 * rate)), n - t)
        f0 = float(rng.uniform(90, 200))
        tt = np.arange(seg) / rate
        v = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 8))
        v = v + 0.4 * rng.standard_normal(seg)
        x[t:t + seg] += (0.25 * v / 3.0) * np.hanning(seg)
        t += seg + int(rng.integers(0.1 * rate, 0.5 * rate))
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak * 32767).astype(np.int16).tobytes()


def test_quiet_microphone_still_produces_turns(live):
    """Speech at -57 dBFS used to yield zero turns."""
    pytest.importorskip("numpy")
    det = live.TurnDetector()
    turns = det.feed(_speechlike(20, peak=0.01, seed=0))
    assert len(turns) > 5, (
        f"a quiet but perfectly audible mic produced {len(turns)} turns")


def test_very_quiet_audio_is_still_rejected(live):
    """The fix must not turn the detector into a noise trigger."""
    pytest.importorskip("numpy")
    det = live.TurnDetector()
    assert det.feed(_speechlike(20, peak=0.0008, seed=0)) == []


def test_start_threshold_min_is_the_absolute_gate(live):
    """`noise_floor_min` only guards the floor; it is not the gate."""
    start, end = live.TurnDetector()._thresholds()
    assert start >= live.DETECTOR_DEFAULTS["start_threshold_min"]
    assert end < start, "hysteresis requires the end threshold below start"
    # The old formula was max(floor*ratio, noise_floor_min) = 0.01.
    assert start < 0.01, f"start threshold {start} is back at the old -40 dBFS floor"


def test_min_voiced_gate_compares_milliseconds(live):
    """3 voiced frames = 96ms is below min_voiced_ms (120ms) and is dropped.

    Counting frames instead (int(120/32) = 3) kept these turns while the
    server dropped them, so the two detectors disagreed on short answers.
    """
    from asr_mcp.streaming.turn_detector import TurnDetector as ServerDetector

    pytest.importorskip("numpy")
    # One 64ms burst = 2 voiced frames, below the gate for both detectors.
    audio = _silence(0.4) + _pcm(0.064) + _silence(1.2)
    assert live.TurnDetector().feed(audio) == []
    assert ServerDetector().feed(audio) == []


def test_client_and_server_detectors_agree_across_a_level_sweep(live):
    """The strongest form of the equality requirement.

    One hand-picked clip passed while the detectors were still wrong; a sweep
    over signal level and chunk size is what exposed the float32 sqrt and the
    frame-vs-millisecond gate. Exact equality, on purpose.
    """
    from asr_mcp.streaming.turn_detector import TurnDetector as ServerDetector

    pytest.importorskip("numpy")
    for seed in (0, 1, 2):
        for peak in (0.5, 0.05, 0.01, 0.005):
            pcm = _speechlike(20, peak=peak, seed=seed)
            for chunk in (1024, 2134):
                client, server = live.TurnDetector(), ServerDetector()
                c_turns, s_turns = [], []
                for i in range(0, len(pcm), chunk):
                    c_turns += client.feed(pcm[i:i + chunk])
                    s_turns += server.feed(pcm[i:i + chunk])
                c_tail, s_tail = client.flush(), server.flush()
                if c_tail:
                    c_turns.append(c_tail)
                if s_tail:
                    s_turns.append(s_tail)
                assert [(t.start_sample, t.end_sample, t.reason)
                        for t in c_turns] == \
                       [(t.start_sample, t.end_sample, t.reason)
                        for t in s_turns], (
                    f"seed={seed} peak={peak} chunk={chunk}: "
                    f"client {len(c_turns)} turns, server {len(s_turns)}")


def test_rms_is_computed_in_float32_like_the_server(live):
    """math.sqrt(float(...)) promoted to float64 and moved turn boundaries.

    The difference is ~1 ULP (3.7e-9) — invisible in the value, but enough
    to flip an `rms >= threshold` comparison on a borderline frame.
    """
    from asr_mcp.streaming.turn_detector import rms as server_rms

    pytest.importorskip("numpy")
    det = live.TurnDetector()
    for i in range(200):
        frame = (np_random_frame(i)).tobytes()
        assert det._rms(frame) == server_rms(frame), (
            f"frame {i}: client {det._rms(frame)!r} != server {server_rms(frame)!r}")


def np_random_frame(i):
    import numpy as np

    rng = np.random.default_rng(i)
    return (rng.standard_normal(512) * 3000).astype(np.int16)


# ── Level reporting ───────────────────────────────────────────────────────

def test_level_meter_summary_reports_a_silent_channel(live):
    import io

    meter = live.LevelMeter(interval=0, stream=io.StringIO())
    meter.feed(0, _pcm(0.1))
    meter.tick(now=1.0)
    meter.finalise()
    summary = meter.summary()
    assert "mic: peak" in summary
    assert "SILENT" in summary, "a channel with no audio must say so"


def test_level_meter_explains_a_zero_turn_run(live):
    import io

    buf = io.StringIO()
    meter = live.LevelMeter(interval=0, stream=buf)
    meter.feed(0, _silence(1.0))
    meter.finalise()
    meter.report_if_silent(0, out=buf)
    text = buf.getvalue()
    assert "No turns were detected" in text
    assert "SILENT" in text, "the report must name the measured level"
    # A run that DID send turns must stay quiet.
    buf2 = io.StringIO()
    m2 = live.LevelMeter(interval=0, stream=buf2)
    m2.feed(0, _pcm(0.2))
    m2.finalise()
    m2.report_if_silent(3, out=buf2)
    assert buf2.getvalue() == ""


def test_turns_sent_and_transcripts_received_are_separate(tmp_path, monkeypatch, live):
    """A session can send turns and get no replies back.

    These were one counter, so a run where the server received 24 turns and
    returned nothing reported "Turns sent: 0" and blamed the microphone.
    """
    s = _session(tmp_path, monkeypatch, live)
    assert s.turns_sent == 0
    assert s.transcript_count == 0
    s.on_message({"type": "transcript", "start": 1.0, "end": 2.0,
                  "text": "hello", "channel": 0, "speaker": "You",
                  "speaker_confidence": 1.0, "speaker_source": "input_device"})
    s.turns_sent += 1
    assert s.transcript_count == 1
    data = json.loads(s.write_sidecar({}).read_text(encoding="utf-8"))
    assert data["turns_sent"] == 1
    assert data["transcripts_received"] == 1
