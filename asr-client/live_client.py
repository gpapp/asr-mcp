#!/usr/bin/env python3
"""Live (real-time) transcription client for asr-mcp — Windows.

Records your microphone **and** your default sound device (WASAPI loopback),
streams detected turns to the server for immediate transcription, and on
shutdown re-diarizes the local recording so the final transcript carries real
speaker names.

    python live_client.py [--outdir DIR] [--no-speaker] [--no-rediag]

Why it is separate from ``transcribe_client.py``
-----------------------------------------------
That script is standard-library only and is packaged as a standalone zip with
a CI guard on that.  Live capture needs ``sounddevice`` (WASAPI), ``numpy`` and
``soxr`` (resampling to the 16 kHz the server expects), so this is a separate
file with its own ``requirements-live.txt`` and the CI guard skips it.  The
transcript *format* is shared by importing the render helpers from
``transcribe_client``, so both clients produce identical output.

The loopback channel
--------------------
Your speakers carry *everyone's* voice, including your own (echoed).  A meeting
is therefore recorded as two channels:

* channel 0 — microphone, which is **you**.  The server labels it from the
  input device and never voiceprint-matches it against your own profile
  (matching a voice against its own embedding returns a confident wrong
  answer).
* channel 1 — loopback, which is everyone else.  The server voiceprint-matches
  it, and withholds the name when the match is weak rather than guessing.

Re-diarization on shutdown
--------------------------
``_transcribe_file`` calls the ASR backend exactly once on the whole file and
then attributes its output items onto diarized turns afterwards, so attribution
is a pure function of (items, turns).  The live session already has the items
(they were decoded in real time), so shutdown uploads the recording plus those
items to ``POST /api/asr/attribution``, which runs diarization and nothing
else — roughly a fifth of the cost of a full re-transcription on the
52-minute podcast benchmark.  **The live text is never re-decoded.**

The recording is kept so a session can be re-processed later; it is named
after the session start timestamp, e.g. ``20260929-141203.wav``.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Transcript rendering is shared with the file client so both emit the same
# `[HH:MM:SS] Speaker (NN%):` layout.
from transcribe_client import (  # noqa: E402
    ClientError,
    build_transcript,
    encode_multipart,
    get_config,
    get_language,
    load_env,
    open_request,
    request_json,
)

# ── Wire protocol (mirrors asr_mcp/streaming/protocol.py) ───────────────────
# Re-implemented rather than imported: this file ships as a standalone zip and
# must not reach into the server package.  tests/test_live_client.py asserts
# these constants still match the server module, so they cannot silently desync.
TURN_MAGIC = b"LVT1"
PROTOCOL_VERSION = 1
MSG_TURN = 1
MSG_FLUSH = 2
TURN_HEADER_FMT = "<4sBBHQI I"
TURN_HEADER_SIZE = 24
SAMPLE_RATE = 16000

CHANNEL_MIC = 0
CHANNEL_SPEAKER = 1


def pack_turn(channel: int, start_sample: int, pcm: bytes, sequence: int = 0) -> bytes:
    """One turn frame: magic | version | type | channel | start | count | seq | PCM."""
    if len(pcm) % 2:
        raise ValueError("PCM payload must be a whole number of int16 samples")
    import struct

    header = struct.pack(
        TURN_HEADER_FMT, TURN_MAGIC, PROTOCOL_VERSION, MSG_TURN, int(channel),
        int(start_sample), len(pcm) // 2, int(sequence),
    )
    return header + bytes(pcm)


def pack_flush() -> bytes:
    import struct

    return struct.pack(
        TURN_HEADER_FMT, TURN_MAGIC, PROTOCOL_VERSION, MSG_FLUSH, 0, 0, 0, 0,
    )


# ── Turn detection (mirrors asr_mcp/streaming/turn_detector.py) ────────────
# Same algorithm and the same defaults, re-implemented for the client: the
# server must not be imported, and VAD has to run here anyway so the live and
# offline turn boundaries agree.
DETECTOR_DEFAULTS = {
    "frame_ms": 32.0,
    "noise_floor_ratio": 3.0,
    "noise_floor_min": 0.004,
    "noise_floor_max": 0.05,
    "start_threshold_ratio": 2.5,
    "end_threshold_ratio": 1.6,
    "start_confirm_frames": 2,
    "end_confirm_frames": 3,
    "hangover_ms": 320,
    "pre_roll_ms": 160,
    "post_roll_ms": 200,
    "min_voiced_ms": 120,
    "max_turn_sec": 30.0,
}


class Turn:
    """One detected utterance on a channel."""

    __slots__ = ("start_sample", "end_sample", "pcm", "reason", "peak_rms")

    def __init__(self, start_sample, end_sample, pcm, reason="end", peak_rms=0.0):
        self.start_sample = int(start_sample)
        self.end_sample = int(end_sample)
        self.pcm = pcm
        self.reason = reason
        self.peak_rms = float(peak_rms)

    @property
    def start_sec(self):
        return self.start_sample / SAMPLE_RATE

    @property
    def end_sec(self):
        return self.end_sample / SAMPLE_RATE


class TurnDetector:
    """Adaptive endpointing: noise floor, hysteresis, pre/post-roll, hangover.

    Mirrors ``asr_mcp.streaming.turn_detector.TurnDetector`` so a turn cut here
    is cut identically when the same audio is re-diarized offline.
    """

    def __init__(self, cfg=None, sample_rate=SAMPLE_RATE):
        import numpy as np

        self._np = np
        self.cfg = dict(DETECTOR_DEFAULTS)
        self.cfg.update(cfg or {})
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(self.sample_rate * self.cfg["frame_ms"] / 1000.0))
        self.pre_roll_frames = max(
            0, int(self.cfg["pre_roll_ms"] / self.cfg["frame_ms"]))
        self.hangover_frames = max(
            0, int(self.cfg["hangover_ms"] / self.cfg["frame_ms"]))
        self.min_voiced_frames = max(
            1, int(self.cfg["min_voiced_ms"] / self.cfg["frame_ms"]))
        self.max_turn_frames = max(
            1, int(self.cfg["max_turn_sec"] * 1000.0 / self.cfg["frame_ms"]))

        self._carry = b""
        self._frame_index = 0
        self._noise_floor = float(self.cfg["noise_floor_min"])
        self._pre_roll = []
        self._in_turn = False
        self._confirm_run = 0
        self._silent_run = 0
        self._turn_start_frame = 0
        self._turn_samples = []
        self._turn_peak = 0.0
        self._turn_voiced = 0
        self.dropped_turns = 0

    # -- thresholds ------------------------------------------------------
    def _thresholds(self):
        floor = self._noise_floor
        start = max(floor * self.cfg["start_threshold_ratio"], self.cfg["noise_floor_min"])
        end = max(floor * self.cfg["end_threshold_ratio"], self.cfg["noise_floor_min"])
        return start, end

    def _rms(self, frame):
        np = self._np
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return 0.0
        return float(math.sqrt(float(np.mean(samples * samples))))

    def _track_floor(self, rms):
        """Follow quiet quickly, loud slowly — the floor must not chase speech."""
        if rms < self._noise_floor:
            self._noise_floor = 0.7 * self._noise_floor + 0.3 * rms
        else:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
        lo, hi = self.cfg["noise_floor_min"], self.cfg["noise_floor_max"]
        self._noise_floor = max(lo, min(hi, self._noise_floor))

    # -- main loop -------------------------------------------------------
    def feed(self, pcm: bytes):
        """Feed int16 LE PCM; return the list of completed turns."""
        data = self._carry + pcm if self._carry else pcm
        size = self.frame_samples * 2
        out = []
        offset = 0
        while offset + size <= len(data):
            frame = data[offset:offset + size]
            offset += size
            turn = self._frame(frame)
            if turn is not None:
                out.append(turn)
        self._carry = data[offset:]
        return out

    def _frame(self, frame: bytes):
        rms = self._rms(frame)
        self._track_floor(rms)
        start_th, end_th = self._thresholds()
        idx = self._frame_index
        self._frame_index += 1

        if not self._in_turn:
            if rms >= start_th:
                self._confirm_run += 1
            else:
                self._confirm_run = 0
            # Pre-roll keeps the frames just before the confirmed onset so a
            # word onset is not clipped.
            self._pre_roll.append((idx, frame))
            if len(self._pre_roll) > self.pre_roll_frames:
                self._pre_roll.pop(0)
            if self._confirm_run >= self.cfg["start_confirm_frames"]:
                self._open_turn()
            return None

        self._turn_samples.append(frame)
        self._turn_peak = max(self._turn_peak, rms)
        if rms >= end_th:
            self._turn_voiced += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
            if self._silent_run >= self.hangover_frames:
                return self._close_turn("hangover")
        if len(self._turn_samples) >= self.max_turn_frames:
            return self._close_turn("max_turn")
        return None

    def _open_turn(self):
        """Open a turn, keeping the pre-roll so the onset is not clipped.

        Mirrors the server exactly, including the distinction that only frames
        from the confirmation index onward count as *voiced* — the pre-roll
        before it is padding, so a click cannot pass the minimum-voiced gate.
        """
        need = max(1, int(self.cfg["start_confirm_frames"]))
        confirm_idx = self._frame_index - need
        older = [(i, f) for i, f in self._pre_roll if i < confirm_idx]
        self._in_turn = True
        self._confirm_run = 0
        self._silent_run = 0
        self._turn_voiced = 0
        self._turn_peak = 0.0
        if older:
            self._turn_start_frame = older[0][0]
            self._turn_samples = [f for _, f in older]
        else:
            self._turn_start_frame = confirm_idx
            self._turn_samples = []
        self._turn_samples.extend(f for i, f in self._pre_roll if i >= confirm_idx)
        _, end_th = self._thresholds()
        for i, f in self._pre_roll:
            if i >= confirm_idx:
                self._turn_peak = max(self._turn_peak, self._rms(f))
                if self._rms(f) >= end_th:
                    self._turn_voiced += 1
        self._pre_roll = []

    def _close_turn(self, reason):
        """Close the turn, trimming the trailing hangover back to post_roll."""
        frames = self._turn_samples
        voiced = self._turn_voiced
        self._turn_samples = []
        self._in_turn = False
        self._turn_voiced = 0
        self._confirm_run = 0
        self._silent_run = 0
        self._pre_roll = []
        if not frames:
            return None

        # Trailing silence beyond post-roll is padding, not audio: keeping it
        # would make a live turn longer than the offline turn at the same place.
        hangover = max(1, int(round(self.cfg["hangover_ms"] / self.cfg["frame_ms"])))
        post_roll = max(0, int(round(self.cfg["post_roll_ms"] / self.cfg["frame_ms"])))
        keep = max(1, len(frames) - hangover + post_roll)
        frames = frames[:keep]

        if voiced < self.min_voiced_frames:
            self.dropped_turns += 1
            return None

        pcm = b"".join(frames)
        start = self._turn_start_frame * self.frame_samples
        return Turn(start, start + len(pcm) // 2, pcm,
                    reason=reason, peak_rms=self._turn_peak)

    def flush(self):
        """Emit a turn still open at end of stream, exactly once."""
        if not self._in_turn:
            return None
        return self._close_turn("flush")


# ── Audio capture ──────────────────────────────────────────────────────────

def _require_live_deps():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("sounddevice", "sounddevice"),
                     ("soxr", "soxr")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ClientError(
            f"Live mode needs {', '.join(missing)}. Install them with:\n"
            f"    {sys.executable} -m pip install -r requirements-live.txt\n"
            "(or run transcribe.bat live, which installs them automatically)"
        )


def list_devices():
    """Print capture and playback devices (helps pick a loopback source)."""
    _require_live_deps()
    import sounddevice as sd

    print("Devices (loopback = capture-capable 'Render'/'Speakers' entries):\n")
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) < 1:
            continue
        api = sd.query_hostapis(dev["hostapi"])["name"]
        print(f"  [{idx}] {dev['name']}")
        print(f"        api={api}  in={dev['max_input_channels']} "
              f"out={dev['max_output_channels']}  default_rate={dev['default_samplerate']:.0f}")


def find_loopback_device(sd):
    """Return (index, samplerate) of the default render device, or (None, None).

    WASAPI exposes the speakers as a *capture* device with a hostapi loopback
    flag; ``sd.query_devices()`` reports those as ordinary inputs, so the
    default playback device is located via ``default_speakers`` on its hostapi.
    """
    try:
        hostapis = sd.query_hostapis()
        for api in hostapis:
            if not api.get("default_speakers"):
                continue
            if "wasapi" not in str(api.get("name", "")).lower():
                continue
            for idx, dev in enumerate(sd.query_devices()):
                if (dev.get("hostapi") == api["index"]
                        and dev.get("max_input_channels", 0) >= 1):
                    return idx, float(dev["default_samplerate"])
    except Exception:
        pass
    return None, None


class Resampler:
    """Streaming resample (device rate -> 16 kHz) with ratio-change guard."""

    def __init__(self, in_rate, out_rate=SAMPLE_RATE):
        import soxr

        self._soxr = soxr
        self.in_rate = float(in_rate)
        self.out_rate = int(out_rate)
        self._stream = soxr.ResampleStream(self.in_rate, self.out_rate, 1, "HQ")

    def process(self, pcm_bytes: bytes) -> bytes:
        import numpy as np

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        out = self._stream.resample_chunk(samples)
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def flush(self) -> bytes:
        import numpy as np

        out = self._stream.resample_chunk(np.zeros(0, dtype=np.int16), last=True)
        if out is None or len(out) == 0:
            return b""
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


class ChannelRecorder:
    """Writes 16 kHz mono int16 PCM to a WAV as it arrives."""

    def __init__(self, path: Path):
        self.path = path
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self.samples = 0

    def write(self, pcm: bytes):
        self._wav.writeframes(pcm)
        self.samples += len(pcm) // 2

    def close(self):
        try:
            self._wav.close()
        except Exception:
            pass

    @property
    def duration_sec(self):
        return self.samples / SAMPLE_RATE


# ── WebSocket transport ────────────────────────────────────────────────────

def _ws_url(base: str) -> str:
    """Convert the REST base URL to its ws:// equivalent, keeping the prefix."""
    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/api/asr/ws/stream"


class Transport:
    """Thin websockets wrapper: sends turn frames, receives JSON messages.

    Imported lazily so ``--help`` and the pure-python unit tests work without
    the ``websockets`` package installed.
    """

    def __init__(self, url, token, on_message):
        from websockets.sync.client import connect

        self._connect = connect
        self.url = url
        self.token = token
        self.on_message = on_message
        self.ws = None
        self.closed_by_server = False
        self.stats = None

    def __enter__(self):
        self.ws = self._connect(
            self.url,
            additional_headers={"X-API-Key": self.token},
            max_size=None,
            ping_interval=20,
        )
        return self

    def __exit__(self, *exc):
        self.close()

    def send_turn(self, channel, turn: Turn, sequence):
        self.ws.send(pack_turn(channel, turn.start_sample, turn.pcm, sequence))

    def send_flush(self):
        try:
            self.ws.send(pack_flush())
        except Exception:
            pass

    def receive(self, timeout=None):
        """One message, or None when the socket is closed/times out."""
        try:
            raw = self.ws.recv(timeout=timeout) if timeout else self.ws.recv()
        except Exception:
            return None
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if isinstance(msg, dict) and msg.get("type") == "stats":
            self.stats = msg
            self.closed_by_server = True
        return msg

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


# ── Live session ───────────────────────────────────────────────────────────

class LiveSession:
    def __init__(self, args):
        self.args = args
        self.base, self.token = get_config()
        self.language = args.language or get_language()
        self.started = datetime.now()
        self.stamp = self.started.strftime("%Y%m%d-%H%M%S")
        self.outdir = Path(args.outdir).expanduser()
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.items = []          # live ASR items -> the re-attribution input
        self.gaps = []           # server-reported drops
        self.turn_count = 0
        self.drop_count = 0
        self.sequence = 0
        self.covered_sec = 0.0
        self.server_stats = None
        self.streaming_failed = None
        self._lock = threading.Lock()

    # -- paths -----------------------------------------------------------
    @property
    def mic_path(self):
        return self.outdir / f"{self.stamp}.wav"

    @property
    def speaker_path(self):
        return self.outdir / f"{self.stamp}-speaker.wav"

    @property
    def txt_path(self):
        return self.outdir / f"{self.stamp}.txt"

    @property
    def sidecar_path(self):
        return self.outdir / f"{self.stamp}.asr.json"

    # -- message handling ------------------------------------------------
    def on_message(self, msg):
        kind = msg.get("type")
        if kind == "transcript":
            with self._lock:
                self.items.append({
                    "start": msg.get("start"),
                    "end": msg.get("end"),
                    "text": (msg.get("text") or "").strip(),
                    "channel": msg.get("channel"),
                    "speaker": msg.get("speaker"),
                    "speaker_confidence": msg.get("speaker_confidence"),
                    "speaker_source": msg.get("speaker_source"),
                    "uncertain": bool(msg.get("uncertain")),
                    "attribution_reason": msg.get("attribution_reason"),
                })
                self.turn_count += 1
            speaker = msg.get("speaker") or "UNKNOWN"
            conf = msg.get("speaker_confidence")
            conf_s = f" ({conf:.0%})" if isinstance(conf, (int, float)) and conf > 0 else ""
            stamp = _fmt_hms(msg.get("start") or 0)
            print(f"[{stamp}] {speaker}{conf_s}: {(msg.get('text') or '').strip()}",
                  flush=True)
        elif kind == "empty":
            pass  # a detected turn the decoder produced no words for
        elif kind == "dropped":
            with self._lock:
                self.gaps.append({
                    "start": msg.get("start"), "end": msg.get("end"),
                    "channel": msg.get("channel"),
                    "reason": msg.get("reason"),
                })
                self.drop_count += 1
            print(f"  !! {msg.get('reason')}: dropped turn "
                  f"{msg.get('start')}-{msg.get('end')}s (decoder fell behind)",
                  file=sys.stderr, flush=True)
        elif kind == "stats":
            self.server_stats = msg
            self.covered_sec = float(msg.get("covered_sec") or 0.0)

    # -- sidecar ---------------------------------------------------------
    def write_sidecar(self, extra=None):
        with self._lock:
            payload = {
                "session_started": self.started.isoformat(timespec="seconds"),
                "language": self.language,
                "server_url": self.base,
                "sample_rate": SAMPLE_RATE,
                "channels": {
                    "0": "microphone (local user)",
                    "1": "default sound device loopback (everyone else)",
                },
                "recording": {
                    "mic_wav": str(self.mic_path),
                    "speaker_wav": str(self.speaker_path) if self.speaker_path.exists() else None,
                },
                "asr_source": "live_stream",
                "turns": self.turn_count,
                "gaps": self.gaps,
                "server_stats": self.server_stats,
                "items": self.items,
            }
            if extra:
                payload.update(extra)
        self.sidecar_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.sidecar_path

    # -- re-attribution --------------------------------------------------
    def reattribute(self):
        """Upload the recording + live items; diarize only, never re-decode.

        Returns the server response dict, or None when the user opted out or
        the request failed.  A failure is NOT fatal: the live transcript is
        already on disk, so the worst case is a live-only transcript.
        """
        if self.args.no_rediag:
            return None
        with self._lock:
            items = [
                {"start": it["start"], "end": it["end"], "text": it["text"]}
                for it in self.items if (it["text"] or "").strip()
            ]
        if not items:
            print("\nNo transcribed text to re-attribute.", file=sys.stderr)
            return None

        print(f"\nRe-diarizing {self.mic_path.name} (no re-transcription) ...",
              file=sys.stderr, flush=True)
        audio = self.mic_path
        fields = {"items": json.dumps(items, ensure_ascii=False)}
        try:
            body, ctype = encode_multipart("file", audio, fields=fields)
            resp = request_json(
                "POST",
                f"{self.base}/api/asr/attribution/upload",
                data=body,
                headers={
                    "X-API-Key": self.token,
                    "Content-Type": ctype,
                },
                timeout=3600,
            )
        except ClientError as e:
            print(f"Re-attribution failed: {e}", file=sys.stderr)
            print("The live transcript below is still valid.", file=sys.stderr)
            return None
        return resp

    def build_live_result(self):
        """Shape the live items like a transcribe result for build_transcript."""
        return {
            "total_speakers": len({it["speaker"] for it in self.items if it["speaker"]}),
            "audio_duration_sec": 0.0,
            "results": self.items,
        }

    def final_result(self, rediag):
        """Prefer the re-diarized result; fall back to the live items."""
        if rediag and not rediag.get("error") and rediag.get("results"):
            return rediag, True
        return self.build_live_result(), False

    def write_transcript(self, result, used_rediag):
        date_str = self.started.strftime("%Y-%m-%d %H:%M:%S")
        self.txt_path.write_text(
            build_transcript(self.mic_path.name, result, date_str), encoding="utf-8")
        banner = ("Re-diarized" if used_rediag else "Live (diarization skipped)")
        text = self.txt_path.read_text(encoding="utf-8")
        header = (f"# {banner}\n"
                  f"# Session: {self.started.isoformat(timespec='seconds')}\n"
                  f"# Audio: {self.mic_path}\n"
                  f"# Turns: {self.turn_count}  Gaps: {self.drop_count}\n")
        self.txt_path.write_text(header + text, encoding="utf-8")
        return self.txt_path


def _fmt_hms(sec):
    sec = max(0, int(round(float(sec or 0))))
    hours, rem = divmod(sec, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ── Main loop ──────────────────────────────────────────────────────────────

def run_live(args):
    _require_live_deps()
    import sounddevice as sd

    session = LiveSession(args)
    loop_index, loop_rate = (None, None)
    if not args.no_speaker:
        loop_index, loop_rate = find_loopback_device(sd)
        if loop_index is None:
            print("No WASAPI loopback device found — recording microphone only.\n"
                  "Other people's voices will be attributed from the mic (echo), "
                  "which is much less accurate.", file=sys.stderr)

    try:
        default_in = sd.query_devices(kind="input")
    except Exception:
        default_in = None
    mic_device = args.device if args.device is not None else default_in
    mic_rate = float(args.rate or (default_in["default_samplerate"] if default_in else 48000))
    block_ms = args.block_ms
    block_samples = int(mic_rate * block_ms / 1000)

    print(f"Session   : {session.stamp}")
    print(f"Server    : {session.base}")
    print(f"Language  : {session.language}")
    print(f"Microphone: {mic_device['name'] if mic_device else 'default'} @ {mic_rate:.0f} Hz")
    if loop_index is not None:
        print(f"Loopback  : {sd.query_devices(loop_index)['name']} @ {loop_rate:.0f} Hz")
    print(f"Recording : {session.mic_path}")
    print("Press Ctrl+C to stop.\n")

    mic_rec = ChannelRecorder(session.mic_path)
    spk_rec = (ChannelRecorder(session.speaker_path)
               if loop_index is not None else None)
    mic_det = TurnDetector()
    spk_det = TurnDetector() if spk_rec is not None else None
    mic_res = Resampler(mic_rate)
    spk_res = Resampler(loop_rate) if spk_rec is not None else None

    out_q: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    def mic_callback(indata, frames, tinfo, status):
        if status:
            print(f"  mic status: {status}", file=sys.stderr)
        out_q.put((CHANNEL_MIC, mic_res.process(bytes(indata))))

    def spk_callback(indata, frames, tinfo, status):
        if status:
            print(f"  loopback status: {status}", file=sys.stderr)
        out_q.put((CHANNEL_SPEAKER, spk_res.process(bytes(indata))))

    streams = []
    try:
        streams.append(sd.InputStream(
            device=mic_device, channels=1, samplerate=mic_rate,
            blocksize=block_samples, dtype="int16", callback=mic_callback))
        if loop_index is not None:
            streams.append(sd.InputStream(
                device=loop_index, channels=1, samplerate=loop_rate,
                blocksize=block_samples, dtype="int16", callback=spk_callback))
        for s in streams:
            s.start()
    except Exception as e:
        for s in streams:
            try:
                s.stop()
            except Exception:
                pass
        raise ClientError(f"Could not open audio device: {e}") from None

    transport = None
    send_error = None
    try:
        try:
            transport = Transport(
                _ws_url(session.base), session.token, session.on_message)
            transport.__enter__()
        except Exception as e:
            # No server means no live items at all. Keep recording so the audio
            # survives, and report the failure honestly — an empty transcript
            # file that looks like success is worse than an error.
            send_error = e
            transport = None
            print(f"\nCould not connect to {session.base}: {e}", file=sys.stderr)
            print("Recording locally only — no transcript can be produced "
                  "without the server. The .wav is still saved; re-transcribe "
                  "it later with transcribe.bat.\n", file=sys.stderr)
            stop.set()

        # Receiver thread: the socket must keep draining while audio flows.
        def receiver():
            while not stop.is_set():
                msg = transport.receive(timeout=0.5)
                if msg is None:
                    if transport.closed_by_server:
                        break
                    continue
                transport.on_message(msg)

        rx = threading.Thread(target=receiver, daemon=True)
        rx.start()

        while not stop.is_set():
            try:
                channel, pcm = out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not pcm:
                continue
            if channel == CHANNEL_MIC:
                mic_rec.write(pcm)
                turns = mic_det.feed(pcm)
            else:
                spk_rec.write(pcm)
                turns = spk_det.feed(pcm)
            for turn in turns:
                session.sequence += 1
                try:
                    transport.send_turn(channel, turn, session.sequence)
                except Exception as e:
                    send_error = e
                    print(f"\nSend failed: {e}", file=sys.stderr)
                    stop.set()
                    break
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        stop.set()
        for s in streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

        # Flush both detectors so trailing speech is not lost, and pad both
        # recordings so they stay the same length as the sent audio.
        for det, rec, res, channel in (
            (mic_det, mic_rec, mic_res, CHANNEL_MIC),
            (spk_det, spk_rec, spk_res, CHANNEL_SPEAKER),
        ):
            if det is None or rec is None:
                continue
            tail = det.flush()
            if tail is not None:
                rec.write(tail.pcm)
                session.sequence += 1
                if transport is not None and not send_error:
                    try:
                        transport.send_turn(channel, tail, session.sequence)
                    except Exception:
                        pass
            rec.write(res.flush())

        if transport is not None:
            transport.send_flush()
            # Give the decoder time to drain before the socket closes; the
            # server sends its stats frame last.
            deadline = time.time() + 30
            while time.time() < deadline and not transport.closed_by_server:
                time.sleep(0.2)
            transport.close()
        mic_rec.close()
        if spk_rec is not None:
            spk_rec.close()

    if send_error:
        print(f"\nStreaming failed ({send_error}); no live items were collected.",
              file=sys.stderr)
        session.streaming_failed = str(send_error)
        session.write_sidecar({"asr_source": "failed", "error": str(send_error)})
        return session, None

    return session, session.reattribute()


def build_parser():
    p = argparse.ArgumentParser(
        prog="live_client.py",
        description="Real-time transcription + re-diarization (Windows).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Environment (.env): SERVER_URL, TOKEN, LANGUAGE",
    )
    p.add_argument("--outdir", default=str(SCRIPT_DIR),
                   help="where to write the .wav/.txt/.asr.json (default: script dir)")
    p.add_argument("--device", type=int, default=None,
                   help="microphone device index (default: system default input)")
    p.add_argument("--rate", type=float, default=None,
                   help="microphone sample rate (default: device default)")
    p.add_argument("--block-ms", type=float, default=64.0,
                   help="capture block size in ms (default 64)")
    p.add_argument("--language", default=None,
                   help="ISO 639-1 or 'auto' (default: LANGUAGE from .env)")
    p.add_argument("--no-speaker", action="store_true",
                   help="microphone only; skip the sound-device loopback channel")
    p.add_argument("--no-rediag", action="store_true",
                   help="skip re-diarization on shutdown (live labels only)")
    p.add_argument("--devices", action="store_true",
                   help="list audio devices and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.devices:
        list_devices()
        return 0
    try:
        session, rediag = run_live(args)
    except ClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    if getattr(session, "streaming_failed", None):
        # No server contact meant no text. Do not write a metadata-only .txt
        # that reads as a successful (empty) transcript.
        print(f"\nNo transcript written. Recording kept: {session.mic_path}",
              file=sys.stderr)
        print(f"Sidecar: {session.sidecar_path}", file=sys.stderr)
        return 1

    result, used_rediag = session.final_result(rediag)
    txt = session.write_transcript(result, used_rediag)
    session.write_sidecar({
        "final_transcript": str(txt),
        "attribution_source": "re_diarized" if used_rediag else "live",
    })

    print(f"\nTranscript: {txt}")
    print(f"Sidecar   : {session.sidecar_path}")
    print(f"Recording : {session.mic_path}")
    if session.gaps:
        print(f"WARNING: {len(session.gaps)} turn(s) dropped under decoder lag — "
              "the sidecar records them as gaps.", file=sys.stderr)
    uncertain = sum(1 for r in result.get("results", []) if r.get("uncertain"))
    if uncertain:
        print(f"WARNING: {uncertain} segment(s) have an UNKNOWN speaker "
              "(text kept, identity withheld).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
