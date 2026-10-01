"""Live-path boundary faults: declared span, client timeline, duplicated tail.

Three defects in the live path (plan F3 a/b/c), all of which make the transcript
disagree with the recording it came from:

* a **coalesced turn declared a span longer than the audio it carried** --
  ``head + tail`` skips the pause between the two merged turns, but the turn
  claimed the tail's real end, so every consumer of the span (attribution,
  ``covered_sec``, the shutdown re-attribution against the .wav) was up to
  ``merge_gap_sec`` past the audio that was actually sent;
* a **client-declared ``start_sample`` was trusted verbatim**, so one dropped
  capture block or a late-starting ``getDisplayMedia`` shifted every later
  boundary on that channel with nothing to say so;
* the **flushed tail was written to the .wav a second time**, making the
  recording up to one turn too long with the tail duplicated.

The three detectors (server, browser, Windows client) must stay identical, so
each fix is asserted on every implementation that owns it.
"""

import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_JS = REPO_ROOT / "asr_mcp" / "static" / "live.js"
NODE = shutil.which("node")

from asr_mcp.streaming import protocol as proto  # noqa: E402
from asr_mcp.streaming.turn_detector import (  # noqa: E402
    SAMPLE_RATE, TurnCoalescer, payload_samples,
)

SR = SAMPLE_RATE


@pytest.fixture
def patched(monkeypatch):
    """Neutralise model loading and record the ASR calls (as in the handler tests).

    ``live`` is a module-scoped fixture, so it cannot depend on the
    function-scoped ``monkeypatch``; every client-side test here asks for
    ``(patched, live)`` instead.
    """
    from asr_mcp.core import model_state, transcriber

    calls = []

    def fake_transcribe(audio=None, **kw):
        assert isinstance(audio, np.ndarray), type(audio)
        assert audio.dtype == np.float32, audio.dtype
        calls.append({"dur": len(audio) / SR, "kwargs": kw, "audio": audio})
        return {"text": "hello", "inference_time_sec": 0.01, "tokens_generated": 3}

    monkeypatch.setattr(model_state.state, "ensure_ready", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(transcriber, "transcribe_audio_sync", fake_transcribe)
    return calls


# ══════════════════════════════════════════════════════════════════════════
# 1. The declared span must equal the samples actually sent
# ══════════════════════════════════════════════════════════════════════════

def _server_turn(start_sec, dur_sec):
    from asr_mcp.streaming.turn_detector import Turn

    n = int(dur_sec * SR)
    return Turn(start_sample=int(start_sec * SR), end_sample=int(start_sec * SR) + n,
                audio=np.full(n, 0.1, dtype=np.float32))


def test_payload_samples_reads_every_payload_shape():
    assert payload_samples(b"\x00\x01" * 10) == 10
    assert payload_samples(bytearray(b"\x00\x01" * 10)) == 10
    assert payload_samples(np.zeros(7, dtype=np.float32)) == 7
    assert payload_samples(b"") == 0
    assert payload_samples(None) == 0


def test_server_coalescer_declares_exactly_the_samples_it_sends():
    c = TurnCoalescer({"merge_gap_sec": 1.0, "max_merge_sec": 15.0}, clock=lambda: 0.0)
    c.submit(_server_turn(0.0, 1.0))
    c.submit(_server_turn(1.4, 0.8))            # 0.4s pause, merged
    merged = c.flush()[0]

    # 1.0s head + 0.4s silence + 0.8s tail. The declared span equals the audio
    # actually carried, and the tail sits at its true offset rather than 0.4s
    # early.
    assert payload_samples(merged.audio) == 2.2 * SR
    assert merged.end_sample - merged.start_sample == payload_samples(merged.audio)
    # With no hangover trim in this fixture the two ends coincide; when there is
    # a trim, end_sample is the end of the audio and audio_end_sample is the
    # recording's.
    assert merged.audio_end_sample == int(2.2 * SR)
    assert merged.audio_end_sample >= merged.end_sample


def test_bytes_payload_coalescer_declares_the_same_span(live):
    """The client's bytes payload must obey the same rule as the server's array.

    ``_rebuild_turn`` serves both shapes; if the byte branch declared the span
    differently the two endpoints would disagree about the same merged turn.
    """
    c = TurnCoalescer({"merge_gap_sec": 1.0, "max_merge_sec": 15.0}, clock=lambda: 0.0)
    c.submit(_server_turn(0.0, 1.0))
    c.submit(_server_turn(1.4, 0.8))
    merged = c.flush()[0]
    assert merged.end_sample - merged.start_sample == payload_samples(merged.audio)

    def t(start_sec, dur_sec):
        n = int(dur_sec * SR)
        return live.Turn(start_sample=int(start_sec * SR),
                         end_sample=int(start_sec * SR) + n,
                         pcm=b"\x00\x01" * n, reason="hangover", peak_rms=0.1)

    cc = live.TurnCoalescer({"merge_gap_sec": 1.0, "max_merge_sec": 15.0},
                            clock=lambda: 0.0)
    cc.submit(t(0.0, 1.0))
    cc.submit(t(1.4, 0.8))
    cm = cc.flush()[0]
    assert (cm.start_sample, cm.end_sample) == (merged.start_sample, merged.end_sample)


def _run_ws(frames):
    import asyncio

    from asr_mcp.streaming import handler as hd

    class WS:
        def __init__(self, frames):
            self._frames = list(frames)
            self.sent = []

        async def accept(self):
            pass

        async def receive(self):
            if self._frames:
                return self._frames.pop(0)
            return {"type": "websocket.disconnect"}

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, *a, **k):
            pass

    ws = WS(frames)
    asyncio.run(hd.handle_ws_stream(ws))
    return ws


def _pcm(sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * 32767 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes()


def _raw(channel, payload, seq=0):
    """A legacy raw-PCM packet (channel, sequence, int16 LE PCM)."""
    return struct.pack("<II", channel, seq) + payload


def test_merged_span_reaches_the_client_as_the_audio_length(patched):
    """End to end: the ``end`` a client is told must match what was decoded.

    The legacy raw-PCM path is the only one whose turns the server coalesces,
    so it is the only one where the declared span reaches the wire.  The pause
    is 0.9s -- inside ``merge_gap_sec``, so the turns merge -- and the merged
    payload carries the pause as silence, so what the decoder reads is the whole
    1.9s timeline and the ``end`` the client is told must match it exactly.
    """
    silence = np.zeros(int(1.0 * SR), dtype="<i2").tobytes()
    gap = np.zeros(int(0.9 * SR), dtype="<i2").tobytes()
    ws = _run_ws([
        {"type": "websocket.receive", "bytes": _raw(0, silence + _pcm(0.5))},
        {"type": "websocket.receive", "bytes": _raw(0, gap + _pcm(0.5), 1)},
    ])

    msgs = [m for m in ws.sent if m.get("type") == "transcript"]
    assert len(msgs) == 1, [m.get("type") for m in ws.sent]
    decoded = patched[0]["dur"]
    assert decoded > 1.5, "the two turns did not merge; the gap was too long"
    assert msgs[0]["duration"] == pytest.approx(decoded, abs=0.02), (
        f"declared {msgs[0]['duration']}s but {decoded}s of audio was decoded"
    )
    # covered_sec describes the RECORDING. It must not be less than the span the
    # client was told about, or the client would conclude its tail was dropped.
    stats = [m for m in ws.sent if m.get("type") == "stats"][0]
    assert stats["covered_sec"] >= msgs[0]["end"] - 0.05, (stats["covered_sec"],
                                                          msgs[0]["end"])


_JS_WAV_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(LIVE, 'utf8');
let clock = 0;
const performance = { now: () => clock };
const mod = { exports: {} };
new Function('module', 'exports', 'window', 'document', 'performance', 'location',
             'WebSocket', 'AudioWorkletNode', src)(
  mod, mod.exports, {}, { getElementById: () => null, createElement: () => ({}) },
  performance, { protocol: 'https:', host: 'x' }, function () {}, function () {});
const Live = mod.exports;

function tone(sec, amp) {
  const n = Math.round(sec * 16000);
  const a = new Int16Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.round(amp * Math.sin(2 * np(i)));
  function np(k) { return 2 * Math.PI * 220 * k / 16000; }
  return a;
}

(async () => {
  // 1.0s speech, 0.5s silence, 0.8s speech still open when Stop is pressed:
  // the harness signal of tests/test_live_js_protocol.py, so the flushed tail
  // is a real coalesced turn.
  const parts = [tone(1.0, 9000), new Int16Array(8000), tone(0.8, 9000)];
  const total = parts.reduce((s, a) => s + a.length, 0);
  const signal = new Int16Array(total);
  let at = 0;
  for (const p of parts) { signal.set(p, at); at += p.length; }

  // No WebSocket: _sendTurn is a no-op without one, and the WAV is what is
  // under test here.
  const session = new Live.Session({ language: 'auto' });
  for (let i = 0; i < total; i += 1024) {
    session._onPcm(0, signal.slice(i, i + 1024));
    clock += 1024 / 16000;
    session.poll();
  }
  const before = session.channel(0).chunks.reduce((a, c) => a + c.length, 0);
  await session.finish(0);
  const after = session.channel(0).chunks.reduce((a, c) => a + c.length, 0);
  process.stdout.write(JSON.stringify({ total, before, after,
                                        micSeconds: session.micSeconds() }));
})().catch(e => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_wav_holds_the_stream_exactly_once():
    """The flushed tail must not be appended to the recording a second time.

    Every 1024-sample block is pushed to ``chunks`` as it arrives, so the
    recording already contains the flushed turn; writing it again made the WAV
    up to one turn too long with the tail repeated.
    """
    script = f"const LIVE = {str(LIVE_JS)!r};\n" + _JS_WAV_HARNESS
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["before"] == result["total"]
    assert result["after"] == result["total"], (
        f"the flushed tail was written again: {result['total']} samples in, "
        f"{result['after']} in the WAV"
    )
    assert result["micSeconds"] == pytest.approx(result["total"] / SR, abs=1e-6)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_coalescer_declares_exactly_the_samples_it_sends():
    script = f"""
const fs = require('fs');
const src = fs.readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{ exports: {{}} }};
new Function('module', 'exports', 'window', 'performance', src)(
  mod, mod.exports, {{}}, {{ now: () => 0 }});
const Live = mod.exports;
function t(startSec, durSec) {{
  const n = Math.round(durSec * 16000);
  return new Live.Turn(startSec * 16000, startSec * 16000 + n, new Int16Array(n),
                       'hangover', 0.2);
}}
const c = new Live.TurnCoalescer({{}}, () => 0);
c.submit(t(0.0, 1.0));
c.submit(t(1.4, 0.8));
const m = c.flush()[0];
process.stdout.write(JSON.stringify({{
  start: m.startSample, end: m.endSample, n: m.pcm.length,
  declared: m.endSample - m.startSample,
}}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["declared"] == r["n"], r
    assert r["start"] == 0


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_records_a_timeline_fault_as_a_gap_with_its_detail():
    """The tab keeps the reason, so the saved sidecar can explain the labels."""
    script = f"""
const fs = require('fs');
const src = fs.readFileSync({str(LIVE_JS)!r}, 'utf8');
const notes = [];
const mod = {{ exports: {{}} }};
new Function('module', 'exports', 'window', 'document', 'performance', src)(
  mod, mod.exports, {{}}, {{ getElementById: () => null, createElement: () => ({{}}) }},
  {{ now: () => 0 }});
const Live = mod.exports;
const s = new Live.Session({{
  language: 'auto',
  onDropped: (m) => notes.push(m.reason),
  onStats: () => {{}},
}});
s._onMessage({{ data: JSON.stringify({{
  type: 'dropped', channel: 0, start: 12.5, end: 12.5,
  reason: 'timeline_drift', detail: 'start_sample_regression: backwards',
}}) }});
s._onMessage({{ data: JSON.stringify({{
  type: 'stats', covered_sec: 30.0, timeline_faults: 1, drifted_channels: [0],
}}) }});
process.stdout.write(JSON.stringify({{
  gaps: s.gaps, dropCount: s.dropCount, notes,
  stats: s.serverStats.drifted_channels,
}}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True,
                             timeout=120)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["dropCount"] == 1
    assert r["gaps"][0]["reason"] == "timeline_drift"
    assert r["gaps"][0]["detail"].startswith("start_sample_regression")
    assert r["notes"] == ["timeline_drift"]
    assert r["stats"] == [0], "the shutdown stats must reach the session"


# ══════════════════════════════════════════════════════════════════════════
# 2. A client-declared timeline must be checked, not trusted
# ══════════════════════════════════════════════════════════════════════════

def test_unpack_turn_rejects_a_turn_with_no_audio():
    with pytest.raises(ValueError):
        proto.unpack_turn(proto.pack_turn(0, 0, b""))
    # MSG_FLUSH legitimately carries no payload.
    header, pcm = proto.unpack_turn(proto.pack_control(proto.MSG_FLUSH))
    assert header["msg_type"] == proto.MSG_FLUSH and pcm == b""


def test_guard_accepts_a_monotonic_timeline():
    g = proto.TimelineGuard()
    start, fault = g.observe(0, 0, 16000, elapsed_sec=1.0)
    assert (start, fault) == (0, None)
    start, fault = g.observe(0, 40000, 16000, elapsed_sec=4.0)
    assert (start, fault) == (40000, None)
    assert g.faults == 0 and g.stats()["channels"] == []


def test_guard_flags_and_clamps_a_regression():
    g = proto.TimelineGuard()
    g.observe(0, 0, 32000, elapsed_sec=2.0)
    start, fault = g.observe(0, 16000, 16000, elapsed_sec=3.0)

    assert fault["reason"] == proto.FAULT_REGRESSION
    assert start == 32000, "a backwards boundary must be clamped, not forwarded"
    assert g.faults == 1
    assert g.drifted_channels() == [0]
    assert g.stats()["by_reason"][proto.FAULT_REGRESSION] == 1


def test_guard_tolerates_a_hair_of_overlap():
    """Pre/post-roll on two adjacent turns can graze; that is not a fault."""
    g = proto.TimelineGuard()
    g.observe(0, 0, 16000, elapsed_sec=1.0)
    start, fault = g.observe(0, 15900, 16000, elapsed_sec=2.0)
    assert fault is None and start == 15900


def test_guard_flags_a_start_beyond_everything_seen():
    """A late-starting second channel, or a counter that jumped."""
    g = proto.TimelineGuard({"timeline_max_lead_sec": 1.0})
    g.observe(0, 0, 16000, elapsed_sec=1.0)
    start, fault = g.observe(1, 600 * SR, 16000, elapsed_sec=2.0)
    assert start == 600 * SR, "an implausible-but-forward claim is not rewritten"
    assert fault["reason"] == proto.FAULT_AHEAD
    assert g.drifted_channels() == [1]
    assert 0 not in g.drifted_channels(), "the mic channel was never contradicted"


def test_the_lead_allowance_grows_with_a_long_session():
    """A resampler a fraction of a percent off must not be called impossible.

    An hour of silence in which the server saw no turn frames at all leaves the
    high-water mark far behind the connection's elapsed time; a client whose
    clock has drifted a few percent must still be believed, because that is
    exactly what a 0.05% resample-rate error accumulates into over 3 600s.
    """
    g = proto.TimelineGuard({"timeline_max_lead_sec": 10.0,
                             "timeline_max_lead_ratio": 0.05})
    g.observe(0, 0, 16000, elapsed_sec=3600.0)      # nothing seen for an hour
    # 3 700s declared at 3 600s elapsed: 2.8% ahead, inside the 5% allowance.
    _, fault = g.observe(0, 3700 * SR, 16000, elapsed_sec=3600.0)
    assert fault is None, fault
    # 5 000s declared: 39% ahead, far outside any plausible drift.
    _, fault = g.observe(0, 5000 * SR, 16000, elapsed_sec=3600.0)
    assert fault["reason"] == proto.FAULT_AHEAD


def test_guard_is_per_channel():
    """The speaker channel must not be judged against the mic channel's clock."""
    g = proto.TimelineGuard()
    g.observe(0, 0, 16000, elapsed_sec=1.0)
    start, fault = g.observe(1, 0, 16000, elapsed_sec=1.0)
    assert (start, fault) == (0, None)


def test_guard_detail_is_bounded():
    g = proto.TimelineGuard()
    for _ in range(50):
        g.observe(0, 0, 16000, elapsed_sec=1.0)
    assert g.faults == 49, "the first frame establishes the timeline, it is not a fault"
    assert len(g.stats()["detail"]) <= 10


def test_start_sample_regression_is_flagged_counted_and_reported(patched):
    """One bad frame must not end the session, and must not pass unnoticed.

    Frame 2 declares a start *before* frame 1 ended -- a client whose sample
    counter reset, or whose turns arrived out of order.  Every later boundary on
    that channel would silently shift, so the fault has to reach the client and
    the shutdown stats.
    """
    ws = _run_ws([
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 0, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 800, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 40000, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_control(proto.MSG_FLUSH)},
    ])

    # The session survives: all three frames were still transcribed.
    texts = [m for m in ws.sent if m.get("type") == "transcript"]
    assert len(texts) == 3, [m.get("type") for m in ws.sent]

    drift = [m for m in ws.sent
             if m.get("type") == "dropped" and m.get("reason") == "timeline_drift"]
    assert len(drift) == 1, "the client was not told its channel drifted"
    assert drift[0]["channel"] == 0
    assert proto.FAULT_REGRESSION in drift[0]["detail"]

    stats = [m for m in ws.sent if m.get("type") == "stats"][0]
    assert stats["timeline_faults"] == 1
    assert stats["drifted_channels"] == [0]
    assert stats["timeline"]["by_reason"][proto.FAULT_REGRESSION] == 1

    # ...and the boundary it emitted does not go backwards.
    starts = [m["start"] for m in texts]
    assert starts == sorted(starts), starts
    # Frame 1 declared 0.0s for 0.4s of audio, so frame 2 is clamped to 0.4s.
    assert texts[1]["start"] == pytest.approx(0.4, abs=0.01), (
        "the regressed frame was not clamped to the previous turn's end"
    )


def test_a_mic_regression_does_not_condemn_the_speaker_channel(patched):
    ws = _run_ws([
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 0, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 800, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_turn(1, 16000, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_control(proto.MSG_FLUSH)},
    ])
    stats = [m for m in ws.sent if m.get("type") == "stats"][0]
    assert stats["drifted_channels"] == [0]


def test_a_clean_session_reports_no_faults(patched):
    ws = _run_ws([
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 0, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_turn(0, 16000, _pcm(0.4))},
        {"type": "websocket.receive", "bytes": proto.pack_control(proto.MSG_FLUSH)},
    ])
    stats = [m for m in ws.sent if m.get("type") == "stats"][0]
    assert stats["timeline_faults"] == 0
    assert stats["drifted_channels"] == []
    assert not [m for m in ws.sent if m.get("reason") == "timeline_drift"]


# ══════════════════════════════════════════════════════════════════════════
# 3. The flushed tail reaches the WAV exactly once (Windows client)
# ══════════════════════════════════════════════════════════════════════════

class _FakeRecorder:
    def __init__(self):
        self.writes = []

    def write(self, pcm):
        self.writes.append(pcm)


class _FakeResampler:
    def __init__(self, tail=b""):
        self.tail = tail
        self.flushed = 0

    def flush(self):
        self.flushed += 1
        return self.tail


class _Clock:
    """Deterministic monotonic clock (the coalescer must never be polled early)."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def test_shutdown_flush_does_not_rewrite_the_tail(live):
    """The whole point: a turn's PCM is never appended to the recording.

    The tail turn's samples arrived as capture blocks and were already written,
    so writing them again made the .wav one turn too long with the tail
    duplicated -- an over-reported duration and a repeated region for the
    re-attribution to diarize.
    """
    stream = (_pcm(0.4) + np.zeros(int(0.6 * SR), dtype="<i2").tobytes()
              + _pcm(0.4))

    def feed(det):
        turns = []
        for i in range(0, len(stream), 4096):
            turns += det.feed(stream[i:i + 4096])
        return turns

    det = live.TurnDetector()
    turns = feed(det)
    # A second detector on the same audio, to see the tail the first one is
    # about to flush -- ``detector.flush()`` is once-only, so it cannot be
    # asked twice.
    probe = live.TurnDetector()
    feed(probe)
    tail = probe.flush()
    assert tail is not None, "the harness left no open turn to flush"

    coalescer = live.TurnCoalescer()
    for turn in turns:
        coalescer.submit(turn)

    rec = _FakeRecorder()
    res = _FakeResampler(tail=b"\x00" * 64)          # soxr's own trailing samples
    ready = live.flush_at_shutdown([(det, rec, res, live.CHANNEL_MIC)],
                                   {live.CHANNEL_MIC: coalescer})

    assert rec.writes == [b"\x00" * 64], (
        f"the shutdown wrote {len(rec.writes)} buffers; only the resampler's "
        f"draining samples are new audio"
    )
    assert ready, "the flushed tail was not returned to be sent"
    # Merged into the held turn or released on its own, either way its audio is
    # what goes out -- and it is not what goes into the WAV.
    assert any(t.pcm.endswith(tail.pcm) for _, t in ready), [
        (t.start_sec, t.end_sec) for _, t in ready]


def test_shutdown_flush_still_sends_the_last_sentence(live):
    """The fix must not cost the tail: it is submitted, not merely recorded."""
    det = live.TurnDetector()
    stream = (np.zeros(int(0.4 * SR), dtype="<i2").tobytes() + _pcm(0.4)
              + np.zeros(int(0.6 * SR), dtype="<i2").tobytes() + _pcm(0.4))
    turns = []
    for i in range(0, len(stream), 4096):
        turns += det.feed(stream[i:i + 4096])
    assert turns, "harness produced no turns"
    coalescer = live.TurnCoalescer()
    for turn in turns:
        coalescer.submit(turn)

    ready = live.flush_at_shutdown(
        [(det, _FakeRecorder(), _FakeResampler(), live.CHANNEL_MIC)],
        {live.CHANNEL_MIC: coalescer})
    assert len(ready) == 1
    channel, turn = ready[0]
    assert channel == live.CHANNEL_MIC
    assert turn.pcm, "an empty turn was sent at shutdown"
    assert turn.end_sample - turn.start_sample == len(turn.pcm) // 2


def test_shutdown_flush_skips_a_channel_without_a_recorder(live):
    ready = live.flush_at_shutdown(
        [(None, None, None, live.CHANNEL_MIC), (None, None, None, live.CHANNEL_SPEAKER)],
        {live.CHANNEL_MIC: live.TurnCoalescer(),
         live.CHANNEL_SPEAKER: live.TurnCoalescer()})
    assert ready == []


# ══════════════════════════════════════════════════════════════════════════
# 4. A synthetic live session survives shutdown re-attribution
# ══════════════════════════════════════════════════════════════════════════

def test_refined_boundaries_survive_shutdown_reattribution(live, tmp_path):
    """End to end, on the recording rather than on a summary of it.

    A live session is judged by what it hands to the shutdown re-attribution:
    the items (timestamps taken from the turns the client declared) and the
    .wav they are re-attributed against.  If a declared span runs past the audio
    that produced it, the items stop lining up with the recording and the
    re-attribution -- which re-runs the offline pipeline over the .wav -- puts
    the words on the wrong turns.
    """
    from asr_mcp.speaker import attribution as attr

    # Two people taking turns, with a pause inside the first one's utterance
    # (so the coalescer merges it) and a real pause between the two people --
    # longer than merge_gap_sec, or the coalescer would merge across speakers.
    parts = [
        _pcm(0.5), np.zeros(int(0.35 * SR), dtype="<i2"), _pcm(0.5),
        np.zeros(int(1.6 * SR), dtype="<i2"), _pcm(0.6),
        np.zeros(int(0.8 * SR), dtype="<i2"),
    ]
    stream = b"".join(parts)
    duration = len(stream) // 2 / SR

    # ── the live session: capture, detect, coalesce, record ──
    det = live.TurnDetector()
    rec = live.ChannelRecorder(tmp_path / "live.wav")
    clock = _Clock()
    coalescer = live.TurnCoalescer(clock=clock)
    sent = []
    for i in range(0, len(stream), 4096):
        block = stream[i:i + 4096]
        rec.write(block)
        for turn in det.feed(block):
            sent += [(live.CHANNEL_MIC, t) for t in coalescer.submit(turn)]
        sent += [(live.CHANNEL_MIC, t) for t in coalescer.poll()]
        clock.advance(len(block) / 2 / SR)
    sent += live.flush_at_shutdown([(det, rec, _FakeResampler(), live.CHANNEL_MIC)],
                                   {live.CHANNEL_MIC: coalescer})
    rec.close()

    assert len(sent) == 2, [(t.start_sec, t.end_sec) for _, t in sent]
    assert sent[0][1].reason == "merged", "the intra-sentence pause was not merged"
    # The recording is the stream, once.
    import wave
    with wave.open(str(tmp_path / "live.wav")) as w:
        recorded = w.getnframes() / w.getframerate()
    assert recorded == pytest.approx(duration, abs=1e-6), (
        f"the .wav is {recorded}s for {duration}s of audio -- the tail is "
        f"written more than once"
    )

    # ── what the server would return for each sent turn ──
    items = [{"start": t.start_sec, "end": t.end_sec, "text": "..."}
             for _, t in sent]
    for it, (_, turn) in zip(items, sent):
        assert it["end"] - it["start"] == pytest.approx(
            len(turn.pcm) / 2 / SR, abs=1e-6), (
            "an item's span does not match the audio that was sent for it"
        )
        assert it["end"] <= recorded + 1e-6

    # ── the offline turn set the re-attribution would build ──
    first_speaker_end = 0.5 + 0.35 + 0.5 + 1.6
    turns = [
        {"start": 0.0, "end": first_speaker_end, "speaker": "Speaker 1"},
        {"start": first_speaker_end, "end": duration, "speaker": "Speaker 2"},
    ]
    starts = [t["start"] for t in turns]

    runs = attr.attribute_items(items, turns, starts)
    speakers = [r["speaker"] for r in runs]
    assert speakers == ["Speaker 1", "Speaker 2"], runs
    assert None not in speakers, f"an item was left unattributable: {runs}"
    # Every item sits inside the recording and resolves to one turn.
    for run in runs:
        for it in run["items"]:
            assert 0.0 <= it["start"] < it["end"] <= recorded + 1e-6, it
            assert attr.turn_index_for_time(
                (it["start"] + it["end"]) / 2, turns, starts) is not None