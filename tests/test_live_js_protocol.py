"""Browser live client (``asr_mcp/static/live.js``) agreement with the server.

The web UI's Live tab is a second implementation of the live wire protocol and
of the turn detector, alongside ``asr-client/live_client.py``.  It has to
re-implement both -- the browser cannot import the server package, and it must
not reach into the standalone-zip client either -- so the same drift risk
applies and needs the same guard.

The tests that matter here are the constant comparisons.  They read the values
out of the JS source rather than executing it, so they need neither node nor a
browser and run anywhere pytest runs.  A test that actually drives the JS
turn detector lives at the bottom and skips without node.
"""

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_JS = REPO_ROOT / "asr_mcp" / "static" / "live.js"
WORKLET_JS = REPO_ROOT / "asr_mcp" / "static" / "live-worklet.js"

pytestmark = pytest.mark.skipif(
    not LIVE_JS.is_file(), reason="asr_mcp/static/live.js not present"
)


def _js_source() -> str:
    return LIVE_JS.read_text(encoding="utf-8")


def _const(name):
    """Read ``const NAME = <int literal>;`` out of live.js."""
    m = re.search(rf"const {re.escape(name)} = (-?\d+);", _js_source())
    assert m, f"{name} not found in live.js"
    return int(m.group(1))


def _js_object(name):
    """Read ``const NAME = { key: value, ... };`` out of live.js as a dict."""
    src = _js_source()
    m = re.search(rf"const {re.escape(name)} = \{{(.*?)\n\}};", src, re.S)
    assert m, f"{name} not found in live.js"
    out = {}
    for key, value in re.findall(r"(\w+):\s*(-?[\d.]+)", m.group(1)):
        out[key] = float(value) if "." in value else int(value)
    return out


# ── Protocol must match the server byte for byte ───────────────────────────

def test_js_protocol_constants_match_server():
    from asr_mcp.streaming import protocol as proto

    assert _const("PROTOCOL_VERSION") == proto.PROTOCOL_VERSION
    assert _const("MSG_TURN") == proto.MSG_TURN
    assert _const("MSG_FLUSH") == proto.MSG_FLUSH
    assert _const("SAMPLE_RATE") == proto.SAMPLE_RATE
    assert _const("TURN_HEADER_SIZE") == proto.TURN_HEADER_SIZE
    assert _const("CHANNEL_MIC") == 0
    assert _const("CHANNEL_SPEAKER") == 1


def test_js_magic_is_lvt1():
    """The magic is written as byte literals; they must spell "LVT1"."""
    from asr_mcp.streaming import protocol as proto

    src = _js_source()
    for byte in proto.TURN_MAGIC:
        assert f"0x{byte:02x}" in src, f"magic byte {byte:#04x} missing from live.js"


def test_js_header_layout_matches_the_struct():
    """packTurn must write each field at the offset struct assigns it.

    The header is built with a DataView because JS has no struct module, so
    nothing enforces the layout. Writing ``sequence`` at 16 instead of 20 (the
    struct has 4 bytes of padding there) yields a frame the server still parses
    but with a wrong sample count -- so the offsets are pinned here, and the
    round-trip test below proves the result parses.
    """
    from asr_mcp.streaming import protocol as proto

    assert proto.TURN_HEADER.format == "<4sBBHQI I"
    src = _js_source()
    body = src[src.index("function packTurn("):src.index("/** End-of-stream marker")]
    # (offset, DataView writer, value expression) for every field after the magic.
    for call in (
        "setUint8(4, PROTOCOL_VERSION)",
        "setUint8(5, MSG_TURN)",
        "setUint16(6, channel, true)",
        "setUint32(8, startSample >>> 0, true)",
        "setUint32(12, Math.floor(startSample / 4294967296), true)",
        "setUint32(16, n, true)",
        "setUint32(20, sequence || 0, true)",
    ):
        assert call in body, f"packTurn no longer writes {call}"


# ── Turn detection must match the server detector ───────────────────────────

def test_js_detector_defaults_match_server_config():
    from asr_mcp.streaming.turn_detector import config as server_config

    server = server_config()
    js = _js_object("DETECTOR_DEFAULTS")
    assert js, "DETECTOR_DEFAULTS not parsed from live.js"
    for key, default in js.items():
        assert server.get(key, default) == default, (
            f"{key}: server={server.get(key)} browser default={default}"
        )
    assert not (set(js) - set(server)), (
        f"live.js sets keys the server does not know: {set(js) - set(server)}"
    )
    # The server's streaming section also carries knobs that have no browser
    # equivalent. Three groups, all deliberately server-side:
    #   * the decode queue and its overflow policy,
    #   * the Silero speech gate and the edge trim that runs just before it --
    #     both need the VAD model, which lives on the server.  The client
    #     declares the boundary; the SERVER moves it, and reports the refined
    #     span back, so the client never needs these knobs.
    #   * the timeline guard that validates the client's declared start_sample.
    # Every other key must be mirrored, or the browser silently falls back to
    # its own default and moves turn boundaries away from the server's.
    server_only = set(server) - set(js)
    assert server_only == {
        "queue_size",             # server-side decode queue
        "drop_on_overflow",       # ditto
        "vad_frame_threshold",    # server-side Silero gate
        "speech_gate_enabled",    # ditto
        "min_speech_prob",        # ditto
        "min_speech_ratio",       # ditto
        "edge_trim_enabled",      # server-side Silero edge refinement
        "edge_frame_threshold",   # ditto
        "edge_max_trim_sec",      # ditto
        "edge_max_trim_ratio",    # ditto
        "timeline_overlap_tolerance_sec",   # server-side timeline guard
        "timeline_max_lead_sec",            # ditto
        "timeline_max_lead_ratio",          # ditto
    }, f"unmirrored server endpointing keys: {server_only}"


def test_js_worklet_targets_the_same_rate():
    """The worklet resamples to the rate the protocol declares."""
    from asr_mcp.streaming import protocol as proto

    assert f"const OUTPUT_RATE = {proto.SAMPLE_RATE};" in \
        WORKLET_JS.read_text(encoding="utf-8")


# ── The frames the JS builds must satisfy the server's unpacker ─────────────

NODE = shutil.which("node")


def _node_eval(body: str) -> str:
    """Run a snippet in node and return stdout.

    The snippet goes through a temp file, not ``node -e``: the detector test
    embeds thousands of samples and would blow the exec argument limit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.js"
        script.write_text(body, encoding="utf-8")
        result = subprocess.run(
            [NODE, str(script)],
            capture_output=True, text=True, timeout=120,
        )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_packed_turn_parses_with_the_server_unpacker():
    """A turn frame built in the browser must be readable by protocol.unpack_turn.

    This is the real interop test: packTurn writes the header by hand because
    JS has no struct module, so the bytes are produced by node, written to
    stdout, and parsed here by the server's own code.
    """
    import numpy as np

    from asr_mcp.streaming import protocol as proto

    n = 1000
    pcm = (np.arange(-500, 500, dtype=np.int16)).tobytes()
    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
new Function('module', 'exports', 'window', src)(mod, mod.exports, {{}});
const Live = mod.exports;
const samples = new Int16Array({n});
for (let i = 0; i < samples.length; i++) samples[i] = (i - {n // 2});
const buf = Buffer.from(Live.packTurn(1, 16000, samples, 7));
process.stdout.write(buf.toString('base64'));
"""
    frame = __import__("base64").b64decode(_node_eval(script))

    assert proto.is_turn_frame(frame)
    header, payload = proto.unpack_turn(frame)
    assert header["channel"] == 1
    assert header["start_sample"] == 16000
    assert header["n_samples"] == n
    assert header["sequence"] == 7
    assert payload == pcm


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_flush_frame_is_accepted_by_the_server():
    from asr_mcp.streaming import protocol as proto

    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
new Function('module', 'exports', 'window', src)(mod, mod.exports, {{}});
const buf = Buffer.from(mod.exports.packFlush());
process.stdout.write(buf.toString('base64'));
"""
    frame = __import__("base64").b64decode(_node_eval(script))
    header, payload = proto.unpack_turn(frame)
    assert header["msg_type"] == proto.MSG_FLUSH
    assert payload == b""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_and_server_detectors_agree_on_turn_boundaries():
    """Feed the same audio through both detectors; boundaries must be identical.

    The whole reason the browser runs its own endpointing is that a turn cut
    live is cut the same way when the recording is re-diarized offline. If the
    two implementations drift, every boundary moves and the re-attribution
    lands on the wrong turns -- which is exactly the bug this class of test
    exists to prevent.
    """
    np = pytest.importorskip("numpy")

    from asr_mcp.streaming.turn_detector import TurnDetector as ServerDetector

    # 1.2s tone, 0.6s silence, 0.9s tone, 1.0s silence, 0.5s tone.
    def tone(sec, amp=9000, hz=220):
        t = np.arange(int(sec * 16000)) / 16000
        return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)

    def silence(sec):
        return np.zeros(int(sec * 16000), dtype=np.int16)

    signal = np.concatenate([
        tone(1.2), silence(0.6), tone(0.9), silence(1.0), tone(0.5),
    ])
    block = 1024  # the worklet's block size

    server = ServerDetector()
    server_turns = []
    for i in range(0, len(signal), block):
        server_turns.extend(server.feed(signal[i:i + block].tobytes()))
    tail = server.flush()
    if tail is not None:
        server_turns.append(tail)
    expected = [(t.start_sample, t.end_sample, t.reason) for t in server_turns]

    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
new Function('module', 'exports', 'window', 'performance', src)(
    mod, mod.exports, {{}}, {{ now: () => 0 }});
const Live = mod.exports;
const signal = {json.dumps([int(v) for v in signal])};
const det = new Live.TurnDetector();
const turns = [];
const BLOCK = {block};
for (let i = 0; i < signal.length; i += BLOCK) {{
    turns.push(...det.feed(Int16Array.from(signal.slice(i, i + BLOCK))));
}}
const tail = det.flush();
if (tail) turns.push(tail);
process.stdout.write(JSON.stringify(
    turns.map(t => [t.startSample, t.endSample, t.reason])));
"""
    got = json.loads(_node_eval(script))
    assert got == [list(t) for t in expected], (
        "browser and server detectors disagree on turn boundaries:\n"
        f"  server = {expected}\n  browser = {got}"
    )


# ── Transcript rendering must match the Windows client ─────────────────────

def test_js_transcript_format_matches_the_python_client():
    """buildTranscript must produce the asr-client's exact line format.

    A .txt from the web UI and one from transcribe.bat are meant to be
    interchangeable, so the header, the [HH:MM:SS] Speaker (NN%): line, the
    confidence suffix rule and the uncertainty footer are all pinned.
    """
    client = (REPO_ROOT / "asr-client" / "transcribe_client.py")
    if not client.is_file():
        pytest.skip("asr-client/transcribe_client.py not present")

    node = NODE
    if node is None:
        pytest.skip("node not installed")

    result = {
        "total_speakers": 2,
        "audio_duration_sec": 12.5,
        "results": [{
            "speaker": "Alice",
            "start": 0.0,
            "end": 3.0,
            "uncertain": False,
            "segments": [
                {"start": 0.0, "end": 1.2, "text": "Hello there.", "confidence": 0.9},
                {"start": 1.4, "end": 2.4, "text": "How are you?", "confidence": 0.7},
            ],
        }],
    }
    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
new Function('module', 'exports', 'window', src)(mod, mod.exports, {{}});
process.stdout.write(mod.exports.buildTranscript('a.wav',
    {json.dumps(result)}, '2026-09-29 14:12:03'));
"""
    js_text = _node_eval(script)

    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("_tc_undertest", client)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    py_text = mod.build_transcript('a.wav', result, '2026-09-29 14:12:03')

    assert js_text == py_text, (
        "live.js buildTranscript diverged from transcribe_client.build_transcript:\n"
        f"--- python ---\n{py_text}\n--- javascript ---\n{js_text}"
    )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_transcript_confidence_suffix_is_omitted_without_confidence():
    """cohere/qwen3 leave confidence None, so no "(NN%)" may be emitted."""
    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
new Function('module', 'exports', 'window', src)(mod, mod.exports, {{}});
process.stdout.write(mod.exports.buildTranscript('a.wav', {{
    total_speakers: 1, audio_duration_sec: 1.0,
    results: [{{ speaker: 'You', uncertain: false,
        segments: [{{ start: 0, end: 1, text: 'Hi.', confidence: null }}] }}],
}}, '2026-09-29 14:12:03'));
"""
    text = _node_eval(script)
    assert "[00:00:00] You: Hi." in text
    assert "%" not in text


# ── The module must be loadable and fully wired ─────────────────────────────

@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_live_module_evaluates_and_init_resolves_every_binding():
    """``Live.init()`` must not hit a temporal-dead-zone binding.

    The whole module is one IIFE, so a top-level ``return`` placed before the
    ``class UI`` declaration silently turns everything after it into dead code:
    the file still parses, ``node --check`` passes, and the failure only shows
    up in the browser as "Cannot access 'UI' before initialization" when the
    user opens the tab. Evaluating the module and calling init() is the only
    way to catch that ordering mistake.
    """
    script = f"""
const src = require('fs').readFileSync({str(LIVE_JS)!r}, 'utf8');
const mod = {{exports: {{}}}};
const document = {{ getElementById: () => null, createElement: () => ({{}}) }};
new Function('module', 'exports', 'window', 'document', src)(
    mod, mod.exports, {{}}, document);
const Live = mod.exports;
const missing = ['packTurn', 'packFlush', 'buildTranscript', 'TurnDetector',
    'TurnCoalescer', 'Session', 'fmtHms', 'wavBlob', 'profilesBanner']
    .filter(k => typeof Live[k] === 'undefined');
if (missing.length) throw new Error('missing exports: ' + missing.join(','));
const ui = Live.init();
if (ui === null || typeof ui.start !== 'function') {{
    throw new Error('init() did not return a UI controller');
}}
if (Live.init() !== ui) throw new Error('init() is not idempotent');
process.stdout.write('ok');
"""
    assert _node_eval(script) == "ok"


_SESSION_HARNESS = r"""
// Drive a full Session connect -> capture -> finish() cycle with a fake
// WebSocket and a fake AudioContext, and report the frames that reached the
// socket. Replaces a browser smoke test, which cannot run in CI.
const fs = require('fs');
const src = fs.readFileSync(LIVE, 'utf8');
const sent = [];

class FakeWS {
  constructor(url) {
    this.url = url; this.binaryType = ''; this.readyState = 1; // OPEN
    // The real socket completes its handshake asynchronously; firing onopen
    // synchronously would make connect() settle before its handler is set.
    setTimeout(() => this.onopen && this.onopen(), 0);
  }
  send(d) {
    const u = new Uint8Array(d);
    // MSG_FLUSH carries the LVT1 magic too, so the msg type must be checked
    // as well or the control frame is counted as a turn.
    sent.push({ magic: u[0], msgType: u[5], len: d.byteLength });
  }
  close() { this.onclose && this.onclose({ code: 1000, reason: 'bye' }); }
}
FakeWS.OPEN = 1;

class FakeNode {
  constructor() { this.port = { onmessage: null }; }
  connect() {} disconnect() {}
}
const fakeCtx = {
  sampleRate: 48000,
  // A real context is `suspended` until the page has user activation, and a
  // suspended one never runs the graph. The fake starts `running` so the
  // harness exercises the capture path, not the autoplay guard.
  state: 'running',
  audioWorklet: { addModule: async () => {} },
  createMediaStreamSource: () => ({ connect() {}, disconnect() {} }),
  resume: async () => { fakeCtx.state = 'running'; },
  close: async () => {},
};
const document = { getElementById: () => null, createElement: () => ({}) };
// attach() resolves the Web Audio classes off `window`, not the bare globals.
const window = {
  livePrefix: '',
  AudioContext: function () { return fakeCtx; },
  AudioWorkletNode: FakeNode,
};
// _wsUrl() builds the socket URL from location, so a session cannot connect
// without one.
const location = { protocol: 'https:', host: 'example.test' };

// The coalescer's merge window is measured against performance.now(), so the
// clock must advance with the audio or the held turn is never released.
let clock = 0;
const performance = { now: () => clock };

const mod = { exports: {} };
new Function('module', 'exports', 'window', 'document', 'performance',
             'location', 'WebSocket', 'AudioWorkletNode', src)(
  mod, mod.exports, window, document, performance, location, FakeWS, FakeNode);
const Live = mod.exports;
global.WebSocket = FakeWS;

function tone(sec, amp) {
  const n = Math.round(sec * 16000);
  const a = new Int16Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.round(amp * Math.sin(2 * Math.PI * 220 * i / 16000));
  return a;
}

(async () => {
  const session = new Live.Session({ language: 'auto' });
  await session.connect();
  await session.attach(0, { getTracks: () => [{ stop() {} }] });

  // 1.0s of speech, 0.5s of silence, then 0.8s of speech that is still open
  // when the user hits Stop -- this is the tail that used to be dropped.
  const parts = [tone(1.0, 9000), new Int16Array(8000), tone(0.8, 9000)];
  const total = parts.reduce((s, a) => s + a.length, 0);
  const signal = new Int16Array(total);
  let at = 0;
  for (const p of parts) { signal.set(p, at); at += p.length; }
  for (let i = 0; i < signal.length; i += 1024) {
    session._onPcm(0, signal.slice(i, i + 1024));
    clock += 1024 / 16000;
    session.poll();
  }

  const openBefore = session.channel(0).detector.inTurn;
  const finish = session.finish(3000);
  // finish() is async and yields at `await ctx.close()` before flushing.
  await new Promise(r => setTimeout(r, 20));

  const turns = sent.filter(f => f.magic === 0x4c && f.msgType === 1);
  process.stdout.write(JSON.stringify({
    openBefore,
    url: 'ok',
    turns: turns.length,
    turnBytes: turns.map(t => t.len),
    control: sent.filter(f => f.msgType === 2).length,
  }));
  finish.catch(() => {});
})().catch(e => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_finish_sends_the_open_turn_before_the_flush_frame():
    """The last sentence of a session must reach the socket.

    A turn that is still open when the user hits Stop is only released by
    ``detector.flush()``.  Recording it into the WAV without submitting it to
    the coalescer leaves it in no turn list at all, so the final sentence of
    every session is silently never sent -- and because the recording still
    contains it, the resulting .txt and the audio disagree.
    """
    from asr_mcp.streaming import protocol as proto

    script = f"const LIVE = {str(LIVE_JS)!r};\n" + _SESSION_HARNESS
    result = json.loads(_node_eval(script))
    header = proto.TURN_HEADER_SIZE

    assert result["openBefore"] is True, "harness signal ended mid-turn"
    # The two tones merge into one 34304-sample turn -- 19456 + a 2560-sample
    # silence pad for the 0.5s gap (inside merge_gap_sec) + 12288 -- plus a
    # 24-byte header, so 68632 bytes of frame. The pad is what keeps the tail's
    # audio on its true timeline; see TurnCoalescer in turn_detector.py.
    assert result["turns"] == 1, f"expected the flushed turn to be sent, got {result}"
    assert result["turnBytes"] == [header + 34304 * 2], (
        f"the flushed turn lost audio: {result['turnBytes']} "
        f"(want [{header + 34304 * 2}])"
    )
    assert result["control"] == 1, "MSG_FLUSH must follow the turns, not precede them"


def test_live_js_return_statement_is_the_last_thing_in_the_iife():
    """Static guard for the same ordering mistake, without needing node.

    The IIFE's ``return { ... }`` must be the last statement before its
    ``})();`` -- otherwise any declaration below it is unreachable.
    """
    lines = _js_source().split("\n")
    close = next(i for i, l in enumerate(lines) if l.strip() == "})();")
    candidates = [i for i, l in enumerate(lines[:close]) if l == "return {"]
    assert candidates, "the IIFE no longer returns an export object"
    start = candidates[-1]
    # The return statement ends at the next column-0 "};".
    end = next(i for i in range(start + 1, close) if lines[i] == "};")
    assert all(not l.strip() for l in lines[end + 1:close]), (
        "unreachable code between the IIFE's return and its close:\n"
        + "\n".join(f"{i + 1}: {lines[i]}" for i in range(end + 1, close))
    )


# ── Capture must survive the browser's autoplay policy ─────────────────────

def test_start_creates_the_audio_context_before_any_await():
    """ensureAudio() must run before the permission prompts.

    Chrome hands back a *suspended* AudioContext when the page has no user
    activation, and a suspended context never runs the audio graph: process()
    is never called, the worklet posts nothing, the detector never opens a
    turn, and the socket receives zero bytes. There is no exception and no
    console error -- the tab just looks alive and records silence. Each await
    before the context is constructed (the two permission prompts, the WS
    connect) consumes the activation, so the order in start() is load-bearing
    and nothing at runtime can catch it.
    """
    src = _js_source()
    start = src.index("    async start() {")
    body = src[start:src.index("\n    async stop() {", start)]
    audio = body.index("ensureAudio()")
    for later, why in (
        ("getUserMedia(", "the microphone permission prompt"),
        ("getDisplayMedia(", "the screen-share picker"),
        ("session.connect()", "the WebSocket connect"),
    ):
        assert audio < body.index(later), (
            f"start() must build the AudioContext before {why}; a context "
            "created after that await is suspended and the tab records nothing"
        )
    assert "await session.ensureAudio();" in body
    # ensureAudio must actually enforce the state rather than trust it.
    ensure = src[src.index("async ensureAudio() {"):src.index("async attach(")]
    assert "resume()" in ensure, "ensureAudio never calls ctx.resume()"
    assert "state !== 'running'" in ensure, (
        "ensureAudio does not reject a context that is still not running"
    )


def test_the_stop_button_survives_the_recording_state():
    """start() hides the settings row; the Stop button must live outside it.

    Both elements were originally inside one #liveConfig wrapper, so hiding it
    on start took the only way to end a session with it -- and the recording
    could not be stopped from the UI at all.
    """
    template = (REPO_ROOT / "asr_mcp" / "templates" / "app.html").read_text(
        encoding="utf-8")
    assert "liveConfig" not in _js_source(), (
        "live.js still toggles the old #liveConfig wrapper"
    )
    for hidden in ("liveSettings", "liveConnection"):
        assert f'id="{hidden}"' in template, f"#{hidden} is missing from app.html"
    settings_at = template.index('id="liveSettings"')
    connection_at = template.index('id="liveConnection"')
    assert 'id="liveStop"' not in template[settings_at:connection_at], (
        "the Stop button sits inside the settings block that start() hides"
    )
    assert template.index('id="liveStop"') > connection_at


def test_the_level_meter_fill_is_not_an_inline_box():
    """`width` does not apply to a non-replaced inline element.

    The meter fill was a <span>, so every width write below was dropped and the
    bar read as a dead capture no matter what the mic was doing. The fill is now
    a <div> inside the shared `.mini-bar` track.
    """
    template = (REPO_ROOT / "asr_mcp" / "templates" / "app.html").read_text(
        encoding="utf-8")
    m = re.search(r'<div class="mini-bar meter-bar"[^>]*>\s*<div id="liveMeterMic"',
                  template)
    assert m, "the mic meter fill is no longer a <div> inside a .mini-bar"
    rule = next(l for l in template.split("\n")
                if ".mini-bar > div {" in l and "display: block" in l)
    assert "display: block" in rule
    live_js = LIVE_JS.read_text(encoding="utf-8")
    assert "el.style.width" in live_js, "the meter no longer writes a width"


# ── Input levelling must behave, and must behave the same in both clients ──

AGC_SCRIPT = r"""
const fs = require('fs');
class AudioWorkletProcessor {
  constructor() { this.port = { onmessage: null, postMessage: () => {} }; }
}
let Registered = null;
const src = fs.readFileSync(WORKLET, 'utf8');
new Function('AudioWorkletProcessor', 'sampleRate', 'registerProcessor', src)(
  AudioWorkletProcessor, 48000, (n, c) => { Registered = c; });

// Drive the real processor with a 220 Hz tone at a given amplitude, in
// 128-sample render quanta as the graph delivers them, and report the level it
// settles at. `modulate` adds a 3 Hz envelope, which is what exposes the
// fast-attack/slow-release bias; a steady tone is what pins convergence and is
// therefore the signal both implementations are compared on.
function drive(amp, blocks, modulate) {
  const p = new Registered({ processorOptions: { sampleRate: 48000 } });
  const out = [];
  p.port.postMessage = (m) => out.push(m);
  // The phase is continuous across blocks: restarting the sine at 0 every
  // 128 samples biases its RMS by ~6%, which is a measurement artifact, not
  // a property of the leveller.
  const n = 128, step = 2 * Math.PI * 220 / 48000;
  let phase = 0;
  for (let b = 0; b < blocks; b++) {
    const ch = new Float32Array(n);
    const env = modulate
      ? 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * b * n / 48000) : 1.0;
    for (let i = 0; i < n; i++) ch[i] = amp * Math.sin(phase + i * step) * env;
    phase += n * step;
    p.process([[ch]]);
  }
  // Steady state is the last half of the POSTED blocks, not of the render
  // quanta: one output block spans ~24 quanta, so slicing the quanta count
  // would leave nothing to measure.
  const tail = out.slice(Math.floor(out.length / 2));
  let sum = 0, count = 0, peak = 0;
  for (const m of tail) {
    for (let i = 0; i < m.pcm.length; i++) {
      const v = m.pcm[i] / 32768;
      sum += v * v;
      count++;
      const a = v < 0 ? -v : v;
      if (a > peak) peak = a;
    }
  }
  return {
    rms: count ? Math.sqrt(sum / count) : 0,
    peak,
    gain: p.lastGain,
    blocks: out.length,
  };
}

const amps = [0.003, 0.01, 0.03, 0.1, 0.3];
const rows = amps.map((a) => drive(a, 4000, false));
const modulated = drive(0.01, 4000, true);
const silence = drive(0.0, 400, false);
const room = drive(0.0002, 400, false);
const loud = drive(0.6, 400, false);
process.stdout.write(JSON.stringify({
  rows: rows.map((r, i) => ({ amp: amps[i], rms: r.rms, peak: r.peak,
                               gain: r.gain, blocks: r.blocks })),
  modulated: { rms: modulated.rms, peak: modulated.peak, gain: modulated.gain },
  silence: { rms: silence.rms, gain: silence.gain },
  room: { rms: room.rms, gain: room.gain },
  loud: { rms: loud.rms, gain: loud.gain },
}));
"""


def _drive_worklet() -> dict:
    import json
    return json.loads(
        _node_eval(f"const WORKLET = {str(WORKLET_JS)!r};\n" + AGC_SCRIPT))


def _py_drive(live, amp, blocks=4000, modulate=False):
    """Feed the same signal through the Python AutoGain, block for block."""
    np = pytest.importorskip("numpy")
    ag = live.AutoGain()
    n = 128
    # Continuous phase across blocks -- see the note in AGC_SCRIPT: a per-block
    # reset restarts the sine at 0 and biases its RMS by ~6%.
    step = 2 * np.pi * 220 / 48000
    phase = 0.0
    tail = []
    for b in range(blocks):
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * b * n / 48000) if modulate else 1.0
        x = (amp * np.sin(phase + np.arange(n) * step) * env).astype(np.float32)
        phase += n * step
        y = ag.apply(x)
        if b >= blocks // 2:
            tail.append(y)
    if not tail:
        return {"rms": 0.0, "peak": 0.0, "gain": ag.gain}
    flat = np.concatenate(tail).astype(np.float64)
    return {
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "peak": float(np.max(np.abs(flat))),
        "gain": ag.gain,
    }


def _db(x):
    return 20 * math.log10(x) if x > 0 else -120.0


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_worklet_levels_a_quiet_mic_up_to_the_target():
    """A quiet microphone is lifted to a normal speech level.

    A quiet mic costs far more at the recogniser than at the turn detector,
    whose thresholds are ratios against a tracked noise floor, and it costs
    again in the offline pass: the recorded WAV is re-uploaded for
    re-attribution, where VAD and the ECAPA embeddings are both level
    sensitive. Levelling therefore lives in the worklet, so the frames on the
    wire and the WAV on disk carry the same level.
    """
    pytest.importorskip("numpy")
    report = _drive_worklet()
    by_amp = {r["amp"]: r for r in report["rows"]}
    for amp in (0.03, 0.1):
        db = _db(by_amp[amp]["rms"])
        # A steady tone must converge on the -18 dBFS target, not merely move
        # in its direction -- an earlier version of this loop settled on a
        # geometric mean of where it started and where it was going, which
        # looked like convergence and left a -40 dBFS mic at -32 dBFS.
        assert -19.5 < db < -16.5, (
            f"a {amp} tone settled at {db:.1f} dBFS, not the "
            f"{_db(0.125):.1f} dBFS target"
        )
        assert by_amp[amp]["peak"] < 0.99, "levelling clipped"

    # A very quiet mic is lifted as far as the ceiling allows -- a bounded
    # booster, not an unlimited one.
    very_quiet = by_amp[0.01]
    assert very_quiet["gain"] > 8.0
    assert _db(very_quiet["rms"]) > _db(0.01) + 12


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_the_worklet_levelling_has_the_invariants():
    """Silence is not lifted, a hot source is not attenuated, modulation is safe.

    The two failure modes that matter: a booster that keeps winding up during
    silence turns room tone into hiss, and one that attenuates makes a hot
    source worse. Fast attack with slow release biases a modulated signal
    upward, so the peak limiter is what keeps a real speech envelope in range
    -- that is asserted too, rather than assumed.
    """
    pytest.importorskip("numpy")
    report = _drive_worklet()
    assert report["silence"]["rms"] == 0.0, "digital silence was amplified"
    assert report["room"]["gain"] < 1.01, "room tone is being lifted"
    assert report["loud"]["gain"] <= 1.01, "a hot source is being attenuated"
    # A sine's RMS is its amplitude over sqrt(2); an untouched source must
    # come out at exactly that, which is the real statement of "never attenuate".
    assert abs(_db(report["loud"]["rms"]) - _db(0.6 / math.sqrt(2))) < 0.5, (
        "a 0.6 full-scale source lost level"
    )
    assert report["modulated"]["peak"] < 0.99, (
        "a 3 Hz envelope pushed the output into the limiter"
    )
    for row in report["rows"]:
        assert row["blocks"] > 0, "the worklet stopped posting blocks"


def test_the_two_clients_level_identically(live):
    """The browser worklet and the Windows client must share the constants.

    They are two implementations of the same capture chain and a user moves
    between them; if one targets -18 dBFS and the other -12, the same
    microphone behaves differently depending on which client recorded it.
    """
    worklet = WORKLET_JS.read_text(encoding="utf-8")

    def js_const(name):
        m = re.search(rf"const {name} = ([0-9.]+);", worklet)
        assert m, f"{name} missing from live-worklet.js"
        return float(m.group(1))

    assert js_const("TARGET_RMS") == live.TARGET_RMS
    assert js_const("MIN_ENV") == live.MIN_ENV
    assert js_const("MAX_GAIN") == live.MAX_GAIN
    assert js_const("ATTACK") == live.AGC_ATTACK
    assert js_const("RELEASE") == live.AGC_RELEASE
    assert js_const("PEAK_CEILING") == live.AGC_PEAK_CEILING
    # Sanity on the values themselves, so a "both are wrong" edit still fails.
    assert live.TARGET_RMS == 0.125
    assert live.MAX_GAIN == 16.0
    assert 0 < live.AGC_RELEASE < live.AGC_ATTACK < 1


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("amp", [0.003, 0.01, 0.03, 0.1])
def test_the_python_autogain_reaches_the_same_level_as_the_worklet(live, amp):
    """Same steady tone through both implementations -> the same level.

    Compared on a steady tone because that is where both loops converge to a
    single answer; a modulated signal measures the envelope response, which
    also depends on how the resampler groups blocks.
    """
    np = pytest.importorskip("numpy")
    report = _drive_worklet()
    js_row = next(r for r in report["rows"] if r["amp"] == amp)
    py = _py_drive(live, amp)
    assert abs(_db(py["rms"]) - _db(js_row["rms"])) < 1.0, (
        f"python {_db(py['rms']):.1f} dBFS vs worklet {_db(js_row['rms']):.1f} dBFS"
    )
    assert abs(py["gain"] - js_row["gain"]) < 0.5


def test_the_python_autogain_protects_silence_and_hot_sources(live):
    np = pytest.importorskip("numpy")
    assert _py_drive(live, 0.0)["rms"] == 0.0, "digital silence was amplified"
    assert _py_drive(live, 0.0002)["gain"] < 1.01, "room tone is being lifted"
    loud = _py_drive(live, 0.6, blocks=400)
    assert loud["gain"] <= 1.01, "a hot source is being attenuated"
    assert abs(_db(loud["rms"]) - _db(0.6 / math.sqrt(2))) < 0.5, (
        "a hot source lost level"
    )
    # And the flag that turns the whole thing off.
    ag = live.AutoGain(enabled=False)
    x = np.full(128, 0.001, dtype=np.float32)
    assert float(np.max(np.abs(ag.apply(x)))) == pytest.approx(0.001)


# ── The server endpoint the tab posts to must exist ─────────────────────────

def test_live_save_endpoint_is_registered():
    """The tab posts to /api/asr/live/save; a rename must break this loudly."""
    router = (REPO_ROOT / "asr_mcp" / "api" / "asr_router.py").read_text(
        encoding="utf-8")
    assert '@router.post("/live/save")' in router
    js = LIVE_JS.read_text(encoding="utf-8")
    assert "/api/asr/attribution/upload" in js, "reattribute() lost its endpoint"
    assert "/api/asr/ws/stream" in js, "connect() lost its websocket endpoint"
    assert "/api/asr/live/save" in js, "the History save post lost its endpoint"


def test_no_js_identifier_collides_with_a_render_placeholder():
    """No JS identifier may be spelled like a _render() substitution token.

    ``server.py::_render`` substitutes with a plain global ``str.replace``, so
    every ``__PREFIX__`` in a template is replaced wherever it appears -- not
    just inside a URL. A JS global named ``window.__PREFIX__`` therefore
    rendered as ``window.`` and took the whole SPA down with a SyntaxError
    pointing at the page, not at the file that broke it.

    Allowed: the token inside a string literal or an attribute value
    (``src="__PREFIX__/static/..."``, ``const P = '__PREFIX__'``). Rejected:
    anywhere else, i.e. as code.
    """
    app = (REPO_ROOT / "asr_mcp" / "templates" / "app.html").read_text(
        encoding="utf-8")
    # Every occurrence must sit inside quotes, a comment, or a tag attribute.
    # __NAV__ legitimately stands alone as an HTML body placeholder, so only
    # the tokens that also appear inside URLs and script are checked.
    for m in re.finditer(r"__(?:PREFIX|ACT_[A-Z]+)__", app):
        before = app[:m.start()].rsplit("\n", 1)[-1]
        line = before + app[m.start():m.end()]
        quoted = before.count("'") % 2 == 1 or before.count('"') % 2 == 1
        in_tag = line.lstrip().startswith("<")
        assert quoted or in_tag, (
            f"{m.group(0)} appears as code, not inside a string/attribute "
            f"(app.html: {before.strip()[-60:]!r}) -- _render() will replace "
            f"it and break the page"
        )

    # The browser client reads the prefix from a global set by the page; keep
    # the two names in step so the global can never be renamed on one side.
    assert "window.livePrefix = P;" in app
    assert "window.livePrefix || ''" in LIVE_JS.read_text(encoding="utf-8")


# ── A failed session must not throw away a usable one ───────────────────────

def test_stop_keeps_downloads_when_the_socket_dropped_after_transcripts():
    """The cached items are the point; a dropped socket is not a reason to lose them.

    ``stop()`` used to bail out on ``!wasStreaming`` before rendering anything,
    so a connection that died after the last transcript silently cost the user
    the recording, the .txt and the History save. Static guard: the bail-out may
    only be reached when no transcript arrived at all.
    """
    src = _js_source()
    stop_body = src[src.index("    async stop() {"):src.index("    _renderDownloads(")]
    # The bail-out is the `return` that ends the "no transcript" branch -- not
    # the `if (!session) return` guard at the top of the method.
    for marker in ("try {", "} finally {", "if (!wasStreaming) {"):
        assert marker in stop_body, f"stop() lost its {marker!r} block"
    branch = stop_body[stop_body.index("if (!session.transcriptCount) {"):]
    assert "return;" in branch
    assert "if (!session.transcriptCount) {" in stop_body, (
        "the early return must be gated on transcriptCount alone, not on the "
        "socket state"
    )
    assert "!wasStreaming ||" not in stop_body and (
        "!wasStreaming &&" not in stop_body
    ), "stop() still discards a usable session when the socket failed"
    # The drain must be guarded so a throw cannot leave the tab locked.
    assert stop_body.index("} finally {") < stop_body.index("if (!session.transcriptCount) {")
    assert "if (!wasStreaming) {" in stop_body, (
        "a dropped socket should be reported, not acted on by discarding data"
    )


def test_the_turn_split_is_reported_whenever_turns_are_counted():
    """A silent mic channel is the one failure the user cannot otherwise see.

    With no mic turns the uploaded mic recording has no speech to diarize, so
    every item collapses onto a single speakerless turn and the whole transcript
    comes back UNKNOWN. The split has to reach the note, the sidecar and the
    .txt header.
    """
    src = _js_source()
    assert "turnSplit()" in src
    for where in ("No turn was detected on the microphone", "turns_sent_per_channel"):
        assert where in src, f"{where!r} missing from live.js"
    assert src.count("session.turnSplit()") >= 2, (
        "the split must appear in more than the note (sidecar/txt header too)"
    )


def test_the_live_save_route_takes_no_audio():
    """``/api/asr/live/save`` stores text, so it must not accept an audio blob.

    An inline base64 field would make this an unbounded JSON body -- the
    sibling upload route caps files at 200MB -- and the browser never sent it,
    so the decode and the ``data/live/`` write were unreachable code.
    """
    # Read the fields statically: importing the schema pulls in fastapi via
    # asr_mcp.api.__init__, which the host does not have, and this guard is
    # about the declared shape, not about pydantic.
    schemas = (REPO_ROOT / "asr_mcp" / "api" / "schemas.py").read_text(encoding="utf-8")
    body = schemas[schemas.index("class LiveSessionSave(BaseModel):"):]
    body = body[:body.index("\nclass ")]
    fields = set(re.findall(r"^\s{4}(\w+)\s*:", body, re.M))
    assert "audio_base64" not in fields, (
        "the live/save schema still declares an audio blob"
    )
    assert fields == {"audio_filename", "result", "stats", "sidecar"}, fields
    js = _js_source()
    assert "audio_base64" not in js, "the client still posts an audio blob"
    router = (REPO_ROOT / "asr_mcp" / "api" / "asr_router.py").read_text(
        encoding="utf-8")
    route = router[router.index('@router.post("/live/save")'):
                   router.index('@router.websocket("/ws/stream")')]
    assert "base64" not in route and "convert_to_wav" not in route, (
        "the live/save route still carries an audio write path"
    )
