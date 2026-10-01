/**
 * Live (real-time) transcription for the browser, mirroring
 * asr-client/live_client.py.
 *
 * The Windows client is the reference implementation and the two are kept
 * deliberately parallel: same wire protocol, same turn-detection algorithm and
 * defaults, same re-attribution-on-shutdown path, same transcript format. This
 * file re-implements all of that in the browser because the server package
 * cannot be imported here -- and because a browser has no WASAPI loopback, so
 * the second channel comes from getDisplayMedia (tab / system audio) instead.
 *
 * Three things are asserted by tests/test_live_js_protocol.py, which reads the
 * constants out of this file and compares them to the server's modules:
 *   - TURN_* / MSG_* / SAMPLE_RATE vs streaming/protocol.py
 *   - DETECTOR_DEFAULTS vs streaming/turn_detector.py::config()
 * Do not change one side without the other.
 *
 * Structure:
 *   1. Wire protocol (packTurn / packFlush)
 *   2. TurnDetector + TurnCoalescer (ports of turn_detector.py)
 *   3. Transcript rendering (port of transcribe_client.build_transcript)
 *   4. Session: capture -> detect -> coalesce -> send -> receive -> stop flow
 */

const Live = (() => {

// ── 1. Wire protocol (mirrors asr_mcp/streaming/protocol.py) ───────────────

const TURN_MAGIC = 0x4c565431;          // "LVT1" little-endian
const PROTOCOL_VERSION = 1;
const MSG_TURN = 1;
const MSG_FLUSH = 2;
const TURN_HEADER_SIZE = 24;            // 4+1+1+2+8+4+4 (4 pad bytes in the struct)
const SAMPLE_RATE = 16000;

const CHANNEL_MIC = 0;
const CHANNEL_SPEAKER = 1;

/**
 * One turn frame: magic | version | type | channel | start | count | seq | PCM.
 *
 * The 24-byte header is written by hand because JS has no struct module. The
 * layout must match TURN_HEADER = struct.Struct("<4sBBHQI I") exactly --
 * note the explicit 4 pad bytes before the trailing uint32, which is what
 * makes the header 24 bytes and not 20. Setting `sequence` on a 24-byte buffer
 * via DataView.setUint32(20, ...) is what the padding is for.
 *
 * pcm is an Int16Array. The server reads it as int16 LE, which is what
 * Int16Array is in every browser we target, so the samples are copied
 * verbatim into the buffer rather than byte-swapped.
 */
function packTurn(channel, startSample, pcm, sequence) {
    const n = pcm.length;
    const buf = new ArrayBuffer(TURN_HEADER_SIZE + n * 2);
    const view = new DataView(buf);
    // magic: write the 4 ASCII bytes individually so no endianness question.
    view.setUint8(0, 0x4c);   // L
    view.setUint8(1, 0x56);   // V
    view.setUint8(2, 0x54);   // T
    view.setUint8(3, 0x31);   // 1
    view.setUint8(4, PROTOCOL_VERSION);
    view.setUint8(5, MSG_TURN);
    view.setUint16(6, channel, true);
    // start_sample is uint64 in the struct; JS numbers are exact below 2^53, so
    // a low/high split is safe. A session would need ~10^8 years to overflow.
    view.setUint32(8, startSample >>> 0, true);
    view.setUint32(12, Math.floor(startSample / 4294967296), true);
    view.setUint32(16, n, true);
    view.setUint32(20, sequence || 0, true);
    new Int16Array(buf, TURN_HEADER_SIZE).set(pcm);
    return buf;
}

/** End-of-stream marker: same header, MSG_FLUSH, no payload. */
function packFlush() {
    const buf = new ArrayBuffer(TURN_HEADER_SIZE);
    const view = new DataView(buf);
    view.setUint8(0, 0x4c);
    view.setUint8(1, 0x56);
    view.setUint8(2, 0x54);
    view.setUint8(3, 0x31);
    view.setUint8(4, PROTOCOL_VERSION);
    view.setUint8(5, MSG_FLUSH);
    return buf;
}

// ── 2. Turn detection (mirrors asr_mcp/streaming/turn_detector.py) ─────────
//
// The whole point of detecting turns here rather than on the server is that a
// turn cut live must be cut the same way when the recording is re-diarized
// offline. Every constant below matches the server's streaming config, and
// tests/test_live_js_protocol.py fails if it drifts.

const DETECTOR_DEFAULTS = {
    frame_ms: 32.0,
    noise_floor_min: 0.0005,
    noise_floor_max: 0.05,
    start_threshold_ratio: 2.5,
    end_threshold_ratio: 1.6,
    start_threshold_min: 0.0012,
    end_threshold_min: 0.0006,
    start_confirm_frames: 2,
    hangover_ms: 320,
    pre_roll_ms: 160,
    post_roll_ms: 200,
    min_voiced_ms: 120,
    max_turn_sec: 30.0,
    merge_gap_sec: 1.0,
    max_merge_sec: 15.0,
};

const UNKNOWN_SPEAKER = 'UNKNOWN';

class Turn {
    constructor(startSample, endSample, pcm, reason, peakRms) {
        this.startSample = startSample;
        this.endSample = endSample;
        this.pcm = pcm;
        this.reason = reason || 'end';
        this.peakRms = peakRms || 0;
    }
    get startSec() { return this.startSample / SAMPLE_RATE; }
    get endSec() { return this.endSample / SAMPLE_RATE; }
}

class TurnDetector {
    /** Adaptive endpointing: noise floor, hysteresis, pre/post-roll, hangover. */
    constructor(cfg) {
        this.cfg = Object.assign({}, DETECTOR_DEFAULTS, cfg || {});
        const c = this.cfg;
        this.frameSamples = Math.max(1, Math.floor(SAMPLE_RATE * c.frame_ms / 1000));
        this.preRollFrames = Math.max(0, Math.floor(c.pre_roll_ms / c.frame_ms));
        this.hangoverFrames = Math.max(0, Math.floor(c.hangover_ms / c.frame_ms));
        this.maxTurnFrames = Math.max(1, Math.floor(c.max_turn_sec * 1000 / c.frame_ms));

        this.carry = null;        // Int16Array tail of a partial frame
        this.frameIndex = 0;
        this.noiseFloor = c.noise_floor_min;
        this.preRoll = [];        // [index, Int16Array] ring
        this.inTurn = false;
        this.confirmRun = 0;
        this.silentRun = 0;
        this.turnStartFrame = 0;
        this.turnFrames = [];     // Int16Array[] accumulated for the open turn
        this.turnPeak = 0;
        this.turnVoiced = 0;
        this.droppedTurns = 0;
    }

    /**
     * start_threshold_min is the absolute gate; noise_floor_min only keeps the
     * tracked floor from collapsing to zero. Using noise_floor_min as the gate
     * here is what previously made the client and server disagree (19 vs 23
     * turns on the same audio at -43 dBFS), which moved every boundary on the
     * offline re-diarization.
     */
    _thresholds() {
        const floor = this.noiseFloor;
        const start = Math.max(floor * this.cfg.start_threshold_ratio,
                               this.cfg.start_threshold_min);
        const end = Math.max(floor * this.cfg.end_threshold_ratio,
                             this.cfg.end_threshold_min, start * 0.6);
        return [start, end];
    }

    /**
     * RMS of one int16 frame, scaled to 0..1.
     *
     * Math.fround on the accumulator keeps this in float32 like the server's
     * np.sqrt(np.mean(samples ** 2)); a float64 sum differs by ~1 ULP, which is
     * enough to flip an `rms >= threshold` comparison on a borderline frame and
     * move a turn boundary by one frame.
     */
    static _rms(frame) {
        if (!frame.length) return 0.0;
        let acc = 0;
        for (let i = 0; i < frame.length; i++) {
            const s = frame[i] / 32768;
            acc = Math.fround(acc + s * s);
        }
        return Math.fround(Math.sqrt(Math.fround(acc / frame.length)));
    }

    /** Follow quiet quickly, loud slowly -- the floor must not chase speech. */
    _trackFloor(rms) {
        if (rms < this.noiseFloor) {
            this.noiseFloor = 0.7 * this.noiseFloor + 0.3 * rms;
        } else {
            this.noiseFloor = 0.995 * this.noiseFloor + 0.005 * rms;
        }
        this.noiseFloor = Math.max(this.cfg.noise_floor_min,
                                   Math.min(this.cfg.noise_floor_max, this.noiseFloor));
    }

    /** Feed int16 samples; return the completed turns. */
    feed(pcm) {
        // Stitch the partial frame carried from the previous call.
        let data = pcm;
        if (this.carry) {
            const joined = new Int16Array(this.carry.length + pcm.length);
            joined.set(this.carry, 0);
            joined.set(pcm, this.carry.length);
            data = joined;
            this.carry = null;
        }
        const out = [];
        const frameLen = this.frameSamples;
        let offset = 0;
        while (offset + frameLen <= data.length) {
            const turn = this._frame(data.subarray(offset, offset + frameLen));
            offset += frameLen;
            if (turn) out.push(turn);
        }
        if (offset < data.length) {
            this.carry = data.slice(offset);
        }
        return out;
    }

    _frame(frame) {
        const rms = TurnDetector._rms(frame);
        this._trackFloor(rms);
        const th = this._thresholds();
        const startTh = th[0];
        const endTh = th[1];
        const idx = this.frameIndex;
        this.frameIndex += 1;

        if (!this.inTurn) {
            this.confirmRun = rms >= startTh ? this.confirmRun + 1 : 0;
            // Pre-roll keeps the frames just before the confirmed onset so a
            // word onset is not clipped.
            this.preRoll.push([idx, frame]);
            while (this.preRoll.length > this.preRollFrames) {
                this.preRoll.shift();
            }
            if (this.confirmRun >= this.cfg.start_confirm_frames) {
                this._openTurn(endTh);
            }
            return null;
        }

        this.turnFrames.push(frame);
        this.turnPeak = Math.max(this.turnPeak, rms);
        if (rms >= endTh) {
            this.turnVoiced += 1;
            this.silentRun = 0;
        } else {
            this.silentRun += 1;
            if (this.silentRun >= this.hangoverFrames) {
                return this._closeTurn('hangover');
            }
        }
        if (this.turnFrames.length >= this.maxTurnFrames) {
            return this._closeTurn('max_turn');
        }
        return null;
    }

    /**
     * Open a turn, keeping the pre-roll so the onset is not clipped.
     *
     * Mirrors the server exactly, including that only frames from the
     * confirmation index onward count as voiced -- the earlier pre-roll is
     * padding, so a click cannot pass the minimum-voiced gate. confirmRun is
     * deliberately NOT reset here: the server resets it only on close, and
     * clearing it here is what made the two detectors disagree.
     */
    _openTurn(endTh) {
        const need = Math.max(1, this.cfg.start_confirm_frames);
        const confirmIdx = this.frameIndex - need;
        const older = this.preRoll.filter((p) => p[0] < confirmIdx);
        this.inTurn = true;
        this.silentRun = 0;
        this.turnVoiced = 0;
        this.turnPeak = 0;
        this.turnFrames = older.map((p) => p[1]);
        this.turnStartFrame = older.length ? older[0][0] : confirmIdx;
        for (const p of this.preRoll) {
            if (p[0] < confirmIdx) continue;
            this.turnFrames.push(p[1]);
            const r = TurnDetector._rms(p[1]);
            if (r > this.turnPeak) this.turnPeak = r;
            if (r >= endTh) this.turnVoiced += 1;
        }
        this.preRoll = [];
    }

    /** Close the turn, trimming the trailing hangover back to post_roll. */
    _closeTurn(reason) {
        let frames = this.turnFrames;
        const voiced = this.turnVoiced;
        this.turnFrames = [];
        this.inTurn = false;
        this.turnVoiced = 0;
        this.confirmRun = 0;
        this.silentRun = 0;
        this.preRoll = [];
        if (!frames.length) return null;

        // Trailing silence beyond post-roll is padding, not audio: keeping it
        // would make a live turn longer than the offline turn at the same place.
        const hangover = Math.max(1, Math.round(this.cfg.hangover_ms / this.cfg.frame_ms));
        const postRoll = Math.max(0, Math.round(this.cfg.post_roll_ms / this.cfg.frame_ms));
        const keep = Math.max(1, frames.length - hangover + postRoll);
        frames = frames.slice(0, keep);

        // The min-voiced gate compares MILLISECONDS, like the server. Counting
        // frames (int(120/32) = 3) kept a 3-voiced-frame turn the server
        // correctly dropped.
        const voicedMs = voiced * this.cfg.frame_ms;
        if (voicedMs < this.cfg.min_voiced_ms) {
            this.droppedTurns += 1;
            return null;
        }

        const total = frames.reduce((n, f) => n + f.length, 0);
        const pcm = new Int16Array(total);
        let o = 0;
        for (const f of frames) { pcm.set(f, o); o += f.length; }
        const start = this.turnStartFrame * this.frameSamples;
        return new Turn(start, start + pcm.length, pcm, reason, this.turnPeak);
    }

    /** Emit a turn still open at end of stream, exactly once. */
    flush() {
        if (!this.inTurn) return null;
        return this._closeTurn('flush');
    }
}

/**
 * Hold a closed turn briefly and merge it into the next one.
 *
 * hangover_ms is 320 ms so a live turn closes promptly, but natural speech has
 * 300-800 ms pauses mid-sentence and every one of them used to cut a turn. A
 * 0.4 s fragment is useless twice over: Whisper hallucinates on it (half a
 * second of room noise reliably decodes to "Thank you.") and ECAPA cannot
 * identify anybody from 0.4 s of audio.
 *
 * poll() releases the held turn once merge_gap_sec has elapsed, which is what
 * bounds the added latency -- without it the last sentence of a session (or the
 * only sentence of a short one) is never sent.
 *
 * The merged payload is `head + tail` -- the pause is not concatenated in -- so
 * the merged turn declares the span it carries and keeps the real end of the
 * audio separately, for the next merge decision. Declaring the tail's real end
 * claimed up to merge_gap_sec of audio the server never received.
 */
class TurnCoalescer {
    constructor(cfg, clock) {
        this.cfg = Object.assign({}, DETECTOR_DEFAULTS, cfg || {});
        this.clock = clock || (() => performance.now() / 1000);
        this.pending = null;
        // Real end of the pending turn's audio; equal to pending.endSample
        // until a merge shortens the declared span.
        this.audioEnd = null;
        this.due = null;
        this.merged = 0;
        this.released = 0;
    }

    _defer(now) {
        this.due = (now !== null && now !== undefined ? now : this.clock())
            + this.cfg.merge_gap_sec;
    }

    _hold(turn) {
        this.pending = turn;
        this.audioEnd = turn.endSample;
    }

    /** Offer a completed turn; return the turns that are ready to send. */
    submit(turn, now) {
        if (!turn) return [];
        const pending = this.pending;
        if (!pending) {
            this._hold(turn);
            this._defer(now);
            return [];
        }
        const pendingEnd = this.audioEnd === null ? pending.endSample : this.audioEnd;
        const gap = turn.startSample - pendingEnd;
        const span = (turn.endSample - pending.startSample) / SAMPLE_RATE;
        if (gap >= 0 && gap <= this.cfg.merge_gap_sec * SAMPLE_RATE
                && span <= this.cfg.max_merge_sec) {
            const headLen = pending.pcm.length;
            const tailLen = turn.pcm.length;
            const start = pending.startSample;
            // Digital silence so the tail's audio lands where it actually
            // occurred. The pad is measured from the head's own length, not from
            // the turn boundary: a payload is only a slice of its turn's span
            // (the detector trims hangover frames), so head+tail would place the
            // tail up to merge_gap_sec early. With the pad the declared span ends
            // at the tail's true end.
            const pad = Math.max(0, turn.startSample - start - headLen);
            const total = headLen + pad + tailLen;
            const pcm = new Int16Array(total);
            pcm.set(pending.pcm, 0);
            pcm.set(turn.pcm, headLen + pad);
            const declaredEnd = start + pcm.length;
            this.pending = new Turn(start, declaredEnd, pcm, 'merged',
                Math.max(pending.peakRms, turn.peakRms));
            this.audioEnd = turn.endSample;
            this.merged += 1;
            this._defer(now);
            return [];
        }
        this.released += 1;
        this._hold(turn);
        this._defer(now);
        return [pending];
    }

    /** Release the held turn once the merge window has elapsed. */
    poll(now) {
        if (!this.pending || this.due === null) return [];
        const t = now !== null && now !== undefined ? now : this.clock();
        if (t < this.due) return [];
        const pending = this.pending;
        this.pending = null;
        this.audioEnd = null;
        this.due = null;
        this.released += 1;
        return [pending];
    }

    /** Release the held turn (end of stream / shutdown). */
    flush() {
        if (!this.pending) return [];
        const pending = this.pending;
        this.pending = null;
        this.audioEnd = null;
        this.due = null;
        this.released += 1;
        return [pending];
    }
}

// ── 3. Transcript rendering (port of transcribe_client.build_transcript) ───
//
// Kept byte-identical to the Windows client's output so a .txt from the web UI
// and one from transcribe.bat are interchangeable. Do not "improve" the rules
// here without changing transcribe_client.py too.

/** HH:MM:SS, hours zero-padded, never truncated. */
function fmtHms(sec) {
    const s = Math.max(0, Math.round(Number(sec) || 0));
    const hours = Math.floor(s / 3600);
    const rem = s % 3600;
    const minutes = Math.floor(rem / 60);
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(rem % 60).padStart(2, '0')}`;
}

function profilesBanner(result) {
    const profiles = result.profiles;
    if (!profiles || typeof profiles !== 'object' || !Object.keys(profiles).length) {
        return [];
    }
    const rows = [];
    for (const name of Object.keys(profiles)) {
        const p = profiles[name];
        if (!p || typeof p !== 'object') continue;
        const parts = [];
        const pitch = p.pitch_hz;
        if (typeof pitch === 'number' && pitch > 0) {
            const std = p.pitch_std;
            if (typeof std === 'number' && std > 0) {
                parts.push(`pitch=${pitch.toFixed(0)}Hz (±${std.toFixed(0)}Hz)`);
            } else {
                parts.push(`pitch=${pitch.toFixed(0)}Hz`);
            }
        }
        if (typeof p.energy_rms === 'number') {
            parts.push(`energy=${p.energy_rms.toFixed(4)}`);
        }
        if (typeof p.total_speech_sec === 'number') {
            parts.push(`speech=${p.total_speech_sec.toFixed(0)}s`);
        }
        if (parts.length) {
            const speech = typeof p.total_speech_sec === 'number' ? p.total_speech_sec : 0;
            rows.push([speech, `  ${name}: ` + parts.join('  ')]);
        }
    }
    if (!rows.length) return [];
    // Sorted by descending total speech: the most present speaker first.
    rows.sort((a, b) => b[0] - a[0]);
    const rule = '='.repeat(60);
    return [rule, 'SPEAKER VOICE PROFILES', rule]
        .concat(rows.map((r) => r[1])).concat([rule]);
}

/**
 * Group a run's segments into paragraphs.
 *
 * Break on a pause >= 1.5s, or at 800 chars provided the previous segment ends
 * a sentence (mid-text wrapping only at a sentence end), or hard at 1600.
 */
function paragraphsFromSegments(segments) {
    const paragraphs = [];
    let current = [];
    let currentLen = 0;
    for (const seg of segments) {
        const text = (seg.text || '').trim();
        if (!text) continue;
        const start = Number(seg.start) || 0;
        if (current.length) {
            const prev = current[current.length - 1];
            const prevEnd = Number(prev.end) || Number(prev.start) || 0;
            const gap = start - prevEnd;
            const lastText = (prev.text || '').trim();
            const sentenceEnd = /[.!?…]$/.test(lastText);
            if (gap >= 1.5 || (currentLen >= 800 && sentenceEnd) || currentLen >= 1600) {
                paragraphs.push(current);
                current = [];
                currentLen = 0;
            }
        }
        current.push(seg);
        currentLen += text.length + 1;
    }
    if (current.length) paragraphs.push(current);
    return paragraphs;
}

/** Duration-weighted mean confidence as a 0..100 int, or null if unavailable. */
function paragraphConfidence(segs) {
    let num = 0;
    let den = 0;
    for (const s of segs) {
        const c = s.confidence;
        if (c === null || c === undefined) continue;
        let w = (Number(s.end) || 0) - (Number(s.start) || 0);
        if (!(w > 0)) w = 1.0;
        num += c * w;
        den += w;
    }
    if (den <= 0) return null;
    return Math.max(0, Math.min(100, Math.round(100 * num / den)));
}

/** The full `<name>.txt` body. See transcribe_client.build_transcript. */
function buildTranscript(filename, result, dateStr) {
    const lines = [];
    lines.push(`Audio: ${filename}`);
    lines.push(`Date: ${dateStr}`);
    lines.push(`Speakers: ${result.total_speakers || 0}`);
    lines.push(`Duration: ${(Number(result.audio_duration_sec) || 0).toFixed(1)}s`);
    lines.push('');

    const banner = profilesBanner(result);
    if (banner.length) {
        lines.push(...banner);
        lines.push('');
    }

    const results = result.results || [];
    for (const r of results) {
        let speaker = r.speaker || UNKNOWN_SPEAKER;
        if (r.uncertain) {
            speaker = `${UNKNOWN_SPEAKER} (${r.attribution_reason || 'unspecified'})`;
        }
        const segs = (r.segments || []).filter(
            (s) => s && (s.text || '').trim());
        if (!segs.length) {
            // No segments: fall back to the coarse start/end form.
            const text = r.text || '';
            if (r.start !== null && r.start !== undefined
                    && r.end !== null && r.end !== undefined) {
                lines.push(`[${speaker}] ${r.start.toFixed(1)}s - ${r.end.toFixed(1)}s: `
                    + text.split('\n').join('\n    '));
            } else if (text) {
                lines.push(text);
            }
            continue;
        }
        for (const para of paragraphsFromSegments(segs)) {
            const pStart = Number(para[0].start) || 0;
            const text = para.map((s) => (s.text || '').trim()).join(' ').trim();
            if (!text) continue;
            const conf = paragraphConfidence(para);
            const confStr = conf === null ? '' : ` (${conf}%)`;
            lines.push(`[${fmtHms(pStart)}] ${speaker}${confStr}: ${text}`);
        }
    }

    const errors = results.filter((r) => r.error).map((r) => r.error);
    if (errors.length) {
        lines.push('');
        lines.push('Errors: ' + errors.join('; '));
    }

    const uncertain = results.filter((r) => r.uncertain);
    if (uncertain.length) {
        lines.push('');
        lines.push(`WARNING: ${uncertain.length} of ${results.length} segment(s) have `
            + 'an UNKNOWN speaker (identity could not be established; the text was '
            + 'kept). Re-run if you need clean labels.');
    }
    return lines.join('\n');
}

// ── 4. Session ─────────────────────────────────────────────────────────────

/**
 * Build a 16 kHz mono WAV Blob from recorded Int16 chunks.
 *
 * The recording is kept so the session can be re-processed later and so the
 * re-attribution upload has audio to diarize -- the server attributes text onto
 * diarized turns, so without the recording there is nothing to attribute
 * against.
 */
function wavBlob(chunks, sampleRate) {
    let total = 0;
    for (const c of chunks) total += c.length;
    const pcm = new Int16Array(total);
    let o = 0;
    for (const c of chunks) { pcm.set(c, o); o += c.length; }
    const buf = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buf);
    const writeStr = (off, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + pcm.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // PCM chunk size
    view.setUint16(20, 1, true);           // format = PCM
    view.setUint16(22, 1, true);           // channels = mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);   // byte rate
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, pcm.length * 2, true);
    new Int16Array(buf, 44).set(pcm);
    return new Blob([buf], { type: 'audio/wav' });
}

function isoStamp(date) {
    // Matches the client's %Y%m%d-%H%M%S (local time).
    const d = date || new Date();
    const p = (n, w) => String(n).padStart(w || 2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`
        + `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

class Channel {
    constructor(name) {
        this.name = name;
        this.detector = new TurnDetector();
        this.coalescer = new TurnCoalescer(this.detector.cfg);
        this.samples = 0;            // absolute position on this channel's timeline
        this.chunks = [];            // recorded PCM, for the WAV
        this.peak = 0;               // rolling peak for the level meter
        this.globalPeak = 0;
        this.gain = 1;                // last input gain the worklet applied
        this.everAudio = false;      // sticky: "has audio EVER reached this channel"
        this.turnsSent = 0;          // turns handed to the socket from this channel
    }
}

class Session {
    constructor(opts) {
        this.opts = Object.assign({
            // Reverse-proxy mount point. Every URL this class builds goes
            // through it, so an undefined value would emit "...hostundefined/..."
            // rather than fail loudly.
            prefix: (typeof window !== 'undefined' && window.livePrefix) || '',
            language: 'auto',
            token: '',
            autoLevel: true,
        }, opts);
        this.language = this.opts.language || 'auto';
        this.started = new Date();
        this.stamp = isoStamp(this.started);
        this.items = [];             // live ASR items -> the re-attribution input
        this.gaps = [];              // server-reported drops
        this.skipped = [];           // turns the server's speech gate rejected
        // Turns cut and handed to the socket. Distinct from transcripts
        // received back: a session can send 24 turns and get 0 replies (server
        // error, or a backend that returns nothing), and conflating the two
        // made "Turns sent: 0" print while the server logged queued_turns: 24.
        this.turnsSent = 0;
        this.transcriptCount = 0;
        this.dropCount = 0;
        this.skipCount = 0;
        this.sequence = 0;
        this.coveredSec = 0;
        this.serverStats = null;
        this.channels = { 0: new Channel('mic'), 1: new Channel('speakers') };
        this.ws = null;
        this.ctx = null;
        this.workletLoaded = false;
        this.nodes = [];
        this.stopped = false;
        // Set when the socket dies mid-session, so stop() can tell "the user
        // spoke and nothing came back" from "the connection never worked".
        this.streamingFailed = false;
    }

    channel(id) { return this.channels[id]; }
    get active() { return this.ws !== null; }

    // -- connection -----------------------------------------------------
    _wsUrl() {
        const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
        // Browser WebSockets cannot set headers, so auth rides the session
        // cookie (the handler checks scope["session"]["user"] first) or the
        // ?token= query param as a fallback.
        const token = this.opts.token ? `&token=${encodeURIComponent(this.opts.token)}` : '';
        return `${scheme}://${location.host}${this.opts.prefix}`
            + `/api/asr/ws/stream?language=${encodeURIComponent(this.language)}${token}`;
    }

    connect() {
        return new Promise((resolve, reject) => {
            let settled = false;
            let ws;
            try {
                ws = new WebSocket(this._wsUrl());
            } catch (e) {
                reject(e);
                return;
            }
            ws.binaryType = 'arraybuffer';
            this.ws = ws;

            ws.onopen = () => { settled = true; resolve(); };
            ws.onerror = () => {
                if (!settled) {
                    settled = true;
                    reject(new Error('Could not connect to the transcription server'));
                }
            };
            ws.onclose = (ev) => {
                if (!settled) {
                    settled = true;
                    reject(new Error(
                        ev.reason || `Connection refused (code ${ev.code}). `
                        + 'Check you are signed in.'));
                }
                this.streamingFailed = true;
                this.ws = null;
                if (this.opts.onClosed) this.opts.onClosed();
            };
            ws.onmessage = (ev) => this._onMessage(ev);
        });
    }

    _onMessage(ev) {
        let msg;
        try {
            msg = JSON.parse(typeof ev.data === 'string' ? ev.data : '');
        } catch (e) {
            return;
        }
        if (!msg || typeof msg !== 'object') return;
        const kind = msg.type;
        if (kind === 'transcript') {
            this.items.push({
                start: msg.start,
                end: msg.end,
                text: (msg.text || '').trim(),
                channel: msg.channel,
                speaker: msg.speaker,
                speaker_confidence: msg.speaker_confidence,
                speaker_source: msg.speaker_source,
                uncertain: !!msg.uncertain,
                attribution_reason: msg.attribution_reason,
            });
            this.transcriptCount += 1;
            this.opts.onTranscript(msg);
        } else if (kind === 'empty') {
            // A detected turn the decoder produced no words for. When the
            // server's speech gate rejected it, `skipped` says why -- worth
            // counting, because "the server discarded 31 of my turns" and "the
            // server transcribed 31 noise turns" look identical otherwise.
            if (msg.skipped) {
                this.skipped.push({
                    start: msg.start, end: msg.end, channel: msg.channel,
                    reason: msg.skipped, speech_score: msg.speech_score,
                });
                this.skipCount += 1;
                this.opts.onSkip(msg);
            }
        } else if (kind === 'dropped') {
            this.gaps.push({
                start: msg.start, end: msg.end,
                channel: msg.channel, reason: msg.reason, detail: msg.detail,
            });
            this.dropCount += 1;
            this.opts.onDropped(msg);
        } else if (kind === 'error') {
            this.opts.onError(msg.message || 'server error');
        } else if (kind === 'stats') {
            this.serverStats = msg;
            this.coveredSec = Number(msg.covered_sec) || 0;
            this.opts.onStats(msg);
            // Release finish()'s drain wait: stats is the end-of-session
            // signal, and waiting for socket close would discard every item
            // still decoding.
            if (this.opts.onDrained) this.opts.onDrained(msg);
        }
    }

    // -- capture --------------------------------------------------------
    /**
     * Create the AudioContext, get it *running*, and load the worklet.
     *
     * This must be awaited from inside the Start click handler, before any
     * other await. Chrome's autoplay policy starts a context created without
     * user activation in the `suspended` state, and a suspended context never
     * runs the audio graph -- so process() is never called, the worklet posts
     * nothing, the detector never opens a turn, and not a single byte reaches
     * the socket. The failure is completely silent: no exception, no console
     * error, an open WebSocket, and a server log showing packets: 0.
     *
     * Constructing the context inside the click also matters for its own sake:
     * every await() before this point (the two permission prompts, the WS
     * connect) consumes the activation, so creating it late guarantees
     * suspended in Chrome and a "no audio" tab in Firefox.
     */
    async ensureAudio() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.ctx.state === 'suspended') {
            await this.ctx.resume();
        }
        if (this.ctx.state !== 'running') {
            throw new Error(
                `the browser blocked audio (AudioContext is ${this.ctx.state}). `
                + 'Click Start again, or allow audio for this site in the '
                + 'address-bar permissions.');
        }
        if (!this.workletLoaded) {
            await this.ctx.audioWorklet.addModule(
                `${this.opts.prefix}/static/live-worklet.js`);
            this.workletLoaded = true;
        }
        return this.ctx;
    }

    /**
     * Attach a MediaStream to a channel: getUserMedia for the mic,
     * getDisplayMedia for tab/system audio (the browser's stand-in for the
     * Windows client's WASAPI loopback).
     */
    async attach(channelId, stream) {
        await this.ensureAudio();
        const ch = this.channel(channelId);
        const source = this.ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(this.ctx, 'live-capture', {
            numberOfInputs: 1,
            numberOfOutputs: 0,
            channelCount: 1,
            channelCountMode: 'explicit',
            processorOptions: {
                sampleRate: this.ctx.sampleRate,
                autoLevel: this.opts.autoLevel !== false,
            },
        });
        node.port.onmessage = (ev) => {
            const data = ev.data;
            if (!data || !data.pcm) return;
            this._onPcm(channelId, data.pcm, data.gain);
        };
        source.connect(node);
        this.nodes.push({ source, node, stream, channelId });
    }

    _onPcm(channelId, pcm, gain) {
        const ch = this.channel(channelId);
        ch.chunks.push(pcm);
        const peak = Session._peak(pcm);
        ch.peak = peak;
        if (typeof gain === 'number' && isFinite(gain)) ch.gain = gain;
        if (peak > ch.globalPeak) ch.globalPeak = peak;
        ch.everAudio = true;
        // The detector's frame index is the timeline: sample counts are per
        // channel, so each channel has its own clock (the client does the same,
        // and the server's Turn.start_sample is a per-channel position).
        for (const turn of ch.detector.feed(pcm)) {
            for (const ready of ch.coalescer.submit(turn)) {
                this._sendTurn(channelId, ready);
            }
        }
    }

    static _peak(pcm) {
        let m = 0;
        for (let i = 0; i < pcm.length; i++) {
            const a = Math.abs(pcm[i]);
            if (a > m) m = a;
        }
        return m / 32768;
    }

    _sendTurn(channelId, turn) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.sequence += 1;
        this.ws.send(packTurn(channelId, turn.startSample, turn.pcm, this.sequence));
        this.turnsSent += 1;
        this.channel(channelId).turnsSent += 1;
    }

    /**
     * "mic 3, speaker 12" -- the per-channel turn split.
     *
     * This is the single most diagnostic number the client has. A session where
     * the mic sent nothing but the speaker channel did is the signature of a
     * too-quiet microphone: the mic recording has no speech for the
     * re-attribution to diarize, every item collapses onto one speakerless
     * turn and the whole transcript comes back UNKNOWN with no explanation.
     */
    turnSplit() {
        return `mic ${this.channel(CHANNEL_MIC).turnsSent}, `
            + `speaker ${this.channel(CHANNEL_SPEAKER).turnsSent}`;
    }

    /**
     * Release any turn whose merge window has elapsed.
     *
     * Called on a timer rather than only when audio arrives: without it a turn
     * waits for the NEXT turn, so the last sentence of a session -- and the only
     * sentence of a short one -- would never be sent.
     */
    poll() {
        for (const id of [CHANNEL_MIC, CHANNEL_SPEAKER]) {
            const ch = this.channel(id);
            if (!this.nodes.some((n) => n.channelId === id)) continue;
            for (const ready of ch.coalescer.poll()) {
                this._sendTurn(id, ready);
            }
        }
    }

    // -- shutdown -------------------------------------------------------
    /**
     * Stop capture, flush both channels, send MSG_FLUSH and wait for the
     * server's `stats` frame.
     *
     * `stats` is the end-of-session signal, not socket close: the server drains
     * its ASR queue and only then emits stats, and a transcript that arrives
     * after close would be lost. 30 s is the same drain window the Windows
     * client uses.
     */
    async finish(waitMs) {
        this.stopped = true;
        // 1. Stop the audio graph so no more PCM arrives mid-flush.
        for (const n of this.nodes) {
            try { n.node.port.postMessage({ type: 'stop' }); } catch (e) { /* closed */ }
            try { n.source.disconnect(); } catch (e) { /* already gone */ }
            for (const track of n.stream.getTracks()) track.stop();
        }
        this.nodes = [];
        if (this.ctx) {
            try { await this.ctx.close(); } catch (e) { /* already closed */ }
            this.ctx = null;
        }

        // 2. Flush detectors (trailing speech) then coalescers (held turns), so
        //    the last sentence is not lost. Neither is appended to the
        //    recording: every block the detector saw was already pushed to
        //    chunks in _onPcm, so writing a turn's PCM again made the WAV up to
        //    one turn too long with the tail duplicated -- which over-reported
        //    audio_duration_sec / micSeconds() and gave the shutdown
        //    re-attribution a repeated region to diarize into extra segments
        //    (and spurious pending profiles).
        for (const id of [CHANNEL_MIC, CHANNEL_SPEAKER]) {
            const ch = this.channel(id);
            const tail = ch.detector.flush();
            // The tail must be submitted to the coalescer, not just recorded:
            // pushing it into chunks alone left it in no turn list at all, so
            // the last sentence of every session was silently never sent.
            if (tail) ch.coalescer.submit(tail, 0);
            for (const ready of ch.coalescer.flush()) {
                this._sendTurn(id, ready);
            }
        }

        // 3. MSG_FLUSH, then wait for stats.
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(packFlush());
            await this._awaitStats(waitMs || 30000);
        }
        if (this.ws) {
            try { this.ws.close(); } catch (e) { /* already closed */ }
            this.ws = null;
        }
    }

    _awaitStats(ms) {
        return new Promise((resolve) => {
            if (this.serverStats) { resolve(this.serverStats); return; }
            let done = false;
            const finish = (stats) => {
                if (done) return;
                done = true;
                clearTimeout(timer);
                clearInterval(poll);
                this.opts.onDrained = null;
                resolve(stats);
            };
            this.opts.onDrained = finish;
            const timer = setTimeout(() => finish(null), ms);
            // Belt and braces: onclose fires after stats normally, but if the
            // server died first we must not wait the full window.
            const poll = setInterval(() => {
                if (this.serverStats) finish(this.serverStats);
            }, 250);
        });
    }

    // -- results --------------------------------------------------------
    /**
     * Shape the live items like a transcribe result.
     *
     * Sorted by start time: the server drains one serial ASR worker, so items
     * arrive in decode-completion order, not chronological. Left unsorted the
     * output reads "[00:00:30] You ... [00:00:00] UNKNOWN ...".
     */
    buildLiveResult() {
        const items = this.items.slice().sort((a, b) => (a.start || 0) - (b.start || 0));
        const speakers = new Set();
        for (const it of items) if (it.speaker) speakers.add(it.speaker);
        return {
            total_speakers: speakers.size,
            audio_duration_sec: this.micSeconds(),
            results: items,
        };
    }

    micSeconds() {
        let n = 0;
        for (const c of this.channels[0].chunks) n += c.length;
        return n / SAMPLE_RATE;
    }

    /** The machine-readable sidecar, matching the client's .asr.json. */
    buildSidecar(extra) {
        return Object.assign({
            session_started: this.started.toISOString(),
            language: this.language,
            sample_rate: SAMPLE_RATE,
            channels: {
                0: 'microphone (local user)',
                1: 'screen/tab audio (everyone else)',
            },
            asr_source: 'live_stream',
            turns_sent: this.turnsSent,
            turns_sent_per_channel: {
                mic: this.channel(CHANNEL_MIC).turnsSent,
                speaker: this.channel(CHANNEL_SPEAKER).turnsSent,
            },
            transcripts_received: this.transcriptCount,
            turns_skipped_non_speech: this.skipCount,
            skipped: this.skipped,
            gaps: this.gaps,
            server_stats: this.serverStats,
            items: this.items,
        }, extra || {});
    }

    /**
     * Upload the recording + cached items; diarize only, never re-decode.
     *
     * See docs/lessons/live-client-design.md: the ASR backend is called once per
     * file and attribution is a pure function of (items, turns), so re-running
     * diarization over the recording and re-attributing the text we already
     * have costs ~a fifth of a full re-transcription. Returns null on any
     * failure -- the live transcript is already in hand, so this is never fatal.
     */
    async reattribute() {
        const items = this.items
            .filter((it) => (it.text || '').trim())
            .map((it) => ({ start: it.start, end: it.end, text: it.text }));
        if (!items.length) return null;

        const form = new FormData();
        form.append('items', JSON.stringify(items));
        form.append('file', this.recording(CHANNEL_MIC), `${this.stamp}.wav`);
        try {
            const r = await fetch(`${this.opts.prefix}/api/asr/attribution/upload`, {
                method: 'POST',
                credentials: 'same-origin',
                body: form,
            });
            if (!r.ok) {
                throw new Error(`server returned HTTP ${r.status}`);
            }
            const payload = await r.json();
            if (payload && payload.error) throw new Error(payload.error);
            return payload;
        } catch (e) {
            this.opts.onError(`Re-attribution failed: ${e.message}`);
            return null;
        }
    }

    recording(channelId) {
        return wavBlob(this.channel(channelId).chunks, SAMPLE_RATE);
    }
}

/** Decode a turn frame -- used by the test that pins this side to the server. */
function unpackTurnForTest(buf) {
    const view = new DataView(buf);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1),
        view.getUint8(2), view.getUint8(3));
    if (magic !== 'LVT1') throw new Error('bad magic: ' + magic);
    const version = view.getUint8(4);
    if (version !== PROTOCOL_VERSION) throw new Error('bad version');
    const msgType = view.getUint8(5);
    const channel = view.getUint16(6, true);
    const start = view.getUint32(8, true) + view.getUint32(12, true) * 4294967296;
    const count = view.getUint32(16, true);
    const sequence = view.getUint32(20, true);
    const pcm = new Int16Array(buf, TURN_HEADER_SIZE, count);
    return { version, msgType, channel, start, count, sequence, pcm };
}

// ── UI controller ──────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

/**
 * Show/hide by toggling .hidden.
 *
 * Visibility is a class so the stylesheet keeps ownership of every element's
 * display mode -- the same rule the rest of the SPA follows. live.js defines
 * its own copy rather than reaching for app.html's `show()`: this file is
 * loaded and exercised by the node test harness with no page around it.
 */
function _show(id, on) {
    const el = $(id);
    if (el) el.classList.toggle('hidden', !on);
    return el;
}

class UI {
    constructor() {
        this.session = null;
        this.poller = null;
        this.meterTimer = null;
        this.audioWatchdog = null;
        this.lastResult = null;
        this.usedRediag = false;
        this._bind();
    }

    _bind() {
        const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
        on('liveStart', 'click', () => this.start());
        on('liveStop', 'click', () => this.stop());
        on('liveSpeaker', 'change', (e) => { this.wantSpeaker = e.target.checked; });
        this.wantSpeaker = true;
        on('liveSaveHistory', 'change', (e) => { this.saveHistory = e.target.checked; });
        this.saveHistory = true;
    }

    setStatus(text, cls) {
        const el = $('liveStatus');
        if (!el) return;
        el.textContent = text;
        el.className = 'pill' + (cls ? ' ' + cls : '');
    }

    clear() {
        const pane = $('liveText');
        if (pane) pane.innerHTML = '';
    }

    _appendTranscript(msg) {
        const pane = $('liveText');
        if (!pane) return;
        const block = document.createElement('div');
        block.className = 'tp-block';
        if (!msg.speaker) block.className += ' tp-block-uncertain';
        const head = document.createElement('div');
        head.className = 'tp-head';
        let label = msg.speaker || `${UNKNOWN_SPEAKER} (${msg.attribution_reason || 'unspecified'})`;
        if (typeof msg.speaker_confidence === 'number' && msg.speaker_confidence > 0) {
            label += ` (${(msg.speaker_confidence * 100).toFixed(0)}%)`;
        }
        head.textContent = `[${fmtHms(msg.start || 0)}] ${label}`;
        const text = document.createElement('div');
        text.className = 'tp-text';
        text.textContent = (msg.text || '').trim();
        block.appendChild(head);
        block.appendChild(text);
        pane.appendChild(block);
        // Autoscroll only when already at the bottom, so reading back is not
        // yanked away by each new line.
        const near = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 80;
        if (near) pane.scrollTop = pane.scrollHeight;
    }

    _note(text, cls) {
        const pane = $('liveText');
        if (!pane) return;
        const d = document.createElement('div');
        d.className = 'note' + (cls ? ' ' + cls : '');
        d.textContent = text;
        pane.appendChild(d);
    }

    /**
     * Drive the two level meters.
     *
     * The fill is the inner <div> of the shared .mini-bar -- a block box on
     * purpose, because width does not apply to an inline element and the
     * original <span> fill silently dropped every write.
     */
    _meter() {
        for (const id of [CHANNEL_MIC, CHANNEL_SPEAKER]) {
            const el = id === CHANNEL_MIC ? $('liveMeterMic') : $('liveMeterSpk');
            if (!el || !this.session) continue;
            const bar = el.parentElement;
            const lbl = bar && bar.parentElement
                ? bar.parentElement.querySelector('.meter-label') : null;
            const ch = this.session.channel(id);
            const running = this.session.nodes.some((n) => n.channelId === id);
            if (!running) {
                el.style.width = '0%';
                if (bar) bar.classList.remove('hot');
                if (lbl) lbl.textContent = '';
                continue;
            }
            // -60..0 dB mapped onto the bar width.
            const db = ch.peak > 0 ? 20 * Math.log10(ch.peak) : -60;
            const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
            el.style.width = `${pct.toFixed(1)}%`;
            if (bar) bar.classList.toggle('hot', db > -1);
            if (lbl) {
                // Show the boost too, so a quiet input reads as "levelled up"
                // rather than as a meter that is stuck low.
                const boost = ch.gain > 1.01
                    ? ` (+${(20 * Math.log10(ch.gain)).toFixed(0)} dB)` : '';
                lbl.textContent = `${db.toFixed(1)} dB${boost}`;
            }
        }
    }

    async start() {
        if (this.session && !this.session.stopped) return;
        this.lastResult = null;
        this.usedRediag = false;
        this.clear();
        const langEl = $('liveLang');
        const tokenEl = $('liveToken');
        const language = langEl && langEl.value ? langEl.value : 'auto';
        // The Settings tab writes the client token here; a browser WebSocket
        // cannot set headers, so ?token= is the fallback when there is no
        // session cookie (e.g. a reverse proxy that strips it).
        const token = tokenEl && tokenEl.value ? tokenEl.value.trim() : '';

        // The Session is created FIRST so its AudioContext is constructed
        // inside the click handler: Chrome starts a context built without user
        // activation in the `suspended` state, and a suspended context never
        // runs the audio graph -- no process() calls, no worklet messages, no
        // turns, no bytes on the socket, and no error anywhere. Every await
        // below consumes the activation, so the order here is load-bearing.
        const session = new Session({
            prefix: window.livePrefix || '',
            language,
            token,
            onTranscript: (m) => this._appendTranscript(m),
            onDropped: (m) => {
                if (m.reason === 'timeline_drift') {
                    // Not a dropped turn: the server could not believe the
                    // timestamps this channel declared. Every word is still
                    // here, but the labels come from a shifted timeline.
                    const which = m.channel === CHANNEL_MIC ? 'Microphone' : 'Speaker';
                    this._note(
                        `${which} timestamps look wrong (${m.detail || m.reason}); `
                        + 'speaker labels on that channel may be off.', 'warn');
                    return;
                }
                this._note(
                    `Decoder fell behind: dropped turn ${m.start}-${m.end}s`, 'warn');
            },
            onSkip: () => { /* counted; summarised at the end, not per turn */ },
            onError: (m) => this._note(m, 'warn'),
            onStats: (s) => { this.lastStats = s; },
            onClosed: () => { if (!session.stopped) this.setStatus('Disconnected', 'bad'); },
        });
        this.session = session;
        try {
            await session.ensureAudio();
        } catch (e) {
            this.setStatus(`Audio blocked: ${e.message}`, 'bad');
            return;
        }

        let micStream;
        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
            });
        } catch (e) {
            this.setStatus(`Microphone denied: ${e.message}`, 'bad');
            this.session = null;
            return;
        }

        let spkStream = null;
        if (this.wantSpeaker) {
            try {
                spkStream = await navigator.mediaDevices.getDisplayMedia({
                    audio: { channelCount: 1 },
                    video: true,   // Chrome refuses tab audio unless video is requested
                });
            } catch (e) {
                // Not fatal: channel 0 alone still transcribes, it just cannot
                // name anyone (see the note appended below).
                this._note(
                    'Tab/system audio was not shared, so only the microphone is '
                    + 'transcribed. Other people cannot be named.', 'italic');
            }
        }

        this.setStatus('Connecting...', '');
        try {
            await session.connect();
        } catch (e) {
            micStream.getTracks().forEach((t) => t.stop());
            if (spkStream) spkStream.getTracks().forEach((t) => t.stop());
            this.session = null;
            this.setStatus(e.message, 'bad');
            return;
        }

        try {
            await session.attach(CHANNEL_MIC, micStream);
            if (spkStream) {
                // The user can stop sharing from the browser's own bar; treat
                // that as "the speaker channel ended" rather than an error.
                spkStream.getVideoTracks()[0].addEventListener('ended', () => {
                    this._note('Tab/system audio sharing stopped.', 'italic');
                });
                await session.attach(CHANNEL_SPEAKER, spkStream);
            }
        } catch (e) {
            micStream.getTracks().forEach((t) => t.stop());
            this.setStatus(`Could not start audio: ${e.message}`, 'bad');
            return;
        }

        $('liveStart').disabled = true;
        $('liveStop').disabled = false;
        // Hide the settings, never the transport controls: they live in their
        // own row, so the Stop button stays reachable for the whole session.
        _show('liveSettings', false);
        _show('liveConnection', false);
        _show('liveOutput', true);
        this.setStatus('Recording', 'ok');
        this.lastStats = null;

        this.poller = setInterval(() => session.poll(), 200);
        this.meterTimer = setInterval(() => this._meter(), 100);
        this.audioWatchdog = setTimeout(() => this._warnIfSilent(), 3000);
    }

    /**
     * If neither channel has produced a single audio block a few seconds in,
     * say so. A dead capture is otherwise indistinguishable from a quiet room:
     * no turns, no transcript, no console error, just an unresponsive tab.
     */
    _warnIfSilent() {
        const session = this.session;
        if (!session || session.stopped) return;
        const dead = [CHANNEL_MIC, CHANNEL_SPEAKER]
            .filter((id) => !session.channel(id).everAudio);
        if (!dead.length) return;
        const which = dead.length === 2 ? 'Neither channel'
            : (dead[0] === CHANNEL_MIC ? 'The microphone' : 'Tab / system audio');
        this._note(
            `${which} produced no audio at all. Check that the `
            + (dead[0] === CHANNEL_MIC ? 'microphone is unmuted and allowed' : 'shared tab is playing audio')
            + ` (the address-bar permissions icon) -- the server has received `
            + `${session.turnsSent} turn(s) so far.`, 'warn');
    }

    async stop() {
        const session = this.session;
        if (!session) return;
        $('liveStop').disabled = true;
        this.setStatus('Finishing...', '');
        clearInterval(this.poller);
        clearInterval(this.meterTimer);
        clearTimeout(this.audioWatchdog);
        this.poller = null;
        this.meterTimer = null;
        this.audioWatchdog = null;
        this._meter();

        const wasStreaming = !session.streamingFailed;
        // A failed drain must not strand the tab: the buttons come back
        // whatever happens, and whatever was received is still usable.
        try {
            await session.finish(30000);
        } catch (e) {
            this._note(`Drain failed: ${e && e.message ? e.message : e}. `
                + 'Keeping whatever the server already sent.', 'warn');
        } finally {
            $('liveStart').disabled = false;
            _show('liveSettings', true);
            _show('liveConnection', true);
            _show('liveOutput', false);
        }

        if (!session.transcriptCount) {
            // Turns went out but no text came back. That is a capture or server
            // problem, so say so in those terms rather than writing an empty
            // transcript that reads as success.
            this.setStatus('No transcript', 'bad');
            this._note(
                `Turns sent: ${session.turnsSent} (${session.turnSplit()}), `
                + `transcripts received: ${session.transcriptCount}. Microphone `
                + `recorded ${session.micSeconds().toFixed(1)}s. If you spoke, `
                + 'capture or the server is the problem.', 'warn');
            return;
        }

        // Text arrived, so the session is worth finishing regardless of how
        // the socket ended: a dropped connection after the last transcript
        // must not throw away the recording and the text with it.
        if (!wasStreaming) {
            this._note('The connection dropped before the session ended. '
                + 'Re-attributing the audio received so far.', 'warn');
        }
        if (!session.channel(CHANNEL_MIC).turnsSent) {
            this._note('No turn was detected on the microphone, so its '
                + 'recording may have no speech to attribute against. Speakers '
                + 'will fall back to UNKNOWN.', 'warn');
        }

        this.setStatus('Re-diarizing (no re-transcription)...', '');
        const rediag = await session.reattribute();
        const result = (rediag && !rediag.error && rediag.results && rediag.results.length)
            ? rediag : session.buildLiveResult();
        this.usedRediag = !!(rediag && !rediag.error && rediag.results && rediag.results.length);
        this.lastResult = result;

        this._note(
            `${this.usedRediag ? 'Re-diarized' : 'Live labels only'} -- `
            + `${session.transcriptCount} segment(s), `
            + `${session.skipCount} non-speech turn(s) filtered, `
            + `${session.dropCount} dropped.`, 'ok');
        this.setStatus('Done', 'ok');

        this._renderDownloads(session, result, rediag);
        if (this.saveHistory) await this._saveHistory(session, result, rediag);
    }

    _renderDownloads(session, result, rediag) {
        const box = $('liveDownloads');
        if (!box) return;
        box.innerHTML = '';
        const stamp = session.stamp;
        const add = (label, blob, name) => {
            const a = document.createElement('a');
            a.className = 'btn btn-sm';
            a.href = URL.createObjectURL(blob);
            a.download = name;
            a.textContent = label;
            box.appendChild(a);
        };
        const header = [
            `# ${this.usedRediag ? 'Re-diarized' : 'Live (diarization skipped)'}`,
            `# Session: ${session.started.toISOString()}`,
            `# Audio: ${stamp}.wav`,
            `# Turns sent: ${session.turnsSent} (${session.turnSplit()})  Transcripts: ${session.transcriptCount}`
            + `  Gaps: ${session.dropCount}`,
        ].join('\n') + '\n';
        const dateStr = session.started.toISOString().slice(0, 19).replace('T', ' ');
        const txt = header + buildTranscript(`${stamp}.wav`, result, dateStr);
        add('Transcript (.txt)', new Blob([txt], { type: 'text/plain' }), `${stamp}.txt`);

        const sidecar = session.buildSidecar({
            final_transcript: `${stamp}.txt`,
            attribution_source: this.usedRediag ? 're_diarized' : 'live',
            final_result: result,
            rediarization: rediag ? { total_speakers: rediag.total_speakers,
                processing_time_sec: rediag.processing_time_sec } : null,
        });
        add('Sidecar (.json)', new Blob([JSON.stringify(sidecar, null, 2)],
            { type: 'application/json' }), `${stamp}.asr.json`);
        if (session.channel(CHANNEL_MIC).chunks.length) {
            add('Recording (.wav)', session.recording(CHANNEL_MIC), `${stamp}.wav`);
        }
        if (session.channel(CHANNEL_SPEAKER).chunks.length) {
            add('Speaker recording (.wav)', session.recording(CHANNEL_SPEAKER),
                `${stamp}-speaker.wav`);
        }
    }

    /**
     * Persist the session so it appears in the History tab.
     *
     * A failure is not fatal: the downloads above are already on the page, so
     * this is reported and otherwise ignored.
     */
    async _saveHistory(session, result, rediag) {
        const stamp = session.stamp;
        const payload = {
            audio_filename: `${stamp}.wav`,
            result: {
                total_speakers: result.total_speakers || 0,
                audio_duration_sec: session.micSeconds(),
                processing_time_sec: rediag ? rediag.processing_time_sec : 0,
                results: result.results || [],
                profiles: result.profiles || null,
            },
            stats: {
                turns_sent: session.turnsSent,
                turns_sent_per_channel: {
                    mic: session.channel(CHANNEL_MIC).turnsSent,
                    speaker: session.channel(CHANNEL_SPEAKER).turnsSent,
                },
                transcripts_received: session.transcriptCount,
                turns_skipped_non_speech: session.skipCount,
                dropped: session.dropCount,
                covered_sec: session.coveredSec,
                server_stats: session.serverStats,
            },
            sidecar: {
                language: session.language,
                gaps: session.gaps,
                skipped: session.skipped,
                attribution_source: this.usedRediag ? 're_diarized' : 'live',
            },
        };
        try {
            const r = await fetch(`${window.livePrefix || ''}/api/asr/live/save`, {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!r.ok) {
                const body = await r.json().catch(() => ({}));
                throw new Error(body.detail || `HTTP ${r.status}`);
            }
            this._note('Saved to Transcription history.', 'ok');
        } catch (e) {
            this._note(`Not saved to history: ${e.message} (the downloads above are fine)`,
                'warn');
        }
    }
}

return {
    SAMPLE_RATE, CHANNEL_MIC, CHANNEL_SPEAKER,
    MSG_TURN, MSG_FLUSH, TURN_HEADER_SIZE, PROTOCOL_VERSION,
    DETECTOR_DEFAULTS,
    Turn, TurnDetector, TurnCoalescer, Session, Channel,
    packTurn, packFlush, unpackTurnForTest,
    fmtHms, buildTranscript, paragraphsFromSegments, paragraphConfidence,
    profilesBanner, wavBlob, isoStamp,

    // -- the tab's UI controller -----------------------------------------
    _ui: null,

    /** Wire the Live tab. Safe to call more than once. */
    init() {
        if (Live._ui) return Live._ui;
        Live._ui = new UI();
        return Live._ui;
    },
};

})();

if (typeof window !== 'undefined') {
    window.Live = Live;
    // app.html sets this; the worklet URL and API paths need the reverse-proxy
    // prefix and a WebSocket cannot read it from anywhere else.
    window.livePrefix = window.livePrefix || '';
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Live;
}
