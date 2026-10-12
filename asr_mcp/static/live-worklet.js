/**
 * AudioWorkletProcessor for the browser Live tab.
 *
 * Runs in AudioWorkletGlobalScope, so it has no access to the page, the DOM or
 * any import. It receives mono float32 at the AudioContext rate and posts
 * 16 kHz mono int16 blocks to the main thread, which is where turn detection
 * and the WebSocket live (see static/live.js).
 *
 * Why a worklet rather than a ScriptProcessorNode: the turn detector and the
 * socket write both run on the main thread, and a ScriptProcessorNode callback
 * runs there too.  On a busy page that starves the audio callback and the
 * recording drops out.  The worklet only does the resample.
 *
 * Resampling is linear interpolation with a carried fractional read position.
 * soxr (what the Windows client uses) is not available in a browser, and for
 * speech endpointing the difference is not observable -- the detector's
 * thresholds are ratios against a tracked noise floor, and the absolute gate
 * (start_threshold_min) is far above any resampling noise.
 *
 * The posted length is BLOCK_SAMPLES of OUTPUT rate (1024 at 16 kHz = 64 ms),
 * matching the Windows client's --block-ms 64 default, so both clients hand
 * the detector audio in comparable chunks.
 *
 * Input levelling (auto-gain) happens here, in float, before the int16
 * conversion. Doing it in the worklet rather than on the sent frames matters:
 * the same PCM is stored for the WAV that is re-uploaded to
 * /api/asr/attribution/upload, and re-attribution runs VAD and ECAPA-TDNN
 * embeddings over it -- both of which are level sensitive. Levelling only the
 * frames on the wire would leave the offline pass working on quiet audio.
 *
 * The gain is updated ONCE PER POSTED BLOCK, on that block's own RMS: 1024
 * samples at 16 kHz = 64 ms, the same cadence the Windows client gets from its
 * resampled chunk (`--block-ms 64`). The render quantum is 128 input samples
 * (2.7 ms at 48 kHz), and driving the gain from THAT is what the loop must not
 * do: ATTACK 0.5 reaches `desired` within two quanta, so the smoothing is
 * bypassed entirely and the leveller degenerates into a per-quantum
 * normaliser that pins every quantum to the target -- room tone included,
 * because TARGET sits above any realistic room SNR. The speech envelope is
 * flattened to a constant level, the detector's adaptive noise floor rises with
 * the noise it is being fed, and Silero then sees a signal with no speech
 * structure left in it.
 *
 * The gain loop targets a speech RMS and moves in the log domain, so a quiet
 * microphone is pulled up quickly while a loud one is eased down slowly (a
 * linear ramp makes the gain audibly pump on every syllable). It is gated at
 * MIN_ENV so room tone is left alone instead of being dragged up to full
 * scale, and clamped by a per-block peak limiter so the boost cannot
 * clip.
 */

const OUTPUT_RATE = 16000;
const BLOCK_SAMPLES = 1024;

// -18 dBFS RMS: a normal level for speech into a recogniser.
const TARGET_RMS = 0.125;
// ~-62 dBFS. Only real room tone is protected; a quiet talker is lifted all
// the way to MAX_GAIN.
const MIN_ENV = 0.0008;
const MAX_GAIN = 16.0;   // +24 dB, enough for a laptop mic across a room
const ATTACK = 0.5;      // raise the gain fast ...
const RELEASE = 0.02;    // ... lower it slowly
const PEAK_CEILING = 0.98;

class LiveCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = (options && options.processorOptions) || {};
        this.inRate = opts.sampleRate || sampleRate || 48000;
        this.ratio = this.inRate / OUTPUT_RATE;
        // Resampled float samples of the block currently being filled. The
        // gain is measured on these, so it has to stay float until the block
        // is complete -- see _emit().
        this.out = new Float32Array(BLOCK_SAMPLES);
        this.outFill = 0;
        // Fractional read position within the current input block, and the last
        // input sample of the previous block so interpolation has continuity
        // across block boundaries (without it every boundary clicks).
        this.pos = 0;
        this.prev = 0;
        this.closed = false;
        this.autoLevel = opts.autoLevel !== false;
        this.gain = 1;
        // Per-block levelling state, posted with every block so the UI can show
        // the boost it applied.
        this.lastGain = 1;
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'stop') {
                this.closed = true;
            } else if (e.data && e.data.type === 'level') {
                this.autoLevel = !!e.data.autoLevel;
            }
        };
    }

    /** Convert one float sample to int16 with clipping. */
    static toInt16(v) {
        const s = v < -1 ? -1 : (v > 1 ? 1 : v);
        return s < 0 ? s * 32768 : s * 32767;
    }

    /**
     * Move the gain toward TARGET_RMS, in the log domain so the time constant
     * is constant in dB rather than in amplitude.
     */
    _updateGain(rms) {
        // desired is the gain that puts THIS block at the target: TARGET/rms.
        // Dividing by the already-gained level instead (TARGET/(rms*gain))
        // fixes the loop at sqrt(TARGET/rms) -- a geometric mean of where it
        // started and where it was going -- so it settles short of the target
        // and a -40 dBFS mic stays at -32 dBFS.
        let desired = 1;
        if (rms > MIN_ENV) {
            desired = Math.min(MAX_GAIN, TARGET_RMS / rms);
        }
        const k = desired > this.gain ? ATTACK : RELEASE;
        this.gain *= Math.pow(desired / this.gain, k);
        if (this.gain < 1) this.gain = 1;
        if (this.gain > MAX_GAIN) this.gain = MAX_GAIN;
        return this.gain;
    }

    /**
     * Emit one complete output block: measure it, level it, post it.
     *
     * This is the ONLY place the gain is updated, and it runs once per block
     * (BLOCK_SAMPLES at OUTPUT_RATE = 64 ms) -- the same cadence the Windows
     * client gets from its resampled chunk (`--block-ms 64`). One gain for the
     * whole block is also what keeps a word intact: a gain that moves mid-block
     * mixes two levels into one utterance.
     */
    _emit() {
        let sum = 0;
        let peak = 0;
        for (let i = 0; i < BLOCK_SAMPLES; i++) {
            const v = this.out[i];
            sum += v * v;
            const a = v < 0 ? -v : v;
            if (a > peak) peak = a;
        }
        const rms = Math.sqrt(sum / BLOCK_SAMPLES);
        let gain = this.autoLevel ? this._updateGain(rms) : 1;
        // Peak limiter: a gain that targets RMS will still push isolated peaks
        // past full scale, and a clipped peak sounds like a click to the model.
        const scaled = peak * gain;
        if (scaled > PEAK_CEILING) gain *= PEAK_CEILING / scaled;
        this.lastGain = gain;

        const pcm = new Int16Array(BLOCK_SAMPLES);
        for (let i = 0; i < BLOCK_SAMPLES; i++) {
            pcm[i] = LiveCaptureProcessor.toInt16(this.out[i] * gain);
        }
        // One copy, transferred (not cloned) to the main thread.
        this.port.postMessage(
            { type: 'block', pcm, gain: this.lastGain },
            [pcm.buffer]);
        this.out = new Float32Array(BLOCK_SAMPLES);
        this.outFill = 0;
    }

    process(inputs) {
        if (this.closed) {
            return false;
        }
        const input = inputs[0];
        if (!input || !input.length) {
            return true;
        }
        const ch = input[0];
        const n = ch.length;
        if (n === 0) {
            return true;
        }
        // Resample only. No gain is computed here: this quantum is 128 input
        // samples (~2.7 ms at 48 kHz), a rendering artefact rather than an
        // audio unit, and a gain update that fast is what flattened the
        // envelope -- see the module docstring.
        // `pos` is in input-sample units relative to the concatenated stream,
        // so rebase it at the start of every block.
        let pos = this.pos;
        while (pos < n) {
            const i = pos | 0;
            const frac = pos - i;
            const a = i === 0 ? this.prev : ch[i - 1];
            const b = ch[i];
            const v = a + (b - a) * frac;
            this.out[this.outFill++] = v;
            pos += this.ratio;
            if (this.outFill === BLOCK_SAMPLES) {
                this._emit();
            }
        }
        this.pos = pos - n;
        this.prev = ch[n - 1];
        return true;
    }
}

registerProcessor('live-capture', LiveCaptureProcessor);
