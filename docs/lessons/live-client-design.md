# Live Client Design — Decision Record

Status: **accepted**, implemented in `asr-client/live_client.py`.
Covers the question "can re-diarization give speaker attribution without
re-transcribing?" and, more importantly, **where the final transcript text comes
from**.

## The question

`/transcribe` runs diarization, then ASR, then maps ASR output onto diarized
turns. If the speaker labels come out wrong, is a full re-run required?

## Key finding: attribution is a pure function

`_transcribe_file()` calls `transcribe_audio_sync()` **exactly once on the whole
file**. Every backend (`cohere._transcribe_windowed`, `whisper._run`, `qwen3`)
returns timestamped items on the file timeline; the diarized turns are only
consulted *afterwards* by `speaker/attribution.py::attribute_items`. So:

```
results = attribute_items(asr_items, turns)      # pure: f(items, turns) -> runs
```

The ASR items are independent of the turns. Re-diarizing therefore needs the
audio and the items — **not** the ASR backend.

Measured on the 52-minute zo230 podcast (whisper large-v3-turbo, GTX 1650):
478 s total ≈ **96 s diarization + ~380 s ASR**. Re-attribution is ~5× cheaper
than the original run, and ~0 s of that is decoding.

This is also why client-side VAD/turn detection is safe: the turns are an
*input* to the pure function, not a constraint on the ASR.

## Decision: transcript always comes from the live ASR cache

The client records the audio, sends framed turns to the live WebSocket, and
keeps every ASR item it receives. On shutdown it uploads the recording plus the
cached items to `POST /api/asr/attribution/upload`, which re-runs diarization
and re-attributes — **but never re-transcribes**. The cached text is the text
in the final `.txt`, regardless of who the diarization names.

`live_client.py::final_result()` prefers the re-diarized attribution and falls
back to the live one; `write_transcript()` records which was used in a header
line, and the `.asr.json` sidecar records `asr_source: "live_stream"`.

### Why the live cache and not a fresh file transcription

A file transcription of the same recording is *not* guaranteed to be better —
and often is worse for this use case:

- The whole-file decode loses the turn framing that keeps short answers
  ("Yes.", "Mm-hmm.") from being absorbed into a neighbour's window.
- Whisper's `vad_filter` re-segments audio that the client already framed, so a
  fresh file run can disagree with what was actually spoken live.
- It costs a second full ASR pass — on the 52-minute benchmark, ~380 s the user
  would sit and wait for *after* they already have a transcript.

The one thing a fresh run buys is style normalisation (punctuation, casing
drift). That is a real but minor gain, and it is exactly the trade the
`--retranscribe` flag below would buy back on demand.

## Rejected alternatives

### 1. Re-transcribe only the dropped gaps

The server sends `{"type": "dropped", "start", "end", "channel",
"reason": "queue_overflow"}` when its bounded queue overflows, so the client
knows precisely which intervals have no text. Re-decoding just those intervals
and splicing them in would give complete coverage with no wasted work.

**Rejected:** it needs partial-transcription plus splice-and-align logic, and
the loss only happens under sustained decoder lag (a GPU too small for the
model, or a backend on CPU). The complexity is not justified by a failure mode
that should be rare, and the `covered_sec` stat already tells the user when it
happened. The `.asr.json` gaps list is the honest record of the hole.

### 2. Cache by default, `--retranscribe` flag to rebuild

Cache the ASR, but let the user ask for a full file transcription and use that
for the final `.txt` instead.

**Rejected as the default** — it makes the common path surprising: the user
watches a live transcript appear, presses Ctrl-C, and the file that lands on
disk has different text from what they just read. The flag is cheap to add
later and the sidecar already records the ASR source, so the door stays open.

**Revisit this decision if:** a user reports that the live transcript reads
worse than a file transcription of the same recording (typically punctuation or
casing drift, not wording). The fix is to add `--retranscribe` to
`live_client.py`; nothing else has to change.

## Live attribution: channel first, voiceprint second

`streaming/attribution.py::attribute_live_turn` decides the speaker per turn:

| Situation | Result |
|---|---|
| Turn shorter than `min_turn_sec` / longer than `max_turn_sec` | `UNKNOWN` (reason: `live_turn_too_short` / `live_turn_too_long`) |
| **Mic channel** | `LOCAL_SPEAKER_LABEL` (default "You"), `speaker_source: "input_device"` |
| Speaker channel, no voiceprints loaded | `UNKNOWN` (`no_voiceprints`) |
| Speaker channel, best match below `min_match_confidence` (0.60) | `UNKNOWN` (`live_match_weak`) |
| Speaker channel, runner-up too close (`min_match_margin` 0.10) | `UNKNOWN` (`live_match_ambiguous`) |
| Otherwise | matched name, `speaker_source: "known_voiceprint"` |

**The mic is deliberately not embedding-matched.** The default input device
carries the local user, who is by definition in their own voiceprints; matching
that audio against the stored profile returns a confident answer that is wrong
whenever they have more than one profile. The channel *is* the evidence.

### The gates are calibrated, and the first calibration was wrong twice

These gates first shipped at `min_match_confidence: 0.60` /
`min_match_margin: 0.05`, on the reasoning that "live turns are 1–3 s, the low
end for ECAPA, so the bar must be higher than the file path's 0.35". The bar
was raised in the wrong direction, and by more than the width of the genuine
distribution.

I then "corrected" that to 0.15, using 2–6 s excerpts of the *correct* speakers
against the 35-voiceprint menu:

| population | combined distance | confidence | margin |
|---|---|---|---|
| **genuine** (13 excerpts, 6 speakers) | 0.174 – 0.402 | 0.196 – 0.65 | 0.141 – 0.417 |
| **non-match** (corrupted audio) | 0.780 – 0.870 | 0.00 | 0.002 – 0.062 |

The second table looked convincing and was **invalid**: the negative population
was corrupted audio, which is not a false positive. It only ever measured the
extreme garbage band and said nothing about the band a *different but similar
colleague* occupies.

An end-to-end run through the real `handle_ws_stream` — real embedding hooks,
real gates, real `pack_turn` frames, an unregistered speaker — then produced the
thing the table could not have predicted:

```
Bob 5.0s on speaker -> 'Alice'  src=known_voiceprint  conf=0.539
```

A stranger was confidently named. conf 0.539 is *inside* the genuine range
0.196–0.65, so no single confidence threshold separates the two populations.

The gate is back at **0.60**, which rejects the measured false positive. This is
a deliberate asymmetric trade: 6 s and 9 s turns of a genuinely registered
speaker scored 0.22 and 0.19 and are now declined. A wrong name is the
expensive error, the text is never withheld, and the authoritative attribution
remains the client's shutdown re-diarization, which has minutes of context
instead of one turn. `tests/test_live_attribution_gates.py` pins this: it
asserts the measured stranger is rejected, that the *strongest* genuine match
still gets a name (a gate that rejects everything is not a fix), and that the
weak genuine matches are withheld on purpose so nobody "repairs" it by lowering
the bar again.

Two measurement gaps are still open and are stated rather than papered over:

- **Margin is unvalidated.** The single-profile menu used in the end-to-end run
  makes margin meaningless (there is no runner-up). Genuine margin 0.141–0.417
  is measured; a false-positive margin is **not**. The 0.10 margin bar is
  therefore unproven as a discriminator and should not be trusted yet.
- **Contaminated profiles collapse recall.** Registering a voiceprint from a
  two-person conversation blends both speakers; against that profile, genuine
  audio of the registered person scored combined 0.55–0.94 (conf 0.00) — the
  profile stops matching anyone, including its own owner. This is a data
  quality problem upstream of the gates, but it means a low conf can mean "bad
  profile" rather than "different person".

### The bug that produced the non-match population

Every live turn was scoring `dist ≈ 0.78–0.87` — random-pair territory — and
the client showed `UNKNOWN` for all of them, which read as "randomly assigns
speakers". The cause was a dtype mismatch, not the model:

`Turn.audio` is a **float32 numpy array** everywhere on the server
(`turn_detector.Turn._close_turn` and `handler._turn_from_frame` both do
`np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0`). The client's
`Turn` keeps raw int16 `pcm` bytes, because that is what it packs into the wire
frame. Both representations are legitimate. But the embedding hooks in
`handler.py` assumed the second:

```python
samples = np.frombuffer(turn.audio, dtype=np.int16)   # float32 array!
```

`np.frombuffer` on a float32 array reinterprets the **IEEE-754 bit patterns** as
int16 and, because a float is 4 bytes and an int16 is 2, **doubles the length**.
A 4800-sample turn arrived as 9600 samples of noise — a ramp from 0.0 to 0.5
came through as `[0, 0, 0, 0.475, 0.5, 0.479]`, RMS 0.554 instead of 0.212.
ECAPA embedded that noise, so every distance was a random-pair distance.

The fix is `turn_detector.samples_as_float32(audio)`, which accepts bytes,
int16 arrays, float arrays and `None`. It exists because two representations
are in play and any code touching turn audio has to normalise. It is covered by
`test_samples_as_float32_accepts_every_turn_payload_shape`, and the hook test
was rewritten to assert sample **values** survive — the old test used a fake
turn whose `audio` was int16 *bytes*, so it encoded the bug and passed.

Gates run in the order above, so the cheap structural rejections
(too short/long) never pay for an embedding, and the mic never pays for a
match at all.

## Turn detection runs on the client

The client frames turns with the same algorithm as the server
(`streaming/turn_detector.py`) and sends `pack_turn(channel, start_sample, pcm)`,
so a turn's boundaries are identical whether it was cut live or found by
re-diarization later. `tests/test_live_client.py` asserts **exact** sample-and-
reason equality between the two implementations on the same audio.

Both frame shapes are accepted by the server: turn frames (magic `LVT1`) and
legacy raw PCM (8-byte header). The magic discriminates them, so no
configuration flag is needed and old clients keep working.

## 36. A second channel requires LVT1; browsers have no loopback

The legacy raw-PCM path runs server-side endpointing, and it only ever feeds
**channel 0** to the detector — `handler.py:437` `continue`s on any other
channel, so the speaker channel is silently discarded before it reaches the ASR.
A client that wants two channels must send LVT1 turn frames and do its own
endpointing, exactly like the Windows client.

The browser Live tab (`asr_mcp/static/live.js`) does exactly that, which makes
it a *third* implementation of the protocol constants, the turn detector and
the transcript renderer. All three are drift-guarded by
`tests/test_live_js_protocol.py`: constants vs. `protocol.py`, every
`setUintNN(offset, …)` in `packTurn` vs. the struct layout, packed bytes vs.
`unpack_turn`, `DETECTOR_DEFAULTS` vs. `turn_detector.config()` (including the
key set, minus the six server-only queue/VAD knobs), node-vs-server detector
boundary equality on a synthetic signal, and `buildTranscript` byte-equality
with `transcribe_client.build_transcript`.

Capture has no browser equivalent for WASAPI loopback: `getUserMedia` gives the
mic only, and `getDisplayMedia({audio, video: true})` gives tab/system audio.
The `video: true` is required — Chrome returns no tab audio unless video is also
requested — so this channel is Chrome/Edge only, and a rejected permission
degrades to mic-only with a visible note rather than failing the session.

The server's queue is 32 turns and overflow evicts the *oldest*, so a slow
decode shows up as `dropped` gaps rather than latency. Both clients cache every
`transcript` item and close the session by re-attributing the recording; the
`stats` frame (not socket close) is the drain signal, and a `stats` frame with
`covered_sec` well below the recording length is how you notice the drops.

### The per-channel turn split is the diagnostic

A real session logged in `logs/app.log` — 13 turn frames, 12 transcripts,
voiceprint matches, `dropped_turns: 0` — ended with
`Re-attributed 12 item(s) onto 1 turn(s) ... 7 result(s), 0 speaker(s),
7 uncertain` and a transcript that was entirely UNKNOWN, with nothing in the UI
to explain it. The one number that explains it is in the same `stats` frame:

```
'mic_packets': 0, 'speaker_packets': 12, 'turn_frames': 13
```

The microphone produced no turns at all, so the mic recording uploaded for
re-attribution contained no speech, and every speaker-channel item collapsed
onto a single speakerless turn, which the uncertainty policy then suppressed
outright. A quiet microphone is not a cosmetic problem in this pipeline: it is
the difference between named speakers and no speakers, and the failure is
silent. That is what motivated input levelling (§37) and why `mic 0, speaker
12` is now printed in the end-of-session note, the `.asr.json` sidecar and the
`.txt` header.

Corollary: never discard a session because the socket failed *after* text
arrived. The client caches every transcript item, so the recording and the text
are still worth re-attributing and downloading.

## 37. Level the input once, in the capture chain, in both clients

A quiet microphone does far more damage than it looks. The turn detector is
unaffected — its thresholds are ratios against a tracked noise floor, so
scaling the signal scales the floor with it — but the recogniser and the
offline pass both care about absolute level. And the offline pass is the one
that is easy to forget: on shutdown the recording is re-uploaded to
`/api/asr/attribution/upload`, where VAD and the ECAPA-TDNN embeddings run over
it, so levelling only the frames on the wire leaves re-attribution working on
quiet audio. **Level once, in the capture chain, before the int16 conversion**,
so the socket, the detector and the WAV on disk all carry the same signal.

Two clients implement the capture chain, so both carry the loop: the browser
worklet (`static/live-worklet.js`) and `asr-client/live_client.py::AutoGain`
(inside `Resampler.process`, one per channel). The constants are pinned
against each other by `test_the_two_clients_level_identically`, and the
behaviour by tests that drive the real worklet in node and the real
`AutoGain` in numpy and compare them.

Three things the loop has to get right:

- **`desired = TARGET_RMS / rms`,** the gain that puts *this block* at the
  target. The first version computed `TARGET_RMS / (rms * gain)` — a feedback
  expression whose fixed point is `sqrt(TARGET / rms)`, i.e. a geometric mean
  of where the signal started and where it was going. It rose smoothly toward
  the target, which read exactly like convergence, and left a -40 dBFS mic at
  -32 dBFS. Compare the achieved level against the target, not against the
  starting level.
- **Smoothing in the log domain** (`gain *= (desired/gain)**k`), so the time
  constant is constant in dB. A linear ramp pumps audibly on every syllable.
  Fast attack / slow release biases an amplitude-modulated signal upward, which
  is what the peak limiter is for — a real speech envelope will sit above the
  nominal RMS.
- **Boost-only, with a gate.** The gain decays back to 1.0 and is never pushed
  below it, so a hot source is never made worse. `MIN_ENV` (room tone) is set
  well below quiet speech so a genuinely quiet talker is still lifted, and
  `MAX_GAIN` is what bounds it, not the gate.

## Auth

`WS /asr/ws/stream` was previously **unauthenticated**. It now resolves the user
in the same order as `get_current_user`: session → `X-API-Key` header or
`?token=` → `is_valid_api_key` → `DEFAULT_USER`, and closes with code 1008
*before* `accept()` so the client gets a 403 rather than a socket that dies
immediately.

## Capture library: PyAudioWPatch, not sounddevice

The two-channel design needs "what the speakers are playing". That is WASAPI
**loopback**, which PortAudio gained in 2022 (PR #672) as *virtual input
devices* named `<render device> [Loopback]`. The stock PortAudio binaries that
`sounddevice` ships still predate that patch, so those devices simply do not
appear — and `sounddevice.WasapiSettings` has no loopback option at all
(only `exclusive`).

`sounddevice`'s absence of loopback caused a silent, dangerous bug: the first
implementation looked for "the first WASAPI device with an input channel",
which on an ordinary machine is a **microphone**. It opened a second mic,
labelled it "everyone else", and raised no error. The fix is
`PyAudioWPatch` (a PortAudio fork carrying the patch) plus a `select_devices`
function that returns `loopback=None` unless a device is genuinely marked
`isLoopbackDevice` / named `[Loopback]`.

Loopback devices are usually **stereo**, so capture also has to downmix
interleaved int16 to mono (`to_mono`) instead of assuming mono bytes.

## Known limitation

The browser Live tab has been run end to end in Chrome, and `logs/app.log`
records a successful session (tab audio through the worklet, the socket, the
decoder, voiceprint matching, the flush, the re-attribution and the History
save). Two things about that run are still unproven. The microphone channel
contributed **no turns** — see the turn-split note above — so the levelled mic
path has never carried a turn end to end, and the level meters have not been
checked against a known signal. Separately, the *diagnosis* of the capture
failure that preceded it (a suspended `AudioContext`, built after two
permission prompts) was never confirmed; `numberOfOutputs: 0` on the worklet
node, which may leave the graph unpulled in some builds, produces identical
symptoms. The fix is defensive on both counts and the watchdog note names the
silent channel, so a recurrence is diagnosable rather than silent.

The audio-capture path (device enumeration, loopback capture, the WebSocket
`Transport`) has never been executed against real hardware — no Windows host
and no audio device were available during development. Framing, endpointing,
resampling arithmetic, device selection and session bookkeeping are covered by
tests (including the real device table from a Windows machine, which is what
exposed the loopback bug); the actual audio callback and loopback capture are
not.

## 42. A live boundary is the client's CLAIM and the server's DECISION

The live detector places a boundary by an **energy crossing plus a fixed
pre/post-roll**: the start is the first frame at/above the start threshold minus
up to `pre_roll_ms` (160 ms) of pre-roll, the end is 5 frames (160 ms) *after*
the last frame above the end threshold, both on a 32 ms grid. So a live boundary
runs up to ~192 ms off a true speech onset, and up to ~0.5 s for quiet speech
where the adaptive floor is high. Nothing ever refined it. That matters because
the item's `start`/`end` are what the shutdown re-attribution matches against
the diarization turns — a 0.2 s disagreement past
`uncertainty.max_boundary_cross_sec` is what turns a correct transcript into
`UNKNOWN (boundary_crossing)`.

### Three defects, all found by reading rather than by a test

1. **A merged turn declared a span it did not carry.** The coalescer built
   `head + tail` — **no gap padding** — while setting
   `start = pending.start_sample, end = turn.end_sample`. The decoder read a
   *discontinuous* turn, and every attribution span was up to `merge_gap_sec`
   (1.0 s) longer than the audio it described. Two fixes were possible: pad with
   the gap samples, or declare `end` from the samples actually sent. The second
   was implemented first and then **overridden**, because it silently moves the
   tail's text earlier than it occurred — the exact error the plan exists to
   remove. The shipped rule is to pad with silence, which is what a pause sounds
   like to a decoder anyway:
   ```
   pad = turn.start_sample - pending.start_sample - payload_samples(head)
   audio = head + silence_like(head, pad) + tail
   ```
   `pad` is computed against the *raw* next turn, so it also absorbs the
   trailing-hangover trim, not just the inter-turn gap.
2. **The server trusted a client-declared `start_sample`.** `unpack_turn`
   validated length, magic, version and the sample count — nothing else. One
   dropped worklet block, or a `getDisplayMedia` that started later than the mic,
   shifted every later boundary on that channel with no signal at all.
   `protocol.TimelineGuard` now keeps a per-channel high-water mark, flags a
   regression or an implausible lead, clamps it, and reports
   `timeline_faults` / `drifted_channels` in `stats` and in the shutdown frame.
   `unpack_turn` also rejects a turn declaring zero samples.
3. **The session tail was written to the WAV twice.** `finish()` pushed the
   flushed turn's PCM into the recording *after* every block had already been
   written, so the file was up to one turn too long with the last turn
   duplicated — over-reporting `audio_duration_sec`/`micSeconds()` and handing
   the learner a duplicate region that can spawn a spurious pending profile.
   `flush_at_shutdown()` now gives the recorder only the resampler's drain.

### Two spans, two meanings — do not conflate them

`Turn.end_sample` is the end of the **audio carried** (what the decoder reads).
`Turn.audio_end_sample` is where the audio really **ended on the recording**, and
it is what the *next* merge decision and the handler's `covered_sec` use. Setting
`audio_end_sample` to the trimmed end was the first implementation and it was
wrong: measuring the next gap against trimmed audio inflates it by the hangover
(2048 samples) and silently stops merges that used to happen. The raw-PCM test
caught it.

### The server moves the boundary back (step 21)

`streaming/speech_gate.py::trim_turn_edges` now trims each turn's leading and
trailing frames by **Silero frame probability**. The VAD model is already
resident for the speech gate, so this is one more scoring pass over frames that
were already computed: no extra model, no measurable latency, and it removes
the pre/post-roll padding from *both* the decoded audio and the declared span.
The **timeline is never shortened** — `start_sample` advances by exactly the
trimmed samples and `audio_end_sample` is carried through, so the turn still
occupies its true interval; only its contents and its declared span change.
`edge_max_trim_sec` 0.40 and `edge_max_trim_ratio` 0.30 bound how much it may
move, and every unavailable path fails open.

Because shutdown re-attribution re-runs the whole offline pipeline, the same
`boundary.py` engine then governs the final `.txt`. **One fix, both paths** —
which is the argument for keeping the client's detector as the claim and the
server as the decision.

### Two smaller things found on the way

- `handler.py` decided **once at connect** whether to install the speech probe
  (`if state.vad_session is not None`). With lazy model loading, a session that
  connects before the VAD is resident — the common case — ran its whole duration
  with no gate and, once the edge trim existed, no trim either. The probe is now
  always installed and reports `speech_probe_unavailable` per turn, which fails
  open with a reason that names itself.
- `live_attribution.min_match_confidence` = 0.60 still rests on a **single**
  false-positive measurement. The margin gate is still **unvalidated** (a
  one-profile menu produces no runner-up, so no false-positive margin was ever
  measured). Do not read either number as calibrated.
