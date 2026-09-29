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
| Speaker channel, best match below `min_match_confidence` (0.15) | `UNKNOWN` (`live_match_weak`) |
| Speaker channel, runner-up too close (`min_match_margin` 0.10) | `UNKNOWN` (`live_match_ambiguous`) |
| Otherwise | matched name, `speaker_source: "known_voiceprint"` |

**The mic is deliberately not embedding-matched.** The default input device
carries the local user, who is by definition in their own voiceprints; matching
that audio against the stored profile returns a confident answer that is wrong
whenever they have more than one profile. The channel *is* the evidence.

### The gates are calibrated, and the first calibration was wrong

These gates originally shipped at `min_match_confidence: 0.60` /
`min_match_margin: 0.05`, on the reasoning that "live turns are 1–3 s, the low
end for ECAPA, so the bar must be higher than the file path's 0.35". The bar
was raised in the wrong direction, and by more than the width of the genuine
distribution.

Measured on real 2–6 s excerpts of the *correct* speakers, fed through the
handler's own embedding hooks against the 35-voiceprint menu:

| population | combined distance | confidence | margin |
|---|---|---|---|
| **genuine** (13 excerpts, 6 speakers) | 0.174 – 0.402 | 0.196 – 0.65 | 0.141 – 0.417 |
| **non-match** (corrupted audio, see below) | 0.780 – 0.870 | 0.00 | 0.002 – 0.062 |

A 0.60 confidence bar sits *above the entire genuine population*: it rejected
11 of 13 real matches, which is exactly the "the correct speaker is found but
not attributed" report. Meanwhile the margin gate at 0.05 was far too loose to
compensate — a non-match scored a margin of 0.002.

The two populations leave an **empty band**: no non-match exceeded confidence
0.00, and no genuine match fell below 0.196. So the confidence floor now sits
in that gap (0.15) as a sanity check only, and the **margin does the
discriminating** (0.10 — below the weakest genuine margin of 0.141, above the
worst non-match of 0.062). The shipped values are pinned to these measurements
by `tests/test_live_attribution_gates.py`, which fails if either gate is moved
back to a value that would misclassify a measured population.

Caveat worth keeping: these excerpts come from the same source audio the
voiceprints were built from, so they are a best case. Real loopback audio
arrives through a different signal path (room, headset EQ, cross-talk) and will
score worse. The non-match side is also only sampled at the extremes — nothing
was measured in the 0.4–0.78 combined band, which is where a genuinely
different-but-similar colleague would land. Treat the margin gate as the real
safety mechanism, not the confidence floor.

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

The audio-capture path (device enumeration, loopback capture, the WebSocket
`Transport`) has never been executed against real hardware — no Windows host
and no audio device were available during development. Framing, endpointing,
resampling arithmetic, device selection and session bookkeeping are covered by
tests (including the real device table from a Windows machine, which is what
exposed the loopback bug); the actual audio callback and loopback capture are
not.
