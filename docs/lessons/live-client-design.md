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
| Speaker channel, runner-up too close (`min_match_margin` 0.05) | `UNKNOWN` (`live_match_ambiguous`) |
| Otherwise | matched name, `speaker_source: "known_voiceprint"` |

**The mic is deliberately not embedding-matched.** The default input device
carries the local user, who is by definition in their own voiceprints; matching
that audio against the stored profile returns a confident answer that is wrong
whenever they have more than one profile. The channel *is* the evidence.

Speaker-channel turns are 1–3 s, which is the low end for ECAPA-TDNN512. The
0.60 confidence bar is therefore higher than the file path's
`uncertainty.min_speaker_confidence` (0.35) — that is honest, not a bug, and
`live_attribution` is a separate config section precisely so it can be tuned
without touching file-path behaviour.

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

## Known limitation

The audio-capture path (`sounddevice`, WASAPI loopback, the WebSocket
`Transport`) has never been executed against real hardware — no Windows host
and no audio device were available during development. Framing, endpointing,
resampling arithmetic, and session bookkeeping are covered by tests; device
enumeration and loopback capture are not.
