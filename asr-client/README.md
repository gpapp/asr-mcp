# asr-client — standalone transcription client

Drop-file Windows client for the ASR MCP server. Pure Python standard
library — nothing to install beyond Python 3.8+. The package is
self-contained: it carries its own `.env` config, `requirements.txt`,
and (optionally) a private `.venv`. No server code or server
dependencies are required.

## Setup

1. Unzip anywhere.
2. Run `transcribe.bat` once — it creates `.env` from `.env.example`
   (and a private `.venv` on the next run).
3. Edit `.env`:
   - `SERVER_URL` — full base URL, include the `/asr-mcp` prefix when
     behind nginx (e.g. `http://host:8087/asr-mcp`)
   - `TOKEN` — generate in the web UI ("Windows Client Token" card)
     or `POST /api/token` while logged in; one token per user
   - `LANGUAGE` — ISO 639-1 code (e.g. `hu`) or `auto` (default)
   - `WRITE_JSON` — `1` also writes `<name>.json` next to each `.txt`
     (raw segments plus per-result `speaker_confidence`, `speaker_source`,
     `uncertain`, `attribution_reason`)
4. Drag audio files onto `transcribe.bat`, or use the CLI below.

## Commands

| Command | Description |
|---------|-------------|
| Drag files onto `transcribe.bat` | Transcribe; writes `<name>.txt` next to each file |
| `transcribe.bat status` | Check server/token connectivity |
| `transcribe.bat voiceprints` | List speakers and voiceprints |
| `transcribe.bat voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip |
| `transcribe.bat voiceprint-refine <name>` | Rebuild a voiceprint from its snippets |
| `transcribe.bat <file...> [--language <code>] [--json]` | Transcribe (flags override `.env`) |

Supported inputs: wav, mp3, flac, ogg, m4a, webm, opus, mkv, mp4.

## Output format

Each `<name>.txt` contains:

```
Audio: <filename>
Date: <transcription run time>
Speakers: N
Duration: NNN.Ns

============================================================
SPEAKER VOICE PROFILES
============================================================
  Gergely Papp: pitch=123Hz (±82Hz)  energy=0.0089  speech=313s
============================================================

[00:00:12] Gergely Papp (77%): paragraph text
[00:04:05] Speaker 3: paragraph without confidence suffix
[00:05:40] UNKNOWN (boundary_crossing): text the server could not attribute

WARNING: 3 of 41 segment(s) have an UNKNOWN speaker (identity could not be
established; the text was kept). Re-run if you need clean labels.
```

- The profiles banner appears when the server reports voice profiles.
- Paragraph lines are `[HH:MM:SS] <speaker>: text` (absolute start
  time); the `(NN%)` decode confidence is included only when the ASR
  backend reports it (whisper does; cohere/qwen3 do not).
- `UNKNOWN (<reason>)` marks speech the server could not attribute to a
  confident speaker (short answer crossing a diarization boundary, a
  ghost/minority speaker, no diarization at all). **The text is kept** —
  only the identity is suppressed, so nothing spoken is lost. The
  reason codes come from the server's uncertainty policy and are also
  present in the `.json` sidecar.
- The CLI exits non-zero only for real failures (unreachable server,
  bad token, error event). Uncertain/partial speaker data is reported on
  stderr but keeps exit code 0, and a metadata-only `.txt` is always
  flagged with a warning.

## Notes

### Upload size and the proxy

The server accepts audio up to **200 MB**, and the reverse proxy in front of it
enforces the same limit. The proxy refuses an oversized body **before the
application ever sees it**, so a too-large upload produces an nginx HTML error
page rather than a JSON message from the server. The client recognises that
specifically and says so instead of printing markup.

To keep large files off the wire, the client **transcodes before uploading**:
any video container (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`, `.m4v`), and any
file over 24 MB, is converted to 16 kHz mono FLAC — exactly what the server
would extract anyway, so nothing is lost — and the result is saved next to the
original as `<name>.16k.flac` so you can see what was sent. This needs
`ffmpeg` on `PATH`, beside the script, or in `C:\Program Files\ffmpeg\bin`.

Audio that already fits the limit is uploaded untouched, even when it is large.
Use `--no-convert` to disable all conversion; a file still over the limit then
fails immediately with its actual size instead of at the proxy.

### One transcription at a time

The server runs **one job at a time** — one GPU, one decoder. A second upload
while the first is running is refused with `HTTP 409`. That is not an error the
user caused, so the client names the file that owns the server (and whether it
can be cancelled from the web UI), then waits on the server's activity stream
and retries the upload once. `--no-wait` turns that off and fails immediately
instead.

### More notes

- Client transcriptions use `?save=false` — nothing is stored in
  server transcript history. If the connection drops mid-job, the
  server finishes the run and saves the result to transcript history
  (download from the web UI). Voiceprint snippets uploaded by the
  client ARE kept (they feed refinement).
- `transcribe.bat` creates `.venv` next to the script and runs the
  client with it; `requirements.txt` is installed there when it
  declares dependencies. Without a working venv it falls back to
  `py`/`python` on PATH (the client is stdlib-only, so this always
  works).
- `.env`, `.venv/`, and `__pycache__/` are ignored by git and are
  never included in the distributed zip.

### New speakers

A speaker the server hears who you have not added becomes an **unnamed
profile**: snippets are collected, but it is excluded from every voiceprint
match until you name it — an unnamed profile has no identity to claim. The
client tells you when this happened, and the web UI (Voiceprints tab → Unnamed
speakers) is where you resolve it. If the profile turns out to be somebody you
already added, merge it rather than naming it again.

If the server learns a speaker it has no voiceprint for, both clients say so:

```
  Learned 2 new speaker(s), not yet named: Pending 2026-09-30 12:45 Speaker_5 talk.mp4, ...
  They are EXCLUDED from matching until named. Name them in the
  web UI: Voiceprints tab -> Unnamed speakers.
```

That message matters: the learned snippets are saved, but the person still reads
`UNKNOWN` in the transcript — and in every later run too — until you give them a
name in the web UI. A learned profile is never auto-named, because a cluster has
no identity to claim.

If nothing is reported, that may be deliberate: a cluster is only learned when the
server can verify it is **one** voice. When it cannot — too little speech, the
model unavailable, or the segments not separating cleanly — nothing is saved,
because a profile blending two people would then be used to decide whose words
are whose in every later transcription. Speak more, or add a named colleague so
there is something to match against.

## Live mode (`transcribe.bat live`)

Real-time transcription of a meeting. Requires `PyAudioWPatch`, `numpy`, `soxr`
and `websockets` (installed automatically from `requirements-live.txt`).

> **Not `sounddevice`.** Capturing the *other* speakers needs the WASAPI
> loopback patch, which was merged into PortAudio in 2022 but is not in the
> prebuilt PortAudio that `sounddevice`'s wheels bundle
> ([portaudio-binaries#6](https://github.com/spatialaudio/portaudio-binaries/issues/6)
> is still open). With `sounddevice` there is no way to capture the speakers at
> all, which is the whole basis of the two-channel design.

```
transcribe.bat live
transcribe.bat live --language hu --outdir C:\meetings
transcribe.bat live --devices
```

It captures the **default microphone** and the **default sound device**
(WASAPI loopback) as two channels, cuts turns locally, and streams them to
`WS /asr/ws/stream`. Microphone turns are labelled `You`
(`speaker_source: "input_device"`); loopback turns are matched against your
registered voiceprints, and stay `UNKNOWN` when the match is weak or ambiguous —
the text is always kept, only the name is withheld.

Each channel is levelled on the way in: a quiet microphone is lifted toward
-18 dBFS RMS (up to +24 dB) with a peak limiter, because a low level costs the
recogniser and the re-diarization pass far more than it costs the turn
detector. Room tone is left alone and a hot source is never attenuated. Pass
`--no-agc` to record the raw levels instead. The browser Live tab
(`/live` in the web UI) uses the same loop with the same constants.

Press Ctrl-C to stop. On shutdown the client writes:

| File | Contents |
|---|---|
| `<timestamp>.wav` | microphone channel, 16 kHz mono |
| `<timestamp>-speaker.wav` | loopback channel |
| `<timestamp>.txt` | final transcript, `[HH:MM:SS] <speaker> (NN%):` format |
| `<timestamp>.asr.json` | every ASR item, gaps, server stats, `asr_source` |

The final `.txt` re-runs diarization on the recording and re-attributes the
**cached** ASR items — it never re-transcribes, so the text is exactly what you
watched appear live. If re-diarization fails (or `--no-rediag` is passed), the
live attribution is used and the header line says so.

`Write JSON` is not needed for live mode; the `.asr.json` sidecar is always
written. Set `TRANSCRIBE_LIVE_LOCAL_LABEL` on the server to change the
microphone speaker label.

#### What gets sent to the server

The client cuts turns with an energy detector, which cannot tell a breath from
a syllable — a sniff, a keyboard click or a chair creak is a perfectly good
turn as far as it knows. Two filters keep that out of the transcript:

1. **Turn coalescing.** A closed turn is held for up to
   `streaming.merge_gap_sec` (1 s on the server) and merged with the next one on
   the same channel, so a 300–800 ms pause in the middle of a sentence no longer
   splits it into two. Real conversational pauses are longer and still submit
   immediately; the added latency is bounded by the merge window. The client
   prints how many turns were merged when the session ends.
2. **A server-side speech gate.** Silero VAD runs on the turn before the
   decoder. A turn with no speech is reported as skipped, never transcribed, and
   the summary ends with `Filtered: N turn(s) contained no speech`. This is the
   *only* effective non-speech filter on the live path — Whisper's own
   `no_speech_prob` was measured at 0.0000 for speech and for noise alike in
   this stack, so it cannot be used. See `asr_mcp/transcribers/whisper.py`.

If the mic level column stays flat, capture is broken. If both levels look right
but nothing arrives, raise `streaming.min_speech_prob` only after checking the
`Filtered:` count — that number is the direct measure of what the gate rejected.

#### Turn boundaries are the client's claim, the server's decision

The energy detector is a good *segmenter* but its edges are a fixed padding
away from the real speech: it starts up to `pre_roll_ms` (160 ms) early and ends
`post_roll_ms` (200 ms) late, on a 32 ms grid. That is small per turn but it is
what turns a speaker change into `UNKNOWN (boundary_crossing)` in the final
transcript, and it is measured against the diarization turns at shutdown. So the
server corrects it:

- **Edges are trimmed with Silero** (`streaming.edge_trim_enabled`), removing
  padded frames from the start and end of each turn up to
  `edge_max_trim_sec` / `edge_max_trim_ratio`. The turn still occupies its true
  interval on the channel — the timeline is never shortened — only the audio
  inside it changes. Each transcript line reports `edge_trimmed_sec` when it
  applied.
- **A merged turn carries its pause as silence**, so the span the server decodes
  is exactly as long as the audio it holds. The tail's words keep their real
  position instead of jumping forward.
- **A timeline guard checks `start_sample`** for monotonicity and plausibility. A
  dropped capture block or a late `getDisplayMedia` would otherwise silently
  shift every later boundary on that channel. It is reported, not silently
  fixed, and the shutdown summary lists `timeline_faults` and
  `drifted_channels`.

If a turn's text appears to start before it was spoken, or the summary reports a
drifted channel, that is this guard — not a transcription error.
