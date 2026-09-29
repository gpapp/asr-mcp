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
