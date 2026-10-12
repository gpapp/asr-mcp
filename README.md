# asr-mcp

GPU-accelerated ASR MCP server with speaker diarization, voiceprint recognition, and streaming transcription. Built with FastAPI + ONNX Runtime CUDA.

## Features

- **ASR with Diarization**: Detect and separate multiple speakers in audio. Pluggable backends via `TRANSCRIBE_ASR_MODEL`: Cohere Transcribe ONNX (default, English), Qwen3-ASR (4-bit, transformers) or Whisper large-v3-turbo (faster-whisper, int8-quantized).
- **Language Selection**: `?language=hu` (ISO 639-1) or `auto` on the upload endpoint (default `auto`); pick it from the GUI dropdown or the client's `LANGUAGE=` in `.env` — per-backend lists served by `GET /api/asr/languages`.
- **Voiceprint Recognition**: Register, identify, and manage speaker profiles with ECAPA-TDNN embeddings.
- **Streaming Transcription**: Real-time WebSocket dual-channel transcription.
- **Auto Voiceprint Collection**: Snippets auto-collected from diarization for registered speakers.
- **Form-Based Auth**: Session-based login with htpasswd password files.
- **API Tokens**: One token per user, generated from the web UI; token-authenticated API access without a session.
- **Self-Service Registration**: Optional email sign-up with a confirmation link
  (`TRANSCRIBE_REGISTRATION_ENABLED` + SMTP) and Google sign-in
  (`TRANSCRIBE_GOOGLE_CLIENT_ID`/`_SECRET`). Both off by default; htpasswd keeps
  working either way.
- **Per-User Isolation**: Speakers, voiceprints, snippets, transcripts and jobs are
  scoped to the signed-in identity. One API key has one voiceprint library on both
  the HTTP and WebSocket transports, and you can only see and cancel your own jobs.
- **Admin Accounts**: One deployment-wide administrator (`TRANSCRIBE_ADMIN_USER`)
  can list accounts, suspend them, delete them (optionally with their data) and
  stop the server. Empty means nobody.
- **Windows Client**: Drop-file transcription client (`asr-client/`) with live progress and `<name>.txt` output.
- **Reverse Proxy Support**: Configurable URL prefix (`TRANSCRIBE_PREFIX`) for nginx/Caddy.
- **Multi-Format Input**: Accepts mp3, mp4, mkv, flac, ogg, m4a — auto-converts via ffmpeg.
- **Compressed Snippets**: Voiceprint snippets stored as FLAC for efficient storage.
  Re-processing a recording REPLACES that recording's snippets for the speaker
  (matched by the filename hash, so moved spans and a new extension still match),
  keeping every clip from every other recording — a profile is never fitted to two
  copies of one take. `?replace_source=false` keeps the older take.
- **Voiceprint Integrity**: A startup audit checks the snippet store and repairs the
  findings that have one correct answer — dead snippet rows are deleted and the
  affected profiles are rebuilt from their surviving audio. Merging similar profiles
  is left to you (`GET /api/voiceprint/integrity`, `POST .../integrity/cleanup`).
- **Consolidation & Review**: A per-speaker (or whole-library) pass re-fits each
  profile from all its snippets, finds the snippets whose voice does not belong
  (intruders, crossed channels, mis-attributed spans), and marks them for review with
  the distance that convicted them. Flagged snippets are already excluded from the
  profile; nothing is deleted until you remove them yourself.
- **Web Dashboard**: Single-page app at `/gui` with Transcribe, Voiceprints, History and Settings tabs.

## Documentation

| Document | Covers |
|---|---|
| **[INSTALLATION.md](INSTALLATION.md)** | Install, reverse proxy, every server setting, and how to configure each authentication method |
| **[docs/api.md](docs/api.md)** | Every route and its authentication |
| **[docs/processing.md](docs/processing.md)** | How a boundary is decided and how audio becomes a transcript |
| **[docs/voiceprints.md](docs/voiceprints.md)** | How a voice profile is created, kept honest, and reviewed |
| **[docs/uncertain-speakers.md](docs/uncertain-speakers.md)** | What the `Speaker N?` labels mean and when a speaker is left unnamed |
| **[docs/ui.md](docs/ui.md)** | The five tabs, the design system, and the traps |
| [AGENTS.md](AGENTS.md) | Project layout, architecture, and the behavioural rules that must not be violated |
| [docs/lessons/](docs/lessons/) | The evidence behind each rule in AGENTS.md |
| [docs/boundary-measurements.md](docs/boundary-measurements.md) | The measured A/B behind the boundary engine, and what is still uncalibrated |

## Installation

See **[INSTALLATION.md](INSTALLATION.md)** for the full setup guide: install,
reverse proxy, every server setting and what it does, and step-by-step
configuration of each authentication method (htpasswd, email registration,
Google sign-in, API keys, admin).

```bash
cp .env.example .env
# Edit .env: set TRANSCRIBE_SESSION_SECRET (required)
docker compose up -d --build     # serves 8087 on the host
```

## API

Full reference: **[docs/api.md](docs/api.md)** — every route with its
authentication, plus the conventions that hold across all of them (per-user
scoping, `?purge=`, SSE stages, 404-for-another-user's-job).

The short version:

| | |
|---|---|
| **Pages** | `/gui` `/live` `/voices` `/transcriptions` `/settings` `/login` |
| **Transcribe** | `POST /api/asr/diarize[/upload]`, `POST /api/asr/transcribe[/upload]` |
| **Live** | `WS /api/asr/ws/stream`, `GET /api/asr/active/stream`, `POST /api/asr/active/cancel` |
| **Voiceprints** | `/api/voiceprint/*` — CRUD, merge, rename, rescan, refine, integrity, consolidate, flagged |
| **Auth** | `/api/auth/*` — login, logout, register, verify, Google start/callback, `/api/user` |
| **Admin** | `/api/admin/users` + disable/enable/delete, `POST /shutdown` |
| **Clients** | `GET/POST /api/token`, `/api/mcp/*`, `GET /api/asr/languages` |


## Architecture

```
Browser → nginx (/asr-mcp/) → FastAPI (port 8087) → ONNX Runtime CUDA
                                  ├── Cohere Transcribe (encoder CUDA, decoder CPU)
                                  ├── ECAPA-TDNN512 (embedding, CUDA)
                                  ├── Silero VAD (CPU)
                                  └── SQLite (voiceprints, sessions, transcripts, snippets)
```

Models are **lazy-loaded on first use** (nothing loads at server start) and idle models
unload after `TRANSCRIBE_MODEL_TTL_MINUTES` (default 5; skipped while a job runs).
CUDA arenas are capped (encoder ≈2.5 GB, embedding ≤768 MiB on a 4 GB card), shrink
after every run (`memory.enable_memory_arena_shrinkage=gpu:0`), and OOM recovery
reloads a fresh arena before falling back to CPU.

### Authentication Flow

> Setup instructions — htpasswd, email registration, Google, API keys, admin —
> are in **[INSTALLATION.md](INSTALLATION.md#authentication)**. This section is
> how identity resolution works once configured.

Every request resolves an identity in a fixed order (`get_current_user`):

1. **Session cookie** — written by `/api/auth/login`, which checks the credential
   store first (as typed *and* lowercased) and the htpasswd file second, so the
   form and `Authorization: Basic` cannot disagree about who can log in.
2. **DB token** in the `X-API-Key` header (sha256-hashed in `api_tokens`).
3. **Static key** from `TRANSCRIBE_API_KEYS` → the synthetic user
   `user_<sha256(key)[:16]>`. **Never `default`** — that is what keeps a shared
   key's voiceprints out of the default user's library. The WebSocket endpoint
   uses the same resolver, so one key has one library on both transports.
4. **401.** An unresolvable key refuses; it never falls back to a shared identity.

With `TRANSCRIBE_API_KEYS` empty, unauthenticated callers would otherwise all
become `default` and share one library — so once
`TRANSCRIBE_REGISTRATION_ENABLED` is on, open mode is refused (401) instead.

Everything downstream — speakers, voiceprints, snippets, transcripts and the
snippet audio under `voices/<user_id>/` — is scoped to that identity, and
`/api/asr/activity/stream` only shows *your* running job. Another user's job is
a 404, and the Cancel button is only offered for a job you own.

### Registration and admin (optional, off by default)

Setup is in **[INSTALLATION.md](INSTALLATION.md#2-email-registration)** and
**[INSTALLATION.md](INSTALLATION.md#5-admin)**. In brief:

- **Email sign-up** needs `TRANSCRIBE_REGISTRATION_ENABLED=true` plus
  `TRANSCRIBE_SMTP_HOST` + `TRANSCRIBE_SMTP_FROM`. It creates an unverified
  account and mails a single-use link; it never signs anyone in, and a mail that
  cannot be sent deletes the account rather than leaving a half-made one.
- **Google sign-in** needs a client id and secret. It is gated on *configuration*,
  not the registration flag, so turning registration off does not sign out
  existing users. A first sign-up's account is named after the normalised email
  address Google verified; identity resolves by subject, never by address.
- **Admin** is `TRANSCRIBE_ADMIN_USER`, one account, empty meaning nobody. A
  delete keeps the recordings and reports counts unless `?purge=true` is passed.
- There is **no password-reset flow** — an admin re-issues a password or deletes
  the account.

> If a confirmation link 404s, the token is almost certainly fine — check
> `TRANSCRIBE_EXTERNAL_URL`. See
> [the symptom table](INSTALLATION.md#verifying-a-deployment).

### The pipeline in one paragraph

Audio is diarized into speaker segments, those segments are turned into a
**timeline of turns covering every second of the recording exactly once**, and
every boundary is cut by **one** engine
(`speaker/boundary.py::cut_between`) that tries VAD-section ownership decided by
voiceprint embedding, then spectral change, then an energy dip, then the
quietest frame, then the gap midpoint. The whole file is decoded in **one** ASR
call and the returned items are attributed onto the turns, so audio is never cut
at a speaker boundary. Voiceprint learning runs *after* that and is passed the
**turns**, so a learned snippet carries the same boundaries the transcript was
attributed on.

Measured against raw VAD sections as ground truth, acoustic refinement wins at
every percentile on all five test files — 0.448 s → 0.000 s median on a
57-minute two-speaker podcast. Full pipeline, the measured A/B and its two
recorded negatives: **[docs/processing.md](docs/processing.md)**.

### Voiceprints in one paragraph

An unknown speaker is learned automatically but as a **pending profile**, never
as a usable voiceprint — an unnamed profile has no identity, so it is excluded
from every match and that person's speech reads `UNKNOWN` (with the text kept)
until you name them in **Voiceprints → Unnamed**. Naming is the only action that
makes them recognisable. Re-processing a recording *replaces* that recording's
contribution rather than stacking a second copy onto a profile, and a profile
whose audio holds more than one person is caught by an integrity audit at
startup and an on-demand outlier review — both of which **flag and report**, and
neither of which deletes anything without you asking.

Full policy, including the cohesion split, the re-join, one-file-one-contribution
and the distance-vs-confidence distinction:
**[docs/voiceprints.md](docs/voiceprints.md)**.

### When a speaker is left unnamed

When an identity cannot be established the server **suppresses the name and
keeps the text** rather than force-fitting the nearest speaker:

```json
{ "start": 9.2, "end": 10.6, "text": "Yes.",
  "speaker": null, "speaker_confidence": 0.0, "speaker_source": "unknown",
  "uncertain": true, "attribution_reason": "boundary_crossing" }
```

Every result carries `speaker_confidence`, `speaker_source`
(`known_voiceprint` | `diarization_cluster` | `unknown`) and, when the identity
was withheld, an `attribution_reason`. The UI and the Windows client render
`null` as `UNKNOWN (<reason>)`. **Text is never dropped, only the name.** Every
`attribution_reason`, both naming gates, and the known limitation on short
recordings: **[docs/uncertain-speakers.md](docs/uncertain-speakers.md)**.


## Further reading

| If you want to know | Read |
|---|---|
| How a boundary is decided, and how a transcript is built | **[docs/processing.md](docs/processing.md)** |
| How a voice profile is created, kept honest, and reviewed | **[docs/voiceprints.md](docs/voiceprints.md)** |
| What `UNKNOWN` labels mean and when a speaker is left unnamed | **[docs/uncertain-speakers.md](docs/uncertain-speakers.md)** |
| How the web UI is put together, and its design system | **[docs/ui.md](docs/ui.md)** |
| Every route and its authentication | **[docs/api.md](docs/api.md)** |
| How the code is laid out, and the rules that must not be broken | **[AGENTS.md](AGENTS.md)** |
| The evidence behind each rule | **[docs/lessons/](docs/lessons/)** |
