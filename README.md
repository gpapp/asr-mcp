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
- **Windows Client**: Drop-file transcription client (`asr-client/`) with live progress and `<name>.txt` output.
- **Reverse Proxy Support**: Configurable URL prefix (`TRANSCRIBE_PREFIX`) for nginx/Caddy.
- **Multi-Format Input**: Accepts mp3, mp4, mkv, flac, ogg, m4a — auto-converts via ffmpeg.
- **Compressed Snippets**: Voiceprint snippets stored as FLAC for efficient storage.
- **Web Dashboard**: Single-page app at `/gui` with Transcribe, Voiceprints, History and Settings tabs.

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Edit .env: set TRANSCRIBE_SESSION_SECRET
docker compose up -d --build
```

### Local dev (CUDA GPU required)

```powershell
cp .env.example .env
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m asr_mcp.server
```

## Configuration

All settings use the `TRANSCRIBE_` env prefix. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_SESSION_SECRET` | (required) | Secret key for session cookies |
| `TRANSCRIBE_HTPASSWD_PATH` | `/app/data/.htpasswd` | Path to htpasswd file |
| `TRANSCRIBE_PREFIX` | `""` | URL prefix for reverse proxy (e.g. `/asr-mcp`) |
| `TRANSCRIBE_CUDA_DEVICE` | `cuda:0` | CUDA device ordinal |
| `TRANSCRIBE_DATA_DIR` | `./data` | Data directory (SQLite DB) |
| `TRANSCRIBE_VOICES_DIR` | `./voices` | Voiceprint snippets directory |
| `TRANSCRIBE_DB_PATH` | `./data/asr_mcp.db` | SQLite database path |
| `TRANSCRIBE_PORT` | `8080` | Server port |
| `TRANSCRIBE_MODEL_TTL_MINUTES` | `5` | Idle minutes before GPU models unload (0 = never; skipped while a job is active) |
| `TRANSCRIBE_GPU_MEMORY_LIMIT_GB` | `4.0` | GPU size hint for CUDA arena caps (encoder ×0.625, embedding min(÷4, 768 MiB)) |
| `TRANSCRIBE_ASR_MODEL` | `cohere` | ASR backend: `cohere` \| `qwen3-asr` \| `whisper` |
| `TRANSCRIBE_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name/repo |
| `TRANSCRIBE_WHISPER_COMPUTE_TYPE` | `auto` | CT2 compute type; `auto` = int8_float16 on CUDA / int8 on CPU |

## Windows Client (`asr-client/`)

Stdlib-only Python client — no installs needed beyond Python 3. Distributed
as a standalone zip built by GitHub Actions (`.github/workflows/asr-client.yml`,
triggers on `asr-client/**` changes); the zip is a workflow artifact containing
`transcribe_client.py`, `transcribe.bat`, `requirements.txt`, `README.md`, and
`.env.example` — never `.env` or `.venv`.

```powershell
# From the unzipped folder — first run creates .env from .env.example
transcribe.bat status
# Edit .env: SERVER_URL (include the /asr-mcp prefix when behind nginx) + TOKEN
# Generate the token in the web UI ("Windows Client Token" card) or via POST /api/token
```

`transcribe.bat` also creates a private `.venv` next to the script and runs the
client with it (falls back to `py`/`python` on PATH); `requirements.txt` is
installed there when it declares dependencies. The client has no server
code/dependency cross-links — CI enforces stdlib-only imports.

| Command | Description |
|---------|-------------|
| Drag files onto `transcribe.bat` | Transcribe with live progress; writes `<name>.txt` next to each file |
| `transcribe.bat status` | Check server/token connectivity |
| `transcribe.bat voiceprints` | List speakers and voiceprints |
| `transcribe.bat voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip |
| `transcribe.bat voiceprint-refine <name>` | Rebuild a voiceprint from its snippets |
| `transcribe.bat <file...> [--language <code>]` | Transcribe (language overrides `.env LANGUAGE`) |

Client transcriptions use `?save=false` — nothing is stored in server transcript history.
If the connection drops mid-job, the server finishes the run and saves the result to
transcript history so it can be downloaded from the web UI. Voiceprint snippets uploaded
by the client ARE kept (they feed refinement).

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/gui` | GET | Session | SPA — Transcribe tab (same app.html serves all tabs) |
| `/voices` | GET | Session | SPA — Voiceprints tab |
| `/transcriptions` | GET | Session | SPA — History tab |
| `/settings` | GET | Session | SPA — Settings tab (status, client token, session) |
| `/login` | GET | No | Login page |
| `/api/asr/diarize` | POST | API key | Diarize audio by file path |
| `/api/asr/diarize/upload` | POST | Session | Diarize uploaded audio |
| `/api/asr/transcribe` | POST | API key | Transcribe by file path |
| `/api/asr/transcribe/upload` | POST | Session/API key | Transcribe uploaded audio (`?save=false` skips server-side save; `?language=hu` or `auto`, default `auto`) |
| `/api/asr/languages` | GET | Session/API key | Static language list of the configured backend (`{backend, supports_auto, languages}`) |
| `/api/asr/activity/stream` | GET | Session/API key | SSE push of job start/finish (snapshot on connect + keep-alive pings; replaces polling `/active`) |
| `/api/asr/active/stream` | GET | Session/API key | SSE replay+follow of the current job; non-owners get events with `result` stripped |
| `/api/asr/active/cancel` | POST | Session/API key | Cancel your own active transcription job |
| `/api/asr/ws/stream` | WS | No | Real-time streaming transcription |
| `/api/speaker/register` | POST | API key | Register voiceprint |
| `/api/speaker/identify` | POST | API key | Identify speaker |
| `/api/speaker/list` | GET | API key | List all voiceprints |
| `/api/speaker/{name}` | DELETE | API key | Delete voiceprint |
| `/api/voiceprint/speakers` | GET | Session | List speakers (web UI) |
| `/api/voiceprint/snippets/{speaker}` | GET | Session | List snippets |
| `/api/voiceprint/upload` | POST | Session | Register voiceprint from upload |
| `/api/voiceprint/speakers/{name}/upload` | POST | Session/API key | Add snippet (optional `?start_sec=&end_sec=`) |
| `/api/voiceprint/speakers/{name}/refine` | POST | Session/API key | Rebuild a voiceprint from its snippets |
| `/api/voiceprint/merge` | POST | Session | Merge speakers |
| `/api/voiceprint/rename` | POST | Session | Rename speaker |
| `/api/voiceprint/rescan` | POST | Session | Rescan voices directory |
| `/api/token` | GET/POST | Session/API key | Client API token status / generate (one per user) |
| `/api/auth/login` | POST | No | Login (returns JSON) |
| `/api/auth/logout` | POST | No | Logout |
| `/api/user` | GET | No | Current user info |
| `/api/mcp/tools` | GET | No | List MCP tools |
| `/api/mcp/resources` | GET | No | List MCP resources |
| `/api/mcp/call` | POST | No | Call MCP tool |

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

1. Browser requests a protected page
2. Auth middleware checks session cookie
3. If not authenticated → redirect to `/login`
4. User submits credentials → `/api/auth/login` verifies against htpasswd file
5. Session cookie set → redirect to `/gui`

API clients (Windows client, scripts) send `X-API-Key: <token>` instead of a
session. Valid tokens are stored as SHA-256 hashes in the `api_tokens` table
(one per user, plaintext shown once); token requests return 401 JSON rather
than a login redirect.

### Uncertain Speakers

When a speaker identity cannot be established, the server **suppresses the
identity and keeps the text** instead of force-fitting the segment to the
nearest or most common speaker:

```json
{ "start": 9.2, "end": 10.6, "text": "Yes.",
  "speaker": null, "speaker_confidence": 0.0, "speaker_source": "unknown",
  "uncertain": true, "attribution_reason": "boundary_crossing" }
```

- Every result carries `speaker_confidence`, `speaker_source`
  (`known_voiceprint` | `diarization_cluster` | `unknown`) and, when the
  identity was withheld, `attribution_reason` (e.g. `boundary_crossing`,
  `ghost_speaker`, `low_span_turn_overlap`).
- The `done` SSE event reports `status` = `ok` | `partial` | `error` and
  `uncertain_segments`.
- The web UI and the Windows client render `null` as `UNKNOWN (<reason>)`;
  the client also prints a warning and can write a `.json` sidecar
  (`WRITE_JSON=1` or `--json`). **Text is never dropped** — only the name is.
- A cluster is renamed to a registered person only if the match clears two
  gates: a minimum confidence (`second_pass.min_identity_confidence`) and a
  one-to-one claim (two acoustically different clusters cannot share a
  voiceprint). Below the gate it stays `Speaker N` — with a large voiceprint
  menu the best of a bad lot is not an identification.
- `num_speakers` selects a *different* clustering path (hard k, no greedy
  merge). If a result looks lopsided, read the `Cluster balance: ...` line in
  the server log before touching the naming policy.
- Thresholds: `asr_mcp/config/thresholds.json` → `uncertainty`, `streaming`,
  `second_pass`. Set `"uncertainty": {"enabled": false}` for the full legacy
  behaviour, or keep the policy and only restore absorption with
  `"suppress_ghost_speakers": false, "suppress_minority_speakers": false`.
- Known limitation: short recordings (<1 min) over-cluster, so one person's
  turns can come back `UNKNOWN`. See **[docs/uncertain-speakers.md](docs/uncertain-speakers.md)**.

### Learning New Speakers

A speaker who has no voiceprint yet is learned automatically — but as a
**pending profile**, never as a voiceprint:

- After a file upload or a live session's re-attribution, an unknown speaker
  (≥10 s of speech across ≥2 segments) is saved as
  `Pending <date> Speaker_5 <source>.flac` with its snippets, and the client
  prints what it learned.
- A pending profile is **excluded from every voiceprint match** — an unnamed
  profile has no identity, so it can never win a nearest-neighbour vote. That
  person's speech therefore still reads `UNKNOWN` (with the text kept) until
  you name them.
- Name them in the web UI: **Voiceprints → Unnamed speakers**, type a name and
  press *Save name*. That is the only action that makes them recognisable, and
  it also moves their snippets into a normal voiceprint.
- Confirming refuses a name already in use, so it cannot overwrite a real
  colleague's profile.
- Re-running the same recording **extends** the existing pending profile
  instead of creating a second one.

#### "Is this someone I already added?"

Every pending profile is scored against your registered speakers. When one
looks like somebody you already have, the card offers a **Merge into …** button
instead of making you invent a second name for the same person — two profiles
for one person would split their speech between two names in every transcript.
Merging moves the snippets into the existing profile and rebuilds it. Nothing
is ever merged automatically; the flag is a suggestion.

### Diarization Pipeline

Step numbers match the `# Step N` comments in
`asr_mcp/diarization/pipeline.py::Diarizer.run`.

1. **VAD** — Silero ONNX → raw sections, merge gaps <0.5s
2. **Energy-dip splitting** — split long segments at quiet dips
3. **Sliding windows** — 2.0s window, 1.2s stride (embeddings only)
4. **Embedding** — ECAPA-TDNN512 ONNX (192-dim), MD5-keyed LRU cache
5. **Clustering** — AgglomerativeClustering (cosine): threshold path uses average linkage at `distance_threshold`, or hard k when `num_speakers` is given (`forced_k_linkage`, default `complete`)
5b/5c. **Label assignment** to short segments, then **overlap detection**
6. **Build segments** — "Speaker N" labels, same-speaker merge (max gap 1.0s)
7. **Cleanup** — split OVERLAP, absorb islands, refine boundaries
8. **Speaker profiling** — pitch, energy, spectral, MFCC stats
9. **Voiceprint matching** — Multi-feature distance (emb 0.6 + pitch 0.15 + spectral 0.1 + MFCC 0.1)
10. **Ghost elimination** — <10s total speech → suppressed to `UNKNOWN` (not reassigned)
11. **Minority suppression** — short speakers suppressed, matched voiceprints protected
12. **Second pass** — re-identify unknown speakers (gated by `second_pass.min_identity_confidence`)
13. **Exact boundary refinement** — from the raw VAD sections

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + auth routes
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe (+upload, SSE), /activity/stream, /active/stream
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list
│   │   ├── voiceprint_router.py # CRUD + upload + merge + rename + rescan + refine
│   │   ├── transcript_router.py # User-scoped transcript CRUD + download
│   │   ├── token_router.py    # GET/POST /token — client API token status + generation
│   │   ├── mcp_router.py      # MCP tools + resources
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key + DB token auth + path validation
│   │   ├── auth.py            # htpasswd + session auth (token header bypass)
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging, CORS
│   ├── core/
│   │   ├── model_state.py     # ModelState, is_gpu_oom, GPU_SHRINK_RUN_OPTIONS, lazy-load/TTL
│   │   ├── model_loader.py    # HuggingFace download + ORT session init (arena caps)
│   │   ├── job_state.py       # Active job tracking (start/publish/finish/attach/activity subs)
│   │   └── transcriber.py     # Thin facade → state.backend (Cohere ONNX / Qwen3-ASR / Whisper)
│   ├── transcribers/
│   │   ├── base.py            # ASRBackend ABC (context/progress_cb interface)
│   │   ├── cohere.py          # Cohere ONNX: windowed decode + encoder chunking
│   │   ├── qwen3.py           # Qwen3-ASR + forced aligner (4-bit), OOM-backoff chunker
│   │   └── whisper.py         # faster-whisper large-v3-turbo (int8-quantized CTranslate2)
│   ├── diarization/
│   │   ├── pipeline.py        # Diarizer: 13-step pipeline
│   │   ├── clustering.py      # AgglomerativeClustering, greedy merge
│   │   └── segment_ops.py     # Collapse, absorb islands, eliminate ghosts
│   ├── speaker/
│   │   ├── audio.py           # fbank, sliding windows, boundary refinement
│   │   ├── embedding.py       # ONNX embedding, batch embed, pitch, energy
│   │   ├── vad.py             # VAD (energy-dip splitting, chunked, ONNX)
│   │   ├── matcher.py         # Multi-feature distance
│   │   ├── profiling.py       # Pitch/energy/MFCC profiling
│   │   ├── uncertainty.py     # Uncertain-speaker policy (single source of truth)
│   │   ├── attribution.py     # Post-hoc span→turn attribution (boundary crossings)
│   │   └── service.py         # SpeakerService (SQLite-backed)
│   ├── voiceprint/
│   │   ├── service.py         # VoiceprintService (snippet CRUD, auto-collect, refine)
│   │   └── utils.py           # Audio load/convert (ffmpeg), FLAC snippets
│   ├── db/
│   │   ├── models.py          # SQLAlchemy: voiceprints, snippets, sessions, transcripts, api_tokens
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, TokenDB, TranscriptDB, etc.
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed)
│   ├── streaming/
│   │   ├── turn_detector.py   # Adaptive energy turn detector + turn coalescer
│   │   ├── speech_gate.py    # Silero speech gate before ASR (fail-open)
│   │   ├── protocol.py       # Turn-frame wire format (LVT1)
│   │   ├── attribution.py    # Channel-aware live speaker attribution
│   │   └── handler.py         # WebSocket dual-channel handler
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (TRANSCRIBE_ prefix)
│   │   ├── logging.py         # Structured logging (stdlib)
│   │   └── thresholds.json    # All tunable params (diarization/uncertainty/streaming)
│   ├── static/
│   └── templates/
│       ├── _nav.html          # SPA tab bar
│       ├── login.html          # Login form
│       └── app.html            # Unified SPA (Transcribe / Voiceprints / History / Settings)
├── tests/                    # Pure-python unit tests (pytest, no GPU required)
├── asr-client/
│   ├── transcribe_client.py    # Stdlib Windows client: SSE progress, <name>.txt, voiceprints
│   ├── transcribe.bat          # Drop-target wrapper: bootstraps .env + private .venv
│   ├── requirements.txt        # Empty (stdlib-only); installed into .venv when it has lines
│   ├── README.md               # Standalone-zip setup guide
│   └── .env.example            # SERVER_URL + TOKEN template
├── .github/workflows/asr-client.yml  # Standalone client zip on asr-client/** changes
├── Dockerfile                 # nvidia/cuda:12.2.0 base
├── docker-compose.yml         # GPU passthrough + named volumes
├── nginx_snippet.conf         # nginx location block for /asr-mcp/
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── AGENTS.md
├── docs/
│   ├── uncertain-speakers.md   # Uncertainty policy reference
│   └── lessons/                # Evidence behind each AGENTS.md rule
└── README.md
```
