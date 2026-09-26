# asr-mcp

GPU-accelerated ASR MCP server with speaker diarization, voiceprint recognition, and streaming transcription. Built with FastAPI + ONNX Runtime CUDA.

## Features

- **ASR with Diarization**: Detect and separate multiple speakers in audio using Cohere Transcribe ONNX.
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

## Windows Client (`asr-client/`)

Stdlib-only Python client — no installs needed beyond Python 3.

```powershell
cd asr-client
copy .env.example .env
# Edit .env: SERVER_URL (include the /asr-mcp prefix when behind nginx) + TOKEN
# Generate the token in the web UI ("Windows Client Token" card) or via POST /api/token
```

| Command | Description |
|---------|-------------|
| Drag files onto `transcribe.bat` | Transcribe with live progress; writes `<name>.txt` next to each file |
| `python transcribe_client.py status` | Check server/token connectivity |
| `python transcribe_client.py voiceprints` | List speakers and voiceprints |
| `python transcribe_client.py voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip |
| `python transcribe_client.py voiceprint-refine <name>` | Rebuild a voiceprint from its snippets |

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
| `/api/asr/transcribe/upload` | POST | Session/API key | Transcribe uploaded audio (`?save=false` skips server-side save) |
| `/api/asr/active` | GET | Session/API key | Current job (any user) + `can_cancel` for requester |
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

### Diarization Pipeline (13 steps)

1. **VAD** — Silero ONNX → speech regions, merge gaps <1s
2. **Energy-dip splitting** — split long segments at quiet dips
3. **Sliding windows** — 3.0s window, 2.5s stride (embeddings only)
4. **Embedding** — ECAPA-TDNN512 ONNX (192-dim), MD5-keyed LRU cache
5. **Clustering** — AgglomerativeClustering (cosine, average linkage)
6. **Greedy merge** — centroid distance < 0.25 merged
7. **Collapse** — same-speaker merge (max gap 0.5s) + absorb islands
8. **Boundary refinement** — re-embedding at transitions
9. **Speaker profiling** — Pitch, energy, spectral, MFCC stats
10. **Merge similar speakers** — embed-only threshold 0.2
11. **Relabel by pitch** — "Speaker 1" = highest pitch (always `Speaker N`, 1-indexed)
12. **Ghost elimination** — speakers with < 5s total speech absorbed
13. **Voiceprint matching** — Multi-feature distance (emb 0.6 + pitch 0.15 + spectral 0.1 + MFCC 0.1)

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + auth routes
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe (+upload, SSE), /active/stream
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
│   │   ├── job_state.py       # Active job tracking (start/publish/finish/attach)
│   │   └── transcriber.py     # Thin facade → state.backend (Cohere ONNX / Qwen3-ASR)
│   ├── transcribers/
│   │   ├── base.py            # ASRBackend ABC (context/progress_cb interface)
│   │   ├── cohere.py          # Cohere ONNX: windowed decode + encoder chunking
│   │   └── qwen3.py           # Qwen3-ASR + forced aligner (4-bit), OOM-backoff chunker
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
│   │   └── handler.py         # WebSocket dual-channel handler
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (TRANSCRIBE_ prefix)
│   │   ├── logging.py         # Structured logging (stdlib)
│   │   └── thresholds.json    # All tunable params
│   ├── static/
│   └── templates/
│       ├── _nav.html          # SPA tab bar
│       ├── login.html          # Login form
│       └── app.html            # Unified SPA (Transcribe / Voiceprints / History / Settings)
├── asr-client/
│   ├── transcribe_client.py    # Stdlib Windows client: SSE progress, <name>.txt, voiceprints
│   ├── transcribe.bat          # Drop-target wrapper (py/python launcher)
│   └── .env.example            # SERVER_URL + TOKEN template
├── Dockerfile                 # nvidia/cuda:12.2.0 base
├── docker-compose.yml         # GPU passthrough + named volumes
├── nginx_snippet.conf         # nginx location block for /asr-mcp/
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── AGENTS.md
└── README.md
```
