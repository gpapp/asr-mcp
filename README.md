# asr-mcp

GPU-accelerated ASR MCP server with speaker diarization, voiceprint recognition, and streaming transcription. Built with FastAPI + ONNX Runtime CUDA.

## Features

- **ASR with Diarization**: Detect and separate multiple speakers in audio using Cohere Transcribe ONNX.
- **Voiceprint Recognition**: Register, identify, and manage speaker profiles with ECAPA-TDNN embeddings.
- **Streaming Transcription**: Real-time WebSocket dual-channel transcription.
- **Auto Voiceprint Collection**: Snippets auto-collected from diarization for registered speakers.
- **Form-Based Auth**: Session-based login with htpasswd password files.
- **Reverse Proxy Support**: Configurable URL prefix (`TRANSCRIBE_PREFIX`) for nginx/Caddy.
- **Multi-Format Input**: Accepts mp3, mp4, mkv, flac, ogg, m4a — auto-converts via ffmpeg.
- **Compressed Snippets**: Voiceprint snippets stored as FLAC for efficient storage.
- **Web Dashboard**: Upload audio, view results, manage voiceprints at `/gui` and `/voices`.

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

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/gui` | GET | Session | Dashboard — upload & process audio |
| `/voices` | GET | Session | Voiceprint management dashboard |
| `/login` | GET | No | Login page |
| `/api/asr/diarize` | POST | API key | Diarize audio by file path |
| `/api/asr/diarize/upload` | POST | Session | Diarize uploaded audio |
| `/api/asr/transcribe` | POST | API key | Transcribe by file path |
| `/api/asr/transcribe/upload` | POST | Session | Transcribe uploaded audio |
| `/api/asr/ws/stream` | WS | No | Real-time streaming transcription |
| `/api/speaker/register` | POST | API key | Register voiceprint |
| `/api/speaker/identify` | POST | API key | Identify speaker |
| `/api/speaker/list` | GET | API key | List all voiceprints |
| `/api/speaker/{name}` | DELETE | API key | Delete voiceprint |
| `/api/voiceprint/speakers` | GET | Session | List speakers (web UI) |
| `/api/voiceprint/snippets/{speaker}` | GET | Session | List snippets |
| `/api/voiceprint/upload` | POST | Session | Register voiceprint from upload |
| `/api/voiceprint/merge` | POST | Session | Merge speakers |
| `/api/voiceprint/rename` | POST | Session | Rename speaker |
| `/api/voiceprint/rescan` | POST | Session | Rescan voices directory |
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

### Authentication Flow

1. Browser requests a protected page
2. Auth middleware checks session cookie
3. If not authenticated → redirect to `/login`
4. User submits credentials → `/api/auth/login` verifies against htpasswd file
5. Session cookie set → redirect to `/gui`

### Diarization Pipeline

1. **VAD** — Silero VAD + energy-dip splitting
2. **Sliding windows** — 2.0s window, 1.2s stride
3. **FBank extraction** — 80-dim log-mel filterbanks + CMN
4. **Embedding** — ECAPA-TDNN ONNX (192-dim), MD5-keyed LRU cache
5. **Clustering** — AgglomerativeClustering (cosine, max 15 clusters)
6. **Greedy merge** — Clusters with centroid distance < 0.25 merged
7. **Boundary refinement** — Batched ONNX re-embedding at transition points
8. **Speaker profiling** — Pitch, energy, spectral, MFCC stats
9. **Relabel by pitch** — SPEAKER_00 = lowest pitch
10. **Ghost elimination** — Reassign speakers with < 10s total speech
11. **Voiceprint matching** — Multi-feature distance (emb 0.6 + pitch 0.15 + spectral 0.1 + MFCC 0.1)

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + auth routes
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe, WS /ws/stream
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list
│   │   ├── voiceprint_router.py # CRUD + upload + merge + rename + rescan
│   │   ├── mcp_router.py      # MCP tools + resources
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key auth + path validation
│   │   ├── auth.py            # htpasswd + session auth
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging, CORS
│   ├── core/
│   │   ├── model_state.py     # ModelState, KVCachePool, LRUCache
│   │   ├── model_loader.py    # HuggingFace download + ORT session init
│   │   └── transcriber.py     # Cohere ASR: mel-spec → encoder → decoder → text
│   ├── diarization/
│   │   ├── pipeline.py        # Diarizer: 11-step pipeline
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
│   │   ├── models.py          # SQLAlchemy: voiceprints, snippets, sessions, transcripts
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, SnippetDB, etc.
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed)
│   ├── streaming/
│   │   └── handler.py         # WebSocket dual-channel handler
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (TRANSCRIBE_ prefix)
│   │   ├── logging.py         # Structured logging (structlog)
│   │   └── thresholds.json    # All tunable params
│   ├── static/
│   └── templates/
│       ├── login.html          # Login form
│       ├── index.html          # Audio processing dashboard
│       └── voices.html         # Voiceprint management dashboard
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
