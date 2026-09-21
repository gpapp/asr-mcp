# AGENTS.md - ASR MCP Server

## Dev Commands

```powershell
# Setup
cp .env.example .env
# Edit .env: set SESSION_SECRET and any CUDA/DB settings

# Infra
docker-compose up -d

# Local dev (CUDA GPU required)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m asr_mcp.server
```

## Code Quality

After completing any code changes:
1. Run `python -m py_compile <file.py>` to verify syntax
2. If running in Docker, rebuild: `docker-compose up -d --build asr-mcp`
3. Run `git diff --check` after documentation or code edits.

## Architecture

| Component | Port | Notes |
|-----------|------|-------|
| App | 8086 (Docker), 8080 (internal) | FastAPI + ONNX Runtime CUDA |
| Nginx | /asr-mcp | Proxy mount point for MCP endpoints |

### GPU Backend
- **ASR Encoder**: ONNX Runtime CUDA EP (float16)
- **ASR Decoder**: ONNX Runtime CPU
- **ECAPA-TDNN Embedding**: ONNX Runtime CUDA EP
- **Silero VAD**: ONNX Runtime CPU
- Peak VRAM: ~2.6GB

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan
│   ├── api/
│   │   ├── router.py          # Combines sub-routers
│   │   ├── asr_router.py      # POST /asr/diarize, /asr/transcribe, WS /asr/ws/stream
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list, DELETE /{name}
│   │   ├── mcp_router.py      # MCP tools + resources
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key auth + path validation
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging, CORS
│   ├── core/
│   │   ├── model_state.py     # ModelState, KVCachePool, LRUCache
│   │   ├── model_loader.py    # HuggingFace download + ORT session init
│   │   └── transcriber.py     # Cohere ASR: mel-spec → encoder → decoder → text
│   ├── diarization/
│   │   ├── pipeline.py        # Diarizer: VAD → embed → cluster → match → refine
│   │   ├── clustering.py      # AgglomerativeClustering, greedy merge, voiceprint matching
│   │   └── segment_ops.py     # Collapse, absorb islands, eliminate ghosts
│   ├── speaker/
│   │   ├── audio.py           # fbank, sliding windows, boundary refinement
│   │   ├── embedding.py       # ONNX embedding, batch embed, pitch, energy
│   │   ├── vad.py             # VAD (energy-dip splitting, chunked, ONNX)
│   │   ├── matcher.py         # Multi-feature distance (emb+pitch+energy+spectral+MFCC)
│   │   ├── profiling.py       # Pitch/energy/MFCC profiling, relabel by pitch
│   │   └── service.py         # SpeakerService (SQLite-backed CRUD)
│   ├── voiceprint/
│   │   ├── service.py         # VoiceprintService (register, refine, identify)
│   │   └── utils.py           # Audio extraction, time parsing, WAV conversion
│   ├── db/
│   │   ├── models.py          # SQLAlchemy: voiceprints, sessions, transcripts
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, SessionDB, TranscriptDB
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed)
│   ├── streaming/
│   │   └── handler.py         # WebSocket dual-channel real-time transcription
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (CUDA, DB, model fields)
│   │   ├── logging.py         # Structured logging (structlog)
│   │   └── thresholds.json    # All tunable diarization/matching/VAD params
│   ├── static/
│   └── templates/
├── Dockerfile                 # nvidia/cuda:12.2.0 base
├── docker-compose.yml         # GPU passthrough + Nginx
├── nginx.conf
├── requirements.txt
├── .env.example
├── AGENTS.md
└── README.md
```

## SQLite Schema

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `voiceprints` | `name` (TEXT) | Speaker embeddings, pitch, energy, MFCC stats |
| `sessions` | `id` (TEXT) | Session data with TTL expiration |
| `transcripts` | `id` (INTEGER) | Stored transcription results |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/gui` | GET | No | Dashboard UI |
| `/api/asr/diarize` | POST | Optional | Diarize audio by path |
| `/api/asr/diarize/upload` | POST | Optional | Diarize uploaded audio |
| `/api/asr/transcribe` | POST | Optional | Transcribe with diarization |
| `/api/asr/ws/stream` | WS | No | Real-time streaming transcription |
| `/api/speaker/register` | POST | Optional | Register voiceprint |
| `/api/speaker/register/upload` | POST | Optional | Register voiceprint from upload |
| `/api/speaker/identify` | POST | Optional | Identify speaker from audio |
| `/api/speaker/list` | GET | Optional | List all voiceprints |
| `/api/speaker/{name}` | DELETE | Optional | Delete voiceprint |
| `/api/mcp/tools` | GET | No | List MCP tools |
| `/api/mcp/resources` | GET | No | List MCP resources |
| `/api/mcp/call` | POST | No | Call MCP tool |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_CUDA_DEVICE` | `cuda:0` | CUDA device ordinal |
| `TRANSCRIBE_HOST` | `0.0.0.0` | Server bind host |
| `TRANSCRIBE_PORT` | `8080` | Server port |
| `TRANSCRIBE_DATA_DIR` | `./data` | Data directory |
| `TRANSCRIBE_LOG_DIR` | `./logs` | Log directory |
| `TRANSCRIBE_DB_PATH` | `./data/asr_mcp.db` | SQLite database path |
| `TRANSCRIBE_DIARIZATION_THRESHOLD` | `0.35` | Clustering cosine threshold |
| `TRANSCRIBE_VAD_THRESHOLD` | `0.5` | VAD speech probability cutoff |
| `TRANSCRIBE_HF_TOKEN` | - | HuggingFace token for gated models |
| `SESSION_SECRET` | - | Session management secret key |
| `API_KEYS` | - | Comma-separated API keys |

## Diarization Pipeline

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

## Known Issues

- **CUDA OOM on large files**: Reduce max_audio_duration_sec or use --num-speakers to limit clusters
- **No CUDA available**: Server starts but models fail to load; API returns errors for ASR operations
- **SQLite locking**: Concurrent writes may fail under heavy load; WAL mode recommended for production
