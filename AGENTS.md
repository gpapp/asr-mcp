# AGENTS.md - ASR MCP Server

## Dev Commands

```powershell
# Setup
cp .env.example .env
# Edit .env: set TRANSCRIBE_SESSION_SECRET and TRANSCRIBE_HTPASSWD_PATH

# Infra
docker compose up -d

# Local dev (CUDA GPU required)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m asr_mcp.server
```

## Code Quality

After completing any code changes:
1. Run `python -m py_compile <file.py>` to verify syntax
2. If running in Docker, rebuild: `docker compose up -d --build asr-mcp`
3. Run `git diff --check` after documentation or code edits.

## Architecture

| Component | Port | Notes |
|-----------|------|-------|
| App | 8087 (Docker), 8080 (internal) | FastAPI + ONNX Runtime CUDA |
| Nginx | /asr-mcp/ | Proxy mount point (trailing slash!) |

### Auth Flow
- Session-based auth via htpasswd file (bcrypt/apr1/sha/plain)
- SessionMiddleware cookie → AuthMiddleware innermost → public paths bypass
- Middleware order: AuthMiddleware (innermost) → SessionMiddleware → CORSMiddleware → logging
- `TRANSCRIBE_PREFIX` sets URL prefix for redirects behind reverse proxy

### GPU Backend
- **ASR Encoder**: ONNX Runtime CUDA EP (float16) — chunked input for long audio (>30s)
- **ASR Decoder**: ONNX Runtime CPU — receives `encoder_hidden_states` + KV caches (8 layers)
- **ECAPA-TDNN512 Embedding**: ONNX Runtime CUDA EP (192-dim) — fbank chunking (60s max) + CPU OOM fallback
- **Silero VAD**: ONNX Runtime CPU — with state/sr inputs, h/c hidden state updates
- Peak VRAM: ~2.6GB

### Transcription Chunking (Long Audio)

Long diarized segments (>30s) are split at **VAD energy-dip boundaries** before encoding, never at arbitrary time points. This prevents cutting mid-word.

**Router-level splitting** (`asr_router.py`):
1. Each diarized segment is extracted from the full audio
2. If segment > 30s: `split_at_energy_dips()` finds natural pause boundaries (dip_ratio=0.35, min_dip_dur=0.3s)
3. Sub-segments are accumulated into a buffer up to 30s
4. **Buffer flushes at >0.5s pauses**: if gap between consecutive sub-segments > 0.5s, the buffer is flushed and a new chunk starts — this preserves speaker-change boundaries within diarized segments
5. Each buffer is concatenated and sent to `transcribe_audio_sync()`

**Encoder-level chunking** (`transcriber.py`) — safety fallback if router chunking is bypassed:
- Mel spectrogram is split into overlapping windows (MAX_ENCODER_SEC=30s, 25% overlap)
- Each window is encoded independently
- Encoder outputs are concatenated along sequence dimension
- Overlap is trimmed from non-first chunks to avoid duplication

**Decoder interface**: The decoder ONNX model requires `encoder_hidden_states` as a direct input (not just KV caches). `_parse_encoder_outputs()` extracts the hidden states tensor from encoder output and passes it through. Cross-attention KV caches (8 layers × key/value) are initialized as empty (seq_len=0).

### Diarization Pipeline (13 steps)

```
1. VAD (Silero ONNX) → speech regions
1b. Merge nearby speech (<1s silence gap)
2. Energy-dip splitting (dip_ratio=0.35, min_dip_dur=0.5s, min_split_piece=2.0s)
3. Sliding windows (3.0s window, 2.5s stride) → fbank features
4. ECAPA-TDNN512 embedding per window (192-dim, ONNX CUDA)
5. AgglomerativeClustering (distance_threshold=0.50, cosine, average linkage)
6. Greedy merge clusters (merge_threshold=0.25)
7. Map labels → segments ("Speaker 1", "Speaker 2", ...)
8. Collapse same-speaker (max_gap=0.5s) + absorb islands
9. Boundary refinement (re-embed boundary frames)
10. Speaker profiling (pitch, energy, MFCC)
11. Relabel by pitch (highest = Speaker 1)
12. Ghost elimination (total_dur < 5s → absorb to nearest neighbor)
13. Known-speaker matching (multi-feature: 60% embedding + 15% pitch + 10% spectral + 10% MFCC)
```

### Audio Format Support
- Input: mp3, mp4, mkv, flac, ogg, m4a, wav, webm, opus
- Auto-converts to WAV PCM 16kHz mono via ffmpeg before processing
- Voiceprint snippets stored as **FLAC** (compressed, lossless)

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + auth routes
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe, /diarize/upload, /transcribe/upload
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list, DELETE /{name}
│   │   ├── voiceprint_router.py # CRUD + upload + merge + rename + rescan
│   │   ├── mcp_router.py      # MCP tools + resources + /call
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key auth + path validation
│   │   ├── auth.py            # htpasswd parse + session auth + require_auth
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging
│   ├── core/
│   │   ├── model_state.py     # ModelState, KVCachePool, LRUCache
│   │   ├── model_loader.py    # HuggingFace download + ORT session init (CUDA EP)
│   │   └── transcriber.py     # Cohere ASR: mel-spec → encoder → decoder → text
│   ├── diarization/
│   │   ├── pipeline.py        # Diarizer: 11-step pipeline
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
│   │   ├── service.py         # VoiceprintService (snippet CRUD, auto-collect, refine, merge)
│   │   └── utils.py           # Audio load/convert (ffmpeg), FLAC snippets
│   ├── db/
│   │   ├── models.py          # SQLAlchemy: voiceprints, snippets, sessions, transcripts
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, SnippetDB, SessionDB, TranscriptDB
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed)
│   ├── streaming/
│   │   └── handler.py         # WebSocket dual-channel real-time transcription
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (TRANSCRIBE_ prefix)
│   │   ├── logging.py         # Structured logging (stdlib)
│   │   └── thresholds.json    # All tunable diarization/matching/VAD params
│   ├── static/
│   └── templates/
│       ├── login.html          # Dark-themed login form
│       ├── index.html          # Audio processing dashboard (upload + results)
│       └── voices.html         # Voiceprint management dashboard
├── Dockerfile                 # nvidia/cuda:12.2.0 base
├── docker-compose.yml         # GPU passthrough + named volumes + port 8087
├── nginx_snippet.conf         # nginx location /asr-mcp/ with proxy_pass
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── AGENTS.md
└── README.md
```

## SQLite Schema

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `voiceprints` | `(user_id, name)` composite | Speaker embeddings, pitch, energy, MFCC stats |
| `snippets` | `id` (INTEGER) | Auto-collected audio snippets (FLAC files) |
| `sessions` | `id` (TEXT) | Session data with TTL expiration |
| `transcripts` | `id` (INTEGER) | Stored transcription results |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/gui` | GET | Session | Audio processing dashboard |
| `/voices` | GET | Session | Voiceprint management dashboard |
| `/login` | GET | No | Login page |
| `/api/asr/diarize` | POST | API key | Diarize audio by file path |
| `/api/asr/diarize/upload` | POST | Session | Diarize uploaded audio |
| `/api/asr/transcribe` | POST | API key | Transcribe by file path |
| `/api/asr/transcribe/upload` | POST | Session | Transcribe uploaded audio |
| `/api/asr/ws/stream` | WS | No | Real-time streaming transcription |
| `/api/speaker/register` | POST | API key | Register voiceprint |
| `/api/speaker/register/upload` | POST | API key | Register voiceprint from upload |
| `/api/speaker/identify` | POST | API key | Identify speaker from audio |
| `/api/speaker/list` | GET | API key | List all voiceprints |
| `/api/speaker/{name}` | DELETE | API key | Delete voiceprint |
| `/api/voiceprint/speakers` | GET | Session | List speakers (web UI) |
| `/api/voiceprint/snippets/{speaker}` | GET | Session | List snippets |
| `/api/voiceprint/upload` | POST | Session | Register voiceprint from upload |
| `/api/voiceprint/merge` | POST | Session | Merge speakers |
| `/api/voiceprint/rename` | POST | Session | Rename speaker |
| `/api/voiceprint/rescan` | POST | Session | Rescan voices directory |
| `/api/auth/login` | POST | No | Login (JSON + session cookie) |
| `/api/auth/logout` | POST/GET | No | Logout |
| `/api/user` | GET | No | Current user info |
| `/api/mcp/tools` | GET | No | List MCP tools |
| `/api/mcp/resources` | GET | No | List MCP resources |
| `/api/mcp/call` | POST | No | Call MCP tool |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_SESSION_SECRET` | (required) | Secret key for session cookies |
| `TRANSCRIBE_HTPASSWD_PATH` | `/app/data/.htpasswd` | Path to htpasswd file |
| `TRANSCRIBE_PREFIX` | `""` | URL prefix for reverse proxy |
| `TRANSCRIBE_CUDA_DEVICE` | `cuda:0` | CUDA device ordinal |
| `TRANSCRIBE_HOST` | `0.0.0.0` | Server bind host |
| `TRANSCRIBE_PORT` | `8080` | Server port |
| `TRANSCRIBE_DATA_DIR` | `./data` | Data directory |
| `TRANSCRIBE_LOG_DIR` | `./logs` | Log directory |
| `TRANSCRIBE_VOICES_DIR` | `./voices` | Voiceprint snippets directory |
| `TRANSCRIBE_DB_PATH` | `./data/asr_mcp.db` | SQLite database path |
| `TRANSCRIBE_MODEL_CACHE_DIR` | `./models` | ONNX model cache |
| `TRANSCRIBE_DIARIZATION_THRESHOLD` | `0.35` | Clustering cosine threshold |
| `TRANSCRIBE_VAD_THRESHOLD` | `0.5` | VAD speech probability cutoff |
| `TRANSCRIBE_HF_TOKEN` | - | HuggingFace token for gated models |
| `API_KEYS` | - | Comma-separated API keys |

## Known Issues

- **CUDA OOM on large files**: Reduce max_audio_duration_sec or use --num-speakers to limit clusters
- **No CUDA available**: Server starts but models fail to load; API returns errors for ASR operations
- **SQLite locking**: Concurrent writes may fail under heavy load; WAL mode recommended for production
- **ffmpeg required**: Audio format conversion requires ffmpeg installed in Docker image
