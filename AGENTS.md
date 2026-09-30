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
2. Run `python -m pytest tests/ -q` — pure-python unit tests (uncertainty policy, attribution, turn detector, streaming handler, segment_ops). The `test_result_merge.py`, `test_clustering_linkage.py` and `test_second_pass_claim.py` modules `importorskip` torch/sklearn and are skipped outside the container; run them with `docker compose exec -T asr-mcp python3 -m pytest tests/ -q` after a rebuild
3. For threshold work, **cache the per-window embeddings** and sweep offline — see lesson 33
4. If running in Docker, rebuild: `docker compose up -d --build asr-mcp`
5. Run `git diff --check` after documentation or code edits

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
- **API token auth**: `X-API-Key` header with a DB token (or static `TRANSCRIBE_API_KEYS` value). `require_auth` returns 401 JSON for an invalid header (never a login redirect); valid header passes without a session. Tokens: `api_tokens` table, sha256-hashed, one per user (`TokenDB.create` replaces), plaintext returned once by `POST /api/token` (session UI card "Windows Client Token"). `get_current_user` order: session → DB token → static key → default/401.
- **Client no-save**: Windows client calls `/api/asr/transcribe/upload?save=false` → TranscriptDB.save skipped; voiceprint snippet uploads are still persisted. Exception: if the SSE consumer disconnects before `done` (`event_stream` sets `client_disconnected` on GeneratorExit/CancelledError), the transcript IS saved so it appears in the web UI history. Save logic lives in `run_transcribe`'s `_persist_transcript(reason)` (both the main path and the no-segments path call it on disconnect).

### GPU Backend
- **CUDA library preload**: `model_loader.py` runs `_preload_cuda_libs()` at module import — `import torch` then `ctypes.CDLL(libcudnn.so.9/libcublas.so.12/libcublasLt.so.12, RTLD_GLOBAL)`. The base image has no cuDNN in system paths (only the pip `nvidia-cudnn-cu12` wheel); ORT cannot find it alone and would silently fall back to CPU, then crash on `GPU_SHRINK_RUN_OPTIONS` (`Did not find an arena based allocator ... gpu:0`). Any refactor that removes a module-level `import torch` (e.g. from `streaming/handler.py`) can regress this — keep the preload or re-add torch import.
- **ASR Encoder** (CohereBackend): ONNX Runtime CUDA EP — chunked input for long audio (>30s). Arena cap `gpu_memory_limit_gb×0.625` (2560 MiB @ 4GB) + `cudnn_conv_algo_search: HEURISTIC`. OOM escalation (`_run_encoder`): reload fresh arena → retry GPU → cached CPU session (success = no error field).
- **ASR Decoder**: ONNX Runtime CPU — receives `encoder_hidden_states` + KV caches (8 layers)
- **ECAPA-TDNN512 Embedding**: ONNX Runtime CUDA EP (192-dim) — fbank chunking (60s max); arena cap `min(gpu_memory_limit_gb/4, 768 MiB)`; OOM → cached CPU session (`_run_with_cpu_fallback`)
- **Silero VAD**: ONNX Runtime CPU — with state/sr inputs, h/c hidden state updates
- **Arena shrinkage**: every GPU run passes `GPU_SHRINK_RUN_OPTIONS` (`memory.enable_memory_arena_shrinkage=gpu:0`, defined in `model_state.py`) so arenas release after each run instead of holding peak until session reload. Reload helpers clear the old session + `gc.collect()` BEFORE constructing the new one (no double-arena peak).
- Peak VRAM: ~2.6GB with caps (target card: GTX 1650, 4096 MiB)

### Whisper Backend (`TRANSCRIBE_ASR_MODEL=whisper`)
- **faster-whisper / CTranslate2**, default model `large-v3-turbo`, always **quantized**: `TRANSCRIBE_WHISPER_COMPUTE_TYPE=auto` → `int8_float16` on CUDA / `int8` on CPU (~1.0GB VRAM measured on the GTX 1650; fp16 would be ~1.6GB + spikes).
- **Load escalation**: requested compute type → `int8` on CUDA → `int8` on CPU; each failed attempt is logged and the model freed first. Runtime CUDA OOM rebuilds the model on CPU (`_fallback_to_cpu`) and retries the whole transcription once.
- **Timestamps**: passes `without_timestamps=False` — segment-level start/end feed `_transcribe_file`'s speaker attribution (the faster-whisper default `True` yields one coarse segment per file). Progress events map segment end times onto 30s windows (`progress_cb(win, est_windows, partial_text, new_segments)`).
- **Confidence**: each whisper segment carries `confidence = exp(avg_logprob)` clamped to 0..1 (`TimedSegment.confidence` is optional; cohere/qwen3 leave it `None` → client omits the `(NN%)` suffix). Note faster-whisper hard-sets `language_probability=1.0` when language is forced — the log line prints `requested=<x> lang=<detected>` to distinguish forced from detected.
- **Style anchor (`STYLE_ANCHOR` initial_prompt)**: every decode gets a punctuated English meeting-style prompt unless caller `context` is non-empty. Without it, long-form/room-audio decodes can lock into a lowercase, unpunctuated style from the FIRST segment (0/251 segments punctuated on a 34-min recording) and `condition_on_previous_text=True` propagates it for the whole file. Verified: degraded-audio sentence density 4.6 → 17.6 per 1000 letters; clean English unchanged; Hungarian 98.5% identical with slightly MORE punctuation and no English word injection. Do NOT "fix" this by flipping `condition_on_previous_text=False` — it *reduces* punctuation density on clean English (15.6 → 8.2 per 1000 letters, measured).
- **Dependency hazard**: `faster-whisper` depends on the CPU `onnxruntime` wheel; both wheels write the same `site-packages/onnxruntime/` files, so installing the CPU wheel last silently kills the CUDA EP (then `GPU_SHRINK_RUN_OPTIONS` crashes — lesson 5). Keep `onnxruntime-gpu` LAST in `requirements.txt` and the Dockerfile's `pip install --force-reinstall --no-deps onnxruntime-gpu` step as guarantee.
- Model download lands in `models/faster-whisper/<spec>/` (mounted volume); `_ensure_local_model` skips the download once `config.json` + `model.bin` exist.

### Model Lifecycle (lazy load + TTL + phase unloads)
- **No auto-load at server start** — `server.py` lifespan does NOT load models; first GPU request calls `state.ensure_ready()`. Endpoint rule: any endpoint that embeds/transcribes/diarizes must call `ensure_ready()` (or be behind a handler that does).
- **Backend-driven state** — `ModelState.backend` holds the active ASR backend instance (`cohere` | `qwen3-asr` | `whisper`, chosen by `TRANSCRIBE_ASR_MODEL`; compare via `transcribers.resolve_backend_name()` so case/whitespace can't cause a reload loop). `state.is_ready` = backend loaded **and** `embedding_session` **and** `vad_session`; `reload_models()` picks/loads the backend via `transcribers.get_backend()`, then fills vad+embedding. `unload_models()` delegates to `backend.unload()` + drops embedding/vad. Transcriber is a thin facade: `core/transcriber.py` forwards `transcribe_audio_sync(audio=...)` to `state.backend`.
- **TTL**: `model_ttl_minutes` (default **5**) — monitor in `server.py` unloads idle models only when `state.any_loaded and idle > ttl` AND `job_state.get_running() is None`. Never unload mid-job.
- **Phase unloads** (models not needed in a phase get freed for the next):
  - `/diarize` + `/diarize/upload`: `unload_encoder()` at job start (diarize never uses the encoder; `Diarizer.run()` gates on `vad_session`+`embedding_session` only, NOT `is_ready` — the encoder is intentionally unloaded)
  - `/transcribe/upload`: `unload_embedding()` right after the `diarization_complete` SSE emit (auto-collect + boundary refinement already done; embedding never used again)
- **Live session resolution**: NEVER pin a copy of `state.*_session` into other objects. `VoiceprintService._emb_session()` resolves `state.embedding_session` at call time (`set_embedding_session` has zero callers — pins caused `'NoneType' has no attribute get_inputs` after lazy unload).

### Transcription Chunking (Long Audio)

**Turn preparation** (`asr_router.py::_prepare_turns`) — every second of the timeline is covered by exactly one turn. Turns feed the timeline/display and post-hoc speaker attribution; they are NOT decode units:
1. Merge same-speaker diarized segments into turns (gap ≤1.5s)
2. Split turns >120s (`MAX_TURN_SEC`) at their largest internal diarized-segment gap
3. **Exact boundary refinement** (`_refine_boundaries_with_vad`): raw uncollapsed VAD over the full file; each VAD section spanning a gap is embedded and attributed to the better-matching adjacent speaker (known DB voiceprint, else ≤30s of that speaker's own turn audio); `_best_split` picks the ownership cut (margin-weighted); both turns move to one shared cut at the exact start of the first section owned by the incoming speaker (or gap end if the gap is entirely the left speaker's)
4. **Fallback chain for any remaining gap**: energy-dip cut (`_gap_boundary`, quietest pause centre) → midpoint
5. Edges extended to 0.0 / audio_duration
6. **Transcription** (`asr_router.py::_transcribe_file`): ONE whole-file `transcribe_audio_sync()` call; the backend decodes in its own windows (Qwen `_transcribe_chunked`, Cohere `_transcribe_windowed`), then `speaker/attribution.py::attribute_items` maps each returned item onto the turns and groups consecutive same-speaker items into `TranscribeResult` runs — audio is never cut at speaker boundaries

**Decode windowing** (`transcribers/cohere.py::_transcribe_windowed`) — triggered when mel > 3000 frames (30s) and no KV/prefix bridge:
- Plans ≤30s windows (`_plan_window_bounds`), each cut snapped to the minimum frame-energy point inside a ±100-frame band (never mid-word); tails <300 frames merge into the previous window
- Each window decoded separately (`_no_window=True` avoids recursion); segment times offset by window start; texts joined
- Window errors are ALWAYS surfaced in `result["error"]`, even when earlier windows produced text (otherwise failures look like silent truncation)

**Encoder-level chunking** (`transcribers/cohere.py`) — safety fallback inside a single decode when a window still exceeds the encoder limit:
- Mel split into overlapping windows (MAX_ENCODER_SEC=30s, 25% overlap), encoded independently, concatenated along sequence dimension
- Overlap trim must happen in OUTPUT space: `trim = min(round(overlap_frames * out_seq/in_len), out_seq)` (subsample-safe — input-frame counts don't match output-frame counts)

**Decoder interface**: The decoder ONNX model requires `encoder_hidden_states` as a direct input (not just KV caches). `_parse_encoder_outputs()` extracts the hidden states tensor from encoder output and passes it through. Cross-attention KV caches (8 layers × key/value) are initialized as empty (seq_len=0).

### Diarization Pipeline (13 steps)

Numbering below matches the `# Step N` comments in
`diarization/pipeline.py::Diarizer.run` — that file is the source of truth.

```
1.   VAD (Silero ONNX) → raw uncollapsed sections (kept for step 13)
1b.  Merge sections <0.5s apart for clustering
2.   Energy-dip splitting (min_segment_dur=3.0s, dip_ratio=0.35,
     min_dip_dur=0.5s, min_split_piece=2.0s)
3.   Sliding windows (2.0s window, 1.2s stride) → fbank features
4.   ECAPA-TDNN512 embedding per window (192-dim, ONNX CUDA; MD5 LRU cache)
5.   AgglomerativeClustering (cosine) — threshold path: average linkage at
     `distance_threshold`; forced-k path (`num_speakers` given):
     `n_clusters=k` with `forced_k_linkage` (default `complete`) and **no**
     greedy merge
5b.  Assign cluster labels to every segment, including short ones
5c.  Overlap detection (proximity_ratio 0.08, min_distance 0.40)
6.   Map labels → "Speaker N" and build segments (same-speaker merge,
     max_speaker_gap 1.0s; `build_overlap_segments`)
7.   Split single-speaker vs OVERLAP; absorb islands; boundary refinement
8.   Speaker profiling (pitch, energy, MFCC)
8b.  Relabel by pitch (highest = Speaker 1); inject cluster centroids
9.   Known-speaker matching (multi-feature: 60% embedding + 15% pitch +
     10% spectral + 10% MFCC) — BEFORE ghost elimination so it can populate
     `alternatives`
10.  Ghost elimination (<10s total speech → suppressed to UNKNOWN, or to a
     matched voiceprint alternative; `ghost_max_share` rescues a large share)
11.  Absorb/suppress minority speakers (max_utterance 5.0s, min_speaker_dur
     8.0s; matched voiceprints protected)
12.  Second-pass re-identification of unknown speakers (gated by
     `second_pass.min_identity_confidence`, one-to-one claim)
13.  Exact turn-boundary refinement from the raw VAD sections
```

### Audio Format Support
- Input: mp3, mp4, mkv, flac, ogg, m4a, wav, webm, opus
- Auto-converts to WAV PCM 16kHz mono via ffmpeg before processing
- Voiceprint snippets stored as **FLAC** (compressed, lossless)

### Browser Live Tab (`static/live.js`)
A third live client alongside `asr-client/live_client.py`, for `/live` in the SPA. It re-implements the wire protocol and the endpointing in JS, so both are drift-guarded by `tests/test_live_js_protocol.py` (constants vs. `protocol.py`, `DETECTOR_DEFAULTS` vs. `turn_detector.config()`, packed frames vs. `unpack_turn`, node-vs-server detector boundary equality, and `buildTranscript` vs. `transcribe_client.build_transcript`).

- **Capture** — `getUserMedia` (mono, AEC/NS) → `AudioWorkletNode` (`live-worklet.js`) that resamples to 16 kHz, **levels the input** (auto-gain, see below) and posts 1024-sample int16 blocks. Channel 1 is `getDisplayMedia({audio, video: true})` tab/system audio — Chrome only returns tab audio when video is requested, and a failed display-media call degrades to mic-only with a visible note.
- **Input levelling** — both live clients apply the same boost-only AGC in the capture chain, before the int16 conversion, so the frames on the wire AND the WAV re-uploaded for re-attribution carry the same level: `TARGET_RMS 0.125`, `MIN_ENV 0.0008`, `MAX_GAIN 16.0`, log-domain attack/release, per-block peak ceiling. `desired = TARGET_RMS / rms` — dividing by the *already-gained* level settles at `sqrt(TARGET/rms)`, which looks like convergence and is not. Constants, mechanism and the `--no-agc` flag: lesson 37.
- **Endpointing in JS** — `TurnDetector`/`TurnCoalescer` ports; the server runs neither on LVT1 frames (`handler.py:101` builds the `Turn` straight from the frame). One coalescer per channel, `poll()` every iteration.
- **Auth** — the browser WS cannot set headers, so the session cookie is used; `?token=` is the fallback (from the Settings "Windows Client Token" card).
- **Stop** — `MSG_FLUSH`, then wait for the server's `stats` frame (the drain signal) before building the transcript. A failed drain must not strand the tab: the transport buttons are restored in a `finally`, and a socket that dropped *after* transcripts arrived still gets downloads and the History save. Re-attribution via `/api/asr/attribution/upload` (no re-decode), optional `POST /api/asr/live/save` to History (text only — audio goes through the upload route, which is the capped one). Downloads are generated client-side: `.txt`, `.asr.json`, and per-channel `.wav`.
- **Per-channel turn split** (`Session.turnSplit()`, in the end-of-session note, the sidecar and the `.txt` header) is the one number that explains a bad transcript. `mic 0, speaker 12` means the mic recording has no speech for the re-attribution to diarize, so every item collapses onto one speakerless turn and the whole session comes back UNKNOWN with no reason shown.
- Live transcript items arrive in **decode-completion order** — sort by `start` before rendering or re-attributing.

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + auth routes
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe, /transcribe/upload (SSE), /live/save, WS /ws/stream
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list, DELETE /{name}
│   │   ├── voiceprint_router.py # CRUD + upload + merge + rename + rescan
│   │   ├── transcript_router.py # User-scoped transcript CRUD + download
│   │   ├── token_router.py     # GET/POST /token — client API token status + generation
│   │   ├── mcp_router.py      # MCP tools + resources + /call
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key auth + path validation
│   │   ├── auth.py            # htpasswd parse + session auth + require_auth
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging
│   ├── core/
│   │   ├── model_state.py     # ModelState, is_gpu_oom, log_gpu_memory, GPU_SHRINK_RUN_OPTIONS, run_embedding
│   │   ├── model_loader.py    # HuggingFace download + ORT session init (CUDA EP, arena caps, _preload_cuda_libs)
│   │   ├── job_state.py       # Active job tracking: start/publish/finish/attach + activity subscribe (import the MODULE)
│   │   └── transcriber.py     # Thin facade: forwards transcribe_audio_sync/mel to state.backend
│   ├── transcribers/
│   │   ├── base.py            # ASRBackend ABC (load/unload/is_loaded/transcribe_audio_sync)
│   │   ├── __init__.py        # BACKENDS_BY_NAME + get_backend(settings) factory
│   │   ├── cohere.py          # CohereBackend: ONNX encoder/decoder (windowed decode, chunking)
│   │   ├── qwen3.py           # Qwen3Backend: Qwen3-ASR via qwen_asr.Qwen3ASRModel (transformers)
│   │   └── whisper.py         # WhisperBackend: faster-whisper/CTranslate2, int8-quantized large-v3-turbo
│   ├── diarization/
│   │   ├── pipeline.py        # Diarizer: 13-step pipeline
│   │   ├── clustering.py      # AgglomerativeClustering, greedy merge, voiceprint matching
│   │   └── segment_ops.py     # Collapse, absorb islands, eliminate ghosts
│   ├── speaker/
│   │   ├── audio.py           # fbank, sliding windows, boundary refinement
│   │   ├── embedding.py       # ONNX embedding, batch embed, pitch, energy
│   │   ├── vad.py             # VAD (energy-dip splitting, chunked, ONNX)
│   │   ├── matcher.py         # Multi-feature distance (emb+pitch+energy+spectral+MFCC)
│   │   ├── profiling.py       # Pitch/energy/MFCC profiling, relabel by pitch
│   │   ├── uncertainty.py     # Uncertain-speaker policy (dependency-free, lesson 30)
│   │   ├── attribution.py     # Post-hoc span→turn attribution (pure, lesson 30)
│   │   └── service.py         # SpeakerService (SQLite-backed CRUD)
│   ├── voiceprint/
│   │   ├── service.py         # VoiceprintService (snippet CRUD, auto-collect, refine, merge)
│   │   └── utils.py           # Audio load/convert (ffmpeg), FLAC snippets
│   ├── db/
│   │   ├── models.py          # SQLAlchemy: voiceprints, snippets, sessions, transcripts
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, SnippetDB, SessionDB, TranscriptDB
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed) — initialized, no consumer (see History)
│   ├── streaming/
│   │   ├── turn_detector.py   # Adaptive energy turn detector + turn coalescer (lesson 31)
│   │   ├── speech_gate.py    # Silero speech gate before ASR (fail-open)
│   │   ├── protocol.py       # Turn-frame wire format (LVT1)
│   │   ├── attribution.py    # Channel-aware live speaker attribution
│   │   └── handler.py         # WebSocket dual-channel real-time transcription
│   ├── config/
│   │   ├── settings.py        # Pydantic BaseSettings (TRANSCRIBE_ prefix)
│   │   ├── logging.py         # Structured logging (stdlib)
│   │   └── thresholds.json    # All tunable diarization/matching/VAD params
│   ├── static/
│   │   ├── viewer.js          # Speaker colors, block rendering, transcript viewer
│   │   ├── live.js            # Browser live client: LVT1 packing, JS TurnDetector/TurnCoalescer, live pane UI
│   │   └── live-worklet.js    # AudioWorkletProcessor: getUserMedia/getDisplayMedia -> 16 kHz mono int16 blocks + input auto-gain
│   └── templates/
│       ├── _nav.html          # SPA tab bar snippet (__NAV__ + __ACT_*__ markers)
│       ├── login.html         # Dark-themed login form
│       └── app.html           # Unified SPA: Transcribe / Live / Voiceprints / History / Settings tabs
├── tests/                    # Pure-python unit tests (pytest, no GPU needed)
├── asr-client/
│   ├── transcribe_client.py    # Stdlib Windows client: SSE progress, <name>.txt output, voiceprints
│   │                           # .txt = profiles banner + [HH:MM:SS] Speaker (NN%): paragraphs —
│   │                           # format change ⇒ update mem-mcp process-transcription skill step 2
│   ├── transcribe.bat          # Drop-target wrapper: bootstraps .env + private .venv, runs client
│   ├── requirements.txt        # Empty (stdlib-only); installed into .venv when it has lines
│   ├── README.md               # Standalone-zip setup guide
│   └── .env.example            # SERVER_URL + TOKEN template
├── .github/workflows/asr-client.yml  # Builds standalone client zip on asr-client/** changes
│                                    # (stdlib-import guard + no server cross-refs; zip = artifact)
├── Dockerfile                 # nvidia/cuda:12.2.0 base
├── docker-compose.yml         # GPU passthrough + named volumes + port 8087
├── nginx_snippet.conf         # nginx location /asr-mcp/ with proxy_pass
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── AGENTS.md
├── docs/
│   ├── uncertain-speakers.md   # User-facing policy reference (JSON shape, reasons, gates)
│   └── lessons/                # Per-subsystem evidence behind each numbered rule
├── README.md
```

## SQLite Schema

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `voiceprints` | `(user_id, name)` composite | Speaker embeddings, pitch, energy, MFCC stats |
| `snippets` | `id` (INTEGER) | Auto-collected audio snippets (FLAC files) |
| `sessions` | `id` (TEXT) | Session data with TTL expiration |
| `transcripts` | `id` (INTEGER) | Stored transcription results |
| `api_tokens` | `user_id` (TEXT) | One client API token per user (SHA-256 hash) |

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/gui` | GET | Session | SPA — Transcribe tab (also serves /live, /voices, /transcriptions, /settings with different active tab) |
| `/live` | GET | Session | SPA — Live tab (browser real-time transcription, `static/live.js`) |
| `/voices` | GET | Session | SPA — Voiceprints tab |
| `/transcriptions` | GET | Session | SPA — History tab |
| `/settings` | GET | Session | SPA — Settings tab (status, client token, session) |
| `/login` | GET | No | Login page |
| `/api/asr/diarize` | POST | API key | Diarize audio by file path |
| `/api/asr/diarize/upload` | POST | Session | Diarize uploaded audio (`?num_speakers=` is a query param, not form) |
| `/api/asr/transcribe` | POST | API key | Transcribe by file path (`"language"` in the body, ISO 639-1 or `auto`, default `auto`) |
| `/api/asr/transcribe/upload` | POST | Session/API key | Transcribe uploaded audio (`?save=false` skips server-side transcript save; `?language=hu` ISO 639-1 or `auto`, default `auto`) |
| `/api/asr/languages` | GET | Session/API key | Static language list of the configured backend (no model load) — `{backend, supports_auto, languages:[{code,name}]}` |
| `/api/asr/activity/stream` | GET | Session/API key | SSE push of job start/finish (snapshot on connect + keep-alive pings; replaces polling `/active`) |
| `/api/asr/active/stream` | GET | Session/API key | SSE replay+follow of the current job; non-owners get events with `result` stripped |
| `/api/asr/active/cancel` | POST | Session/API key | Cancel your own active transcription job |
| `/api/asr/ws/stream` | WS | Session/API key | Real-time streaming transcription (auth: session, `X-API-Key` header or `?token=`; closes 1008 *before* `accept()`) |
| `/api/asr/stream` | POST | No | Always 501 — a stub that points at the WebSocket endpoint |
| `/api/asr/attribution` | POST | Session/API key | Re-attribute cached ASR items to speakers from a server-side `wav_path` — **no ASR** (`items` is a list of `{start, end, text, confidence}`) |
| `/api/asr/attribution/upload` | POST | Session/API key | Same, for a client-recorded upload: `file` + `items` (JSON array form field) |
| `/api/asr/live/save` | POST | Session/API key | Persist a browser Live-tab session to History: JSON `{audio_filename, result, stats?, sidecar?}` — text only, no audio field; stores `metadata.asr_source="live_stream"` |
| `/api/speaker/register` | POST | API key | Register voiceprint |
| `/api/speaker/register/upload` | POST | API key | Register voiceprint from upload |
| `/api/speaker/identify` | POST | API key | Identify speaker from audio |
| `/api/speaker/list` | GET | API key | List all voiceprints |
| `/api/speaker/{name}` | DELETE | API key | Delete voiceprint |
| `/api/voiceprint/speakers` | GET | Session | List speakers (web UI); each entry carries `pending` |
| `/api/voiceprint/speakers/{speaker_name}/snippets` | GET | Session | List snippets for a speaker |
| `/api/voiceprint/speakers/{speaker_name}/rename` | POST | Session | Rename speaker |
| `/api/voiceprint/speakers/merge` | POST | Session | Merge speakers |
| `/api/voiceprint/upload` | POST | Session | Register voiceprint from upload |
| `/api/voiceprint/rescan` | POST | Session | Rescan voices directory |
| `/api/voiceprint/rescan/stream` | POST | Session | Rescan as an SSE stream |
| `/api/voiceprint/pending` | GET | Session | Learned-but-unnamed speakers awaiting a name |
| `/api/voiceprint/pending/{speaker_name}/confirm` | POST | Session | Give a pending profile a real name — the only route that makes it matchable |
| `/api/voiceprint/snippets/{snippet_id}` | DELETE | Session | Delete one snippet |
| `/api/voiceprint/snippets/{snippet_id}/audio` | GET | Session | Stream snippet audio (FLAC) |
| `/api/voiceprint/speakers/{speaker_name}` | DELETE | Session | Delete a voiceprint |
| `/api/voiceprint/speakers/{name}/upload` | POST | Session/API key | Add snippet (optional `?start_sec=&end_sec=`) |
| `/api/voiceprint/speakers/{name}/refine` | POST | Session/API key | Rebuild voiceprint from snippets |
| `/api/token` | GET/POST | Session/API key | Client API token status / generate (one per user) |
| `/api/transcripts` | GET | Session | List all transcriptions for user |
| `/api/transcripts/{id}` | GET | Session | Get transcription by ID |
| `/api/transcripts/{id}/download` | GET | Session | Download as plain text |
| `/api/transcripts/{id}` | DELETE | Session | Delete transcription |
| `/api/auth/login` | POST | No | Login (JSON + session cookie) |
| `/api/auth/logout` | POST/GET | No | Logout |
| `/api/user` | GET | No | Current user info |
| `/api/mcp/tools` | GET | No | List MCP tools |
| `/api/mcp/resources` | GET | No | List MCP resources |
| `/api/mcp/call` | POST | No | Call MCP tool |
| `/api/mcp/resource/{uri}` | GET | No | Read one MCP resource |

## Environment Variables

**The clustering thresholds live in `asr_mcp/config/thresholds.json`, not in the
environment.** `TRANSCRIBE_DIARIZATION_THRESHOLD` is declared in
`config/settings.py` but never read — the pipeline reads
`thresholds.json → diarization.distance_threshold` (or the per-request
`diarization_threshold` field, which overrides it for one call). Editing
`.env.example` to change clustering behaviour has no effect.

`docker-compose.yml` only forwards the variables listed below; anything else
set in `.env` stays in the host and never reaches the container. Copy
`.env.example`, edit it, and add a `- TRANSCRIBE_X=${TRANSCRIBE_X}` line to
`docker-compose.yml` for anything new.

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_SESSION_SECRET` | (required) | Secret key for session cookies |
| `TRANSCRIBE_HTPASSWD_PATH` | `/app/htpasswd` | Path to htpasswd file (compose sets `/app/data/.htpasswd`) |
| `TRANSCRIBE_API_KEYS` | `[]` | JSON list of static API keys; a request with a matching `X-API-Key` runs as the `default` user |
| `TRANSCRIBE_PREFIX` | `""` | URL prefix for reverse proxy |
| `TRANSCRIBE_CUDA_DEVICE` | `cuda:0` | CUDA device ordinal |
| `TRANSCRIBE_HOST` | `0.0.0.0` | Server bind host |
| `TRANSCRIBE_PORT` | `8080` | Server port |
| `TRANSCRIBE_DATA_DIR` | `./data` | Data directory |
| `TRANSCRIBE_LOG_DIR` | `./logs` | Log directory |
| `TRANSCRIBE_VOICES_DIR` | `./voices` | Voiceprint snippets directory |
| `TRANSCRIBE_DB_PATH` | `./data/asr_mcp.db` | SQLite database path |
| `TRANSCRIBE_MODEL_CACHE_DIR` | `./models` | ONNX model cache |
| `TRANSCRIBE_VAD_THRESHOLD` | `0.5` | VAD speech probability cutoff (not forwarded by compose) |
| `TRANSCRIBE_MODEL_TTL_MINUTES` | `5` | Idle minutes before GPU models unload (0 = disabled); skipped while a job is active |
| `TRANSCRIBE_GPU_MEMORY_LIMIT_GB` | `4.0` | GPU size hint for CUDA arena caps (encoder = ×0.625, embedding = ÷4 capped at 768 MiB); not forwarded by compose |
| `TRANSCRIBE_LOG_LEVEL` | `INFO` | Log level |
| `TRANSCRIBE_ASR_MODEL` | `cohere` | ASR backend: `cohere` (ONNX, default), `qwen3-asr` (transformers/Qwen3-ASR) or `whisper` (faster-whisper) |
| `TRANSCRIBE_QWEN_MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | Qwen3-ASR model name (HF) |
| `TRANSCRIBE_QWEN_MODEL_DIR` | `./models/qwen3-asr` | Qwen3-ASR local cache dir |
| `TRANSCRIBE_QWEN_FORCED_ALIGNER_NAME` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner model name (HF) |
| `TRANSCRIBE_QWEN_FORCED_ALIGNER_DIR` | `./models/qwen3-forced-aligner` | Qwen3-Forced-Aligner local cache dir |
| `TRANSCRIBE_QWEN_TORCH_DTYPE` | `float16` | Torch dtype for Qwen3 model/aligner |
| `TRANSCRIBE_QWEN_MAX_NEW_TOKENS` | `256` | Max decode tokens for Qwen3-ASR |
| `TRANSCRIBE_QWEN_MAX_INFERENCE_BATCH_SIZE` | `8` | Qwen3-ASR inference batch size |
| `TRANSCRIBE_QWEN_QUANTIZE_4BIT` | `true` | Load Qwen3-ASR via BitsAndBytes load_in_4bit |
| `TRANSCRIBE_QWEN_ALIGNER_QUANTIZE_4BIT` | `true` | Load the Qwen3 aligner in 4-bit (saves ~0.9GB VRAM); fp16 fallback (not forwarded by compose) |
| `TRANSCRIBE_WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model size name or HF repo id |
| `TRANSCRIBE_WHISPER_MODEL_DIR` | `./models/faster-whisper` | Whisper local download dir (subdir per model spec) |
| `TRANSCRIBE_WHISPER_COMPUTE_TYPE` | `auto` | CTranslate2 compute type; `auto` = int8_float16 on CUDA / int8 on CPU |
| `TRANSCRIBE_WHISPER_BEAM_SIZE` | `5` | Whisper beam size |
| `TRANSCRIBE_WHISPER_VAD_FILTER` | `true` | faster-whisper built-in Silero VAD filter (disabled for live turns) |
| `TRANSCRIBE_WHISPER_CPU_THREADS` | `0` | Threads for Whisper CPU decode (0 = CT2 default; not forwarded by compose) |
| `TRANSCRIBE_HF_TOKEN` | - | HuggingFace token for gated models (not forwarded by compose) |

## Known Issues

- **CUDA OOM on large files**: arena caps + per-run shrinkage + reload-fresh-arena + CPU fallback now recover automatically (window may be slow on CPU, but no silent holes); reduce max_audio_duration_sec if still OOM
- **No CUDA available**: Server starts but models fail to load; API returns errors for ASR operations
- **SQLite locking**: Concurrent writes may fail under heavy load; WAL mode recommended for production
- **ffmpeg required**: Audio format conversion requires ffmpeg installed in Docker image

## Lessons Learned (Critical Behavior Rules)

These are hard-won bugs that **will** reappear if violated. The rule is stated
here; the evidence, measurements and mechanism live in the linked detail file.
Follow these when modifying any code — and read the detail file before changing
anything the rule covers.

### Diarization, speaker identity, uncertainty policy
| # | Rule | Detail |
|---|---|---|
| 1 | Speaker labels are ALWAYS `Speaker N`, 1-indexed — never `SPEAKER_00` | [diarization](docs/lessons/diarization-and-speakers.md) |
| 3 | Segments are speech-length, not window-length; windows are for embeddings only | [diarization](docs/lessons/diarization-and-speakers.md) |
| 4 | Merge nearby VAD regions (<0.5s gap) before splitting | [diarization](docs/lessons/diarization-and-speakers.md) |
| 11 | Reordering the pipeline requires updating ALL downstream code | [diarization](docs/lessons/diarization-and-speakers.md) |
| 29 | Auto-collect gates on the speaker's CUMULATIVE total, not a per-chunk cap | [diarization](docs/lessons/diarization-and-speakers.md) |
| 30 | Uncertain speaker: suppress the IDENTITY, keep the text; one policy module, `uncertainty.enabled: false` is the rollback | [diarization](docs/lessons/diarization-and-speakers.md) · [user docs](docs/uncertain-speakers.md) |
| 32 | `num_speakers` selects a DIFFERENT clustering path (hard k, no greedy merge) | [diarization](docs/lessons/diarization-and-speakers.md) |
| 33 | Verify diarization on a SHORT clip too — 52-min files hide over-clustering | [diarization](docs/lessons/diarization-and-speakers.md) |
| 39 | A learned-but-unnamed speaker is a PENDING profile: excluded from every match until the user names it, and the client must say so out loud | [diarization](docs/lessons/diarization-and-speakers.md) |

### ASR decoding internals (Cohere ONNX)
| # | Rule | Detail |
|---|---|---|
| 7 | Mel must be split into overlapping ≤30s windows for the encoder | [decoding](docs/lessons/asr-decoding.md) |
| 8 | KV cache names map `present.` → `past_key_values.` after step 0 | [decoding](docs/lessons/asr-decoding.md) |
| 12 | Prompt tokens come from `token_to_id` lookups in a fixed exact order | [decoding](docs/lessons/asr-decoding.md) |
| 13 | Mel features must match the reference pipeline exactly (nperseg/noverlap/mode/z-norm) | [decoding](docs/lessons/asr-decoding.md) |
| 14 | `attention_mask` covers full past + current length; `position_ids` offset by `past_seq_len` | [decoding](docs/lessons/asr-decoding.md) |
| 15 | Cohere timestamps are `<\|spltokenN\|>` tokens, not `<\|1.23\|>` | [decoding](docs/lessons/asr-decoding.md) |
| 16 | Decode long audio in ≤30s windows; never swallow a partial window error | [decoding](docs/lessons/asr-decoding.md) |

### GPU / ONNX Runtime
| # | Rule | Detail |
|---|---|---|
| 5 | ORT has no `ORTRuntimeError`; catch `Exception` and gate on `is_gpu_oom(e)` | [gpu](docs/lessons/gpu-and-onnx-runtime.md) |
| 6 | Embedding GPU OOM → CPU fallback with fbank chunking | [gpu](docs/lessons/gpu-and-onnx-runtime.md) |
| 27 | Arena discipline: cap at creation → shrink per run → clear the old session before reloading | [gpu](docs/lessons/gpu-and-onnx-runtime.md) |

### API, SSE and auth
| # | Rule | Detail |
|---|---|---|
| 9 | Long-running upload endpoints return `text/event-stream` | [api](docs/lessons/api-sse-and-auth.md) |
| 10 | Authenticated `fetch()` needs `credentials: 'same-origin'` | [api](docs/lessons/api-sse-and-auth.md) |
| 19 | SSE producers must yield (`sleep(0.01)`) AND run heavy work off-loop | [api](docs/lessons/api-sse-and-auth.md) |
| 20 | Last middleware added is outermost; do not reorder Session/Auth | [api](docs/lessons/api-sse-and-auth.md) |
| 25 | `stage: "done"` finishes the job — progress events must not use it | [api](docs/lessons/api-sse-and-auth.md) |
| 28 | `language` must be threaded router → prompt → decode, not forced to English | [api](docs/lessons/api-sse-and-auth.md) |
| 38 | A proxy refuses the body before the app sees it — a 413 may be HTML, and a 409 means "wait", not "failed" | [api](docs/lessons/api-sse-and-auth.md) |

### Streaming, frontend and output
| # | Rule | Detail |
|---|---|---|
| 17 | Every second of the timeline is covered by exactly one turn | [turns](docs/lessons/transcription-turns.md) |
| 18 | Exact turn boundaries come from uncollapsed VAD + voiceprint attribution | [turns](docs/lessons/transcription-turns.md) |
| 21 | CSS for JS-generated elements must not be scoped under classes JS never adds | [frontend](docs/lessons/frontend-and-client.md) |
| 22 | Display the REFINED segments, not raw diarization | [frontend](docs/lessons/frontend-and-client.md) |
| 23 | Progress UI needs per-window time interpolation | [frontend](docs/lessons/frontend-and-client.md) |
| 24 | Paragraph breaks at pauses snap to a sentence end within 40 chars | [frontend](docs/lessons/frontend-and-client.md) |
| 26 | Never pin `state.*_session`; resolve it live at call time | [frontend](docs/lessons/frontend-and-client.md) |
| 31 | Streaming: one speech state per turn, validate `len(data) >= 8`, bounded queue, coalesce short turns, gate non-speech, flush on disconnect | [streaming](docs/lessons/streaming-websocket.md) |
| 34 | Live: the mic channel IS the identity evidence — never voiceprint-match it; WS auth precedes `accept()` | [live design](docs/lessons/live-client-design.md) |
| 35 | `Turn.audio` is float32 on the server, int16 bytes on the client — normalise with `samples_as_float32`; live match gates are calibrated, not guessed | [live design](docs/lessons/live-client-design.md) |
| 36 | A second live channel requires LVT1 turn frames — the legacy raw-PCM path drops channel 1; browsers have no WASAPI loopback, so channel 1 is `getDisplayMedia` tab/system audio | [live design](docs/lessons/live-client-design.md) |
| 37 | Level the input ONCE in the capture chain (both clients, shared constants); `desired = TARGET/rms` — dividing by the already-gained level settles at a geometric mean that looks like convergence | [live design](docs/lessons/live-client-design.md) |

### Reference documents
- [docs/uncertain-speakers.md](docs/uncertain-speakers.md) — user-facing description of the
  uncertainty policy: JSON shape, every `attribution_reason`, both naming gates, and the
  known limitation on short recordings.
- [docs/lessons/](docs/lessons/) — one file per subsystem with the full evidence behind
  each rule above.
- [docs/lessons/live-client-design.md](docs/lessons/live-client-design.md) — why the
  live client's final text comes from the live ASR cache rather than a re-transcription,
  the two rejected alternatives, and the trigger for revisiting the decision.

## Historical notes (no longer true — kept so they are not re-introduced)

These were once accurate and shaped the code. They are archived rather than
deleted because each one explains a decision a reader would otherwise re-litigate.

- **Lesson 2, "pass the session through every call site"** — superseded by
  lesson 26. The current rule is the opposite: never pin a copy of
  `state.*_session`, resolve it at call time (`VoiceprintService._emb_session`).
  Threading a session through call sites was the old fix for the same
  `'NoneType' has no attribute get_inputs` failure that pinning reintroduced.
- **`merge_similar_speakers` / `close_match_threshold`** — the function is
  defined in `diarization/clustering.py` and the key in `thresholds.json`, and
  **neither is called**. Do not treat them as pipeline steps (step 5 uses
  `AgglomerativeClustering` directly). They are leftovers from an earlier
  post-clustering merge pass; the greedy merge in step 5 replaced them.
- **`sessions/manager.py` (`SessionManager`)** — constructed and `initialize()`d
  in `server.py` lifespan, but nothing reads it. The `sessions` table and this
  module are the pre-`transcripts` design; the web UI's History tab is backed by
  `transcripts`, not `sessions`. Remove the module and the lifespan call
  together if you touch it.
- **`POST /api/asr/stream`** — a 501 stub that predates the WebSocket endpoint
  and still points clients at `/ws/stream`. Kept only so old clients get a
  useful error instead of a 404.
- **`/api/mcp/*`** — mounted and reachable, but the tools are not wired to the
  transcription path. Present, not exercised.
