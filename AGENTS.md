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
2. Run `python -m pytest tests/ -q` — pure-python unit tests (uncertainty policy, attribution, turn detector, streaming handler, segment_ops). The `test_result_merge.py`, `test_clustering_linkage.py` and `test_second_pass_claim.py` modules `importorskip` torch/sklearn and are skipped outside the container; run them with `docker compose exec -T asr-mcp python3 -m pytest tests/ -q` after a rebuild. **Always run test suites through a subagent** (delegate the `docker compose cp` + `pytest` invocation to a `general` subagent), never inline in the main conversation — the suite output is long and running two suites concurrently OOM-kills the container
3. For threshold work, **cache the per-window embeddings** and sweep offline — see lesson 33
4. If running in Docker, rebuild: `docker compose up -d --build asr-mcp`
5. Run `git diff --check` after documentation or code edits
6. **A test fixture must PIN every `Settings` field the assertion depends on.**
   `Settings()` reads the environment, so a fixture that only overrides what it
   is testing inherits `TRANSCRIBE_PREFIX`, `TRANSCRIBE_EXTERNAL_URL`,
   `TRANSCRIBE_HTPASSWD_PATH`… from the running deployment. This has produced
   three separate wrong-result runs: an admin table that showed a live
   `memories` account (pinned `htpasswd_path` in `tests/test_admin_routes.py`),
   a Google redirect URI that carried `/asr-mcp/asr-mcp` (pinned `prefix` in
   `tests/test_google_routes.py`), and a confirmation link addressed to
   `mcp.itworks.hu` (pinned `prefix` and `external_url` in
   `tests/test_auth_routes.py`). A test that passes locally and fails in the
   image is a fixture bug, not a flake. The host has no pydantic at all, so
   these fixtures cannot be exercised outside the container — which is exactly
   why the ambient leak survives local runs.
7. **Assert a URL whole, never by substring.** `".../asr-mcp" in body` is
   satisfied by `.../asr-mcp/asr-mcp/api/auth/verify`, which is how lesson 51's
   doubled prefix shipped with a passing test.

## Architecture

| Component | Port | Notes |
|-----------|------|-------|
| App | 8087 (Docker), 8080 (internal) | FastAPI + ONNX Runtime CUDA |
| Nginx | /asr-mcp/ | Proxy mount point (trailing slash!) |

### Auth Flow
- Session-based auth via htpasswd file (bcrypt/apr1/sha/plain) — the legacy
  root, still the first store the login form consults
- SessionMiddleware cookie → AuthMiddleware innermost → public paths bypass
- Middleware order: AuthMiddleware (innermost) → SessionMiddleware → CORSMiddleware → logging
- `TRANSCRIBE_PREFIX` sets URL prefix for redirects behind reverse proxy
- **`get_current_user` order**: session → DB token → static key → 401. A
  **static** `TRANSCRIBE_API_KEYS` value resolves to a SYNTHETIC id
  `user_<sha256(key)[:16]>` (`api/security.py::resolve_api_key_user`), never to
  `default` — that is what keeps a static key's voiceprints out of the default
  user's library. `_websocket_user` calls the SAME resolver, so one key has one
  voiceprint scope on both transports. An unresolvable key returns `None`, so a
  caller refuses rather than falling back to a shared identity.
- **API token auth**: `X-API-Key` header with a DB token (or static `TRANSCRIBE_API_KEYS` value). `require_auth` returns 401 JSON for an invalid header (never a login redirect); valid header passes without a session. Tokens: `api_tokens` table, sha256-hashed, one per user (`TokenDB.create` replaces), plaintext returned once by `POST /api/token` (session UI card "Windows Client Token").
- **Open mode refused while registration is on**: when `TRANSCRIBE_API_KEYS` is
  empty, an unauthenticated caller would otherwise become `default` and share one
  library with every other anonymous caller. With `TRANSCRIBE_REGISTRATION_ENABLED`
  on, `get_current_user` answers 401 instead.
- **Registration** (`TRANSCRIBE_REGISTRATION_ENABLED`, off by default) lives in
  `asr_mcp/auth/` — `passwords.py` (stdlib scrypt, params travel inside the
  `scrypt$n$r$p$dklen$salt$hash` string so N can be raised), `mailer.py`, `store.py`
  (`CredentialStore` on the SQLAlchemy DB) and `api/auth_router.py`. Email sign-up
  is **throttle-then-flag** (10 attempts / 600 s per client, rightmost
  `X-Forwarded-For` hop; then the flag, 404 so a disabled route is
  indistinguishable from an unmounted one), creates an UNVERIFIED row, mails a
  single-use link (token stored as sha256 only), and **never signs anybody in**.
  A mail failure DELETES the row (503) — no half-made accounts.
- **Google sign-in** (`asr_mcp/auth/google_auth.py`, stdlib urllib, no JWT lib):
  `/api/auth/google/start` mints `secrets.token_urlsafe(24)` into
  `session["oauth_state"]` and is gated on **configuration, not the registration
  flag** (turning registration off must not sign out existing users). The callback
  blanks the state first (consumed either way), compares with `hmac.compare_digest`,
  exchanges the code and fetches userinfo. A Google sign-up's `user_id` **is the
  normalised email address** — Google already verified it, so no confirmation mail
  — and it is refused when `email_verified is not True`, when registration is off,
  or when the address is already a vault key. Identity is resolved by **subject,
  never by address**: `email_taken` deliberately does NOT consult
  `google_identities`, so a username signup for an address a Google vault uses is
  allowed and safe (the two accounts stay separate).
- **Admin** (`TRANSCRIBE_ADMIN_USER`, default empty ⇒ **nobody**): `is_admin_user`
  is a plain equality against the deployment constant; `require_admin` guards
  `GET /api/admin/users`, the disable/enable routes, `DELETE /api/admin/users/{id}`
  and `POST /shutdown`. An admin delete **keeps** the user's
  voiceprints/snippets/transcripts/audio and reports the counts, unless the
  opt-in `?purge=true` also drops the rows and `rmtree`s `voices/<user_id>/`.
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
Detail in [docs/processing.md](docs/processing.md#transcription).

- **ONE whole-file decode.** `run_transcribe` calls `transcribe_audio_sync()` once
  and the backend windows internally; ASR items are attributed onto the turns
  afterwards, so audio is never cut at a speaker boundary.
- **≤30 s decode windows**, cut at the minimum frame-energy point inside a ±100-frame
  band (never mid-word); tails <300 frames merge backwards.
- **A window error is ALWAYS surfaced in `result["error"]`**, even when earlier
  windows produced text — otherwise a failure reads as silent truncation (lesson 16).
- **Encoder chunking is a safety fallback**, and its overlap trim happens in
  OUTPUT space: `trim = min(round(overlap_frames * out_seq/in_len), out_seq)`
  (subsample-safe).
- **`language` is threaded** router → prompt → decode, never forced (lesson 28).

### Diarization Pipeline (13 steps)
Detail in [docs/processing.md](docs/processing.md#diarization-pipeline); numbering
matches the `# Step N` comments in `diarization/pipeline.py::Diarizer.run`.

1 VAD (raw sections kept for 13) · 1b merge <0.5 s · 2 energy-dip split ·
3 windows (embeddings only) · 4 ECAPA embedding · 5 clustering (threshold OR
forced-k) · 5b label short segments · 5c overlap · 6 `Speaker N` + merge ·
7 cleanup + boundary refine · 8 profiling · 8b relabel by pitch ·
9 **known-speaker matching** · 10 ghosts · 11 minorities · 12 second pass ·
13 exact turn-boundary refinement.

- **Step 9 runs BEFORE step 10** — matching must populate `alternatives` before
  ghosts are suppressed, or a ghost cluster is lost instead of rescued.
- **`num_speakers` is a different clustering path** (hard k, no greedy merge).
- **`Diarizer.run` never touches the DB.** `_load_known_speakers(settings, user_id)`
  in `asr_router` is the single funnel, and it excludes pending profiles there.
- **Segments are speech-length, not window-length** (lesson 3); **merge VAD
  regions <0.5 s apart before splitting** (lesson 4).

### Audio Format Support
Input: mp3, mp4, mkv, flac, ogg, m4a, wav, webm, opus — auto-converted to WAV PCM
16 kHz mono via ffmpeg. **Snippets are stored as FLAC** (lossless, compressed).

### Voiceprint learning: the cohesion split RE-JOINS, it does not refuse
Detail in [docs/voiceprints.md](docs/voiceprints.md#how-a-profile-is-created).

`_split_cluster_by_cohesion` embeds each span separately and re-clusters at
`learning.max_intra_cluster_dist` before learning, because a diarization cluster
is not a person. If the resulting groups sit closer than `learning.min_inter_dist`
that split is an **artefact and the groups are merged back into ONE part** — which
also brings back the dropped one-span groups. Refusing used to discard the
largest and richest clusters: on `2026-10-07_17-00-32.mp4` fourteen clusters were
refused at 0.175–0.281 apart while two profiles learned from the same run had
intra-group means of 0.170 and 0.179. **Too close is evidence of one voice.**

- **There is NO maximum-based purity gate, on any group, re-joined or not.** A max
  over n(n-1)/2 pairs grows with span count, so it rises with the richness of the
  cluster — which is why gating on it lost every cluster of
  `2026-09-04 12-08-59.mp4` (22 spans, mean 0.21, max 0.379). A real speaker the
  user had already learned (`Speaker 9`, 20 spans) scores mean 0.255 / max **0.463**
  once re-joined; a synthetic blend is mean 0.252 / max 0.42. **Neither statistic
  separates them**, so any such threshold could only choose which real speakers to
  discard, and it chose the largest.
- **The merge window and the blend window are disjoint.** The re-join fires only
  when every group's nearest neighbour is under `min_inter_dist`; ZO249 — the real
  blend this exists for — had its groups 0.368 apart, so it splits and never merges.
  Undoing the merge cannot smuggle a blend past the split.
- **`min_inter_dist` is now a MERGE TRIGGER, not a refusal bar.** Lowering it
  merges more aggressively; raising it above `max_intra_cluster_dist` mostly
  disables the re-join. Lowering merges more; there is no separate refusal.
- **The only refusals** are `too_few_segments`, `too_few_embeddings`,
  `no_embedding_session`, `no_sklearn`, `clustering_failed` — all fail-closed via
  `learning.fail_closed`.
- **Test fixtures must be a WIDE-SPREAD SINGLE VOICE** (`_one_voice(jitter=0.68)`
  over `_fan_spans(34)`). A two-voice fixture proves nothing: with both thresholds
  at 0.32 the merge window is unreachable for two CLEAN voices, because average
  linkage splits on the MEAN cross-pair distance while the check compares centroids,
  and for clean voices those coincide.

### Voiceprint Integrity: audit, automatic rebuild, cleanup
Detail in [docs/voiceprints.md](docs/voiceprints.md#integrity-audit-and-automatic-rebuild).

`voiceprint/audit.py` is read-only and reports 8 findings; `rebuild.py` +
`cleanup_integrity` are the two halves that act. **A finding with ONE correct answer
is repaired automatically; anything needing a human decision is only reported.**

- `TARGET_FINDINGS = ("snippets_without_profile", "profiles_without_snippets",
  "missing_snippet_files")`. NOT targets: `near_duplicate_profiles`,
  `name_collisions`, `duplicate_snippet_content`, `orphan_snippet_files`,
  `stale_speech_totals` — merging is the user's decision.
- **Purge first, rebuild second.** `_purge_dead_snippet_rows` deletes ROWS ONLY (no
  unlink — the audio is already gone; a file that resolves must never be unlinked).
  The first boot of this feature re-embedded 219 snippets (~25 min of GPU) for a
  `missing_snippet_files` finding and fixed nothing, because 73 rows pointed at
  `.wav` files left over from a wav→flac migration.
- **Why dropping the derived row is safe**: the snippets and their FLAC audio are
  the source of truth and are NEVER touched; only the embedding row is recomputed.
- **`pending` and `purity` are read BEFORE the delete and re-applied after**,
  because a `deleted row` destroys them and `VoiceprintDB.save(pending=None)`
  preserves the flag — without the re-apply a learned-but-unnamed profile would
  silently become matchable (lesson 39).
- **Convergence**: `MAX_ATTEMPTS = 3`, then the entry is dropped with an ERROR
  naming `POST /api/voiceprint/integrity/cleanup`; a `skipped` result is NOT an
  attempt.
- The TTL monitor's 60 s tick drains the queue only when it is non-empty AND
  `job_state.get_running() is None`, calling `ensure_diarize_ready()` (VAD +
  ECAPA, **never** the ASR backend) — the documented exception to "no model load
  at startup".

### Voiceprint snippets: ten per file, five seconds minimum
Detail in [docs/voiceprints.md](docs/voiceprints.md#the-per-file-snippet-budget).

- **Per FILE, per profile.** `AUTO_COLLECT_MAX_PER_SOURCE = 10`,
  `AUTO_COLLECT_MIN_DURATION = 5.0`. The per-file cap is what stops one long
  recording from BECOMING a profile: a 50-minute meeting attributed to one
  colleague for 40 of those minutes yields forty clips from one room. Neither
  `AUTO_COLLECT_MAX_TOTAL_SEC` (1200s, cumulative) nor the
  one-file-one-contribution rule covers that case, and the cap is independent of
  the cumulative one in both directions.
- **Best fit first.** `span_fit_key(seg)` ranks by `speaker_confidence` with
  duration as the tiebreak — free, no embedding pass. A span with NO recorded
  verdict (a generic `Speaker N`) is **1.0, not 0.0**: it is an unranked cluster
  rather than a confident mismatch, and 0.0 would starve the learn path of the
  budget it has to spend. Stored rows keep no confidence, so the cleanup ranks
  by duration (longest, then earliest, then lowest id).
- **The 5s floor is measured on the STORED clip, AFTER the VAD trim.** A 5.2s
  turn with a silent head is a 4.6s snippet. A trailing sliver left under
  `AUTO_COLLECT_MAX_SEGMENT_SEC` (300s) or `learning.max_snippet_sec` is dropped
  rather than extended into the next chunk — extending would break the
  per-snippet cap.
- **The live save path shares the floor.** `/live/save` and the upload routes all
  funnel through `asr_router.py::_auto_collect` (946, 1400, 1840), so a sub-5s live
  turn feeds no snippet on any transport. This is a deliberate decision, not an
  oversight: a turn too short to embed is too short to name, whether it arrived
  over the WebSocket or in a file.
- **`trim_snippets` in `cleanup_integrity` applies the budget backwards.** Without
  it the policy only ever reaches the file you re-process next. It runs BEFORE
  the totals are recomputed so its doomed ids feed `_surviving_speech_seconds`,
  and the only rows it NEVER touches are the ones that name no recording at
  all — no stored `source_audio` **and** a filename with no hash in it. A row
  that lost its `source_audio` is still trimmed: `snippet_source_identity`
  recovers the hash `add_snippet` wrote into the filename, so it groups with the
  rows that kept their path, and a reprocess replaces it too. `<hash>_manual.flac`
  hashes the AUDIO, not a recording, so it is not an identity.

## Voiceprint Consolidation: outlier snippets and manual review
Detail in [docs/voiceprints.md](docs/voiceprints.md#consolidation-finding-snippets-that-are-not-this-speaker).

`consolidate_speaker` answers "is every snippet of this profile actually the same
person?". It **flags** the outliers and recomputes from the **kept** snippets only;
**it deletes nothing.** Removal is a separate, dry-run-by-default step.

- **One embedding pass, shared with refine.** `_embed_snippets(...)` loads and
  embeds every usable snippet once; `_auto_refine` and consolidation both consume
  it. Do not reintroduce a direct `extract_embedding()` call in either — a
  219-snippet re-embed is ~25 min of GPU.
- **The detector is a PURE module-level function** `score_outliers(vectors, cfg=None)`,
  iterated MAD reweighting: each round recomputes the centre from the **kept**
  vectors only (one centre over everything is dragged toward the outliers).
- **Accusation requires BOTH** `median + k·MAD·1.4826` (`outlier_mad_k`) **and** an
  absolute floor (`outlier_floor` = **0.35**, the naming bar). The floor is
  deliberately NOT the 0.32 that `learning` reuses (lesson 43).
- **Guardrails**: `outlier_min_keep` (3) accuses nobody below it;
  `outlier_max_fraction` (0.2) is a **whole-pass** budget enforced worst-first and
  floors at 1; `outlier_mad_rounds: 0` = report-only.
- **Marks are durable evidence, not a boolean** (`snippets.review_flag` /
  `review_score` / `review_reason`). Re-running clears marks it no longer makes.
- **Removal never calls `delete_snippet`** (it re-refines per call → N flagged
  snippets = N full re-embeds). `remove_flagged_snippets` deletes via
  `_delete_snippet_row`, then does ONE `_auto_refine` per touched speaker.
- **`POST /consolidate/stream` registers a `job_state` job** — `_auto_refine`
  returns SILENTLY when the embedding session is gone, and the TTL monitor unloads
  it whenever no job is running; a sweep that does not register can have ECAPA
  pulled mid-pass and report success having done nothing.

### Browser Live Tab (`static/live.js`)
Detail in [docs/ui.md](docs/ui.md#live-tab).

- `getUserMedia` (mono, AEC/NS) → `AudioWorkletNode` (`live-worklet.js`) that
  resamples to 16 kHz, **levels the input** (auto-gain, see lesson 37) and posts
  1024-sample int16 blocks.
- **Channel 1 is `getDisplayMedia({audio, video: true})`** tab/system audio — Chrome
  only returns tab audio when video is requested; a failed display-media call
  degrades to mic-only with a visible note.
- The server runs **neither** detector on LVT1 frames (`handler.py:101` builds the
  `Turn` straight from the frame). One coalescer per channel; `poll()` each iteration.
- **Stop**: `MSG_FLUSH`, then wait for the server's `stats` frame (the drain signal)
  before building the transcript. A failed drain must not strand the tab — restore
  the transport buttons in a `finally`; a socket that dropped after transcripts
  arrived still gets downloads and the History save.
- Downloads are client-side: `.txt`, `.asr.json`, per-channel `.wav`.
- **`Session.turnSplit()`** is the one number that explains a bad transcript:
  `mic 0, speaker 12` means the mic recording has no speech for the
  re-attribution to diarize, so the whole session comes back UNKNOWN.
- Transcript items arrive in **decode-completion order** — sort by `start` before
  rendering or re-attributing.

### Frontend: one stylesheet, one design system
Detail in [docs/ui.md](docs/ui.md#design-system).

- All CSS is inline in `templates/app.html` (`<style>`, comments stripped by the
  tests before matching). No CSS file, no build step.
- **A pane composes shared classes; it does not declare its own.** The primitives
  are `.pill`, `.note`, `.mini-bar` + `.meter-row`, `.text-pane` +
  `.pane-short|tall|live`, `.btn` + `.row`/`.row-spread`, `.badge`, `.toast`,
  `.hidden` + `show(id, on)`. Status colours are `:root` tokens and must not be
  re-hardcoded.
- **Two max-width breakpoints only: 860px and 560px.** Base rules are the desktop
  layout, so a `min-width` query is a rule violation. The 560px block is the phone
  layout; `<meta name="viewport">` carries `viewport-fit=cover` or the safe-area
  padding is inert.
- **No inline styles.** Visibility is `.hidden` plus `show()`; no `style.display`
  writes and no `style="..."` in markup.
- `tests/test_ui_design_system.py` pins all of this in both directions (no orphan
  class either way, no private primitive, no inline style, token-only colours,
  exactly two breakpoints).

### Voiceprints tab: two sub-tabs
Detail in [docs/ui.md](docs/ui.md#voiceprints-tab).

- `.pill-tab` is ONE primitive for both strips: the header nav (`_nav.html`
  anchors) and the in-pane sub-tabs (buttons). The old `.tabs a` rule was replaced,
  and the test asserts it is gone — an anchor-scoped rule is what a button sub-tab
  would silently miss.
- **An unnamed profile is a fold** exactly like a named one: same endpoint, same
  player, same remove button. Its audio is the only evidence of who the voice is,
  so it must be playable **in place**. The action row differs on purpose — naming,
  merging and discarding are identity decisions, and only an unnamed profile has a
  "this is ___" input.
- **`snippetListHtml` + `loadSnippetBody` are the single renderer/loader pair**;
  the test pins one definition, one call in the loader, and one call from each fold.
- **Every list is sorted by name** (`byName`, `numeric: true`) so `Speaker 2`
  precedes `Speaker 10`. This applies to the known list, the unnamed list and the
  flagged card's speaker groups.
- **A redraw must not close what the user opened.** `refresh()` captures the open
  fold ids (`captureOpenFolds`) and reopens them afterwards (`restoreOpenFolds`).
  Reviewing means open a profile, remove three snippets, look again.
- **A job in another pane must not hijack the tab.** `/activity/stream` is global,
  so a `consolidate` or `rescan` job reaches the Transcribe pane's
  `attachActiveJob`; the mode is checked against `TRANSCRIPTION_MODES` **before**
  any tab switch. `showVoiceprintSubtab` only toggles visibility and NEVER
  re-renders, for the same reason as the folds.

### Markup inside a JS template literal is not validated by anything
An `onclick="f(${jsStr(n)})""` — a doubled closing quote, which is what a careless
search-and-replace of `'${escHtml(x)}'` into `${jsStr(x)}` produces — silently
kills the button. `node --check` parses past it (the markup is inside a template
literal), and the design-system tests did not catch it either.
`test_no_html_attribute_is_closed_twice` scans for `""` and `">>` outside
`<style>`/`<script>`/line comments. See [docs/ui.md](docs/ui.md#known-traps).

## Project Structure

```
asr-mcp/
├── asr_mcp/
│   ├── server.py              # FastAPI app + lifespan + POST /shutdown (require_admin)
│   ├── api/
│   │   ├── router.py          # Combines sub-routers under /api
│   │   ├── asr_router.py      # POST /asr/diarize, /transcribe, /transcribe/upload (SSE), /live/save, WS /ws/stream
│   │   ├── speaker_router.py  # POST /speaker/register, /identify, GET /list, DELETE /{name}
│   │   ├── voiceprint_router.py # CRUD + upload + merge + rename + rescan
│   │   ├── transcript_router.py # User-scoped transcript CRUD + download
│   │   ├── token_router.py     # GET/POST /token — client API token status + generation
│   │   ├── admin_router.py    # /admin/users list + disable/enable/delete (require_admin)
│   │   ├── auth_router.py     # login/logout/register/verify + Google sign-in
│   │   ├── mcp_router.py      # MCP tools + resources + /call
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── security.py        # API key auth + is_admin_user/require_admin + path validation
│   │   ├── auth.py            # htpasswd parse + session auth + require_auth
│   │   ├── exceptions.py      # Custom exceptions + handlers
│   │   └── middleware.py      # Request logging
│   ├── auth/
│   │   ├── passwords.py       # scrypt hash/check + username/email normalisation (stdlib only)
│   │   ├── mailer.py          # smtp_configured + verification_email + send_mail (injectable client)
│   │   ├── store.py           # CredentialStore on the SQLAlchemy DB (accounts, throttle, google identities)
│   │   └── google_auth.py     # OAuth urls/callback + authorization_url/exchange_code/fetch_userinfo
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
│   │   ├── boundary.py        # THE boundary engine — cut_between (pure, lesson 40)
│   │   └── service.py         # SpeakerService (SQLite-backed CRUD)
│   ├── voiceprint/
│   │   ├── service.py         # VoiceprintService (snippet CRUD, auto-collect, refine, merge)
│   │   ├── audit.py           # Read-only integrity audit (8 findings)
│   │   ├── rebuild.py         # Automatic from-scratch profile rebuild + its JSON queue
│   │   └── utils.py           # Audio load/convert (ffmpeg), FLAC snippets
│   ├── db/
│   │   ├── models.py          # SQLAlchemy: voiceprints, snippets, sessions, transcripts
│   │   └── manager.py         # DatabaseManager, VoiceprintDB, SnippetDB, SessionDB, TranscriptDB
│   ├── sessions/
│   │   └── manager.py         # SessionManager (SQLite-backed) — initialized, no consumer (see History)
│   ├── streaming/
│   │   ├── turn_detector.py   # Adaptive energy turn detector + turn coalescer (lesson 31)
│   │   ├── speech_gate.py    # Silero speech gate + turn-edge trim before ASR (fail-open, lesson 42)
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
│       ├── login.html         # Login + registration form + Google button
│       └── app.html           # Unified SPA: Transcribe / Live / Voiceprints / History / Settings tabs
├── tests/                    # Pure-python unit tests (pytest, no GPU needed)
│                            # test_ui_design_system.py = the CSS design system
│                            # test_live_js_protocol.py = the browser live client
│                            # test_voiceprint_consolidation.py = outlier snippets (load by path)
│                            # test_voiceprint_panes.py = the Voiceprints tab's two sub-tabs
│                            # test_user_scoping.py = per-user isolation (P1)
│                            # test_auth_credentials.py = the credential store; test_auth_routes.py = email sign-up
│                            # test_google_auth.py + test_google_routes.py = the OAuth client + routes
│                            # test_admin_routes.py = the admin gate, user table, delete + /shutdown
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
├── INSTALLATION.md           # install, reverse proxy, every server setting, auth setup
├── AGENTS.md
├── docs/
│   ├── api.md                  # Every route + its auth + cross-cutting conventions
│   ├── processing.md           # Diarization, turns, boundaries, ASR, live streaming
│   ├── voiceprints.md          # Profile creation, pending, integrity, consolidation
│   ├── ui.md                   # The SPA, its design system, its known traps
│   ├── uncertain-speakers.md   # User-facing policy reference (JSON shape, reasons, gates)
│   ├── boundary-measurements.md # The boundary A/B measurement + what is uncalibrated
│   └── lessons/                # Per-subsystem evidence behind each numbered rule
├── README.md                   # Overview + pointers to every doc above
├── INSTALLATION.md             # Install, reverse proxy, every server setting, auth setup
```

## SQLite Schema

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `voiceprints` | `(user_id, name)` composite | Speaker embeddings, pitch, energy, MFCC stats |
| `snippets` | `id` (INTEGER) | Auto-collected audio snippets (FLAC files); `review_flag`/`review_score`/`review_reason` hold the consolidation marks |
| `sessions` | `id` (TEXT) | Session data with TTL expiration |
| `transcripts` | `id` (INTEGER) | Stored transcription results |
| `api_tokens` | `user_id` (TEXT) | One client API token per user (SHA-256 hash) |
| `credentials` | `user_id` (TEXT) | Email-registered accounts: scrypt `password_hash`, `email` (partial UNIQUE index where `email <> ''`), `email_verified_at`, `verification_token` (sha256 only), `disabled_at` |
| `google_identities` | `(provider, subject)` | One Google subject → exactly one `user_id` forever; `display_name`, `last_used_at`. NOT keyed on the email |

## API Endpoints

**The full route table — every endpoint, its auth and its conventions — is in
[docs/api.md](docs/api.md).** It is maintained there and is not duplicated here.

The rules that govern all of them:

- **Identity is a STRING resolved per request** (`get_current_user`, see Auth
  Flow above). Every DB accessor defaults to `DEFAULT_USER`, so a call site that
  forgets `user_id=` silently reads another identity's rows instead of failing.
- **Per-user scoping.** speakers, voiceprints, snippets, transcripts and the
  audio under `voices/<user_id>/` all belong to the resolved identity.
  `/api/asr/active/stream` is a 404 for another user's job; `/activity/stream`
  reports only your own; `/active/cancel` only cancels your own.
- **Admin routes** (`/api/admin/*`) and `POST /shutdown` require `require_admin`,
  which is a plain equality against `TRANSCRIBE_ADMIN_USER`. Empty means **nobody**.
- **SSE routes** yield and run heavy work off-loop (lesson 19), and `stage: "done"`
  finishes the job, so a progress event must never use it (lesson 25).
- **Long uploads return `text/event-stream`**, never JSON.


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
| `TRANSCRIBE_API_KEYS` | `[]` | JSON list of static API keys; a request with a matching `X-API-Key` runs as the synthetic user `user_<sha256(key)[:16]>`, never `default` |
| `TRANSCRIBE_PREFIX` | `""` | URL prefix for reverse proxy |
| `TRANSCRIBE_REGISTRATION_ENABLED` | `false` | Offer email/Google sign-up on the login page; also forces auth (open mode is refused while it is on) |
| `TRANSCRIBE_GOOGLE_CLIENT_ID` | `""` | Google OAuth client id (both halves required, or `/api/auth/google/start` is a 404) |
| `TRANSCRIBE_GOOGLE_CLIENT_SECRET` | `""` | Google OAuth client secret |
| `TRANSCRIBE_EXTERNAL_URL` | `""` | Public base URL for the verification mail and the Google `redirect_uri`; falls back to `request.base_url`. May include `TRANSCRIBE_PREFIX` or not — `_public_base` adds the prefix only when absent (lesson 51) |
| `TRANSCRIBE_SMTP_HOST` | `""` | SMTP host — email sign-up needs this AND `TRANSCRIBE_SMTP_FROM` |
| `TRANSCRIBE_SMTP_PORT` | `587` | SMTP port |
| `TRANSCRIBE_SMTP_USER` | `""` | SMTP user (login is skipped when empty — a relay on a trusted network may take none) |
| `TRANSCRIBE_SMTP_PASSWORD` | `""` | SMTP password |
| `TRANSCRIBE_SMTP_FROM` | `""` | Envelope sender for the confirmation mail |
| `TRANSCRIBE_SMTP_STARTTLS` | `true` | Issue STARTTLS before the login |
| `TRANSCRIBE_SMTP_TIMEOUT` | `20.0` | SMTP socket timeout (seconds) |
| `TRANSCRIBE_ADMIN_USER` | `""` | The one account allowed to reach `/api/admin/*` and `POST /shutdown`; empty means **nobody** |
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

- **Shared GPU**: arena caps come from the card's *total* size, so another workload on the same device can leave too little free and cuBLAS fails at `cublasCreate()` (lesson 44). Services now fall back to CPU rather than dying, but that is slow — free the VRAM for real throughput
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
| 39 | A learned-but-unnamed speaker is a PENDING profile: excluded from every match until the user names it (or merges it into someone they already have), the client must say so out loud, and the UI must let the user LISTEN to it — an unnamed profile whose audio cannot be played is a row of counts, not an identification | [diarization](docs/lessons/diarization-and-speakers.md) |
| 40 | ONE boundary engine (`speaker/boundary.py::cut_between`) decides every cut; a cluster is not a person, so naming is withheld when the largest cluster owns the speech | [diarization](docs/lessons/diarization-and-speakers.md) |
| 41 | Learn a voiceprint from the boundaries the TRANSCRIPT uses (trimmed to whole VAD sections) — a mid-word snippet becomes a reference for cutting the next run's boundaries | [diarization](docs/lessons/diarization-and-speakers.md) |
| 43 | Naming reads a DISTANCE and learning reads a CONFIDENCE; a match in the 0.325–0.35 band gets the speaker's name and is refused for learning, on purpose. Learn from `docs/lessons/diarization-and-speakers.md` before touching either threshold. | [diarization](docs/lessons/diarization-and-speakers.md) · [user docs](docs/uncertain-speakers.md) |
| 45 | Flagging an outlier snippet is an ACCUSATION: conservative by construction (MAD cut AND the 0.35 naming floor), never a deletion — flag, let the user keep or remove, and never let a bulk delete re-embed per snippet. One embedding pass feeds both refine and consolidation. | [diarization](docs/lessons/diarization-and-speakers.md) |
| 46 | A cohesion split whose groups come out too close is RE-JOINED into one part, not refused — refusing discarded the largest clusters on every file, and a split closer than a single voice's own spread is evidence of one voice. Do NOT gate any group, re-joined or not, on its maximum pairwise distance: it grows with the span count, and a real speaker the user had already learned scored higher on it than the synthetic blend the gate was built to reject. | [diarization](docs/lessons/diarization-and-speakers.md) |

### Identity, registration and admin
| # | Rule | Detail |
|---|---|---|
| 47 | Identity is a STRING resolved per request and threaded down: session → DB token → static key (`user_<sha256(key)[:16]>`, **never** `default`) → 401. Every DB accessor defaults to `DEFAULT_USER`, so a call site that forgets `user_id=` silently reads another identity's rows instead of failing. `_websocket_user` and `get_current_user` share ONE resolver (`resolve_api_key_user`) so one key has one scope on both transports. | [auth](docs/lessons/identity-registration-admin.md) |
| 48 | Email sign-up is **throttle-then-flag** (10/600 s per client, rightmost `X-Forwarded-For` hop; then the flag, **404** so a disabled route is indistinguishable from an unmounted one), creates an UNVERIFIED row, and **never signs anybody in**. A mail failure DELETES the row (503). An unverified or disabled account is a **403 with a reason**; a wrong password is a 401. | [auth](docs/lessons/identity-registration-admin.md) |
| 49 | A Google sign-up's `user_id` IS the normalised email address, resolved by **subject, never by address**. `email_taken` deliberately does NOT consult `google_identities`, so a username signup for an address a Google vault uses is allowed and safe. `/api/auth/google/start` is gated on **configuration, not the registration flag** — turning registration off must not sign out existing users. The `oauth_state` lives in the server-side session and is blanked **before** it is compared (consumed either way). | [auth](docs/lessons/identity-registration-admin.md) |
| 50 | Authorization is a DEPLOYMENT CONSTANT, not a role column: `TRANSCRIBE_ADMIN_USER` (default empty ⇒ **nobody**) is a plain equality. An admin delete **keeps** data and reports counts unless `?purge=true` is passed explicitly — one boolean argument must not be able to destroy another user's recordings. `/shutdown` is gated on `require_admin` because stopping the process cancels other users' jobs. | [auth](docs/lessons/identity-registration-admin.md) |
| 51 | An absolute URL must carry the URL prefix EXACTLY once, however the base was spelled. `TRANSCRIBE_EXTERNAL_URL` is set either way in practice (`https://host` or `https://host/asr-mcp`), so `_public_base` appends the prefix only when the base does not already end with it **on a path-segment boundary**. Unconditional appending built `/asr-mcp/asr-mcp/...`, and the symptom was a 404 on the confirmation link — indistinguishable, to the user, from a token being refused. The confirmation link and the Google redirect URI must be built by the SAME helper. | [auth](docs/lessons/identity-registration-admin.md) |

### GPU sharing
| # | Rule | Detail |
|---|---|---|
| 44 | `is_gpu_oom` must recognise every wording a runtime uses, INCLUDING a load-time failure — cuBLAS fails at `cublasCreate()` with `CUBLAS_STATUS_ALLOC_FAILED`, which mentions no allocation of ours, and an unbuilt session kills every embedding endpoint | [gpu](docs/lessons/gpu-and-onnx-runtime.md) |

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
| 21 | CSS for JS-generated elements must not be scoped under classes JS never adds — and a pane must *compose* the shared classes, not declare a private copy | [frontend](docs/lessons/frontend-and-client.md) |
| 22 | Display the REFINED segments, not raw diarization | [frontend](docs/lessons/frontend-and-client.md) |
| 23 | Progress UI needs per-window time interpolation | [frontend](docs/lessons/frontend-and-client.md) |
| 24 | Paragraph breaks at pauses snap to a sentence end within 40 chars | [frontend](docs/lessons/frontend-and-client.md) |
| 26 | Never pin `state.*_session`; resolve it live at call time | [frontend](docs/lessons/frontend-and-client.md) |
| 31 | Streaming: one speech state per turn, validate `len(data) >= 8`, bounded queue, coalesce short turns, gate non-speech, flush on disconnect | [streaming](docs/lessons/streaming-websocket.md) |
| 34 | Live: the mic channel IS the identity evidence — never voiceprint-match it; WS auth precedes `accept()` | [live design](docs/lessons/live-client-design.md) |
| 35 | `Turn.audio` is float32 on the server, int16 bytes on the client — normalise with `samples_as_float32`; live match gates are calibrated, not guessed | [live design](docs/lessons/live-client-design.md) |
| 36 | A second live channel requires LVT1 turn frames — the legacy raw-PCM path drops channel 1; browsers have no WASAPI loopback, so channel 1 is `getDisplayMedia` tab/system audio | [live design](docs/lessons/live-client-design.md) |
| 37 | Level the input ONCE in the capture chain (both clients, shared constants); `desired = TARGET/rms` — dividing by the already-gained level settles at a geometric mean that looks like convergence | [live design](docs/lessons/live-client-design.md) |
| 42 | A live boundary is the client's CLAIM and the server's DECISION: Silero trims the edges, a timeline guard refuses a `start_sample` that goes backwards, and a merged turn carries its gap as silence so the declared span equals the audio sent | [live design](docs/lessons/live-client-design.md) |

### Reference documents

**Topic documentation** — what a subsystem does, written for a reader:

| Document | Covers |
|---|---|
| [docs/api.md](docs/api.md) | Every route, its authentication, and the conventions that hold across all of them |
| [docs/processing.md](docs/processing.md) | The request shape, the 13-step pipeline, turn preparation, the boundary engine, the three ASR backends, live streaming, progress events |
| [docs/voiceprints.md](docs/voiceprints.md) | Profile creation, pending profiles, one-file-one-contribution, the integrity audit and rebuild, outlier consolidation, naming-vs-learning |
| [docs/ui.md](docs/ui.md) | The five tabs, the design system, and the traps (`jsStr` in `onclick`, template-literal markup, redraws that close folds) |
| [docs/uncertain-speakers.md](docs/uncertain-speakers.md) | User-facing description of the uncertainty policy: JSON shape, every `attribution_reason`, both naming gates, and the known limitation on short recordings |
| [INSTALLATION.md](INSTALLATION.md) | Install, reverse proxy, every server setting and what it does, and how to configure each authentication method |
| [README.md](README.md) | Overview and pointers; nothing is documented there that is not documented better elsewhere |

**Evidence and history** — why each rule above exists:

- [docs/lessons/](docs/lessons/) — one file per subsystem with the full evidence behind
  each rule above.
- [docs/lessons/live-client-design.md](docs/lessons/live-client-design.md) — why the
  live client's final text comes from the live ASR cache rather than a re-transcription,
  the two rejected alternatives, and the trigger for revisiting the decision.
- [docs/lessons/identity-registration-admin.md](docs/lessons/identity-registration-admin.md)
  — the auth/registration port: what the per-user separation actually fixed,
  the credential store, email sign-up, Google sign-in and the admin, with the
  port-parity decisions and the call-site traps each phase surfaced.
- [docs/boundary-measurements.md](docs/boundary-measurements.md) — the **measured**
  A/B behind lesson 40, the two results that do not flatter it, the corrections
  to the original plan, and the honest list of what is still uncalibrated.

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
