# API, SSE and session auth

Lessons 9, 10, 19, 20, 25, 28 of `AGENTS.md`. Event streaming, middleware order, job lifecycle and language threading.


### 9. SSE for Long-Running Endpoints
- Upload endpoints that run diarization/transcription MUST use `StreamingResponse(media_type="text/event-stream")`.
- nginx `proxy_read_timeout` defaults to 120s — set to 600s in `nginx_snippet.conf`.
- Frontend reads SSE via `ReadableStream` + `TextDecoder()`, parses `data:` lines.

### 10. Session Auth Requires `credentials: 'same-origin'`
- All `fetch()` calls for authenticated endpoints MUST include `credentials: 'same-origin'`.
- Without it, the session cookie isn't sent, AuthMiddleware redirects to `/login` (HTML), and frontend tries to parse HTML as JSON.
- Use `window.location.replace()` not `window.location.href` for login redirects (avoids back-button loops).


### 19. SSE Producers Must Yield AND Run Heavy Work Off-Loop
- `await queue.put()` on an unbounded `asyncio.Queue` NEVER suspends — the consumer doesn't run and all events flush in one burst when the job ends.
- `_sse_put()` = `queue.put()` + `await asyncio.sleep(0.01)`. `sleep(0)` alone is insufficient: the BaseHTTPMiddleware body pump + uvicorn transport need a real tick to flush bytes.
- Yielding is not enough for CPU-heavy work (ONNX decode, clustering): it starves the loop regardless. Run turns via `loop.run_in_executor(None, ...)`; thread-side progress uses `loop.call_soon_threadsafe(queue.put_nowait, evt)`.

### 20. Starlette Middleware Order — Last Added Runs Outermost
- `app.add_middleware()` prepends to the stack: the LAST middleware added wraps all earlier ones and runs FIRST.
- Working order in `server.py`: AuthMiddleware added first (inner), SessionMiddleware added second (outer) → Session populates `request.scope["session"]` before Auth reads it.
- **Why**: An attempted swap (to "fix" SSE 401s) broke auth and was reverted. SSE endpoints are not in `PUBLIC_PATHS` and depend on this order to see the session cookie. Do not reorder without tracing who populates `scope["session"]` first.


### 25. `stage: "done"` in SSE Finishes the Job — Progress Events Must Not Use It
- `_sse_put` publishes every event to `job_state.publish`, which calls `finish()` when `stage in ("done","error","cancelled")` → `_active = None`. If a mid-job progress event says `done` (pipeline used to emit final diarization progress as `stage: "done"`), the TTL monitor sees no running job and unloads models WHILE the turn loop still runs → `'NoneType' has no attribute 'get_inputs'`. `"cancelled"` is emitted ONLY by run_transcribe's `except JobCancelled` handler (cancel via `POST /api/asr/active/cancel` sets `job.cancel_requested`; `JobCancelled(BaseException)` is raised from progress callbacks so backend `except Exception` blocks cannot swallow it).
- Pipeline final progress stage must be `"diarization_finished"` (NOT `"done"`). `job_state.publish` also guards `evt.get("phase") != "diarization"` so real terminal events (which carry NO phase key) still finish the job.
- `state.touch()` at the start of the transcribe turn loop so idle-TTL never counts job time as idle.
- Import rule: `from asr_mcp.core import job_state` — the MODULE (functions `start_job/get_running/publish/finish/ensure_finished`); there is NO `job_state` symbol to import.
- **Activity channel** (replaces polling `GET /active`): `job_state.subscribe_activity()/unsubscribe_activity()` queues receive only `{"active": true/false, ...}` transitions from `start_job`/`finish`. `GET /api/asr/activity/stream` subscribes BEFORE taking its snapshot (a start/finish racing the connect lands in the queue, not lost), sends the snapshot first, then pushes changes with 20s `: ping` keep-alives. GUI keeps one long-lived `connectActivity()` reader; `POST /active/cancel` and the per-job `/active/stream` are unchanged.


### 28. Language Must Be Threaded End-to-End — Routers, Prompt, and Decode
- `transcribe/upload` accepts `?language=` and `transcribe` (path-based) accepts `"language"` in the body (ISO 639-1 or `auto`, default `auto` on both) → `_transcribe_file(language=...)` → `transcribe_audio_sync(language=...)`. Before this, routers never passed `language` and every backend silently forced English — Hungarian audio decoded as English hallucination. Both endpoints must be checked: `/upload` was threaded first and the path-based `transcribe` kept forcing English for a long time, which the podcast tests exposed.
- Each backend carries a **static `LANGUAGES: list[(code, name)]` class attribute** (+ `SUPPORTS_AUTO`) — available without loading the model; `GET /api/asr/languages` serves it. GUI dropdown (Transcribe options) and the client (`LANGUAGE=` in .env / `--language` flag) both default to `auto`.
- `auto` handling: whisper (`None` → faster-whisper detect), qwen3 (`_to_canonical_language` → `None` = detect), cohere (no `<|auto|>` vocab token → explicit English-prompt fallback, `SUPPORTS_AUTO=False`, GUI labels it "English fallback").
- Cohere: the decoder prompt embeds language tokens (`<|hu|>` id 87, `<|en|>` id 62) — build it **per request** via `_prompt_ids_for(language)` (cached in `_prompt_cache`), NOT once at load with `language="en"`. Qwen maps via `LANGUAGE_MAP` (`"hu" → "Hungarian"`); Whisper passes the ISO code straight to faster-whisper (`auto`/empty → detect).
- **Cohere model limitation**: cohere-transcribe-03-2026-ONNX q4 cannot transcribe Hungarian at all — en prompt → English hallucination, hu prompt → Cyrillic/spam hallucination (verified; English control input is perfect). Use whisper for non-English.
- 56.8-min hu podcast benchmark (GTX 1650): whisper 498s/1290MiB/7.1× (excellent quality), cohere 1295s/2392MiB/2.65× (hu unusable), qwen3 2388s/2830MiB/1.43× (correct content, no punctuation).
