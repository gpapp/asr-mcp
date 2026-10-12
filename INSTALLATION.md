# Installation & configuration

Everything needed to run asr-mcp: install, reverse proxy, every server setting,
and how to set up each authentication method.

- [Quick start](#quick-start)
- [Reverse proxy](#reverse-proxy)
- [Authentication](#authentication) — [htpasswd](#1-htpasswd),
  [email registration](#2-email-registration), [Google](#3-google-sign-in),
  [API keys](#4-api-keys), [admin](#5-admin)
- [Server settings reference](#server-settings-reference)
- [Windows client](#windows-client)
- [Verifying a deployment](#verifying-a-deployment)

---

## Quick start

### Docker (recommended)

```bash
cp .env.example .env
# Edit .env: set TRANSCRIBE_SESSION_SECRET (required — the server refuses to start without it)
docker compose up -d --build
```

The server listens on `0.0.0.0:8080` inside the container, published as
**8087** on the host. Models download on first use, not at startup — nothing is
loaded until the first transcription request, so a fresh boot is fast and the
logs stay quiet until someone actually uses it.

### Local dev (CUDA GPU required)

```powershell
cp .env.example .env
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m asr_mcp.server
```

> **Keep `onnxruntime-gpu` last in `requirements.txt`.** It depends on the CPU
> `onnxruntime` wheel, and both write the same `site-packages/onnxruntime/`
> files. Installing the CPU wheel last silently disables the CUDA execution
> provider, and the first CUDA call then fails with an allocator error that
> reads like a GPU problem rather than a packaging one.

---

## Reverse proxy

Serve behind nginx when the app does not own the whole domain. The snippet in
`nginx_snippet.conf` is complete and can be pasted into a `server` block.

Three settings make this work, and they must agree:

| Setting | Value | Why |
|---|---|---|
| `TRANSCRIBE_PREFIX` | `/asr-mcp` | The public path. nginx strips it (`proxy_pass .../`) before the app sees the request. |
| `TRANSCRIBE_EXTERNAL_URL` | `https://host/asr-mcp` | The URL a user pastes into a browser. Used for the confirmation link and the Google redirect URI. |
| `TRANSCRIBE_PORT` | `8080` | Internal only. nginx proxies to `127.0.0.1:8087`, which compose maps to 8080. |

Three nginx directives are load-bearing and easy to omit:

- `proxy_buffering off` — without it SSE job progress is held back and the UI
  appears to hang until the job finishes.
- `proxy_http_version 1.1` + `Upgrade`/`Connection` headers — without them the
  live WebSocket fails the handshake.
- `client_max_body_size` — the default 1 MB refuses any real audio file.

**The app does not serve the prefix itself.** It answers `/api/...`, not
`/asr-mcp/api/...`. Testing with `curl localhost:8087/asr-mcp/api/...` returns a
redirect from the auth middleware, which looks like a routing fault but is just
the prefix being requested twice.

---

## Authentication

An identity is resolved per request in a fixed order:

1. **Session cookie** — written by the login form.
2. **`X-API-Key` holding a DB token** — generated in the web UI (Settings →
   Windows Client Token) or via `POST /api/token`, sha256-hashed at rest.
3. **`X-API-Key` holding a static key** from `TRANSCRIBE_API_KEYS` — resolves to
   the synthetic user `user_<sha256(key)[:16]>`, **never** `default`.
4. **401.** An unresolvable key is refused; it never falls back to a shared
   identity.

Everything downstream — speakers, voiceprints, snippets, transcripts, the audio
under `voices/<user_id>/` — is scoped to that identity. Job streams show only
your own job; another user's is a 404.

`TRANSCRIBE_API_KEYS` empty means **open mode**: any unauthenticated caller
becomes `default` and they all share one voiceprint library. That is fine on a
laptop and wrong the moment the port is reachable by someone else. Turning on
`TRANSCRIBE_REGISTRATION_ENABLED` refuses open mode automatically (401 instead).

### 1. htpasswd

The legacy store, read-only to this app, and the first thing the login form
consults. Generated with `htpasswd`:

```bash
htpasswd -c data/.htpasswd alice      # create (-c), or omit -c to add
htpasswd data/.htpasswd bob           # add another user
```

Compose mounts `data/` and sets `TRANSCRIBE_HTPASSWD_PATH=/app/data/.htpasswd`.
Supported hashes: bcrypt (`$2b$`/`$2y$`), apr1, SHA, SSHA and plaintext.

These accounts work regardless of registration, and an htpasswd user is
**read-only in the admin UI** — they cannot be suspended or deleted from there,
because the file is not this app's to rewrite.

### 2. Email registration

Off by default. An endpoint that creates accounts is something an operator turns
on deliberately, not something a deployment inherits by upgrading.

```bash
TRANSCRIBE_REGISTRATION_ENABLED=true
TRANSCRIBE_SMTP_HOST=smtp.example.org
TRANSCRIBE_SMTP_FROM=asr@example.org
# Optional, only if your relay requires a login:
TRANSCRIBE_SMTP_USER=
TRANSCRIBE_SMTP_PASSWORD=
TRANSCRIBE_SMTP_STARTTLS=true
```

Only `SMTP_HOST` and `SMTP_FROM` are load-bearing — a relay on a trusted network
may take no credentials, and a deployment with no host can send nothing. Those
two are exactly what the server checks before offering the sign-up form.

How it behaves:

- Sign-up is throttled per client: **10 attempts / 600 s**, keyed on the
  rightmost `X-Forwarded-For` hop (nginx appends, so the rightmost is the one
  the client chose — trusting the leftmost would let a client spoof its way past
  the limit).
- An account is created **unverified** and a single-use link is mailed. Sign-up
  **never signs anyone in**.
- Until the link is opened, login answers **403** "this account has not been
  confirmed yet".
- If the mail cannot be sent the row is **deleted** and the request answers 503
  — no half-made accounts that nobody can confirm.
- The token is stored as a sha256 hash only, and the confirmation route answers
  identically for a bad, spent or missing token, so a probe cannot distinguish
  them.

There is **no password-reset flow**. An admin re-issues a password or deletes
the account.

### 3. Google sign-in

OAuth 2.0 authorization-code flow, stdlib `urllib` only. Gated on the client
id and secret being configured — **not** on the registration flag — so turning
registration off cannot sign out users who already signed in through Google.

#### Step 1 — create the OAuth client

In [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services**
→ **Credentials** → **Create credentials** → **OAuth client ID**:

| Field | Value |
|---|---|
| Application type | **Web application** |
| Name | anything, e.g. `asr-mcp` |
| Authorized redirect URIs | the URL in the box below |

After creating, Google shows a **Client ID** and a **Client secret**.

#### Step 2 — the redirect URI

This must match what the server sends, **verbatim** — Google compares it byte
for byte, so `http://` is refused where `https://` is required, and a trailing
slash is a different URI.

```
{TRANSCRIBE_EXTERNAL_URL}/api/auth/google/callback
```

Concretely, for this deployment:

```
https://mcp.itworks.hu/asr-mcp/api/auth/google/callback
```

The server **derives** this from `TRANSCRIBE_EXTERNAL_URL` plus
`TRANSCRIBE_PREFIX` rather than configuring it separately, because a redirect
URI that disagrees with the URL the app is actually served on fails at Google
with an error page that names neither. To confirm the exact string your
deployment will send:

```bash
docker compose exec -T asr-mcp python3 -c "
from asr_mcp.config.settings import get_settings
from asr_mcp.api.auth_router import _public_base
from asr_mcp.auth.google_auth import callback_url
class R: base_url = 'http://localhost:8080/'
print(callback_url(_public_base(get_settings(), R())))"
```

`TRANSCRIBE_EXTERNAL_URL` may include `TRANSCRIBE_PREFIX` or leave it off — both
spellings produce the same links, because the prefix is appended only when the
base does not already end with it, compared on a path-segment boundary. So
`https://host` and `https://host/asr-mcp` both work. Set it to the URL you can
actually paste into a browser, which is the one you can verify by clicking it.

#### Step 3 — configure and restart

```bash
TRANSCRIBE_GOOGLE_CLIENT_ID=xxxxx.apps.googleusercontent.com
TRANSCRIBE_GOOGLE_CLIENT_SECRET=GOCSPX-xxxxx
docker compose up -d --force-recreate asr-mcp
```

**Both halves are required.** The secret is what makes the code exchange a
confidential-client exchange; with only an id, the browser can start a login
that can never finish.

#### Step 4 — verify

The **Continue with Google** button appears on `/login` only when both halves
are set. Check it reports enabled without leaking anything:

```bash
curl -s https://host/api/auth/methods
# {"email":false,"google":true,"password_hint":10}
```

If `google` is `false` with both values set, the variable did not reach the
container — `docker-compose.yml` only forwards variables it explicitly lists, so
an entry in `.env` alone is inert.

#### How a Google account maps to a user

A first sign-up creates an account **named after the normalised email address**
Google verified, and records the Google **subject** against it. No confirmation
mail: Google already proved the address.

Subsequent sign-ins resolve by **subject, never by address**. That distinction
matters — it means an address cannot be used to walk into an existing account,
and re-assigning an address to a different person later cannot reach the old
account's recordings.

Google sign-in is refused when the address is empty, when Google does not report
`email_verified: true`, when registration is off, or when the address is already
a vault key.

### 4. API keys

Two kinds, and they behave differently.

**DB tokens** — per user, generated in the web UI (Settings → *Windows Client
Token*) or via `POST /api/token`. Stored sha256-hashed; the plaintext is shown
once. Send as `X-API-Key`.

**Static keys** — `TRANSCRIBE_API_KEYS` is a JSON list, useful for a service
account or a CI job:

```bash
TRANSCRIBE_API_KEYS=["a-long-random-string"]
```

Each resolves to the synthetic user `user_<sha256(key)[:16]>`. Never `default` —
that is what keeps a shared key's voiceprints out of the default user's library.
The WebSocket endpoint uses the same resolver, so one key has one library on
both transports.

### 5. Admin

One deployment-wide administrator. **Empty means nobody** — the routes answer
403 and name the variable rather than falling back to "anyone who is logged in".

```bash
TRANSCRIBE_ADMIN_USER=alice      # the account name, exactly as it signs in
```

That account reaches the Settings tab's **Accounts** card, `/api/admin/*` and
`POST /shutdown`. The Accounts card simply does not appear for anyone else.

Deleting an account **keeps** its voiceprints, snippets, transcripts and audio,
and reports the counts. `?purge=true` is the explicit opt-in that also drops
those rows and `rmtree voices/<user_id>/`. Deleting the account you are signed
in as, or the configured admin, is refused.

---

## Server settings reference

Every setting uses the `TRANSCRIBE_` prefix. `docker-compose.yml` forwards only
the variables it lists explicitly — an entry in `.env` for anything else is
inert, and you must add a `- TRANSCRIBE_X=${TRANSCRIBE_X}` line.

### Required

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_SESSION_SECRET` | — | Signs session cookies. **Required: omitting it stops the server from starting.** An *empty* value is accepted, so `SESSION_SECRET=` in `.env` starts fine and then signs cookies with no secret — set a real one. Generate with `openssl rand -hex 32`. Changing it invalidates every session, including the admin's. |

### Authentication and accounts

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_REGISTRATION_ENABLED` | `false` | Offers sign-up on the login page. **Also forces authentication** — open mode is refused while it is on, because an unauthenticated caller would otherwise become `default` and share one voiceprint library. |
| `TRANSCRIBE_API_KEYS` | `[]` | JSON list of static keys. Each resolves to `user_<sha256(key)[:16]>`, never `default`. Empty means open mode. |
| `TRANSCRIBE_SMTP_HOST` | `""` | SMTP host. Sign-up needs this **and** `TRANSCRIBE_SMTP_FROM`; a relay on a trusted network may take no credentials. |
| `TRANSCRIBE_SMTP_PORT` | `587` | SMTP port. |
| `TRANSCRIBE_SMTP_USER` | `""` | SMTP user. Empty skips the login entirely. |
| `TRANSCRIBE_SMTP_PASSWORD` | `""` | Only used when `TRANSCRIBE_SMTP_USER` is set. |
| `TRANSCRIBE_SMTP_FROM` | `""` | Envelope sender for the confirmation mail. |
| `TRANSCRIBE_SMTP_STARTTLS` | `true` | Issue STARTTLS before login. Turn off only for a local relay on port 25. |
| `TRANSCRIBE_SMTP_TIMEOUT` | `20` | Seconds to wait on the SMTP socket. |
| `TRANSCRIBE_GOOGLE_CLIENT_ID` | `""` | OAuth client id. **Both halves required** — with only an id a login can start but never finish. |
| `TRANSCRIBE_GOOGLE_CLIENT_SECRET` | `""` | OAuth client secret. Makes the code exchange a confidential-client exchange. |
| `TRANSCRIBE_ADMIN_USER` | `""` | The one account allowed to reach `/api/admin/*` and `POST /shutdown`. **Empty means nobody.** The account name, exactly as it signs in. |

### Paths

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_DATA_DIR` | `./data` | Root for the SQLite database and the htpasswd file. |
| `TRANSCRIBE_DB_PATH` | `./data/asr_mcp.db` | The database itself. **Back this up** — it holds accounts, tokens and every speaker profile. |
| `TRANSCRIBE_VOICES_DIR` | `./voices` | Snippet audio, in `<voices_dir>/<user_id>/<speaker>/`. The largest directory; the one to mount. |
| `TRANSCRIBE_MODEL_CACHE_DIR` | `./models` | Downloaded model weights. Safe to delete; they re-download. |
| `TRANSCRIBE_LOG_DIR` | `./logs` | Structured logs. |
| `TRANSCRIBE_HTPASSWD_PATH` | `/app/htpasswd` | Operator accounts. Read-only to this app. |

### Network

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_HOST` | `0.0.0.0` | Bind address. |
| `TRANSCRIBE_PORT` | `8080` | Bind port (internal; compose publishes 8087). |
| `TRANSCRIBE_WORKERS` | `1` | Uvicorn worker processes. Only read by `python -m asr_mcp.server`; **keep it at 1** — models load per process, so N workers hold N copies in VRAM. |
| `TRANSCRIBE_PREFIX` | `""` | Public URL prefix behind a reverse proxy, e.g. `/asr-mcp`. nginx strips it. Empty means served at the root. |
| `TRANSCRIBE_EXTERNAL_URL` | `""` | Absolute base URL for the confirmation link and Google redirect URI. Empty derives it from the request, which is right until a proxy rewrites `Host`. |
| `TRANSCRIBE_ENABLE_CORS` | `true` | CORS middleware on. |
| `TRANSCRIBE_CORS_ORIGINS` | `["*"]` | Allowed origins. Tighten if the API is reachable from a browser. |
| `TRANSCRIBE_ENABLE_RATE_LIMIT` | `true` | Request rate limiting. |
| `TRANSCRIBE_RATE_LIMIT` | `60/minute` | The limit itself. |

### GPU and models

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_CUDA_DEVICE` | `cuda:0` | CUDA device ordinal. |
| `TRANSCRIBE_GPU_MEMORY_LIMIT_GB` | `4.0` | **The card's total size, not a cap.** It sets the ONNX arena sizes: encoder × 0.625, embedding min(÷4, 768 MiB). Setting it lower than the real card starves the arena; setting it higher lets a run OOM. Get it from `nvidia-smi`. |
| `TRANSCRIBE_TORCH_DTYPE` | `float16` | Torch dtype for the Qwen3 backend. |
| `TRANSCRIBE_MODEL_TTL_MINUTES` | `5` | Idle minutes before models unload from VRAM. `0` keeps them resident (faster, but holds ~2.6 GB). Never unloads mid-job. |
| `TRANSCRIBE_ASR_MODEL` | `cohere` | `cohere` (ONNX), `qwen3-asr` (transformers), or `whisper` (faster-whisper). |
| `TRANSCRIBE_HF_TOKEN` | — | HuggingFace token, only for gated models. |
| `TRANSCRIBE_MODEL_REPO` | `onnx-community/cohere-transcribe-03-2026-ONNX` | ASR model repo. |
| `TRANSCRIBE_EMBEDDING_MODEL_REPO` | `Wespeaker/wespeaker-ecapa-tdnn512-LM` | ECAPA voiceprint model repo. |
| `TRANSCRIBE_VAD_MODEL_REPO` | `onnx-community/silero-vad` | Silero VAD repo. |

### ASR backends

Used only by the backend named in `TRANSCRIBE_ASR_MODEL`.

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_QWEN_MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | Qwen3 model. |
| `TRANSCRIBE_QWEN_MODEL_DIR` | `./models/qwen3-asr` | Where it is cached. |
| `TRANSCRIBE_QWEN_FORCED_ALIGNER_NAME` | `Qwen/Qwen3-ForcedAligner-0.6B` | Aligner model for timestamps. |
| `TRANSCRIBE_QWEN_FORCED_ALIGNER_DIR` | `./models/qwen3-forced-aligner` | Aligner cache. |
| `TRANSCRIBE_QWEN_TORCH_DTYPE` | `float16` | Qwen3 torch dtype. |
| `TRANSCRIBE_QWEN_MAX_NEW_TOKENS` | `256` | Decode budget per turn. |
| `TRANSCRIBE_QWEN_MAX_INFERENCE_BATCH_SIZE` | `8` | Batch size. |
| `TRANSCRIBE_QWEN_QUANTIZE_4BIT` | `true` | 4-bit load, roughly halves VRAM. |
| `TRANSCRIBE_QWEN_ALIGNER_QUANTIZE_4BIT` | `true` | Same for the aligner, saving ~0.9 GB. Falls back to fp16. |
| `TRANSCRIBE_WHISPER_MODEL` | `large-v3-turbo` | Whisper size or HF repo id. |
| `TRANSCRIBE_WHISPER_MODEL_DIR` | `./models/faster-whisper` | Whisper cache, one subdir per spec. |
| `TRANSCRIBE_WHISPER_COMPUTE_TYPE` | `auto` | `auto` = int8_float16 on CUDA, int8 on CPU. Quantization is not optional in practice — fp16 needs ~1.6 GB and spikes. |
| `TRANSCRIBE_WHISPER_BEAM_SIZE` | `5` | Beam width. |
| `TRANSCRIBE_WHISPER_VAD_FILTER` | `true` | faster-whisper's built-in Silero filter. Off for live turns. |

### Diarization

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_VAD_THRESHOLD` | `0.5` | Silero speech-probability cutoff. |
| `TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS` | `250` | Drops shorter speech regions. |
| `TRANSCRIBE_DIARIZATION_THRESHOLD` | `0.35` | **Declared but unused.** Clustering thresholds live in `asr_mcp/config/thresholds.json` (`diarization.distance_threshold`). Editing this has no effect. |
| `TRANSCRIBE_REQUEST_TIMEOUT` | `120` | **Declared but unused** — nothing reads it. |

### Logging

| Variable | Default | What it does |
|---|---|---|
| `TRANSCRIBE_LOG_LEVEL` | `INFO` | `DEBUG` also logs every boundary decision (`method`, nominal cut, final cut, `shift_sec`) — useful when tuning boundaries, very noisy otherwise. |

---

## Windows client

Stdlib-only Python — no installs beyond Python 3. Distributed as a standalone zip
built by GitHub Actions (`.github/workflows/asr-client.yml`, triggers on
`asr-client/**` changes). The zip contains `transcribe_client.py`,
`transcribe.bat`, `requirements.txt`, `README.md` and `.env.example` — never
`.env` or `.venv`.

```powershell
# From the unzipped folder — first run creates .env from .env.example
transcribe.bat status
# Edit .env: SERVER_URL (include the /asr-mcp prefix when behind nginx) + TOKEN
# Generate the token in the web UI ("Windows Client Token" card) or via POST /api/token
```

`transcribe.bat` creates a private `.venv` next to the script and runs the
client with it (falling back to `py`/`python` on PATH). CI enforces stdlib-only
imports, so the client has no server dependencies to drift out of sync with.

| Command | Description |
|---|---|
| Drag files onto `transcribe.bat` | Transcribe with live progress; writes `<name>.txt` next to each file |
| `transcribe.bat status` | Check server/token connectivity |
| `transcribe.bat voiceprints` | List speakers and voiceprints |
| `transcribe.bat voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip |
| `transcribe.bat voiceprint-refine <name>` | Rebuild a voiceprint from its snippets |
| `transcribe.bat <file...> [--language <code>]` | Transcribe; overrides `.env LANGUAGE` |

Client transcriptions use `?save=false`, so nothing lands in server transcript
history. If the connection drops mid-job the server finishes the run and saves
the result so it can be downloaded from the web UI. Voiceprint snippets the
client uploads **are** kept.

---

## Verifying a deployment

Run these after installing. Each one names a specific way the deployment is
broken, rather than returning a generic health check.

```bash
# 1. Reachable, and models not yet loaded
curl -s https://host/health

# 2. Which sign-up methods are actually offered
curl -s https://host/api/auth/methods
# {"email":false,"google":true,"password_hint":10}

# 3. The confirmation link and Google redirect URI your deployment will send
docker compose exec -T asr-mcp python3 -c "
from asr_mcp.config.settings import get_settings
from asr_mcp.api.auth_router import _public_base
from asr_mcp.auth.google_auth import callback_url
class R: base_url = 'http://localhost:8080/'
print(callback_url(_public_base(get_settings(), R())))"

# 4. Open mode is refused once registration is on
curl -s -o /dev/null -w '%{http_code}\n' https://host/api/transcripts   # 401, not 200
```

**Symptom → cause:**

| Symptom | Cause |
|---|---|
| Confirmation link 404s | The token is almost certainly fine — check `TRANSCRIBE_EXTERNAL_URL`. This is exactly the `/asr-mcp/asr-mcp` doubling bug, fixed by `_public_base`; it means the base already ends with the prefix and got it again. |
| Google button missing | One of id/secret missing, or the variable is in `.env` but not forwarded in `docker-compose.yml`. |
| Google error page naming neither side | Redirect URI mismatch. Re-run check 3 and register that string verbatim. |
| Sign-up form missing | `TRANSCRIBE_REGISTRATION_ENABLED` off, or SMTP host/from incomplete. |
| Server will not start, validation error on `session_secret` | `TRANSCRIBE_SESSION_SECRET` is missing from the container. An *empty* value does not raise — see below. |
| Server runs, but nobody can stay signed in | `TRANSCRIBE_SESSION_SECRET=` set but blank. It starts, then signs cookies with no secret. Generate a real one. |
| 500 on `cublasCreate` or an arena error | CPU `onnxruntime` installed after `onnxruntime-gpu`, or `TRANSCRIBE_GPU_MEMORY_LIMIT_GB` does not match the card. |
| Everything works locally, 404 behind nginx | Requesting `/asr-mcp/asr-mcp/...`. nginx must strip the prefix. |
| Live tab connects then goes quiet | `proxy_buffering` not disabled in nginx. |
| WebSocket fails to open | `proxy_http_version 1.1` and the `Upgrade`/`Connection` headers missing. |
| Uploads rejected | `client_max_body_size` still at the 1 MB default. |