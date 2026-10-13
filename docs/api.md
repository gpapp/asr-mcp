# API reference

Every HTTP route the server exposes. Authentication for each is in the **Auth**
column; the model behind it is [below](#authentication-model).

Server settings are documented in
[INSTALLATION.md](../INSTALLATION.md#server-settings-reference).

---

## Authentication model

Every request resolves an identity in a fixed order (`get_current_user` in
`asr_mcp/api/security.py`):

1. **Session cookie** — written by `/api/auth/login`, which checks the credential
   store first (as typed *and* lowercased) and the htpasswd file second, so the
   form and Basic auth cannot disagree about who can log in.
2. **`X-API-Key` holding a DB token** — generated in the web UI or via
   `POST /api/token`, sha256-hashed at rest.
3. **`X-API-Key` holding a static key** from `TRANSCRIBE_API_KEYS` → the synthetic
   user `user_<sha256(key)[:16]>`. **Never `default`** — that is what keeps a
   shared key's voiceprints out of the default user's library. The WebSocket
   endpoint uses the same resolver, so one key has one library on both transports.
4. **401.** An unresolvable key is refused; it never falls back to a shared identity.

With `TRANSCRIBE_API_KEYS` empty, unauthenticated callers would otherwise all
become `default` and share one voiceprint library — so once
`TRANSCRIBE_REGISTRATION_ENABLED` is on, open mode is refused (401) instead.

Everything downstream is scoped to that identity: speakers, voiceprints,
snippets, transcripts, the audio under `voices/<user_id>/`, and jobs.
`/api/asr/activity/stream` only shows *your* running job; another user's is a 404,
and the Cancel button is only offered for a job you own.

### What the Auth column means

| Value | Meaning |
|---|---|
| **No** | No credential needed. Reachable unauthenticated. |
| **Session** | A signed-in session cookie. |
| **API key** | `X-API-Key` header (or `?token=` on the WebSocket) holding a DB token or a static key. |
| **Session/API key** | Either. |
| **Admin** | Any resolved identity, and it must equal `TRANSCRIBE_ADMIN_USER`. Empty means nobody. |

Setup for each method is in
[INSTALLATION.md — Authentication](../INSTALLATION.md#authentication).

---

## Pages

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check + model status |
| `/login` | GET | No | Login page (standalone; also the registration surface) |
| `/gui` | GET | Session | SPA — Transcribe tab |
| `/live` | GET | Session | SPA — Live tab (browser real-time transcription, `static/live.js`) |
| `/voices` | GET | Session | SPA — Voiceprints tab |
| `/transcriptions` | GET | Session | SPA — History tab |
| `/settings` | GET | Session | SPA — Settings tab (status, client token, session, admin accounts card) |

`/gui`, `/live`, `/voices`, `/transcriptions` and `/settings` all serve the same
`app.html` with a different tab activated.

---

## Authentication routes

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/auth/methods` | GET | No | Which sign-up methods this deployment offers — `{email, google, password_hint}` |
| `/api/auth/login` | POST | No | Login — 401 wrong password, **403 with a reason** (unverified/disabled) |
| `/api/auth/login/session` | POST | No | Login from the form (302) |
| `/api/auth/logout` | POST/GET | No | Logout |
| `/api/auth/register` | POST | No | Email sign-up — **201** + a confirmation mail; never signs anybody in |
| `/api/auth/verify` | GET | No | Consume the single-use email link (302; bad/spent/empty all answer alike) |
| `/api/auth/google/start` | GET | No | Begin Google sign-in (302; **404 when unconfigured**) |
| `/api/auth/google/callback` | GET | No | Finish it — 302 to the SPA, or to `{prefix}/login?google=state\|declined\|failed\|closed` |
| `/api/user` | GET | No | Current user info (**session only** — an API-key caller reads as anonymous) |

### Status codes worth distinguishing

| Response | Meaning |
|---|---|
| **401** | Wrong password, or no usable credential. |
| **403** | The account exists but cannot sign in: unverified email, or disabled. The body carries the reason. |
| **404** on `/auth/google/start` | Google is not configured. |
| **404** on `/auth/register` | Registration is off, or that method is not configured. Deliberately a 404 rather than a 403 so a disabled route is indistinguishable from one never mounted. |
| **429** | Sign-up throttle: 10 attempts / 600 s per client. |
| **503** on `/auth/register` | The account was deleted because the confirmation mail could not be sent. Nothing half-made is left behind. |

---

## Transcription and diarization

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/asr/diarize` | POST | Session/API key | Diarize audio by server-side file path |
| `/api/asr/diarize/upload` | POST | Session | Diarize uploaded audio (`?num_speakers=` is a **query param**, not a form field) |
| `/api/asr/transcribe` | POST | Session/API key | Transcribe by server-side file path (`"language"` in the body, ISO 639-1 or `auto`, default `auto`) |
| `/api/asr/transcribe/upload` | POST | Session/API key | Transcribe uploaded audio — SSE progress. `?save=false` skips the server-side save, `?language=hu` |
| `/api/asr/languages` | GET | Session/API key | Static language list of the configured backend (**no model load**) — `{backend, supports_auto, languages:[{code,name}]}` |
| `/api/asr/stream` | POST | No | Always **501** — a stub that points at the WebSocket endpoint |

Both `/transcribe/upload` and `/diarize/upload` produce the **same refined turn
boundaries** and learn voiceprints from them; diarize-only skips the decode, not
the refinement.

### Re-attribution without re-decoding

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/asr/attribution` | POST | Session/API key | Re-attribute cached ASR items to speakers from a server-side `wav_path` — **no ASR**. `items` is a list of `{start, end, text, confidence}` |
| `/api/asr/attribution/upload` | POST | Session/API key | Same, for a client-recorded upload: `file` + `items` (JSON array form field) |

This is how a Live-tab session is re-attributed after you name a profile, without
paying for the decode again.

---

## Jobs

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/asr/activity/stream` | GET | Session/API key | SSE push of job start/finish (snapshot on connect + keep-alive pings; replaces polling) |
| `/api/asr/active/stream` | GET | Session/API key | SSE replay+follow of the current job; **another user's job is a 404** |
| `/api/asr/active/cancel` | POST | Session/API key | Cancel **your own** active transcription job |

`can_cancel` appears in the activity payload and is owner-gated. A `diarize` job
reports `can_cancel: false` — only transcription is cancellable.

> The `done` stage finishes a job. Progress events must not use it — see
> [docs/lessons/api-sse-and-auth.md](lessons/api-sse-and-auth.md).

---

## Live streaming

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/asr/ws/stream` | WS | Session/API key | Real-time streaming transcription. Auth via session, `X-API-Key` header or `?token=`; closes **1008 before `accept()`** |
| `/api/asr/live/save` | POST | Session/API key | Persist a browser Live-tab session to History: JSON `{audio_filename, result, stats?, sidecar?}` — **text only, no audio field**; stores `metadata.asr_source="live_stream"` |

Live-tab audio reaches the server through `/api/asr/attribution/upload`, which is
the capped upload route. `live/save` deliberately does not take audio.

---

## Speakers (`/api/speaker`)

A thin programmatic wrapper over the voiceprint store, for scripts that do not
want the web UI's shape.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/speaker/register` | POST | Session/API key | Register a voiceprint |
| `/api/speaker/register/upload` | POST | Session/API key | Register a voiceprint from an upload |
| `/api/speaker/identify` | POST | Session/API key | Identify a speaker from audio |
| `/api/speaker/list` | GET | Session/API key | List voiceprints |
| `/api/speaker/{name}` | DELETE | Session/API key | Delete a voiceprint |

All are scoped to the caller's identity. For the richer web-UI surface
(snippets, merges, renaming, review, integrity), use `/api/voiceprint/*`.

---

## Voiceprints (`/api/voiceprint`)

### Profiles

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/voiceprint/speakers` | GET | Session | List speakers; each entry carries `pending` |
| `/api/voiceprint/speakers/{speaker_name}/snippets` | GET | Session | List snippets for a speaker |
| `/api/voiceprint/speakers/{speaker_name}` | DELETE | Session | Delete a voiceprint |
| `/api/voiceprint/speakers/{speaker_name}/rename` | POST | Session | Rename a speaker |
| `/api/voiceprint/speakers/merge` | POST | Session | Merge two speakers |
| `/api/voiceprint/speakers/{name}/refine` | POST | Session/API key | Rebuild a voiceprint from its snippets |
| `/api/voiceprint/upload` | POST | Session | Register a voiceprint from an upload |
| `/api/voiceprint/rescan` | POST | Session | Rescan the voices directory |
| `/api/voiceprint/rescan/stream` | POST | Session | Same, as an SSE stream |

### Snippets

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/voiceprint/speakers/{name}/upload` | POST | Session/API key | Add a snippet from a file. Optional `?start_sec=&end_sec=`. **`?replace_source=false`** keeps earlier snippets from the same file — the default **replaces** them |
| `/api/voiceprint/snippets/{snippet_id}` | DELETE | Session | Delete one snippet |
| `/api/voiceprint/snippets/{snippet_id}/audio` | GET | Session | Stream snippet audio (FLAC) |

**One file, one contribution.** Re-processing a recording replaces that
recording's take rather than stacking a second copy of the same room and channel
onto a profile. The recording is identified by its **filename hash** (md5 of the
source stem), so a reprocess matches across a different directory, extension,
upload name, and every moved span or duration. Pass `?replace_source=false` to opt
out and keep both takes. See
[docs/voiceprints.md](voiceprints.md#one-file-one-contribution).

### Pending profiles

A learned-but-unnamed profile is **excluded from every voiceprint match** until
you name it. These are the only routes that make one matchable.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/voiceprint/pending` | GET | Session | Learned-but-unnamed speakers awaiting a name |
| `/api/voiceprint/pending/{speaker_name}/confirm` | POST | Session | Give a pending profile a real name — the only route that makes it matchable |
| `/api/voiceprint/pending/{speaker_name}/candidates` | GET | Session | Registered speakers this profile resembles, ranked, each with `likely` |
| `/api/voiceprint/pending/{speaker_name}/merge` | POST | Session | Fold a pending profile into an existing speaker (`{"into": "<name>"}`) |

### Integrity and review

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/voiceprint/integrity` | GET | Session | Read-only store audit (8 findings) |
| `/api/voiceprint/integrity/cleanup` | POST | Session | Repair the findings that have one correct answer, and enforce the snippet budget (`IntegrityCleanupRequest{dry_run=True, duplicates=True, dead_snippets=True, stale_totals=True, trim_snippets=True}`) |
| `/api/voiceprint/consolidate/stream` | POST | Session | Flag the outlier snippets of one speaker (`?speaker_name=`, omit = whole library) and re-fit from the kept ones, as SSE |
| `/api/voiceprint/flagged` | GET | Session | Snippets flagged for review, ranked by distance (`?speaker_name=`) — read-only |
| `/api/voiceprint/flagged/{snippet_id}/keep` | POST | Session | Withdraw the flag; re-run consolidation to fold the snippet back into the profile |
| `/api/voiceprint/flagged/remove` | POST | Session | Delete every flagged snippet in **one** pass, then re-refine each touched speaker once (`FlaggedRemoveRequest{dry_run=True, speaker_name?}`) |

`integrity/cleanup` and `flagged/remove` are **dry runs by default**. Pass
`dry_run: false` to actually delete.

`trim_snippets` is the one step that is not an audit finding: it applies the
snippet budget (top `AUTO_COLLECT_MAX_PER_SOURCE` per file per profile, none
under `AUTO_COLLECT_MIN_DURATION`) to the whole library, which a reprocess only
ever applies to the file it re-runs.

See [docs/voiceprints.md](voiceprints.md) for what consolidation flags, what the
audit repairs, and how the per-file snippet budget is spent.

---

## Transcripts and tokens

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/transcripts` | GET | Session | List transcriptions for the user |
| `/api/transcripts/{id}` | GET | Session | Get one transcription |
| `/api/transcripts/{id}/download` | GET | Session | Download as plain text |
| `/api/transcripts/{id}` | DELETE | Session | Delete a transcription |
| `/api/token` | GET/POST | Session/API key | Client API token status / generate — **one per user**, plaintext returned once |

---

## Admin

All four require `require_admin`: the caller must resolve to an identity, and it
must equal `TRANSCRIBE_ADMIN_USER`. **Empty means nobody** — these answer 403 and
name the variable rather than falling back to "anyone who is logged in".

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/admin/users` | GET | Admin | Account table — **never a hash or a verification token** |
| `/api/admin/users/{id}/disable` | POST | Admin | Suspend a registrable account |
| `/api/admin/users/{id}/enable` | POST | Admin | Lift a suspension |
| `/api/admin/users/{id}` | DELETE | Admin | Delete the account; `?purge=true` also drops its rows and `voices/<id>/` |
| `/shutdown` | POST | Admin | Stop the server (it cancels other users' jobs) |

**A delete keeps the data.** Voiceprints, snippets, transcripts and audio are
preserved and their counts are reported, unless `?purge=true` is passed
explicitly. Deleting the account you are signed in as, or the configured admin, is
refused.

htpasswd accounts appear in the table but are **read-only** there — the file is
not this app's to rewrite.

---

## MCP

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/mcp/tools` | GET | No | List MCP tools |
| `/api/mcp/resources` | GET | No | List MCP resources |
| `/api/mcp/call` | POST | Session/API key | Call an MCP tool |
| `/api/mcp/resource/{uri}` | GET | Session/API key | Read one MCP resource |

The data-bearing routes require a credential; the two manifests are public static
descriptions.

> **Present, not exercised.** These are mounted and reachable, but the tools are
> not wired to the transcription path in a meaningful way. Treat them as a
> surface, not a feature.

---

## Conventions

- Long-running upload endpoints return **`text/event-stream`**. If a
  transcription request hangs without streaming, that endpoint is wrong.
- SSE producers must yield (`sleep(0.01)`) **and** run heavy work off the event
  loop.
- Authenticated `fetch()` from the browser needs `credentials: 'same-origin'`.
- A proxy may refuse the body **before** the app sees it: a 413 may be HTML, and
  a 409 means "wait", not "failed".
- The WebSocket closes with **1008 before `accept()`** when unauthenticated —
  accepting first would let an unauthorized client open a stream it cannot use.

Background: [docs/lessons/api-sse-and-auth.md](lessons/api-sse-and-auth.md).