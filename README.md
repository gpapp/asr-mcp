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
- **Self-Service Registration**: Optional email sign-up with a confirmation link
  (`TRANSCRIBE_REGISTRATION_ENABLED` + SMTP) and Google sign-in
  (`TRANSCRIBE_GOOGLE_CLIENT_ID`/`_SECRET`). Both off by default; htpasswd keeps
  working either way.
- **Per-User Isolation**: Speakers, voiceprints, snippets, transcripts and jobs are
  scoped to the signed-in identity. One API key has one voiceprint library on both
  the HTTP and WebSocket transports, and you can only see and cancel your own jobs.
- **Admin Accounts**: One deployment-wide administrator (`TRANSCRIBE_ADMIN_USER`)
  can list accounts, suspend them, delete them (optionally with their data) and
  stop the server. Empty means nobody.
- **Windows Client**: Drop-file transcription client (`asr-client/`) with live progress and `<name>.txt` output.
- **Reverse Proxy Support**: Configurable URL prefix (`TRANSCRIBE_PREFIX`) for nginx/Caddy.
- **Multi-Format Input**: Accepts mp3, mp4, mkv, flac, ogg, m4a — auto-converts via ffmpeg.
- **Compressed Snippets**: Voiceprint snippets stored as FLAC for efficient storage.
  Re-processing a recording REPLACES that recording's snippets for the speaker
  (matched by the filename hash, so moved spans and a new extension still match),
  keeping every clip from every other recording — a profile is never fitted to two
  copies of one take. `?replace_source=false` keeps the older take.
- **Voiceprint Integrity**: A startup audit checks the snippet store and repairs the
  findings that have one correct answer — dead snippet rows are deleted and the
  affected profiles are rebuilt from their surviving audio. Merging similar profiles
  is left to you (`GET /api/voiceprint/integrity`, `POST .../integrity/cleanup`).
- **Consolidation & Review**: A per-speaker (or whole-library) pass re-fits each
  profile from all its snippets, finds the snippets whose voice does not belong
  (intruders, crossed channels, mis-attributed spans), and marks them for review with
  the distance that convicted them. Flagged snippets are already excluded from the
  profile; nothing is deleted until you remove them yourself.
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
| `TRANSCRIBE_API_KEYS` | `[]` | JSON list of static API keys; each maps to the synthetic user `user_<sha256(key)[:16]>`, never `default` |
| `TRANSCRIBE_REGISTRATION_ENABLED` | `false` | Offer email/Google sign-up on the login page; also forces auth (open mode is refused while on) |
| `TRANSCRIBE_GOOGLE_CLIENT_ID` | `""` | Google OAuth client id (both halves required, or `/api/auth/google/start` 404s) |
| `TRANSCRIBE_GOOGLE_CLIENT_SECRET` | `""` | Google OAuth client secret |
| `TRANSCRIBE_EXTERNAL_URL` | `""` | Public base URL for the confirmation mail and Google `redirect_uri`; defaults to the request's own URL. May include the prefix or not — both work |
| `TRANSCRIBE_SMTP_HOST` | `""` | SMTP host — email sign-up needs this **and** `TRANSCRIBE_SMTP_FROM` |
| `TRANSCRIBE_SMTP_PORT` | `587` | SMTP port |
| `TRANSCRIBE_SMTP_USER` | `""` | SMTP user (login is skipped when empty) |
| `TRANSCRIBE_SMTP_PASSWORD` | `""` | SMTP password |
| `TRANSCRIBE_SMTP_FROM` | `""` | Envelope sender for the confirmation mail |
| `TRANSCRIBE_SMTP_STARTTLS` | `true` | Issue STARTTLS before the login |
| `TRANSCRIBE_ADMIN_USER` | `""` | The one account allowed to reach `/api/admin/*` and `POST /shutdown`; empty means **nobody** |

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
| `transcribe.bat voiceprint-add <name> <file> [start] [end]` | Create/refine a voiceprint from a clip (replaces this speaker's earlier snippets from the same file) |
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
| `/api/voiceprint/speakers/{name}/upload` | POST | Session/API key | Add snippet from a file (optional `?start_sec=&end_sec=`; `?replace_source=false` keeps earlier snippets from the same file — the default **replaces** them, see below) |
| `/api/voiceprint/speakers/{name}/refine` | POST | Session/API key | Rebuild a voiceprint from its snippets |
| `/api/voiceprint/merge` | POST | Session | Merge speakers |
| `/api/voiceprint/rename` | POST | Session | Rename speaker |
| `/api/voiceprint/rescan` | POST | Session | Rescan voices directory |
| `/api/voiceprint/integrity` | GET | Session | Read-only store integrity audit (8 findings) |
| `/api/voiceprint/integrity/cleanup` | POST | Session | Repair the findings that have one correct answer (`dry_run=True` by default) |
| `/api/voiceprint/consolidate/stream` | POST | Session | Flag the outlier snippets of one speaker (`?speaker_name=`, omit = whole library) and re-fit from the kept ones (SSE progress) |
| `/api/voiceprint/flagged` | GET | Session | Snippets flagged for review, ranked by distance (`?speaker_name=`) — read-only |
| `/api/voiceprint/flagged/{snippet_id}/keep` | POST | Session | Withdraw one flag; re-run consolidation to fold the snippet back in |
| `/api/voiceprint/flagged/remove` | POST | Session | Delete every flagged snippet in ONE pass, then re-refine each touched speaker once (`dry_run=True` by default) |
| `/api/token` | GET/POST | Session/API key | Client API token status / generate (one per user) |
| `/api/auth/methods` | GET | No | Which sign-up methods this deployment offers — `{email, google, password_hint}` |
| `/api/auth/login` | POST | No | Login (returns JSON; 403 with a reason for an unverified or disabled account) |
| `/api/auth/login/session` | POST | No | Login from the form (302) |
| `/api/auth/logout` | POST/GET | No | Logout |
| `/api/auth/register` | POST | No | Email sign-up — 201 + a confirmation mail; never signs anybody in |
| `/api/auth/verify` | GET | No | Consume the single-use email link (302; bad/spent/empty all answer alike) |
| `/api/auth/google/start` | GET | No | Begin Google sign-in (302; 404 when unconfigured) |
| `/api/auth/google/callback` | GET | No | Finish Google sign-in |
| `/api/user` | GET | No | Current user info |
| `/api/admin/users` | GET | Admin | Account table — never a hash or a verification token |
| `/api/admin/users/{id}/disable` | POST | Admin | Suspend a registrable account |
| `/api/admin/users/{id}/enable` | POST | Admin | Lift a suspension |
| `/api/admin/users/{id}` | DELETE | Admin | Delete the account; `?purge=true` also drops its rows and `voices/<id>/` |
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

Every request resolves an identity in a fixed order (`get_current_user`):

1. **Session cookie** — written by `/api/auth/login`, which checks the credential
   store first (as typed *and* lowercased) and the htpasswd file second, so the
   form and `Authorization: Basic` cannot disagree about who can log in.
2. **DB token** in the `X-API-Key` header (sha256-hashed in `api_tokens`).
3. **Static key** from `TRANSCRIBE_API_KEYS` → the synthetic user
   `user_<sha256(key)[:16]>`. **Never `default`** — that is what keeps a shared
   key's voiceprints out of the default user's library. The WebSocket endpoint
   uses the same resolver, so one key has one library on both transports.
4. **401.** An unresolvable key refuses; it never falls back to a shared identity.

With `TRANSCRIBE_API_KEYS` empty, unauthenticated callers would otherwise all
become `default` and share one library — so once
`TRANSCRIBE_REGISTRATION_ENABLED` is on, open mode is refused (401) instead.

Everything downstream — speakers, voiceprints, snippets, transcripts and the
snippet audio under `voices/<user_id>/` — is scoped to that identity, and
`/api/asr/activity/stream` only shows *your* running job. Another user's job is
a 404, and the Cancel button is only offered for a job you own.

### Registration (optional, off by default)

Set `TRANSCRIBE_REGISTRATION_ENABLED=true` plus **either** an SMTP server (email
sign-up) **or** a Google OAuth pair (Google sign-in); the login page only renders
the methods that could actually work.

- **Email** (`TRANSCRIBE_SMTP_HOST` + `TRANSCRIBE_SMTP_FROM`): sign-up is
  throttled per client (10 attempts / 600 s, keyed on the rightmost
  `X-Forwarded-For` hop), creates an **unverified** account, mails a single-use
  link, and **never signs anybody in**. Until the link is opened, login answers
  403 with "this account has not been confirmed yet". A mail that cannot be sent
  deletes the row (503) — no half-made accounts.
- **Google**: `/api/auth/google/start` is gated on the client id/secret being
  configured, *not* on the registration flag, so turning registration off does
  not sign out existing users. A first sign-up's account is named after the
  normalised email address Google verified; identity is resolved by Google
  **subject**, never by address.

There is **no password-reset flow** — an admin re-issues a password or deletes
the account.

> **If a confirmation link 404s**, the token is almost certainly fine: check
> `TRANSCRIBE_EXTERNAL_URL`. Set it to the URL you can actually paste into a
> browser (`https://host/asr-mcp`). Including or omitting `TRANSCRIBE_PREFIX`
> there both work — the prefix is added only when it is missing — but a base
> that already has it must not get it twice.

### Admin (optional, off by default)

Set `TRANSCRIBE_ADMIN_USER` to the one account allowed to reach the Settings
tab's **Accounts** card, `/api/admin/*` and `POST /shutdown`. Empty means
**nobody**.

A delete **keeps** the user's voiceprints, snippets, transcripts and audio and
reports the counts; pass `?purge=true` to drop those rows and
`rmtree voices/<user_id>/` as well. Deleting the account you are signed in as,
or the configured admin, is refused.

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
- Name them in the web UI: **Voiceprints → Unnamed**, type a name and
  press *Save name*. That is the only action that makes them recognisable, and
  it also moves their snippets into a normal voiceprint.
- Confirming refuses a name already in use, so it cannot overwrite a real
  colleague's profile.
- Re-running the same recording — **or a different one** — extends the existing
  pending profile instead of creating a second one: the match is by
  **embedding distance** to the other pending profiles, with the filename as a
  fallback. Before a profile is extended, the new audio is compared against the
  profile's own embedding and refused if it spreads wider than
  `learning.max_profile_intra_dist`.
- A profile records a **purity** figure (largest distance *within* the group,
  smallest *between* groups) that the Voiceprints tab shows. When a cluster
  cannot be verified as one voice — no embedding session, too few snippets, a
  clustering failure, or groups that sit closer than `learning.min_inter_dist` —
  **nothing is learned** (`learning.fail_closed`), because a blend learned as one
  profile would then be used as a reference for cutting later boundaries.
- Snippets are cut from the **same turns the transcript is attributed on** and
  trimmed to whole VAD sections, so a boundary can never land mid-word.
- **Diarizing without transcribing does the same refinement.** The
  *Diarization only* mode (and `POST /api/asr/diarize[/upload]`) returns those
  refined turn boundaries, not the raw diarization segments, and learns
  voiceprints from them — so a file you only diarized improves your voiceprints
  exactly like a file you transcribed, and reports the pending profiles it
  created.

#### "Is this someone I already added?"

Every pending profile is scored against your registered speakers. When one
looks like somebody you already have, the card offers a **Merge into …** button
instead of making you invent a second name for the same person — two profiles
for one person would split their speech between two names in every transcript.
Merging moves the snippets into the existing profile and rebuilds it. Nothing
is ever merged automatically; the flag is a suggestion.

#### "Who is this?" — listen to the unnamed profiles

The Voiceprints pane has two sub-tabs: **Known speakers** and **Unnamed**
(the second carries a count, because an unnamed profile is otherwise easy to
overlook). Every unnamed profile folds out into its own snippet list with the
same play button, filename, source and *Remove* control a named speaker has —
the audio is the only evidence of who the voice is, so it has to be playable
**there**, not summarised as a row of counts. Each card also offers *Consolidate*
(the same outlier pass the known profiles get), *Save name*, *Merge into …* for a
plausible match, and *Discard*. Naming or merging moves the profile to the Known
tab, which the UI switches to for you.

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

### Boundary accuracy

Every speaker transition is cut by **one** engine,
`asr_mcp/speaker/boundary.py::cut_between`, shared by the pipeline, the turn
timeline and the client-side detector. It tries, in order: VAD-section ownership
decided by voiceprint embedding → spectral change → energy dip → quietest frame
→ gap midpoint. Whichever method wins, the two spans either side move to the
*same* cut, so no second of the timeline is ever unclaimed or double-claimed.

The acoustic step is not decoration. Measured against the raw VAD sections as
ground truth (distance from the cut to the nearest speech edge, lower is
better), on 4 ground-truth clips and a 57-minute podcast:

| material | energy only | acoustic |
|---|---|---|
| 0-four-speakers-zh (56.9s) | 1.397 / 3.970 / 6.738 s | **0.000 / 0.550 / 0.550 s** |
| 1-two-speakers-en (16.0s) | 0.864 / 1.530 / 1.696 s | **0.391 / 0.678 / 0.750 s** |
| 2-two-speakers-en (34.0s) | 0.304 / 0.504 / 0.576 s | **0.106 / 0.227 / 0.398 s** |
| 3-two-speakers-en (54.8s) | 0.806 / 1.488 / 1.616 s | **0.550 / 0.634 / 0.690 s** |
| ZO249 podcast (57min, 471 cuts) | 0.448 / 1.168 / 3.792 s | **0.000 / 0.350 / 1.984 s** |

(p50 / p90 / worst.) `boundary_refine.acoustic: false` gives the energy-only
column and is the A/B switch. **`spectral_change` never fired on any of that
real material** — it is implemented and unit-tested on a synthetic signal only,
and its two thresholds remain uncalibrated, so it is currently inert. Roll back
the engine wholesale with `boundary_refine.enabled: false` (every cut at the gap
midpoint). Full method notes and the caveats: **docs/lessons/diarization-and-speakers.md**
(lesson 40) and **[docs/plans/boundary-and-voice-quality-plan.md](docs/plans/boundary-and-voice-quality-plan.md)**.

**Live boundaries** are the client's claim and the server's decision: Silero
trims each turn's leading and trailing frames, a timeline guard rejects a
`start_sample` that goes backwards or runs implausibly far ahead, and a merged
turn carries its pause as silence so the declared span equals the audio actually
sent. All three are reported back to the client. See
**docs/lessons/live-client-design.md** (lesson 42).

### Voiceprint integrity

The store is audited at every startup (`asr_mcp/voiceprint/audit.py`, read-only, 8
findings) and logged. Findings that have exactly one correct answer are then
repaired automatically; anything that would need a human decision is only reported.

**Automatic.** Three findings — `snippets_without_profile`, `profiles_without_snippets`
and `missing_snippet_files` — queue a from-scratch rebuild
(`voiceprint/rebuild.py` → `<data_dir>/voiceprint_rebuild_queue.json`). The server's
idle tick drains the queue when nothing else is running, loading only the VAD and
embedding models (never the ASR backend), and for each flagged speaker:

1. **purge** snippet rows whose audio no longer exists — rows only, never a file that
   resolves, and never a file that merely fails to load;
2. **drop** the derived `voiceprints` row;
3. **rebuild** the embedding from the surviving FLAC snippets, re-applying the
   profile's `pending` and `purity` flags.

This cannot lose material: the snippets and their audio are the source of truth and are
never touched — only a derived row is recomputed. It is also self-healing: an
interrupted rebuild leaves a speaker with snippets but no profile, which is itself a
rebuild target. The other five findings (`near_duplicate_profiles`,
`name_collisions`, `duplicate_snippet_content`, `orphan_snippet_files`,
`stale_speech_totals`) are left alone, because merging profiles is your call.

**Manual.** `GET /api/voiceprint/integrity` runs the same audit for the signed-in
user; `POST /api/voiceprint/integrity/cleanup` fixes byte-identical duplicate
snippets and the audit's dead-row list, then re-refines each affected speaker once.
It is a **dry run unless you pass `dry_run: false`** — check the plan first.

### Consolidation: finding snippets that are not this speaker

A profile is the weighted average of its snippets' embeddings, and a snippet that
belongs to somebody else (a bad diarization turn, a crossed channel, a
mis-attributed span) drags that average toward itself — quietly making the profile
worse for every later recording. **Consolidation** checks that, per speaker or
across the whole library, from the Voiceprints tab (*Consolidate* on a speaker card,
*Consolidate All* in the toolbar).

Each pass embeds every snippet once, scores the snippets against a robust centre of
the profile, and marks the outliers for review. The report carries the mean
intra-profile distance **before and after**, so the effect of the re-fit is visible
rather than asserted. A snippet is only accused when it exceeds *both* a robust
cut (`median + 6·MAD·1.4826`) **and** an absolute floor of `0.35` — the same
distance the naming gate uses, because "is this the same person" is the same
question. The pass is deliberately conservative: at most 20% of a speaker's
snippets are flagged, a profile of fewer than 3 is never cut, and the two failures
are asymmetric — under-flagging just means running it again, over-flagging destroys
good audio on one bad run.

**Flagged snippets are excluded from the profile immediately; nothing is deleted.**
A flag is an accusation, so it arrives with its evidence: the score, the reason, and
a play button for the audio. From the **Flagged snippets** card you can

- **Keep** one — the mark is withdrawn; re-run consolidation to fold the audio back
  into the profile;
- **Remove** one — a confirm dialog, then the snippet and its audio are gone;
- **Remove all flagged** — the UI always shows you the dry-run plan (count and
  seconds) first and asks again before deleting.

Re-running consolidation also clears marks it no longer makes, so a profile that
has been fixed stops being reported. If everything would be flagged, nothing is
re-fitted — an unapproved set is worse than the profile you already have.

A whole-library run over many speakers takes real GPU time (roughly 25 min per 219
snippets on a GTX 1650) and the progress is streamed, so it is safe to start and
walk away.

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
│   │   ├── security.py        # API key + DB token auth, admin gate, path validation
│   │   ├── admin_router.py    # /admin/users list + disable/enable/delete (require_admin)
│   │   ├── auth_router.py     # login/logout/register/verify + Google sign-in
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
│   │   ├── audit.py           # Read-only integrity audit
│   │   ├── rebuild.py         # Automatic from-scratch profile rebuild + queue
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
│                            # test_voiceprint_consolidation.py = outlier snippets (load by path)
│                            # test_voiceprint_panes.py = the Voiceprints tab's two sub-tabs
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
│   ├── lessons/                # Evidence behind each AGENTS.md rule
│   └── plans/                  # Improvement plans (boundary + voice quality)
└── README.md
```
