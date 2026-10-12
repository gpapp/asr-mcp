# Identity, registration and the admin

The evidence behind lessons 47–50. This is the port of `mem-mcp`'s
authentication/registration/admin stack into `asr-mcp`, plus the per-user
separation that had to land first.

The work is in five phases: (1) per-user separation, (2) the credential store +
email registration, (3) Google sign-in, (4) the admin, (5) these docs. Each
phase is verified by its own test module, and the whole suite is green at each
step (655 → 760 → 822 → 844 passed, 137 skipped throughout).

---

## 1. Phase 1 — the separation was never a schema problem

The tables were already user-scoped when this started: `voiceprints` has a
composite `(user_id, name)` primary key, `snippets`/`transcripts`/`api_tokens`
carry an indexed `user_id`. What was missing was every *call site* passing one.

The failure mode is specific and silent: **every DB accessor defaults to
`DEFAULT_USER = "default"`.** A call site that forgets `user_id=` does not raise —
it reads and writes the `default` user's rows. The live store was the proof: 47
voiceprints and 2003 snippets under `memories`, while `/health` reported the
`default` user's count (0) on a public endpoint.

Ten defects, all found by reading the code:

| # | Where | Defect |
|---|---|---|
| 1 | `speaker/service.py` | `SpeakerService` wrapped only `VoiceprintDB` and **never** passed `user_id`. Every method — `register`, `identify`, `get`, `list`, `remove`, `to_dict` — bound to `default`. This was a second, parallel voiceprint store view independent of who was signed in. |
| 2 | `api/speaker_router.py` | All five routes used `Depends(verify_api_key)`, which returns the **key**, never the user. No route called `get_current_user`. |
| 3 | `api/mcp_router.py` | `identify_speaker` / `list_speakers` / `read_resource` went through the unscoped `SpeakerService`. Worse, `diarize_audio` passed **no `known_speakers` at all** — MCP diarization never matched voiceprints, so it never even had the chance to scope. |
| 4 | `api/asr_router.py` | `POST /transcribe` used `get_session_user(request) or "default"` — an `X-API-Key` caller matched against the default user's profiles and its run was neither learned nor saved to History. |
| 5 | `api/asr_router.py` | `/activity/stream` broadcast **every** user's job meta (filename, stage, progress, `user_id`) to all subscribers. Only `can_cancel` was owner-checked. |
| 6 | `api/asr_router.py` | `/active/stream` returned 200 + a *scrubbed* stream for another user's job (`result`/`partial_text`/`new_segments` dropped) — progress, filename and the owner's `user_id` still leaked. Now a 404. |
| 7 | `api/asr_router.py` vs `api/security.py` | A static API key resolved to `DEFAULT_USER` on the WebSocket but to `user_<sha256(key)[:16]>` on HTTP. **One key had two different voiceprint scopes depending on the transport.** |
| 8 | `api/speaker_router.py` | Called `SpeakerService.register_from_audio`, which does not exist on that class (only `VoiceprintService` has it) → `AttributeError` → a real 500 on every `POST /api/speaker/register`. |
| 9 | `server.py` + `voiceprint/service.py` | Counts with no `user_id`: `/health` and the startup log reported `default`'s numbers. |
| 10 | `api/voiceprint_router.py` | The single-active-job guard was global — user A's running `consolidate` answered 409 to user B. |

### The fix that matters: one resolver, both transports

`resolve_api_key_user(x_api_key, settings=None)` in `api/security.py` is now the
only place a key becomes an identity:

```python
token_user = lookup_db_token(x_api_key, settings)      # a DB token → its owner
if token_user:
    return token_user
if settings.api_keys and x_api_key in settings.api_key_set:
    return f"user_{hashlib.sha256(x_api_key.encode()).hexdigest()[:16]}"
return None
```

Three deliberate properties:

- **A static key never resolves to `default`.** It resolves to a synthetic
  `user_<sha16>`. Without this, every holder of the one shared key shares one
  voiceprint library — which is the whole problem being solved.
- **An unresolvable key returns `None`, not a fallback id.** A caller that gets
  `None` refuses. A caller that gets `"default"` silently shares.
- **`_websocket_user` calls the same function.** Defect 7 existed because the two
  transports had two derivations; sharing one resolver is what makes it
  structurally impossible.

`lookup_db_token` also gained a `settings` parameter, because it previously
opened the process-global `get_settings().db_path`. The token rung and the
static-key rung now read the SAME store — a mismatch there is another silent
scope split.

### Cancelling another user's job

`/api/asr/active/cancel` already used `job_state.get_for_user(user_id)`, so it
answered 404 for another user's job. What was wrong was everything *around* it:
`/activity/stream` advertised the job as cancellable, and `/active/stream`
served a partial view of it. The `_own_job(meta)` helper now returns `None`
whenever `meta.get("user_id") != user_id`, and the snapshot sets
`"active": own is not None` — a user whose colleague is transcribing sees
"no active job", not a foreign filename with a Cancel button.

`tests/test_user_scoping.py` (22 tests) pins all of this, including a textual
assertion that `_websocket_user` no longer contains `sha256` or `api_key_set`
(it must derive identity through the shared resolver, never re-derive it).

---

## 2. Phase 2 — the credential store

`asr_mcp/auth/` is stdlib-only by design (`passwords`, `mailer`, `store`), so the
store can be tested with no ORM, no web framework and no SMTP server. The tables
live in the existing `db/models.py` on the SQLAlchemy DB (binding decision #2),
reusing `init_db`'s additive-ALTER pattern.

### `check_password`, not `verify_password`

`asr_mcp/api/auth.py` already has `verify_password(stored_hash, password)` for
htpasswd, with the **hash first**. mem-mcp's scrypt function is
`verify_password(password, encoded)` — the **password first**. Porting it under
the same name gives two same-named, opposite-order functions in one codebase, and
the compiler catches neither. The port is named `check_password(password,
encoded)` and the asymmetry is deliberate.

### Three refusals, no upsert

`create(username, password, email="")` refuses a taken username, a taken
address, and an out-of-policy password. It never upserts — `registering twice
refuses AND does not reset the password` is a test, because a store that
"helpfully" updated the row would let anyone take over an account by
re-registering it.

The row is created **UNVERIFIED**, and `check_login` returns a *reason* for an
unverified or disabled account. That is what lets the route answer 403-with-a-
reason instead of a misleading 401.

### Throttle-then-flag, deliberately backwards

`_require_registration(method, request, settings)` evaluates the per-client
throttle (10 attempts / 600 s) **before** the registration flag. The obvious
order — check the flag first, return 404 — is the wrong one: counting only
successful attempts would remove the rate limit from a still-mounted route the
moment the operator turns registration off. A disabled method is a **404**, not a
403, so a disabled route is indistinguishable from an unmounted one.

`_signup_client` takes the **rightmost** `X-Forwarded-For` hop, because nginx
appends. The leftmost hop is client-chosen.

### Rollback, or no half-made account

`POST /auth/register` creates the row, issues the token, builds the link, then
sends the mail off-loop via `asyncio.to_thread`. A mail failure calls
`store.delete` and answers 503. Registration **never signs anybody in** — a
confirmation link in a mailbox is not proof the person who clicked it is the
person who will use the account, and auto-sign-in would make email verification
a no-op.

### Secret hygiene

The verification token is stored as sha256 only; `public()`/`list_all()` drop
`password_hash` and `verification_token`; the hash is stored verbatim as
`scrypt$n$r$p$dklen$salt$hash` so the cost parameters travel with each credential
and N can be raised without locking anyone out. `check_password` never raises on
a malformed hash — it returns `False` — and rejects a digest whose length
disagrees with its stored `dklen` (which is what stops a truncated hash
verifying against itself).

### Open mode is refused while registration is on

With `TRANSCRIBE_API_KEYS` empty, `get_current_user` used to hand every
unauthenticated caller the `default` identity — one shared library. Turning
registration on is exactly the moment that stops being acceptable, so the
open-mode branch now answers 401 when `settings.registration_enabled`.

---

## 3. Phase 3 — Google sign-in

`asr_mcp/auth/google_auth.py` is a stdlib `urllib` port: `authorization_url`,
`exchange_code`, `fetch_userinfo`, one exception type (`GoogleOAuthError`, whose
message never contains a code, token or secret), and a single network seam
(`_request_json`) with an injectable `opener` — which is what makes the whole
client testable with no network and no JWT library.

### The state is session data, and it is consumed either way

`google_start` mints `secrets.token_urlsafe(24)` into `request.session["oauth_state"]`
— not a cookie of our own, because the session is already HttpOnly and
server-side. `google_callback` **blanks the state first and compares second**:

```python
expected = str(request.session.get("oauth_state") or "")
request.session["oauth_state"] = ""
if not expected or not state or not hmac.compare_digest(state, expected):
    return RedirectResponse(url=f"{prefix}/login?google=state", status_code=302)
```

Consumed either way, so a state is replayable exactly once. `hmac.compare_digest`
because the state is compared against session data an attacker may have planted.

### A sign-up's `user_id` IS the address (binding decision #3)

`_google_signup` returns the normalised email address as the user id. Google has
already verified the address, so there is no confirmation mail. It refuses when:

- the address is empty, or `email_verified is not True` (only an explicit `True`
  counts — the truthy string `"true"` is refused);
- registration is off;
- the address is already a vault key — "Google has proved who the person is; it
  has not proved which of their accounts they meant."

`/api/auth/google/start` is gated on **configuration, not the registration
flag**. Turning registration off must not sign out existing users.

### Identity is resolved by subject, never by address

`email_taken` deliberately does **not** consult `google_identities`. A Google
vault is keyed on the address; a password account is keyed on a username that
`normalise_username` refuses to let contain `@`. So a username signup for an
address a Google vault uses is allowed, and the two accounts stay separate. The
consequence is real and pinned by a test: `test_a_password_account_does_not_block_the_address`
— a password account under a different username does **not** block a Google
signup at that address.

The `google_identities` primary key is `(provider, subject)` — **not**
`(provider, subject, user_id)` — so one subject resolves to exactly one vault
forever, and `link_google_identity` raises if that subject is ever asked to
point somewhere else.

---

## 4. Phase 4 — the admin

`TRANSCRIBE_ADMIN_USER` (default empty ⇒ **nobody**) is a deployment constant,
not a role column. `is_admin_user` is a plain `==`; `require_admin` is a
FastAPI dependency that raises 403 naming the variable to set. The 403 text is
deliberately instructional: an operator who forgot to configure it needs to know
the name of the variable, not just that they are refused.

Three things are gated, each because of what it protects:

| Route | What it protects |
|---|---|
| `GET /api/admin/users` | Reads every account's email address |
| `DELETE /api/admin/users/{id}` | With `?purge=true`, drops another user's voiceprints, snippets, transcripts and audio |
| `POST /shutdown` | Stops a process other users queue jobs against — it cancels their jobs |

`POST /shutdown` had **no role check at all** before this phase; the only gate
was the middleware, which any session or any valid API key satisfies.

### Delete keeps the data unless you opt in

An admin delete `store.delete`s the credential row, unlinks the Google
identities and drops the API token, and **keeps** the voiceprints, snippets,
transcripts and audio — reporting the counts so the operator sees what is still
there. `?purge=true` additionally runs the three `delete_all_for_user` calls and
`rmtree`s `voices/<user_id>/`. One boolean argument must not be able to destroy
another user's recordings implicitly.

The counts are read **before** any delete, so the response reports what was there
rather than what is left.

Two refusals come first: the account the caller is signed in as (an admin who
deletes themselves is locked out with no way back), and the configured
`TRANSCRIBE_ADMIN_USER` account itself (an app configured to name an account
that no longer exists has no admin, and the only way back is an env edit and a
restart).

### Rows are emitted for identities that own data but have no credential

This is the asr-mcp divergence from mem-mcp. A static `TRANSCRIBE_API_KEYS`
value maps to a synthetic `user_<hash16>` id with **no credential row at all**,
so mem-mcp's `store.list_all()` alone would render a table that hides a deleted
key's recordings while still holding them. `_user_table` unions
`store.list_all()` → `store.list_all_google_identities()` → `store._htpasswd_keys()`
→ `VoiceprintDB.user_ids() ∪ SnippetDB.user_ids() ∪ TranscriptDB.user_ids()`,
deduplicated, each data-only identity carrying `sources == []`.

htpasswd wins for display and forces `can_manage`/`can_delete` to `False` — a
row that came from a file the operator edits cannot be suspended from the UI.

`_voices_dir` resolves the user id against the base and returns `None` if it
escapes, so a `../other` path parameter cannot delete a neighbour's recordings.

---

## 5. The traps each phase surfaced

1. **`_user_table` is sync.** Wrapping a route helper in `asyncio.run` without
   checking for `async def` gives `ValueError: a coroutine was expected` — and in
   this case the error message *contained the correct table*, which was itself
   useful evidence. Check before wrapping.
2. **Ambiguous response keys.** An early delete response emitted both
   `"purged": bool(purge)` and `"purge": <counts>`. A test failed with
   `TypeError: 'bool' object is not subscriptable`, and it was right. One key:
   `"purged"` holds the counts dict or `None`.
3. **The container's real htpasswd leaks into tests.** `Settings()` picks up
   `TRANSCRIBE_HTPASSWD_PATH` from the environment, so a test that does not pin
   `htpasswd_path` sees the live `memories` account as a table row. Pin it.
   Same for `prefix` (`TRANSCRIBE_PREFIX=/asr-mcp` in the container breaks any
   no-prefix URL assertion) and any test that inherits ambient config — a test
   that only passes locally is a test that will be deleted.
4. **`_store(request, …)` needs a request.** It reads `request.app.state.htpasswd_users`;
   passing an app raises `AttributeError: 'FakeApp' object has no attribute 'app'`.
5. **A fake that bypasses normalisation tests nothing.** The Google route test's
   fake returned `dict(self.info)` verbatim, skipping `fetch_userinfo`'s
   `verified = payload.get("email_verified") is True` — so the truthy *string*
   `"true"` slipped past the refusal gate. The fake was fixed to model the real
   function; the product was correct.
6. **An unverified account cannot log in, even with the right password.** The
   disable/enable test failed with `assert '' == 'ada'` until the fixture issued
   and redeemed a verification token. The fix was a `_verified()` helper, not a
   weakened assertion.
7. **A failing fake must raise, not return.** `RecordingOpener` *returned* a
   queued `HTTPError`, and because an `HTTPError` satisfies the context-manager
   protocol its `.read()` served the error body as a 200 — the fake silently took
   the success path. `__call__` now raises.
8. **An unsatisfiable negative assertion.** `assert "&state=a" not in url` cannot
   pass when the state value itself starts with `a`; assert the positively-encoded
   form (`state=a+b%26c%3Dd`) and the absence of the *raw* form instead.
9. **A success path clears the session.** `assert session.get("oauth_state") == ""`
   fails when the callback succeeds, because `request.session.clear()` removes
   the key entirely (`None`). Assert `"oauth_state" not in session`.
10. **A failure must not log you out.** The failure branches blank `oauth_state`
    and leave a pre-existing session user alone. Asserting the failure branch
    wiped `session["user"]` was asserting the wrong invariant; the right one is
    that the failure branch never *writes* the Google address.
11. **`SnippetDB.count(speaker_name=None)` counted nothing.** It did an
    unconditional `filter_by(user_id, speaker_name)`, so `None` became
    `WHERE speaker_name IS NULL` — exactly what the admin purge needs and could
    not get. Now guarded with `if speaker_name is not None`, the same pattern
    `find_duplicate` already used.
12. **`TokenDB` had no delete method at all**, so an admin delete could not
    remove the token row. It does now.
13. **A URL assertion on a substring.** `assert "https://host/asr-mcp" in body`
    is satisfied by `https://host/asr-mcp/asr-mcp/api/auth/verify`, which is how
    the doubled prefix shipped green. Extract the link and assert
    `link.startswith(...)` on the whole URL.
14. **A fixture that inherits the deployment's `Settings`.** `Settings()` reads
    the environment, so a fixture that pins only what it is *testing* silently
    inherits `TRANSCRIBE_PREFIX`, `TRANSCRIBE_EXTERNAL_URL` and
    `TRANSCRIBE_HTPASSWD_PATH` from the running container. This produced three
    separate wrong-result runs before it was noticed — an admin table showing a
    live account, a Google redirect URI with `/asr-mcp/asr-mcp`, and a
    confirmation link addressed to the real deployment. Pin every field the
    assertion depends on. The host has no pydantic, so the leak cannot surface
    outside the container.

---

## 5b. Lesson 51 — the prefix appeared twice, and the link 404'd

A live deployment reported: *the confirmation mail arrives, but the page does
not accept the token.*

The mail was fine and the token was fine. The **link** was
`https://mcp.itworks.hu/asr-mcp/asr-mcp/api/auth/verify?token=…`.

`_external_url` returned `TRANSCRIBE_EXTERNAL_URL` verbatim, and the route then
appended `settings.prefix`:

```python
link = f"{_external_url(settings, request)}{settings.prefix}/api/auth/verify?token=…"
```

Both spellings of `TRANSCRIBE_EXTERNAL_URL` are reasonable and both occur in
practice:

| Value | What the operator meant | What the code built |
|---|---|---|
| `https://host` | the bare origin; prefix appended here | `https://host/asr-mcp/…` ✓ |
| `https://host/asr-mcp` | the page you actually paste into a browser | `https://host/asr-mcp/asr-mcp/…` ✗ |

The second is arguably the *more* natural value — it is the app's real public
URL — and nothing in the name or the docs said the prefix must be stripped. So
the operator set it, and every confirmation link 404'd.

**The symptom is why this is a lesson and not a typo.** To the user, a dead
confirmation link is *identical* to a token that was refused: the mail arrives,
the page says it did not work, and there is no way to tell "wrong URL" from
"already spent" from "bad token". The route deliberately answers all three with
the same `?verified=0` (that is correct — it stops link-probing), which also
means a URL bug is invisible from the response alone.

`_public_base` now appends the prefix only when the base does not already end
with it, on a **path-segment boundary**:

```python
prefix = str(settings.prefix or "").strip("/")
if base == f"/{prefix}" or base.endswith(f"/{prefix}"):
    return base
return f"{base}/{prefix}"
```

The boundary matters: a prefix of `/asr` against a base of `…/asr-mcp` does
**not** already end with the prefix, so it is appended (`…/asr-mcp/asr`) rather
than silently swallowed by a bare `endswith`. `tests/test_auth_routes.py` pins
all four spellings, the substring case, and the no-prefix case.

### Both links come from one helper

The same expression built the Google `redirect_uri`, so the same bug would have
made every Google sign-in fail at Google with an error page that names neither
the URI nor the reason. `_google_callback_url` now calls `_public_base` too, and
a test asserts the confirmation link and the redirect URI agree on the base —
they are separate call sites, which is exactly how they would drift apart again.

### Why the test did not catch it

The pre-existing test asserted a **substring**:

```python
assert "https://asr.example.org/asr-mcp" in sent["body"]
```

`…/asr-mcp/asr-mcp/api/auth/verify?token=…` *contains* `…/asr-mcp`. The assertion
passed on the broken URL. A URL assertion has to be on the whole URL — the new
tests extract the link and assert `link.startswith(...)` and
`"/asr-mcp/asr-mcp/" not in link`.

The same test also silently depended on ambient config: `Settings()` picks up
`TRANSCRIBE_EXTERNAL_URL` and `TRANSCRIBE_PREFIX` from the environment, so once
a real `.env` set them the test began producing links to `mcp.itworks.hu`. The
`_settings` helper now **pins both to `""`**. That is the second half of the
same lesson — a test that inherits the deployment's config is a test that
passes until somebody deploys.

---

## 6. What is still open

- There is **no password-reset flow**, by design (mem-mcp has none either). A
  forgotten password means an admin re-issues via `set_password` or deletes the
  account. Adding one is a feature, not a bug fix, and it needs a reset-token
  table with the same single-use discipline as the verification token.
- The registration throttle is **in-process and per-worker**. Two workers means
  two windows. A shared store is the correct fix if this is ever run with more
  than one uvicorn worker.
- Google sign-in's `redirect_uri` derives from `TRANSCRIBE_EXTERNAL_URL` (then
  `request.base_url`), so a deployment whose public URL differs from the bind
  address must set it or Google will reject the callback.
- The admin user table reads a live htpasswd file on every request
  (`app.state.htpasswd_users`), which is parsed once at startup — an operator
  editing the file must restart (or the change is invisible), which is the
  pre-existing htpasswd behaviour, not a regression.
