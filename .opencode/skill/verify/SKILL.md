---
name: verify
description: Use when finishing a change in asr-mcp and deciding how much testing it deserves — running pytest, the container suite, docker compose cp, py_compile, git diff --check, or checking whether a change needs tests at all. Also use before writing a test that constructs Settings, before running any suite in the container, and when a test passes locally but fails in the image.
---

# Verifying a change in asr-mcp

The point of this skill is to spend verification effort where it can actually
catch a bug. A doc edit does not need a container suite. A shared resolver does.
Choosing wrong in either direction wastes the user's time.

## Step 1 — size the change, then stop

Pick the cheapest rung that covers the blast radius. Do not climb past it.

| What changed | Verification | Not this |
|---|---|---|
| Markdown, comments, `.env.example`, docs | `git diff --check` | any pytest run |
| One line in a leaf module, no signature change | `python3 -m py_compile <file>` | any pytest run |
| A pure function or its tests | the new test file, on the host | the full suite |
| A signature change or a shared helper | the tests that call it, container | the full suite |
| A router, `security.py`, `db/manager.py`, or anything another module imports | the **full suite**, container | — |

**The full suite is for the last row only.** It is a two-minute, high-output
run; it is not a reflex.

If the change is in the last row, say so and run it. Otherwise stop at the
rung that covers the change and report what you ran.

Do not add a test for a rename, a constant, or a log line. A test that cannot
fail is not a test; it is a maintenance cost.

## Step 2 — can it run on the host?

Host `python3` has **only numpy, pytest and bcrypt**. No pydantic, no fastapi,
no sqlalchemy, no torch, no httpx.

So a test that imports any of those must run in the container. Check the test
file's imports before choosing a rung — `from asr_mcp.config.settings import
Settings` alone is enough to send it to the container.

`py_compile` is the only host-side check that works on everything.

## Step 3 — running in the container

Container service name is `asr-mcp` (container `asr-mcp-app`). **Source is baked
into the image; there is no source bind mount**, so changed files must be copied
in first.

```bash
docker compose cp asr_mcp/. asr-mcp:/app/asr_mcp
docker compose cp tests/. asr-mcp:/app/tests
docker compose exec -T asr-mcp python3 -m pytest tests/test_x.py -q
```

**The trailing `/.` is mandatory.** A bare `docker compose cp asr_mcp
asr-mcp:/app/asr_mcp` NESTS, creating `/app/asr_mcp/asr_mcp`, and you then test
the old code while believing you tested the new. Check after copying:

```bash
docker compose exec -T asr-mcp ls -d /app/asr_mcp/asr_mcp /app/tests/tests
```

**Always delegate the run to a `general` subagent.** The suite output is long
and two concurrent suites OOM-kill the container. Ask the subagent for the
summary line and the failing test ids only.

## Step 4 — while writing a test

**Pin every `Settings` field the assertion depends on.** `Settings` reads the
environment, so a fixture that pins only what it is testing inherits
`TRANSCRIBE_PREFIX`, `TRANSCRIBE_EXTERNAL_URL`, `TRANSCRIBE_HTPASSWD_PATH` from
the running deployment. Three separate wrong-result runs came from this. It
manifests as a test that passes locally and fails in the image, which reads like
a flake and is not.

```python
def _settings(tmp_path, **overrides):
    return Settings(
        session_secret="s", db_path=str(tmp_path / "asr.db"),
        prefix="", external_url="", htpasswd_path=str(tmp_path / "none"),
        **overrides,
    )
```

**Assert a URL whole.** `"/asr-mcp" in link` is satisfied by
`/asr-mcp/asr-mcp/api/auth/verify`. Compare the full prefix and path.

**Check `async def` before wrapping in `asyncio.run`.** A sync helper wrapped
that way raises `ValueError: a coroutine was expected` — and the error message
helpfully contains the value you wanted, so read it before rewriting.

**A helper that needs a request gets a request.** Passing the app where
`request.app.state` is read gives `AttributeError: ... has no attribute 'app'`.

Load `voiceprint/service.py` **by path** with torch/soundfile stubbed. Importing
`asr_mcp.voiceprint` puts the real package in `sys.modules` and breaks the
stubbed loaders in every other test module.

## Step 5 — live probing, when it is genuinely warranted

Only when the user reports a production symptom a unit test cannot reproduce.

**The app does not mount under `TRANSCRIBE_PREFIX`.** nginx strips the prefix.
Curling `http://localhost:8087/asr-mcp/api/...` hits the auth middleware's 302
and looks like a routing bug. Curl `http://localhost:8087/api/...`.

After copying source into a running container you must
`docker compose restart asr-mcp` — the old code is in memory.

## Reporting

State the rung you chose and why, the exact command, and the counts. If you
skipped a rung you considered, say so in one clause. Do not report a test you
did not run.
