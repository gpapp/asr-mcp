"""One user's speakers, history and jobs stay theirs.

The data tables were always user-scoped (``voiceprints`` has the composite
``(user_id, name)`` key, ``snippets``/``transcripts``/``api_tokens`` carry a
``user_id``), and the diarizer never touches the database -- every matching
path funnels through ``asr_router._load_known_speakers(settings, user_id)``.
So the separation was never a schema problem. It was a set of call sites that
forgot to say WHICH user, and every one of them silently fell back to the
``default`` user's rows instead of failing.

That fallback is what makes these tests worth having. ``VoiceprintDB`` takes
``user_id: str = DEFAULT_USER`` on every accessor, so a route that resolves no
identity does not error -- it quietly reads and writes ``default``. On the live
store that meant a second, parallel view of the library that always saw the
same seven profiles no matter who was signed in.

Three concrete holes are pinned here, plus the two stream endpoints that used
to publish other people's jobs:

* ``security.resolve_api_key_user`` -- one definition for both transports.  A
  static ``TRANSCRIBE_API_KEYS`` value used to resolve to ``user_<hash16>`` over
  HTTP and to ``default`` over the WebSocket, so the same key read two
  different voiceprint libraries depending on which door the client used.
* ``/activity/stream`` and ``/active/stream`` -- the first broadcast every
  user's job metadata to every subscriber (only ``can_cancel`` was checked),
  and the second answered a non-owner's request with a scrubbed stream rather
  than a 404, still leaking the owner's filename, progress and ``user_id``.
* ``/active/cancel`` -- already owner-scoped through
  ``job_state.get_for_user``; pinned so it cannot regress.

The route wiring (which user each endpoint resolves, that the MCP diarizer
finally passes ``known_speakers``) is pinned textually at the bottom: those
functions need the GPU models to run, and a wiring change is exactly the kind
of regression that never shows up in a behaviour test because the behaviour
never gets far enough to be wrong.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy", reason="DB layer needed")
pytest.importorskip("fastapi", reason="the routers need the web stack")
pytest.importorskip("torch", reason="asr_router and SpeakerService pull in the audio stack")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, size=192).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeRequest:
    """Just enough for ``get_session_user`` / the SSE handlers."""

    def __init__(self, session: dict | None = None):
        self.scope = {"session": session or {}}


@pytest.fixture
def db(tmp_path):
    from asr_mcp.db.manager import DatabaseManager

    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    try:
        yield mgr
    finally:
        mgr.close()


def _service(db, tmp_path):
    from asr_mcp.speaker.service import SpeakerService

    return SpeakerService(tmp_path, db)


# ---------------------------------------------------------------------------
# SpeakerService
# ---------------------------------------------------------------------------


class TestSpeakerServiceScoping:
    def test_two_users_never_see_each_others_speakers(self, db, tmp_path):
        service = _service(db, tmp_path)
        service.register_speaker("Ada", _unit(1), total_speech_sec=10, user_id="ada")
        service.register_speaker("Bo", _unit(2), total_speech_sec=10, user_id="bo")

        assert set(service.list_speakers(user_id="ada")) == {"Ada"}
        assert set(service.list_speakers(user_id="bo")) == {"Bo"}

    def test_identify_is_scoped_to_the_caller(self, db, tmp_path):
        service = _service(db, tmp_path)
        service.register_speaker("Ada", _unit(1), user_id="ada")
        service.register_speaker("Bo", _unit(2), user_id="bo")

        # Ada's own embedding, asked for as Bo. The search has no distance
        # cutoff, so it still answers -- with Bo, at 0.94 away. What must never
        # happen is Ada's profile appearing in Bo's answer, which is what
        # matching her voice under Bo's name would take.
        hits = service.identify_speaker(_unit(1), top_k=5, user_id="bo")
        assert [h["name"] for h in hits] == ["Bo"]
        assert [h["name"] for h in service.identify_speaker(_unit(1), user_id="ada")] == ["Ada"]

    def test_deleting_yours_leaves_the_other_users_alone(self, db, tmp_path):
        service = _service(db, tmp_path)
        service.register_speaker("Ada", _unit(1), user_id="ada")
        service.register_speaker("Ada", _unit(2), user_id="bo")

        assert service.remove_speaker("Ada", user_id="ada") is True
        assert service.list_speakers(user_id="ada") == {}
        assert set(service.list_speakers(user_id="bo")) == {"Ada"}

    def test_a_caller_who_names_no_user_gets_the_default_rows(self, db, tmp_path):
        """The documented fallback, pinned so it stays a FALLBACK.

        ``user_id=DEFAULT_USER`` is what lets a forgotten argument read the
        ``default`` rows instead of failing. That is the behaviour the routers
        are supposed to avoid, not the behaviour they rely on.
        """
        service = _service(db, tmp_path)
        service.register_speaker("Ada", _unit(1), user_id="default")
        service.register_speaker("Bo", _unit(2), user_id="bo")

        assert set(service.list_speakers()) == {"Ada"}


# ---------------------------------------------------------------------------
# Which user is a key?
# ---------------------------------------------------------------------------


def _settings(**over):
    from asr_mcp.config.settings import Settings

    base = {"session_secret": "test-secret", "api_keys": ["static-key-abc"]}
    base.update(over)
    return Settings(**base)
class TestApiKeyResolution:
    def test_a_static_key_is_the_same_user_on_http_and_the_websocket(self):
        """The parity regression: two transports, ONE answer.

        The WebSocket handshake used to map any valid static key to
        ``default`` while HTTP mapped it to ``user_<sha256(key)[:16]>``. A
        client that matched its own speakers over HTTP lost them all the
        moment it opened a socket, because it was suddenly reading somebody
        else's library.
        """
        from asr_mcp.api.security import resolve_api_key_user

        async def both():
            from asr_mcp.api.asr_router import _websocket_user

            http_user = resolve_api_key_user("static-key-abc", _settings())
            ws_user = await _websocket_user(
                _FakeWebSocket(headers={"x-api-key": "static-key-abc"}), _settings()
            )
            return http_user, ws_user

        http_user, ws_user = asyncio.run(both())
        assert http_user == ws_user
        assert http_user != "default"
        assert http_user.startswith("user_")

    def test_a_db_token_resolves_to_its_owner(self, db, tmp_path):
        from asr_mcp.db.manager import TokenDB
        from asr_mcp.api.security import resolve_api_key_user

        token = TokenDB(db).create("ada")
        # Same settings object that decides the static-key rung, so the token
        # rung reads the same store instead of the process-global db_path.
        assert resolve_api_key_user(
            token, _settings(db_path=str(tmp_path / "asr.db"))) == "ada"

    def test_an_unknown_key_is_not_a_user(self):
        from asr_mcp.api.security import resolve_api_key_user

        assert resolve_api_key_user("not-a-key", _settings()) is None
        assert resolve_api_key_user("", _settings()) is None

    def test_the_session_beats_a_key(self):
        from asr_mcp.api.security import get_current_user

        user = asyncio.run(get_current_user(
            _FakeRequest(session={"user": "ada"}),
            x_api_key="static-key-abc",
            settings=_settings(),
        ))
        assert user == "ada"

    def test_an_api_key_caller_is_not_the_default_user(self):
        """The same call path the routers now use.

        A static key must not collapse onto ``default``: that is how an
        unauthenticated caller ends up reading the shared library.
        """
        from asr_mcp.api.security import get_current_user

        user = asyncio.run(get_current_user(
            _FakeRequest(), x_api_key="static-key-abc", settings=_settings(),
        ))
        assert user != "default"


class _FakeWebSocket:
    """Enough of a WebSocket for ``_websocket_user``."""

    def __init__(self, headers=None, query_params=None):
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = query_params or {}
        self.scope = {"session": {}}


# ---------------------------------------------------------------------------
# Jobs belong to whoever started them
# ---------------------------------------------------------------------------


def _start(mode: str = "transcribe", user_id: str = "ada") -> object:
    from asr_mcp.core import job_state

    return job_state.start_job(mode=mode, filename="a.wav", user_id=user_id)


def _finish(job):
    from asr_mcp.core import job_state

    job_state.finish(job, "done")


def _first_sse_event(response) -> dict:
    """The first ``data:`` payload of a StreamingResponse, then close it."""
    from asr_mcp.core import job_state

    async def _one(agen):
        try:
            async for chunk in agen:
                line = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                for part in line.splitlines():
                    if part.startswith("data: "):
                        return json.loads(part[len("data: "):])
                return None
            return None
        finally:
            await agen.aclose()
            for sub in list(job_state._activity_subs):
                job_state.unsubscribe_activity(sub)

    return asyncio.run(_one(response.body_iterator))


class TestJobIsolation:
    def test_another_users_job_cannot_be_cancelled(self):
        from asr_mcp.api.asr_router import active_job_cancel

        job = _start(user_id="ada")
        try:
            resp = asyncio.run(active_job_cancel(_FakeRequest(), user_id="bo"))
            assert resp.status_code == 404
            assert job.cancel_requested is False
        finally:
            _finish(job)

    def test_the_owner_can_still_cancel(self):
        from asr_mcp.api.asr_router import active_job_cancel

        job = _start(user_id="ada")
        try:
            # Success is a plain dict (FastAPI serialises it to a 200); the
            # 404/409 paths are JSONResponse. Both are pinned below.
            resp = asyncio.run(active_job_cancel(_FakeRequest(), user_id="ada"))
            assert resp["status"] == "cancelling"
            assert job.cancel_requested is True
        finally:
            _finish(job)

    def test_activity_stream_reports_only_the_callers_own_job(self):
        from asr_mcp.api.asr_router import activity_stream

        job = _start(user_id="ada")
        try:
            mine = asyncio.run(activity_stream(_FakeRequest(), user_id="ada"))
            theirs = asyncio.run(activity_stream(_FakeRequest(), user_id="bo"))

            own = _first_sse_event(mine)
            assert own["active"] is True
            assert own["job"]["user_id"] == "ada"
            assert own["job"]["can_cancel"] is True

            # Not a scrubbed job: no job at all. Bo's snapshot must not carry
            # Ada's filename, progress or identity for her to then strip.
            other = _first_sse_event(theirs)
            assert other == {"active": False, "job": None}
        finally:
            _finish(job)

    def test_a_diarization_job_is_never_cancellable(self):
        from asr_mcp.api.asr_router import activity_stream

        job = _start(mode="diarize", user_id="ada")
        try:
            snap = _first_sse_event(
                asyncio.run(activity_stream(_FakeRequest(), user_id="ada")))
            assert snap["job"]["mode"] == "diarize"
            assert snap["job"]["can_cancel"] is False
        finally:
            _finish(job)

    def test_active_stream_refuses_another_users_job(self):
        """A 404, not a scrubbed stream.

        The scrubbed version still streamed the owner's filename, progress
        and ``user_id`` to anybody who asked for it.
        """
        from asr_mcp.api.asr_router import active_job_stream

        job = _start(user_id="ada")
        try:
            resp = asyncio.run(active_job_stream(_FakeRequest(), user_id="bo"))
            assert resp.status_code == 404
        finally:
            _finish(job)

    def test_the_owner_can_reattach_to_its_own_job(self):
        from asr_mcp.api.asr_router import active_job_stream

        job = _start(user_id="ada")
        try:
            resp = asyncio.run(active_job_stream(_FakeRequest(), user_id="ada"))
            assert resp.status_code == 200
            assert resp.media_type == "text/event-stream"
        finally:
            _finish(job)

    def test_a_job_in_another_mode_is_not_a_cancellable_transcription(self):
        """Own job, wrong mode: 409, not a silent cancel of something else."""
        from asr_mcp.api.asr_router import active_job_cancel

        job = _start(mode="diarize", user_id="ada")
        try:
            resp = asyncio.run(active_job_cancel(_FakeRequest(), user_id="ada"))
            assert resp.status_code == 409
            assert job.cancel_requested is False
        finally:
            _finish(job)


# ---------------------------------------------------------------------------
# Wiring pins
# ---------------------------------------------------------------------------


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class TestUserResolutionWiring:
    """Textual pins for the parts that need the GPU to run.

    A route that resolves no user does not fail -- it falls back to
    ``default``. So the only thing that can catch a regression here is a test
    that reads the call site.
    """

    def test_the_speaker_routes_pass_the_caller_into_the_service(self):
        src = _read("asr_mcp/api/speaker_router.py")
        assert src.count("Depends(get_current_user)") >= 5
        assert "user_id=user" in src
        assert "verify_api_key" not in src

    def test_the_mcp_tools_pass_the_caller_and_their_own_voiceprints(self):
        src = _read("asr_mcp/api/mcp_router.py")
        assert src.count("Depends(get_current_user)") >= 2
        assert src.count("user_id=user") >= 3
        # diarize_audio used to pass no known_speakers at all, so an MCP
        # diarization never matched a voiceprint and never scoped to a user.
        assert "known_speakers=_load_known_speakers(settings, user)" in src

    def test_transcribe_resolves_the_caller_not_the_session(self):
        src = _read("asr_mcp/api/asr_router.py")
        assert 'get_session_user(request) or "default"' not in src
        assert "user_id: str = Depends(get_current_user)" in src

    def test_the_websocket_handshake_uses_the_shared_resolver(self):
        src = _read("asr_mcp/api/asr_router.py")
        start = src.index("async def _websocket_user")
        body = src[start:]
        assert "resolve_api_key_user" in body
        # A local re-derivation here is what let the two transports drift.
        assert "sha256" not in body
        assert "api_key_set" not in body

    def test_the_two_streams_are_owner_scoped(self):
        src = _read("asr_mcp/api/asr_router.py")
        assert "_scrub" not in src
        assert 'job.user_id != user_id' in src
        assert "meta.get(\"user_id\") != user_id" in src

    def test_the_consolidate_and_rescan_guards_are_per_user(self):
        src = _read("asr_mcp/api/voiceprint_router.py")
        assert src.count("job_state.get_for_user(user_id)") >= 2
        assert "Another job is already running" not in src
