"""The orderings that make a registration endpoint safe, over the real routes.

``test_auth_credentials.py`` covers the store against a plain SQLAlchemy
install. This module covers everything that is a property of the ROUTE rather
than of the table, and each one is a bug if it regresses:

* **The throttle is evaluated before the registration flag.** Counting only
  successful signups would remove the rate limit from a still-mounted route the
  moment an operator turns registration off -- which is exactly when somebody is
  probing for it.
* **A disabled method answers 404, not 403.** A 403 confirms the route exists
  and is merely closed; a 404 is indistinguishable from a path that was never
  mounted.
* **Registration never signs anybody in.** The account is unverified until the
  link is opened, so a session here would be a session for an account nobody has
  proved they own.
* **A mail that fails leaves nothing behind.** The row is deleted, so a failed
  signup cannot strand a name its registrant can never reclaim.
* **The confirmation link names this deployment, not the request's Host.**
  Behind a proxy that rewrites Host, a derived link is a link nobody can open.

The routes are driven as coroutines against a fake request rather than through a
TestClient, for the same reason the rest of this suite does it: ``await
route(request, ...)`` makes the ordering assertions direct ("the throttle answer
arrived before the flag was consulted"), and it needs no listening socket.

Everything here needs fastapi and pydantic, so it runs in the container.
"""

import asyncio

import pytest

pytest.importorskip("fastapi", reason="the auth routes are FastAPI handlers")
pytest.importorskip("sqlalchemy", reason="the credential tables need the DB layer")

PASSWORD = "correct horse battery"
ADDRESS = "someone@example.com"


@pytest.fixture(autouse=True)
def fresh_throttle():
    """A process-global counter shared by every test in the run."""
    from asr_mcp.auth.store import reset_registration_attempts

    reset_registration_attempts()
    yield
    reset_registration_attempts()


class FakeApp:
    """Stands in for the FastAPI app, carrying only what the routes read."""

    def __init__(self, htpasswd_users=None):
        self.state = type("State", (), {})()
        self.state.htpasswd_users = htpasswd_users


class FakeRequest:
    """Enough of ``starlette.requests.Request`` for the auth routes.

    ``session`` is a plain dict because that is what ``request.session`` is once
    Starlette's SessionMiddleware has decoded the cookie -- and it is where
    ``login_user`` writes, which is what makes "did registration sign anybody in"
    directly assertable.
    """

    def __init__(self, app=None, session=None, headers=None, client_host="10.0.0.9",
                 base_url="http://asr.internal:8080/", form=None, query=None):
        self.app = app if app is not None else FakeApp()
        self.scope = {"session": {} if session is None else session}
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self.client = type("Client", (), {})()
        self.client.host = client_host
        self.base_url = base_url
        self._form = form or {}
        self.query_params = query or {}

    @property
    def session(self):
        """What ``SessionMiddleware`` installs: the scope's session dict."""
        return self.scope["session"]

    async def form(self):
        return self._form


class RecordingMailer:
    """An injectable SMTP seam. Records, or raises."""

    def __init__(self, fail=None):
        self.sent = []
        self.fail = fail

    def send_message(self, message):
        if self.fail is not None:
            raise self.fail
        self.sent.append(message)


def _settings(tmp_path, **overrides):
    from asr_mcp.config.settings import Settings

    kwargs = {
        "session_secret": "test-secret",
        "db_path": str(tmp_path / "asr.db"),
        "data_dir": str(tmp_path),
        # Off unless a test turns it on: an endpoint that creates accounts is
        # something an operator opts into, not something a deployment inherits.
        "registration_enabled": False,
        # Pinned, not inherited. The container exports TRANSCRIBE_PREFIX=/asr-mcp
        # and a real .env sets TRANSCRIBE_EXTERNAL_URL, so a test that leaves
        # these alone passes locally and fails in the image. Every test that
        # cares about a base or a prefix passes one explicitly.
        "prefix": "",
        "external_url": "",
        "smtp_host": "smtp.example.invalid",
        "smtp_from": "noreply@example.invalid",
        "smtp_port": 25,
        "smtp_starttls": False,
        "smtp_timeout": 5.0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture
def db(tmp_path):
    from asr_mcp.db.manager import DatabaseManager

    mgr = DatabaseManager(str(tmp_path / "asr.db"))
    try:
        yield mgr
    finally:
        mgr.close()


def _store(db, htpasswd_users=None):
    from asr_mcp.auth.store import CredentialStore

    return CredentialStore(db, htpasswd_users=htpasswd_users)


def _route():
    from asr_mcp.api import auth_router

    return auth_router


def _register(request, settings, **overrides):
    """Drive the register route, capturing the message it tried to send."""
    auth_router = _route()
    fields = {"username": "alice", "email": ADDRESS, "password": PASSWORD}
    fields.update(overrides)
    sent = {}
    original = auth_router.send_mail

    def fake_send_mail(to, subject, body, *, client=None, settings=None):
        assert settings is not None, "the route must pass its settings down"
        sent["to"], sent["subject"], sent["body"] = to, subject, body

    auth_router.send_mail = fake_send_mail
    try:
        response = asyncio.run(auth_router.register(
            request, type("P", (), fields)(), settings=settings))
    finally:
        auth_router.send_mail = original
    sent["response"] = response
    return sent


# ---------------------------------------------------------------------------
# The throttle runs before the flag
# ---------------------------------------------------------------------------


class TestThrottleBeforeFlag:
    def test_the_throttle_is_evaluated_before_the_flag_is_consulted(self, tmp_path, db):
        """Over the limit, with registration OFF: 429, not 404.

        If the flag were checked first, a deployment with registration turned
        off would answer 404 to a probe -- which is fine -- but a deployment
        with it turned ON and one client hammering the form would never rate
        limit them, because every attempt would be "successful" and counted.
        The order is what makes the limit apply to attempts rather than
        accounts.
        """
        auth_router = _route()
        from asr_mcp.auth.store import REGISTRATION_ATTEMPT_LIMIT, allow_registration_attempt

        settings = _settings(tmp_path, registration_enabled=False)
        request = FakeRequest(client_host="203.0.113.7")

        for _ in range(REGISTRATION_ATTEMPT_LIMIT):
            allow_registration_attempt("203.0.113.7")

        with pytest.raises(Exception) as excinfo:
            asyncio.run(auth_router._require_registration("email", request, settings))
        assert getattr(excinfo.value, "status_code", None) == 429

    def test_a_disabled_method_is_a_404_and_not_a_403(self, tmp_path):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=False)
        request = FakeRequest(client_host="198.51.100.4")
        with pytest.raises(Exception) as excinfo:
            asyncio.run(auth_router._require_registration("email", request, settings))
        # 404, so a caller cannot tell "closed" from "never mounted".
        assert getattr(excinfo.value, "status_code", None) == 404

    def test_an_enabled_and_configured_method_passes(self, tmp_path):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        request = FakeRequest(client_host="198.51.100.5")
        # Returns None rather than raising -- the only success shape here.
        assert asyncio.run(
            auth_router._require_registration("email", request, settings)) is None

    def test_the_client_key_is_the_last_hop_the_client_did_not_choose(self):
        """The rightmost X-Forwarded-For entry, because nginx appends.

        Taking the leftmost one would let a caller rotate their quota by
        inventing a new first hop on every request.
        """
        auth_router = _route()
        request = FakeRequest(headers={"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"},
                              client_host="192.0.2.10")
        assert auth_router._signup_client(request) == "3.3.3.3"

    def test_without_a_proxy_header_the_socket_peer_is_the_client(self):
        auth_router = _route()
        request = FakeRequest(client_host="192.0.2.11")
        assert auth_router._signup_client(request) == "192.0.2.11"


# ---------------------------------------------------------------------------
# Email registration
# ---------------------------------------------------------------------------


class TestEmailRegistration:
    def test_an_account_is_created_and_a_message_is_sent(self, tmp_path, db):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        request = FakeRequest()
        sent = _register(request, settings)

        assert sent["response"]["status"] == "ok"
        assert sent["response"]["user"] == "alice"
        assert sent["response"]["verification_sent"] is True
        assert sent["to"] == ADDRESS
        # The link carries the token that was stored -- hashed -- against the row.
        assert "/api/auth/verify?token=" in sent["body"]

    def test_registration_does_not_sign_anybody_in(self, tmp_path, db):
        """The whole point: the account is not confirmed yet.

        A session written here would be a session for an account nobody has
        proved they own, and the confirmation link would become decorative.
        """
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        request = FakeRequest()
        _register(request, settings)
        assert "user" not in request.scope["session"]

    def test_the_account_starts_unverified(self, tmp_path, db):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        _register(FakeRequest(), settings)
        store = _store(db)
        row = store.get("alice")
        assert row is not None
        assert row["email_verified_at"] is None

    def test_the_confirmation_link_names_the_deployment_not_the_request(self, tmp_path, db):
        """TRANSCRIBE_EXTERNAL_URL wins, because a proxy rewrites Host.

        A link derived from ``request.base_url`` behind nginx names the
        container's own host:port, which is not an address anybody can open.
        """
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True,
                             external_url="https://asr.example.org/asr-mcp")
        request = FakeRequest(base_url="http://asr-internal:8080/")
        sent = _register(request, settings)
        assert "https://asr.example.org/asr-mcp" in sent["body"]
        assert "asr-internal" not in sent["body"]

    def test_the_link_works_when_external_url_is_not_set(self, tmp_path, db):
        """The setting is an override, not a requirement."""
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        request = FakeRequest(base_url="http://asr.internal:8080/")
        sent = _register(request, settings)
        assert "http://asr.internal:8080" in sent["body"]

    # ------------------------------------------------------------------
    # The prefix must appear EXACTLY once in a link.
    #
    # This is a regression test for a real outage: a deployment set
    # TRANSCRIBE_EXTERNAL_URL to the full public URL including the prefix
    # (https://host/asr-mcp, which is what a user actually pastes into a
    # browser), and the route appended settings.prefix on top. The link became
    # https://host/asr-mcp/asr-mcp/api/auth/verify and 404'd. From the user's
    # side that is indistinguishable from "the token was refused" -- the mail
    # arrives, the link is dead, and the account can never be confirmed.
    #
    # The assertion has to be on the WHOLE link, not a substring: a
    # substring assertion is what let this through in the first place,
    # because ".../asr-mcp/asr-mcp/api/auth/verify" *contains*
    # ".../asr-mcp".
    # ------------------------------------------------------------------

    def _link_in(self, body: str) -> str:
        """The confirmation URL from an email body, or '' if there is none."""
        for line in body.splitlines():
            if "/api/auth/verify" in line:
                return line.strip()
        return ""

    @pytest.mark.parametrize("external_url", [
        "https://asr.example.org",              # bare origin, prefix appended
        "https://asr.example.org/",             # trailing slash is harmless
        "https://asr.example.org/asr-mcp",      # public URL as pasted
        "https://asr.example.org/asr-mcp/",     # ... with a trailing slash
    ])
    def test_the_prefix_appears_once_however_the_base_is_spelled(self, tmp_path, db,
                                                                external_url):
        """Every reasonable spelling of the base yields the same link."""
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True, prefix="/asr-mcp",
                             external_url=external_url)
        sent = _register(FakeRequest(base_url="http://asr-internal:8080/"), settings)
        link = self._link_in(sent["body"])
        assert link.startswith("https://asr.example.org/asr-mcp/api/auth/verify?"), link
        assert "/asr-mcp/asr-mcp/" not in link, link

    def test_a_prefix_that_is_a_substring_is_still_appended(self, tmp_path, db):
        """The dedup is on a path-segment boundary, not a bare ``endswith``.

        A base of ``.../asr-mcp`` with a prefix of ``/asr`` does NOT already
        end with the prefix, so the prefix is appended rather than silently
        dropped. Comparing with a bare ``endswith(prefix)`` would wrongly call
        ``/asr-mcp`` an instance of ``/asr``.
        """
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True, prefix="asr",
                             external_url="https://asr.example.org/asr-mcp")
        sent = _register(FakeRequest(base_url="http://asr-internal:8080/"), settings)
        link = self._link_in(sent["body"])
        assert link.startswith("https://asr.example.org/asr-mcp/asr/api/auth/verify?"), link

    def test_the_google_redirect_uri_agrees_with_the_confirmation_link(self, tmp_path, db):
        """Both links must name the same deployment.

        They are built by different call sites, which is exactly why they
        drifted: the confirmation link was fixed and the Google redirect URI
        would have kept the doubled prefix, so a deployment behind a prefix
        would confirm by mail and then fail every Google sign-in.
        """
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True, prefix="/asr-mcp",
                             external_url="https://asr.example.org/asr-mcp")
        sent = _register(FakeRequest(base_url="http://asr-internal:8080/"), settings)
        link = self._link_in(sent["body"])
        callback = auth_router._google_callback_url(settings, FakeRequest())
        assert "/asr-mcp/asr-mcp/" not in link
        assert "/asr-mcp/asr-mcp/" not in callback
        # Same base, so a user who registered by mail can sign in with Google.
        assert callback.startswith(link.split("/api/auth/verify")[0])

    def test_a_deployment_with_no_prefix_is_unaffected(self, tmp_path, db):
        """The dedup must not invent a prefix, and must not eat the base."""
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True, prefix="",
                             external_url="https://asr.example.org")
        sent = _register(FakeRequest(base_url="http://asr-internal:8080/"), settings)
        link = self._link_in(sent["body"])
        assert link.startswith("https://asr.example.org/api/auth/verify?"), link
        assert auth_router._google_callback_url(
            settings, FakeRequest()) == \
            "https://asr.example.org/api/auth/google/callback"

    def test_a_rejected_signup_sends_nothing(self, tmp_path, db):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        sent = {}
        original = auth_router.send_mail

        def fake_send_mail(to, subject, body, *, client=None, settings=None):
            sent["to"] = to

        auth_router.send_mail = fake_send_mail
        try:
            with pytest.raises(Exception) as excinfo:
                asyncio.run(auth_router.register(FakeRequest(),
                                                 type("P", (), {
                                                     "username": "al",
                                                     "email": "",
                                                     "password": PASSWORD})(),
                                                 settings=settings))
        finally:
            auth_router.send_mail = original

        assert getattr(excinfo.value, "status_code", None) == 400
        assert "to" not in sent, "a refused signup must not send mail"
        assert _store(db).get("al") is None

    def test_a_mail_that_cannot_be_sent_leaves_nothing_behind(self, tmp_path, db):
        """503, and the row goes: a half-account can never sign in.

        Leaving it would strand a name its registrant can never reclaim --
        the username is reserved forever by an account that can never be used.
        """
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        request = FakeRequest()

        class Boom(Exception):
            pass

        original = auth_router.send_mail

        def failing_mail(to, subject, body, *, client=None, settings=None):
            raise Boom("smtp down")

        auth_router.send_mail = failing_mail
        try:
            with pytest.raises(Exception) as excinfo:
                asyncio.run(auth_router.register(request,
                                                 type("P", (), {
                                                     "username": "alice",
                                                     "email": ADDRESS,
                                                     "password": PASSWORD})(),
                                                 settings=settings))
        finally:
            auth_router.send_mail = original

        assert getattr(excinfo.value, "status_code", None) == 503
        assert _store(db).get("alice") is None
        assert _store(db).count() == 0

    def test_a_disabled_method_is_not_a_registrable_endpoint(self, tmp_path, db):
        """404 even though the route object is mounted and callable."""
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=False)
        request = FakeRequest()
        with pytest.raises(Exception) as excinfo:
            asyncio.run(auth_router.register(request,
                                             type("P", (), {
                                                 "username": "bob",
                                                 "email": "bob@example.invalid",
                                                 "password": PASSWORD})(),
                                             settings=settings))
        assert getattr(excinfo.value, "status_code", None) == 404
        assert _store(db).count() == 0

    def test_registering_a_name_that_is_taken_is_a_400(self, tmp_path, db):
        auth_router = _route()
        settings = _settings(tmp_path, registration_enabled=True)
        store = _store(db)
        store.create("alice", PASSWORD, ADDRESS)

        with pytest.raises(Exception) as excinfo:
            asyncio.run(auth_router.register(FakeRequest(),
                                             type("P", (), {
                                                 "username": "alice",
                                                 "email": "other@example.invalid",
                                                 "password": PASSWORD})(),
                                             settings=settings))
        assert getattr(excinfo.value, "status_code", None) == 400


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


class TestEmailConfirmation:
    def _confirmed(self, tmp_path, db):
        store = _store(db)
        store.create("alice", PASSWORD, ADDRESS)
        token = store.issue_verification_token("alice")
        return store, token

    def test_a_good_token_redirects_saying_so(self, tmp_path, db):
        auth_router = _route()
        _, token = self._confirmed(tmp_path, db)
        response = asyncio.run(auth_router.verify_email(
            FakeRequest(), token=token, settings=_settings(tmp_path)))
        assert response.status_code == 302
        assert "verified=1" in response.headers["location"]

    def test_a_spent_token_gets_the_same_answer_as_a_bad_one(self, tmp_path, db):
        store, token = self._confirmed(tmp_path, db)
        auth_router = _route()
        settings = _settings(tmp_path)
        first = asyncio.run(auth_router.verify_email(FakeRequest(), token=token,
                                                     settings=settings))
        second = asyncio.run(auth_router.verify_email(FakeRequest(), token=token,
                                                      settings=settings))
        assert first.status_code == second.status_code == 302
        assert "verified=1" in first.headers["location"]
        assert "verified=0" in second.headers["location"]

    def test_an_empty_token_is_not_an_exception(self, tmp_path, db):
        auth_router = _route()
        response = asyncio.run(auth_router.verify_email(FakeRequest(), token="",
                                                        settings=_settings(tmp_path)))
        assert response.status_code == 302
        assert "verified=0" in response.headers["location"]

    def test_garbage_gets_the_same_answer(self, tmp_path, db):
        auth_router = _route()
        response = asyncio.run(auth_router.verify_email(FakeRequest(), token="not-a-token",
                                                        settings=_settings(tmp_path)))
        assert "verified=0" in response.headers["location"]

    def test_the_redirect_target_is_not_request_controlled(self, tmp_path, db):
        """Nothing a caller supplies reaches the Location header.

        Both outcomes are built by the module; the query string is a fixed
        string, so a token cannot be turned into an open redirect.
        """
        auth_router = _route()
        for token in ("", "junk", None):
            response = asyncio.run(auth_router.verify_email(
                FakeRequest(), token=token, settings=_settings(tmp_path)))
            location = response.headers["location"]
            # The whole path after the prefix is one fixed string -- nothing a
            # caller supplied reaches the Location header.
            assert location.endswith("/login?verified=0") or \
                location.endswith("/login?verified=1")

    def test_confirming_makes_the_account_signable(self, tmp_path, db):
        """The link is not paperwork: it is what flips the row to usable."""
        store, token = self._confirmed(tmp_path, db)
        auth_router = _route()
        before = store.check_login("alice", PASSWORD)
        asyncio.run(auth_router.verify_email(FakeRequest(), token=token,
                                             settings=_settings(tmp_path)))
        after = store.check_login("alice", PASSWORD)
        assert before[0] == ""
        assert after == ("alice", "")


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    def _registered(self, db):
        store = _store(db)
        store.create("alice", PASSWORD, ADDRESS)
        store.verify_email_token(store.issue_verification_token("alice"))

    def _login(self, tmp_path, db, request, username="alice", password=PASSWORD,
               **overrides):
        auth_router = _route()
        settings = _settings(tmp_path, **overrides)
        return asyncio.run(auth_router.login_post(
            request, username=username, password=password, settings=settings))

    def test_a_confirmed_account_signs_in(self, tmp_path, db):
        self._registered(db)
        request = FakeRequest()
        response = self._login(tmp_path, db, request)
        assert response.status_code == 200
        assert request.scope["session"]["user"] == "alice"

    def test_an_unverified_account_is_a_403_with_the_reason(self, tmp_path, db):
        """403 rather than 401: the password is right, the account is not usable.

        The distinction is what lets the page say "open your confirmation link"
        instead of "invalid credentials", which would send the person in circles
        trying passwords that already work.
        """
        store = _store(db)
        store.create("alice", PASSWORD, ADDRESS)
        request = FakeRequest()
        response = self._login(tmp_path, db, request)
        assert response.status_code == 403
        assert "confirm" in response.body.decode().lower()

    def test_a_disabled_account_is_a_403_with_a_different_reason(self, tmp_path, db):
        self._registered(db)
        _store(db).disable("alice")
        response = self._login(tmp_path, db, FakeRequest())
        assert response.status_code == 403
        assert "disabled" in response.body.decode().lower()

    def test_a_wrong_password_is_a_401(self, tmp_path, db):
        self._registered(db)
        response = self._login(tmp_path, db, FakeRequest(), password="wrong password")
        assert response.status_code == 401

    def test_the_session_is_cleared_before_the_user_is_written(self, tmp_path, db):
        """Anti-fixation: a cookie captured before login is not a logged-in one."""
        self._registered(db)
        request = FakeRequest(session={"user": "someone-else", "leftover": 1})
        self._login(tmp_path, db, request)
        assert request.scope["session"] == {"user": "alice"}

    def test_the_name_is_matched_case_insensitively(self, tmp_path, db):
        """The table is keyed on the normalised username.

        A user who types their own name with a capital letter must land in the
        vault they registered, not in a second, empty one -- and not be told
        their credentials are wrong.
        """
        self._registered(db)
        request = FakeRequest()
        response = self._login(tmp_path, db, request, username="Alice")
        assert response.status_code == 200
        assert request.scope["session"]["user"] == "alice"

    def test_an_htpasswd_user_still_signs_in(self, tmp_path, db):
        """The operator's own file is not replaced by the new table.

        ``{SHA}`` needs no subprocess, so the htpasswd ladder is exercised here
        rather than by shelling out to ``openssl passwd``.
        """
        import base64
        import hashlib

        digest = base64.b64encode(hashlib.sha1(b"operator-pw").digest()).decode()
        settings = _settings(tmp_path)
        request = FakeRequest(app=FakeApp(htpasswd_users={"operator": "{SHA}" + digest}))
        auth_router = _route()
        response = asyncio.run(auth_router.login_post(
            request, username="operator", password="operator-pw", settings=settings))
        assert response.status_code == 200
        assert request.scope["session"]["user"] == "operator"

    def test_an_htpasswd_user_keeps_the_name_exactly_as_typed(self, tmp_path, db):
        """The file is case-sensitive by nature and this app does not second-guess it."""
        import base64
        import hashlib

        digest = base64.b64encode(hashlib.sha1(b"operator-pw").digest()).decode()
        settings = _settings(tmp_path)
        request = FakeRequest(app=FakeApp(htpasswd_users={"Operator": "{SHA}" + digest}))
        auth_router = _route()
        response = asyncio.run(auth_router.login_post(
            request, username="Operator", password="operator-pw", settings=settings))
        assert response.status_code == 200
        # Not lowercased: the htpasswd name IS the vault key.
        assert request.scope["session"]["user"] == "Operator"

    def test_a_missing_account_and_a_wrong_password_answer_alike(self, tmp_path, db):
        """Neither one says which it was.

        A refusal only for a name that exists is the oracle a credential
        stuffing script uses to enumerate who has an account.
        """
        self._registered(db)
        missing = self._login(tmp_path, db, FakeRequest(),
                              username="nobody", password=PASSWORD)
        wrong = self._login(tmp_path, db, FakeRequest(), password="not-it")
        assert missing.status_code == wrong.status_code == 401
        assert b"Invalid username or password" in missing.body
        assert b"Invalid username or password" in wrong.body


# ---------------------------------------------------------------------------
# What the login page may offer
# ---------------------------------------------------------------------------


class TestRegistrationMethods:
    def test_both_methods_are_off_by_default(self, tmp_path):
        auth_router = _route()
        result = asyncio.run(auth_router.auth_methods(FakeRequest(), settings=_settings(tmp_path)))
        assert result == {"email": False, "google": False, "password_hint": 10}

    def test_email_needs_a_mail_server_and_the_flag(self, tmp_path):
        auth_router = _route()
        result = asyncio.run(auth_router.auth_methods(
            FakeRequest(),
            settings=_settings(tmp_path, registration_enabled=True, smtp_host="")))
        assert result["email"] is False

    def test_the_flag_alone_is_not_enough(self, tmp_path):
        auth_router = _route()
        result = asyncio.run(auth_router.auth_methods(
            FakeRequest(),
            settings=_settings(tmp_path, registration_enabled=False)))
        assert result["email"] is False

    def test_an_unknown_method_raises_rather_than_reading_as_off(self, tmp_path):
        """A typo in a caller is a loud failure, not a silently hidden form.

        It only raises once the flag is on, which is the same order mem-mcp
        uses: a deployment with registration off never reaches the method name,
        and a deployment with it on cannot quietly mislabel a method.
        """
        from asr_mcp.auth.store import registration_enabled

        settings = _settings(tmp_path, registration_enabled=True)
        assert registration_enabled("email", settings) is True
        with pytest.raises(ValueError):
            registration_enabled("bogus", settings)


# ---------------------------------------------------------------------------
# The session-only identity endpoint
# ---------------------------------------------------------------------------


class TestWhoAmI:
    def test_an_api_key_caller_is_not_a_session_user(self):
        """``/api/user`` reads the session only.

        It renders the browser's own header. The identity a request ACTS as is
        resolved by ``get_current_user``, which also understands tokens --
        mixing the two here would show an API-key caller as anonymous on screen
        while their request wrote to a real vault.
        """
        auth_router = _route()
        request = FakeRequest(headers={"X-API-Key": "some-key"})
        result = asyncio.run(auth_router.get_current_user_info(request))
        assert result["authenticated"] is False
        assert result["username"] == "anonymous"

    def test_a_signed_in_session_names_the_user(self):
        auth_router = _route()
        request = FakeRequest(session={"user": "alice"})
        result = asyncio.run(auth_router.get_current_user_info(request))
        assert result == {"username": "alice", "authenticated": True}


# ---------------------------------------------------------------------------
# The mailer's own seam
# ---------------------------------------------------------------------------


class TestConfirmationMail:
    def test_the_message_carries_the_link_and_the_username(self, tmp_path):
        from asr_mcp.auth.mailer import verification_email

        subject, body = verification_email("alice", "https://asr.example.org/verify?token=abc")
        assert "alice" in body
        assert "https://asr.example.org/verify?token=abc" in body
        assert subject

    def test_the_message_says_the_link_works_once(self, tmp_path):
        from asr_mcp.auth.mailer import verification_email

        _, body = verification_email("alice", "https://asr.example.org/verify?token=abc")
        assert "once" in body.lower()

    def test_an_injected_client_is_used_and_the_message_is_formed(self, tmp_path):
        from asr_mcp.auth.mailer import _plain_message, send_mail

        settings = _settings(tmp_path)
        client = RecordingMailer()
        send_mail(ADDRESS, "Subject", "Body", client=client, settings=settings)
        assert len(client.sent) == 1
        message = client.sent[0]
        assert message["To"] == ADDRESS
        assert message["From"] == settings.smtp_from
        assert message["Subject"] == "Subject"

    def test_a_failing_client_raises_rather_than_returning_false(self, tmp_path):
        """The route's whole rollback depends on this raising."""
        from asr_mcp.auth.mailer import send_mail

        settings = _settings(tmp_path)
        client = RecordingMailer(fail=OSError("connection refused"))
        with pytest.raises(OSError):
            send_mail(ADDRESS, "Subject", "Body", client=client, settings=settings)

    def test_an_unconfigured_server_refuses_before_any_connection(self, tmp_path):
        from asr_mcp.auth.mailer import send_mail

        settings = _settings(tmp_path, smtp_host="")
        with pytest.raises(RuntimeError):
            send_mail(ADDRESS, "Subject", "Body", settings=settings)

    def test_a_failure_is_logged_by_type_without_the_address_or_the_link(self, tmp_path, caplog):
        """Neither the address nor the live token may reach a log file."""
        import logging

        from asr_mcp.auth.mailer import log_send_failure

        with caplog.at_level(logging.WARNING, logger="asr_mcp.auth.mailer"):
            log_send_failure(OSError("dead"))
        assert caplog.records
        text = caplog.text.lower()
        assert "oserror" in text
        assert ADDRESS not in caplog.text
        assert "token" not in text
