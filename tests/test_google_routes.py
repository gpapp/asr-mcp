"""The Google sign-in routes: state, the subject link, and first-use creation.

``test_google_auth.py`` covers the OAuth client. This module covers what is a
property of the ROUTES rather than of the client, and each one is a bug if it
regresses:

* **The ``oauth_state`` lives in the server-side session and nowhere else,**
  is compared with ``compare_digest``, and is consumed on both paths -- so a
  state can be replayed exactly once.
* **An unknown subject is only ever created by the callback, never by ``start``.**
* **A first-time sign-in keys the account on the verified ADDRESS**, with no
  confirmation mail, because Google already verified it.
* **Registration being off stops new accounts without signing anybody out** --
  ``/start`` is gated on the client being configured, not on the flag.
* **A failure never writes a session user.**

The routes are driven as coroutines against a fake request, the same way the rest
of this suite does it: ``await route(request, ...)`` makes the ordering
assertions direct and needs no listening socket.
"""

import asyncio

import pytest

pytest.importorskip("fastapi", reason="the auth routes are FastAPI handlers")
pytest.importorskip("sqlalchemy", reason="the credential tables need the DB layer")

SUBJECT = "google-subject-123"
ADDRESS = "someone@example.com"
INFO = {"subject": SUBJECT, "email": ADDRESS, "email_verified": True, "name": "Someone"}
STATE = "the-minted-state"

REDIRECT_URI = "http://asr.internal:8080/api/auth/google/callback"


class FakeApp:
    def __init__(self, htpasswd_users=None):
        self.state = type("State", (), {})()
        self.state.htpasswd_users = htpasswd_users


class FakeRequest:
    def __init__(self, app=None, session=None, headers=None, client_host="10.0.0.9",
                 base_url="http://asr.internal:8080/", query=None):
        self.app = app if app is not None else FakeApp()
        self.scope = {"session": {} if session is None else session}
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self.client = type("Client", (), {})()
        self.client.host = client_host
        self.base_url = base_url
        self.query_params = query or {}

    @property
    def session(self):
        return self.scope["session"]

    async def form(self):
        return {}


class FakeExchange:
    """Stands in for ``google_auth.exchange_code``/``fetch_userinfo``.

    Recording, and able to fail, which is what lets a test assert that a failed
    exchange leaves no session behind.
    """

    def __init__(self, info=INFO, error=None, tokens=()):
        self.info = info
        self.error = error
        self.calls = []
        self._tokens = list(tokens) or ["token-abc"]

    def exchange(self, code, redirect_uri, *, settings=None):
        self.calls.append({"code": code, "redirect_uri": redirect_uri})
        if self.error is not None:
            raise self.error
        return self._tokens.pop(0) if len(self._tokens) > 1 else self._tokens[0]

    def userinfo(self, access_token, *, opener=None):
        """Return the profile the way ``fetch_userinfo`` hands it over.

        The ``is True`` coercion is not decoration: the real function applies it
        (:231 of google_auth.py), and a fake that returns its payload verbatim
        would let a truthy *string* pass a check that exists to refuse it.
        """
        self.calls.append({"token": access_token})
        if self.error is not None:
            raise self.error
        info = dict(self.info)
        info["email_verified"] = info.get("email_verified") is True
        return info


def _settings(tmp_path, **overrides):
    from asr_mcp.config.settings import Settings

    kwargs = {
        "session_secret": "test-secret",
        "db_path": str(tmp_path / "asr.db"),
        "data_dir": str(tmp_path),
        "google_client_id": "client-id-123",
        "google_client_secret": "client-secret-456",
        "external_url": "http://asr.internal:8080",
        "registration_enabled": False,
        # Pinned so the assertions below are about the redirect URI, not about
        # whatever TRANSCRIBE_PREFIX the container happens to export. A test
        # that inherits the ambient prefix is a test that only passes locally.
        "prefix": "",
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


@pytest.fixture
def store(db):
    from asr_mcp.auth.store import CredentialStore

    return CredentialStore(db)


def _route():
    from asr_mcp.api import auth_router

    return auth_router


def _patch_google(monkeypatch, exchange=None, *, configured=True, info=INFO, error=None):
    """Install the network seams the routes use, off-thread.

    ``authorization_url`` is stubbed with a real query string so a test can
    assert the redirect URI and the state really travel to Google.
    """
    from asr_mcp.auth import google_auth

    exchange = exchange or FakeExchange(info=info, error=error)
    seen = {}
    monkeypatch.setattr(google_auth, "configured", lambda settings=None: configured)
    monkeypatch.setattr(
        google_auth, "authorization_url",
        lambda state, redirect_uri, *, client_id_override="", settings=None: (
            seen.setdefault("last", f"https://accounts.google.com/o/oauth2/v2/auth"
                                    f"?state={state}&redirect_uri={redirect_uri}")))
    exchange.authorization = seen
    monkeypatch.setattr(
        google_auth, "exchange_code",
        lambda code, redirect_uri, **kw: exchange.exchange(code, redirect_uri))
    monkeypatch.setattr(
        google_auth, "fetch_userinfo",
        lambda access_token, **kw: exchange.userinfo(access_token))
    return exchange


def _callback(request, settings, **query):
    kwargs = {"code": "the-code", "state": STATE}
    kwargs.update(query)
    return asyncio.run(_route().google_callback(request, settings=settings, **kwargs))


class TestGoogleStart:
    def test_starting_signs_in_to_googles_consent_screen(self, tmp_path, monkeypatch):
        _patch_google(monkeypatch)
        request = FakeRequest()
        response = asyncio.run(_route().google_start(request, settings=_settings(tmp_path)))
        assert response.status_code == 302
        assert response.headers["location"].startswith(
            "https://accounts.google.com/o/oauth2/v2/auth?")

    def test_the_state_is_kept_in_the_session_not_a_cookie_of_our_own(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        request = FakeRequest()
        asyncio.run(_route().google_start(request, settings=_settings(tmp_path)))
        minted = request.session.get("oauth_state")
        assert minted
        # The value in the session is the value sent to Google.
        assert f"state={minted}" in exchange.authorization["last"]

    def test_start_is_a_404_when_google_is_not_configured(self, tmp_path, monkeypatch):
        """Gated on the client being configured, so a half-set deployment shows
        no button rather than one that starts a login and cannot finish it.
        """
        _patch_google(monkeypatch, configured=False)
        with pytest.raises(Exception) as excinfo:
            asyncio.run(_route().google_start(FakeRequest(), settings=_settings(tmp_path)))
        assert getattr(excinfo.value, "status_code", None) == 404

    def test_sign_in_still_works_when_registration_is_closed(self, tmp_path, monkeypatch):
        """Switching registration off has to stop NEW accounts without signing
        out the people who already have one.
        """
        _patch_google(monkeypatch)
        response = asyncio.run(_route().google_start(
            FakeRequest(), settings=_settings(tmp_path, registration_enabled=False)))
        assert response.status_code == 302

    def test_the_redirect_uri_names_the_deployment_not_the_request(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        # A proxy that rewrites Host: the redirect URI registered with Google is
        # still the one used, not whatever this request claims to be.
        response = asyncio.run(_route().google_start(
            FakeRequest(base_url="http://proxy-rewrote-this.example/"),
            settings=_settings(tmp_path)))
        assert REDIRECT_URI in response.headers["location"]
        assert REDIRECT_URI in exchange.authorization["last"]

    def test_the_started_url_uses_the_external_base_url(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        asyncio.run(_route().google_start(
            FakeRequest(base_url="http://proxy-rewrote-this.example/"),
            settings=_settings(tmp_path)))
        assert REDIRECT_URI in exchange.authorization["last"]


class TestGoogleCallback:
    def test_a_linked_subject_signs_in_without_creating_anything(self, tmp_path, db, store, monkeypatch):
        store.link_google_identity(SUBJECT, ADDRESS, email=ADDRESS, name="Someone")
        exchange = _patch_google(monkeypatch)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path))
        assert response.status_code == 302
        assert request.session.get("user") == ADDRESS
        # Nothing new was created -- the exchange ran, the subject resolved.
        assert exchange.calls
        assert store.count() == 0
        assert store.list_all_google_identities()[0]["user_id"] == ADDRESS

    def test_a_first_time_account_is_created_and_signed_in(self, tmp_path, db, store, monkeypatch):
        _patch_google(monkeypatch)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.status_code == 302
        assert response.headers["location"].endswith("/gui")
        assert request.session.get("user") == ADDRESS
        assert store.list_all_google_identities()[0]["user_id"] == ADDRESS

    def test_the_vault_is_keyed_on_the_verified_address(self, tmp_path, db, store, monkeypatch):
        """The whole justification for skipping the confirmation mail: Google has
        already verified that this account controls this address.
        """
        _patch_google(monkeypatch)
        _callback(FakeRequest(session={"oauth_state": STATE}),
                  _settings(tmp_path, registration_enabled=True))
        link = store.list_all_google_identities()[0]
        # The key IS the address, and it is what the session is signed in as.
        assert link["user_id"] == ADDRESS
        assert link["email"] == ADDRESS
        # A Google account has no password row: there is nothing to sign in with
        # apart from the subject link.
        assert store.check_login(ADDRESS, "anything")[0] == ""

    def test_an_unverified_address_creates_no_account(self, tmp_path, db, store, monkeypatch):
        _patch_google(monkeypatch, info=dict(INFO, email_verified=False))
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.headers["location"].endswith("google=closed")
        assert request.session.get("user") is None
        assert store.count() == 0
        assert store.list_all_google_identities() == []

    def test_a_missing_address_creates_no_account(self, tmp_path, store, monkeypatch):
        _patch_google(monkeypatch, info=dict(INFO, email=""))
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.headers["location"].endswith("google=closed")
        assert request.session.get("user") is None
        assert store.list_all_google_identities() == []

    def test_a_password_account_does_not_block_the_address(self, tmp_path, store, monkeypatch):
        """Identity is resolved by SUBJECT, never by address.

        A password account's key is its username, and a username may not contain
        ``@`` -- so a Google address colliding with one is not a second way in.
        The route's ``user_id_taken`` guard is therefore live against another
        Google identity (or an htpasswd name); against a password account it has
        nothing to find, which is why the "taken" case is pinned by
        ``test_a_taken_address_is_not_opened_a_second_way`` instead of here.
        """
        store.create("someone", "correct horse battery", "elsewhere@example.com")
        _patch_google(monkeypatch)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.headers["location"].endswith("/gui")
        assert request.session.get("user") == ADDRESS
        # And the two accounts are separate: the password one still exists.
        assert store.get("someone") is not None

    def test_a_taken_address_is_not_opened_a_second_way(self, tmp_path, store, monkeypatch):
        _patch_google(monkeypatch)
        store.link_google_identity("some-other-subject", ADDRESS, email=ADDRESS)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.headers["location"].endswith("google=closed")
        assert request.session.get("user") is None

    def test_the_code_is_redeemed_with_the_same_redirect_uri(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        _callback(FakeRequest(session={"oauth_state": STATE}),
                  _settings(tmp_path, registration_enabled=True))
        assert exchange.calls[0]["redirect_uri"] == REDIRECT_URI

    def test_a_state_that_is_wrong_or_absent_is_refused(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        for session, state in (({"oauth_state": STATE}, "someone-elses-state"),
                              ({"oauth_state": STATE}, ""),
                              ({}, STATE)):
            request = FakeRequest(session=session)
            response = _callback(request, _settings(tmp_path), state=state)
            assert response.headers["location"].endswith("google=state")
            assert request.session.get("user") is None
        assert exchange.calls == []

    def test_the_state_is_consumed_so_it_cannot_be_replayed(self, tmp_path, monkeypatch):
        """Consumed either way, so a state is replayable exactly once.

        The success path also clears the whole session (anti-fixation), so the
        key is gone rather than left as ``""`` -- what is pinned is that it no
        longer holds a usable state, and a callback arriving against that
        session is refused.
        """
        _patch_google(monkeypatch)
        session = {"oauth_state": STATE}
        _callback(FakeRequest(session=session),
                  _settings(tmp_path, registration_enabled=True))
        assert "oauth_state" not in session
        replay = _callback(FakeRequest(session=dict(session)),
                           _settings(tmp_path, registration_enabled=True))
        assert replay.headers["location"].endswith("google=state")

    def test_a_dismissed_consent_screen_is_not_a_failure(self, tmp_path, monkeypatch):
        exchange = _patch_google(monkeypatch)
        response = _callback(FakeRequest(session={"oauth_state": STATE}),
                             _settings(tmp_path), code="", error="access_denied")
        assert response.headers["location"].endswith("google=declined")
        assert exchange.calls == []

    def test_a_code_that_will_not_exchange_reports_a_failure(self, tmp_path, monkeypatch):
        from asr_mcp.auth.google_auth import GoogleOAuthError

        _patch_google(monkeypatch, error=GoogleOAuthError("invalid grant"))
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path))
        assert response.headers["location"].endswith("google=failed")
        assert request.session.get("user") is None

    def test_a_failure_never_writes_a_session_user(self, tmp_path, store, monkeypatch):
        """The failure path writes nothing, and clears nothing.

        Only the success path clears the session -- a Google login that fails
        must not sign out whoever was already there, it just must not put anyone
        else in. So the assertion is on what was never written, not on what was
        wiped.
        """
        from asr_mcp.auth.google_auth import GoogleOAuthError

        _patch_google(monkeypatch, error=GoogleOAuthError("could not reach Google"))
        request = FakeRequest(session={"oauth_state": STATE, "user": "someone-else"})
        _callback(request, _settings(tmp_path))
        assert request.session.get("user") != ADDRESS
        # The state was consumed on the way out; the pre-existing session
        # survived, which is what a failed login has to do.
        assert request.session.get("oauth_state") == ""
        assert request.session.get("user") == "someone-else"

    def test_no_confirmation_mail_is_sent(self, tmp_path, db, store, monkeypatch):
        """Google already verified the address, so re-verifying it would prove
        nothing and cost a message that could not be delivered anyway.
        """
        from asr_mcp.api import auth_router

        sent = []
        original = auth_router.send_mail

        def fake_send_mail(*args, **kwargs):
            sent.append(args)

        auth_router.send_mail = fake_send_mail
        try:
            _patch_google(monkeypatch)
            _callback(FakeRequest(session={"oauth_state": STATE}),
                      _settings(tmp_path, registration_enabled=True))
        finally:
            auth_router.send_mail = original
        assert sent == []

    def test_a_second_sign_in_finds_the_same_account(self, tmp_path, store, monkeypatch):
        _patch_google(monkeypatch)
        settings = _settings(tmp_path, registration_enabled=True)
        first_request = FakeRequest(session={"oauth_state": STATE})
        _callback(first_request, settings)
        # A fresh session, a fresh state, the same subject.
        second = FakeRequest(session={"oauth_state": "another-state"})
        response = _callback(second, settings, state="another-state")
        assert response.headers["location"].endswith("/gui")
        assert second.session.get("user") == ADDRESS
        assert len(store.list_all_google_identities()) == 1


class TestGoogleSignupRefusals:
    """Every refusal creates no account and no subject link."""

    @pytest.mark.parametrize("info_overrides", [
        {"email": ""},
        {"email": ADDRESS, "email_verified": False},
        {"email": ADDRESS, "email_verified": "true"},
        {"email": ADDRESS, "email_verified": None},
    ])
    def test_a_refusal_creates_nothing(self, tmp_path, store, monkeypatch, info_overrides):
        _patch_google(monkeypatch, info=dict(INFO, **info_overrides))
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=True))
        assert response.headers["location"].endswith("google=closed")
        assert request.session.get("user") is None
        assert store.count() == 0
        assert store.list_all_google_identities() == []

    def test_registration_off_refuses_a_new_account(self, tmp_path, store, monkeypatch):
        _patch_google(monkeypatch)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=False))
        assert response.headers["location"].endswith("google=closed")
        assert store.list_all_google_identities() == []

    def test_an_existing_subject_is_not_gated_on_the_registration_flag(self, tmp_path, store, monkeypatch):
        """The flag stops NEW accounts. It must not lock out the people who
        already have one.
        """
        store.link_google_identity(SUBJECT, ADDRESS, email=ADDRESS)
        _patch_google(monkeypatch)
        request = FakeRequest(session={"oauth_state": STATE})
        response = _callback(request, _settings(tmp_path, registration_enabled=False))
        assert response.headers["location"].endswith("/gui")
        assert request.session.get("user") == ADDRESS
