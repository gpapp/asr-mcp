"""The OAuth client: URL building, parameter validation, and the shape of answers.

``network`` is the one thing these tests never touch. The single seam is
``opener`` -- anything that takes the same call signature as
``urllib.request.urlopen`` -- so a fake responder is enough to exercise the
whole exchange, and what that leaves is exactly the attack surface of an OAuth
client: what we send, what we accept back, and what a failure says.

The module is stdlib-only, which is why these tests can import and *call* it on
a machine with no web framework, no network and no JWT library.
"""

import json
import urllib.parse
import urllib.request

import pytest

from asr_mcp.auth import google_auth
from asr_mcp.auth.google_auth import GoogleOAuthError

STATE = "an-opaque-state-value"


class FakeResponse:
    """What ``urlopen`` returns: a context manager with ``.read()``."""

    def __init__(self, payload, status=200):
        if isinstance(payload, (dict, list)):
            self._raw = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, bytes):
            self._raw = payload
        else:
            self._raw = str(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHTTPError(urllib.error.HTTPError):
    """An HTTPError, built with a body the error parser can read.

    ``body`` is the raw bytes; a JSON payload is encoded for you. Passing
    ``body=b"..."`` directly is how a test produces an error body that is not
    JSON at all -- a captive portal, an intercepting proxy -- which is what the
    "unreadable error body" case is about.
    """

    def __init__(self, code, payload=None, body=None,
                 url="https://oauth2.googleapis.com/token"):
        super().__init__(url, code, "refused", None, None)
        if body is not None:
            self._body = body
        elif isinstance(payload, bytes):
            self._body = payload
        else:
            self._body = json.dumps(payload if payload is not None else {}).encode("utf-8")

    def read(self):
        return self._body


class RecordingOpener:
    """An ``OpenerDirector``-alike that records requests and answers them.

    An ``HTTPError`` is *raised*, the way ``urlopen`` raises it -- returning one
    instead would silently take the success path, because ``HTTPError`` also
    satisfies the context-manager protocol and its ``read()`` serves the error
    body as if it were a 200.
    """

    def __init__(self, *responses):
        self.requests = []
        self._responses = list(responses)

    def __call__(self, request, timeout=None):
        self.requests.append({"url": request.full_url, "data": request.data,
                              "headers": dict(request.header_items())})
        if len(self._responses) == 1:
            response = self._responses[0]
        else:
            response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _settings(**overrides):
    from asr_mcp.config.settings import Settings

    kwargs = {
        "session_secret": "test-secret",
        "google_client_id": "client-id-123",
        "google_client_secret": "client-secret-456",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_both_halves_are_required(self):
        """A setup with only a client id can start a login and cannot finish it,
        so offering the button in that state would send a person to Google and
        then to an error.
        """
        assert google_auth.configured(_settings()) is True
        assert google_auth.configured(_settings(google_client_secret="")) is False
        assert google_auth.configured(_settings(google_client_id="")) is False
        assert google_auth.configured(_settings(google_client_id="",
                                                google_client_secret="")) is False

    def test_a_blank_variable_reads_as_absent(self):
        """"""
        assert google_auth.configured(_settings(google_client_id="   ")) is False

    def test_the_callback_url_is_derived_from_the_base(self):
        assert google_auth.callback_url("https://asr.internal:8080/") == \
            "https://asr.internal:8080/api/auth/google/callback"

    def test_a_base_with_a_trailing_slash_is_not_doubled(self):
        assert google_auth.callback_url("https://asr.internal:8080///") == \
            "https://asr.internal:8080/api/auth/google/callback"


# ---------------------------------------------------------------------------
# The consent-screen URL
# ---------------------------------------------------------------------------


class TestAuthorizationUrl:
    def test_the_url_carries_everything_google_needs(self):
        url = google_auth.authorization_url(STATE, "https://asr.internal/cb",
                                            settings=_settings())
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["client_id"] == ["client-id-123"]
        assert query["redirect_uri"] == ["https://asr.internal/cb"]
        assert query["response_type"] == ["code"]
        assert query["state"] == [STATE]

    def test_only_openid_email_and_profile_are_asked_for(self):
        """No Drive, no Calendar, no "see your Gmail". A consent screen that says
        "ASR-MCP wants to see and change your Gmail" is one nobody accepts.
        """
        url = google_auth.authorization_url(STATE, "https://asr.internal/cb",
                                            settings=_settings())
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["scope"] == ["openid email profile"]

    def test_the_account_picker_is_forced(self):
        """Without ``prompt=select_account`` a browser that has ever signed in
        silently re-authorises the PREVIOUS Google account, which on a shared
        machine signs you in as whoever used it last.
        """
        url = google_auth.authorization_url(STATE, "https://asr.internal/cb",
                                            settings=_settings())
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["prompt"] == ["select_account"]

    def test_building_the_url_does_not_need_a_secret(self):
        """The client id names the app; only the redemption needs the secret.
        """
        url = google_auth.authorization_url(
            STATE, "https://asr.internal/cb",
            client_id_override="other-id", settings=_settings(google_client_secret=""))
        assert "client_id=other-id" in url

    def test_a_missing_state_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            google_auth.authorization_url("", "https://asr.internal/cb",
                                          settings=_settings())

    def test_a_missing_redirect_uri_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            google_auth.authorization_url(STATE, "", settings=_settings())

    def test_without_a_client_id_it_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            google_auth.authorization_url(
                STATE, "https://asr.internal/cb",
                settings=_settings(google_client_id="", google_client_secret=""))

    def test_a_state_with_url_specials_is_encoded_not_concatenated(self):
        """The encoded form travels; the raw form never does.

        A state is logged, compared and eventually put in a Location header, so
        what arrives must parse back to exactly what was minted -- and nothing
        raw may be interpolated. The space becomes ``+`` and the ``&`` becomes
        ``%26``, which is what stops a crafted state from adding parameters.
        """
        url = google_auth.authorization_url("a b&c=d", "https://asr.internal/cb",
                                            settings=_settings())
        assert "state=a+b%26c%3Dd" in url
        assert "state=a b" not in url
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert query["state"] == ["a b&c=d"]

    def test_the_client_id_and_secret_never_appear_in_the_url(self):
        url = google_auth.authorization_url(STATE, "https://asr.internal/cb",
                                            settings=_settings())
        assert "client-secret-456" not in url
        assert "client_id=client-id-123" in url


# ---------------------------------------------------------------------------
# Trading the code
# ---------------------------------------------------------------------------


class TestExchangeCode:
    def test_the_code_is_redeemed_with_the_secret_over_a_form_post(self):
        opener = RecordingOpener(FakeResponse({"access_token": "token-abc"}))
        token = google_auth.exchange_code(
            "the-code", "https://asr.internal/cb", opener=opener, settings=_settings())
        assert token == "token-abc"
        sent = opener.requests[0]
        assert sent["url"] == "https://oauth2.googleapis.com/token"
        assert sent["headers"].get("Content-type") == "application/x-www-form-urlencoded"
        body = urllib.parse.parse_qs(sent["data"].decode("ascii"))
        assert body["code"] == ["the-code"]
        assert body["client_secret"] == ["client-secret-456"]
        assert body["grant_type"] == ["authorization_code"]
        assert body["redirect_uri"] == ["https://asr.internal/cb"]

    def test_the_secret_is_only_ever_in_the_body_never_the_url(self):
        opener = RecordingOpener(FakeResponse({"access_token": "token-abc"}))
        google_auth.exchange_code("the-code", "https://asr.internal/cb",
                                  opener=opener, settings=_settings())
        assert "client-secret-456" not in opener.requests[0]["url"]
        assert b"client-secret-456" in opener.requests[0]["data"]

    def test_a_bad_input_is_refused_before_any_request_is_issued(self):
        opener = RecordingOpener(FakeResponse({"access_token": "never"}))
        for bad in ("", "   ", "x" * (google_auth.MAX_CODE_CHARS + 1)):
            opener.requests.clear()
            with pytest.raises(GoogleOAuthError):
                google_auth.exchange_code(bad, "https://asr.internal/cb",
                                          opener=opener, settings=_settings())
            assert opener.requests == []

    def test_a_token_response_that_is_missing_or_oversized_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=RecordingOpener(FakeResponse({})),
                                      settings=_settings())
        with pytest.raises(GoogleOAuthError):
            google_auth.exchange_code(
                "c", "https://asr.internal/cb",
                opener=RecordingOpener(FakeResponse({"access_token": "x" * 9000})),
                settings=_settings())

    def test_a_body_that_is_not_a_json_object_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=RecordingOpener(FakeResponse(["a", "b"])),
                                      settings=_settings())

    def test_a_200_that_is_not_json_is_refused(self):
        """A captive portal or an intercepting proxy. None of those is a Google
        account, and parsing leniently is how an HTML error page becomes a
        username.
        """
        opener = RecordingOpener(FakeResponse("<html>login required</html>"))
        with pytest.raises(GoogleOAuthError):
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())

    def test_google_refusal_is_reported_in_its_own_words(self):
        opener = RecordingOpener(FakeHTTPError(400, {"error": "invalid_grant",
                                                     "error_description": "Bad Request"}))
        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())
        assert "Bad Request" in str(excinfo.value)
        assert "the-code" not in str(excinfo.value)

    def test_an_unreadable_error_body_still_produces_a_usable_message(self):
        """A 503 whose body is not JSON -- a proxy's HTML error page. The message
        still names the status, so an operator can tell "Google is down" from
        "the code was spent".
        """
        opener = RecordingOpener(FakeHTTPError(503, body=b"<html>502 Bad Gateway</html>"))
        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())
        assert "503" in str(excinfo.value)

    def test_a_network_failure_says_try_again_rather_than_refused(self):
        def boom(request, timeout=None):
            raise OSError("no route to host")

        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=boom, settings=_settings())
        assert "could not reach Google" in str(excinfo.value)

    def test_without_a_configured_client_nothing_is_sent(self):
        opener = RecordingOpener(FakeResponse({"access_token": "token"}))
        with pytest.raises(GoogleOAuthError):
            google_auth.exchange_code("c", "https://asr.internal/cb", opener=opener,
                                      settings=_settings(google_client_secret=""))
        assert opener.requests == []


# ---------------------------------------------------------------------------
# Reading the signed-in user
# ---------------------------------------------------------------------------


class TestFetchUserinfo:
    def _userinfo(self, payload, opener=None):
        if opener is None:
            opener = RecordingOpener(FakeResponse(payload))
        return google_auth.fetch_userinfo("token-abc", opener=opener)

    def test_the_signed_in_user_is_returned_normalised(self):
        info = self._userinfo({"sub": "1234567890",
                               "email": "Someone@Example.COM",
                               "email_verified": True,
                               "name": "Someone"})
        assert info == {"subject": "1234567890", "email": "someone@example.com",
                        "email_verified": True, "name": "Someone"}

    def test_the_token_is_sent_as_a_bearer_credential(self):
        opener = RecordingOpener(FakeResponse({"sub": "1"}))
        self._userinfo({"sub": "1"}, opener=opener)
        assert opener.requests[0]["headers"].get("Authorization") == "Bearer token-abc"
        assert "token-abc" not in opener.requests[0]["url"]

    def test_the_subject_is_never_derived_from_the_address(self):
        """`sub` is Google's stable id and `email` is the display address. They
        are not the same thing and must not be confused: an address can be
        renamed, reassigned or made an alias.
        """
        info = self._userinfo({"sub": "31337", "email": "a@example.com",
                               "email_verified": True})
        assert info["subject"] == "31337"

    def test_a_response_with_no_subject_is_refused(self):
        with pytest.raises(GoogleOAuthError):
            self._userinfo({"email": "a@example.com", "email_verified": True})

    def test_only_an_explicit_true_counts_as_verified(self):
        for value in (False, "true", 1, None, "yes"):
            info = self._userinfo({"sub": "1", "email": "a@example.com",
                                   "email_verified": value})
            assert info["email_verified"] is False

    def test_a_missing_address_is_empty_rather_than_fatal(self):
        info = self._userinfo({"sub": "1", "email_verified": True})
        assert info["email"] == ""

    def test_oversized_display_fields_are_truncated(self):
        info = self._userinfo({"sub": "1", "email": "a@" + "b" * 300 + ".com",
                               "email_verified": True, "name": "n" * 400})
        assert len(info["email"]) == google_auth.MAX_EMAIL_CHARS
        assert len(info["name"]) == google_auth.MAX_NAME_CHARS

    def test_an_implausible_token_is_refused_before_any_request(self):
        opener = RecordingOpener(FakeResponse({"sub": "1"}))
        for bad in ("", "   ", "x" * (google_auth.MAX_TOKEN_CHARS + 1)):
            opener.requests.clear()
            with pytest.raises(GoogleOAuthError):
                google_auth.fetch_userinfo(bad, opener=opener)
            assert opener.requests == []


# ---------------------------------------------------------------------------
# Secrecy
# ---------------------------------------------------------------------------


class TestSecrecy:
    def test_the_code_never_appears_in_a_failure_message(self):
        opener = RecordingOpener(FakeHTTPError(400, {"error_description": "nope"}))
        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("secret-code-value", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())
        assert "secret-code-value" not in str(excinfo.value)

    def test_the_client_secret_never_appears_in_a_failure_message(self):
        opener = RecordingOpener(FakeHTTPError(400, {"error_description": "nope"}))
        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())
        assert "client-secret-456" not in str(excinfo.value)

    def test_a_very_long_error_description_is_bounded(self):
        opener = RecordingOpener(FakeHTTPError(400, {"error_description": "x" * 5000}))
        with pytest.raises(GoogleOAuthError) as excinfo:
            google_auth.exchange_code("c", "https://asr.internal/cb",
                                      opener=opener, settings=_settings())
        assert len(str(excinfo.value)) < 500


class TestEndpoints:
    def test_the_endpoints_are_googles_https_urls(self):
        for url in (google_auth.GOOGLE_AUTH_URL, google_auth.GOOGLE_TOKEN_URL,
                    google_auth.GOOGLE_USERINFO_URL):
            assert url.startswith("https://")
