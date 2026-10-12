"""Google sign-in for the browser: the OAuth 2.0 authorization-code flow.

This module used to verify a Google **ID token** that a person pasted into the
Setup page, which meant it needed a JWT library, a JWKS cache, an audience
check, an algorithm pin and a clock-skew window -- about 270 lines to answer the
question "is this string really from Google?". All of that exists to support a
flow nobody asked for.

The ordinary flow is three HTTP calls and no cryptography at all:

    1. send the browser to Google's consent screen with a `state` we minted;
    2. Google sends it back to our callback URL with a short-lived `code`;
    3. we trade that code plus our client secret for an access token, and read
       the signed-in user from Google's userinfo endpoint.

Because the code is redeemed with the **client secret** over TLS, and the
userinfo response arrives over TLS from Google, the confidentiality and
authenticity the signature check used to provide are provided by the transport
and by the secret. That is how every web app that signs people in with Google
works; the local verification is what a *resource server* does when it has no
secret to present and therefore cannot make a back-channel call.

Deliberately stdlib-only. ``urllib`` is entirely adequate for two form posts and a
GET, and keeping the module importable with nothing but the standard library is
what lets the tests import and *call* it on a machine with no web framework, no
network and no JWT library.

The one seam is ``opener``, an injected ``urllib.request.OpenerDirector``-alike.
Everything above it is pure: URL building, parameter validation, and the shape
of the answers. That is the entire attack surface of an OAuth client, so that is
what the tests exercise.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from asr_mcp.config.settings import Settings, get_settings

# Google's three endpoints. The token one is not the legacy
# accounts.google.com/o/oauth2/token -- the oauth2.googleapis.com host is the
# current one and returns JSON rather than a urlencoded body.
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

# openid is what makes `sub` present in the userinfo response; email and profile
# are what make it useful. Nothing else is asked for: no Drive, no Calendar, no
# "see your Gmail". A consent screen that says "ASR-MCP wants to see and change
# your Gmail" is a consent screen nobody accepts.
GOOGLE_SCOPES = "openid email profile"

# Google's own guidance is that a userinfo call is well under a second. Ten is a
# ceiling, not a target: a hung socket on an unauthenticated login route holds a
# request open rather than failing it.
HTTP_TIMEOUT = 10.0

# A `code` is short-lived and small, and this bound is only there so a caller
# cannot hand us a megabyte to urlencode. An access token from the token endpoint
# is bounded too, for the same reason.
MAX_CODE_CHARS = 4096
MAX_TOKEN_CHARS = 8192

# An address has to fit a header and an SMTP envelope. These are the practical
# maxima, not RFC allowances.
MAX_EMAIL_CHARS = 254   # RFC 5321's practical maximum
MAX_NAME_CHARS = 120


class GoogleOAuthError(Exception):
    """Anything that stopped us learning who the person is.

    One exception type for every failure, on purpose: the message reaches a
    client response body, and a caller that has to tell six exception classes
    apart to render one error message will eventually render the wrong one. The
    message names the cause in words; it never contains a code, a token or the
    client secret.
    """


def _settings(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


def client_id(settings: Optional[Settings] = None) -> str:
    return str(_settings(settings).google_client_id or "").strip()


def client_secret(settings: Optional[Settings] = None) -> str:
    return str(_settings(settings).google_client_secret or "").strip()


def configured(settings: Optional[Settings] = None) -> bool:
    """True when a Google sign-in could actually be completed.

    Both halves are required and neither is optional for this flow. The client id
    names the app to Google; the *secret* is what makes the code redemption a
    confidential-client exchange rather than an unauthenticated one. A setup with
    only a client id can start a login and cannot finish it, so offering the
    button in that state would send a person to Google and then to an error.
    """
    return bool(client_id(settings) and client_secret(settings))


def callback_url(base_url: str) -> str:
    """Where Google sends the browser back to.

    Derived from the deployment's external base URL rather than configured
    separately, because a redirect URI that disagrees with the URL the app is
    actually served on fails at Google with an error page that names neither. It
    must be registered with Google as
    ``{base}/api/auth/google/callback``, verbatim.
    """
    return f"{str(base_url or '').rstrip('/')}/api/auth/google/callback"


def authorization_url(state: str, redirect_uri: str, *,
                     client_id_override: str = "",
                     settings: Optional[Settings] = None) -> str:
    """The Google URL to send the browser to.

    `state` is the CSRF token for the whole flow and is required, not optional:
    without it a code minted for someone else's login can be posted at this app's
    callback and signed in as them. It is the one parameter here with security
    weight rather than plumbing.
    """
    cid = str(client_id_override or "").strip() or client_id(settings)
    if not cid:
        raise GoogleOAuthError("Google sign-in is not configured on this server")
    if not str(state or "").strip():
        raise GoogleOAuthError("a state value is required to start Google sign-in")
    if not str(redirect_uri or "").strip():
        raise GoogleOAuthError("a redirect URI is required to start Google sign-in")
    query = urllib.parse.urlencode({
        "client_id": cid,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        # The one non-standard parameter. Without it, a browser that has ever
        # signed in to this app silently re-authorises the *previous* Google
        # account instead of showing the consent screen -- which on a shared
        # machine signs you in as whoever used it last.
        "prompt": "select_account",
    })
    return f"{GOOGLE_AUTH_URL}?{query}"


def exchange_code(code: str, redirect_uri: str, *, client_id_override: str = "",
                  client_secret_override: str = "", opener=None,
                  settings: Optional[Settings] = None) -> str:
    """Trade an authorization code for an access token. Returns the token.

    A `code` is single-use and short-lived, so there is nothing to cache and
    nothing to refresh: it is spent here and the answer is used immediately to
    read the userinfo endpoint. An access token with no refresh token is the
    correct shape for that -- a refresh token would be a credential at rest for
    a value that is only ever needed once.
    """
    cid = str(client_id_override or "").strip() or client_id(settings)
    secret = str(client_secret_override or "").strip() or client_secret(settings)
    if not cid or not secret:
        raise GoogleOAuthError("Google sign-in is not configured on this server")
    text = str(code or "").strip()
    if not text:
        raise GoogleOAuthError("Google did not send an authorization code")
    if len(text) > MAX_CODE_CHARS:
        # Refused rather than sent. There is no legitimate code anywhere near
        # this size, and a bounded input is a bounded request body.
        raise GoogleOAuthError("that authorization code is not plausible")
    if not str(redirect_uri or "").strip():
        raise GoogleOAuthError("a redirect URI is required to redeem the code")

    body = urllib.parse.urlencode({
        "code": text,
        "client_id": cid,
        "client_secret": secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }).encode("ascii")
    payload = _request_json(
        GOOGLE_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        opener=opener,
        what="authorization code",
    )
    token = str(payload.get("access_token") or "").strip()
    if not token:
        # A 200 with no access_token is not a thing Google's server does; it is
        # a thing a proxy or a captive portal does. Refusing is the only answer
        # that does not sign somebody in as nobody.
        raise GoogleOAuthError("Google's token response contained no access token")
    if len(token) > MAX_TOKEN_CHARS:
        raise GoogleOAuthError("Google's access token is implausibly long")
    return token


def fetch_userinfo(access_token: str, *, opener=None) -> dict:
    """Read the signed-in user. Returns `{subject, email, email_verified, name}`.

    `sub` is Google's stable identifier for the account and `email` is the
    display address; the two are not the same thing and must not be confused.
    An address can be renamed, reassigned or made an alias, which is why the
    vault is looked up by `sub` where a link exists.
    """
    token = str(access_token or "").strip()
    if not token:
        raise GoogleOAuthError("no access token was available to identify the user")
    if len(token) > MAX_TOKEN_CHARS:
        raise GoogleOAuthError("that access token is implausibly long")

    payload = _request_json(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {token}"},
        opener=opener,
        what="user info",
    )
    subject = str(payload.get("sub") or "").strip()
    if not subject:
        # Refused rather than defaulted. A userinfo response with no `sub` is
        # not a user we can key anything on, and inventing one would produce a
        # vault whose owner cannot sign in again.
        raise GoogleOAuthError("Google's user info response carried no account id")
    # `email_verified` comes back as a JSON boolean on this endpoint. Anything
    # other than an explicit true is treated as unverified, because the failure
    # that matters is the one where a non-boolean reads as present.
    verified = payload.get("email_verified") is True
    return {
        "subject": subject,
        "email": str(payload.get("email") or "").strip().lower()[:MAX_EMAIL_CHARS],
        "email_verified": verified,
        "name": str(payload.get("name") or "").strip()[:MAX_NAME_CHARS],
    }


def _request_json(url: str, *, data=None, headers=None, opener=None,
                   what: str = "response") -> dict:
    """One HTTPS call, returning parsed JSON, or raising GoogleOAuthError.

    The single place this module touches the network, so the error text is
    written once and every caller reports failure the same way. `opener` is the
    test seam: it takes the same call signature as `urllib.request.urlopen`.
    """
    request = urllib.request.Request(url, data=data, headers=dict(headers or {}))
    send = opener if opener is not None else urllib.request.urlopen
    logger = logging.getLogger("asr_mcp.auth.google_auth")
    try:
        with send(request, timeout=HTTP_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # Google's error body is JSON and names the cause in
        # error_description -- "invalid_grant" means the code was already spent
        # or expired, which is worth saying because it is the single most common
        # failure here and it looks like nothing else.
        detail = _error_detail(exc)
        logger.warning("google oauth: %s refused with HTTP %s: %s", what, exc.code, detail)
        raise GoogleOAuthError(f"Google refused the {what} ({detail})") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Network failure, DNS, TLS, timeout. One message for all of them
        # because they are one answer to the client's question: not now.
        reason = getattr(exc, "reason", exc)
        logger.warning("google oauth: could not reach Google for %s: %s", what, reason)
        raise GoogleOAuthError("could not reach Google — try again in a moment") from None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        # A 200 that is not JSON is a captive portal, an intercepting proxy or a
        # misconfigured base URL. None of those is a Google account, and parsing
        # it leniently is how an HTML error page becomes a username.
        logger.warning("google oauth: %s was not JSON (%d bytes)", what, len(raw))
        raise GoogleOAuthError(f"Google's {what} response could not be read") from None
    if not isinstance(payload, dict):
        logger.warning("google oauth: %s was JSON but not an object", what)
        raise GoogleOAuthError(f"Google's {what} response could not be read") from None
    return payload


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The human half of Google's error body, without leaking anything.

    Bounded, because this string is interpolated into a message that reaches the
    browser, and a proxy's error page can be arbitrarily long.
    """
    try:
        body = exc.read()[:2048]
        payload = json.loads(body.decode("utf-8", "replace"))
        if isinstance(payload, dict):
            described = str(payload.get("error_description") or payload.get("error") or "").strip()
            if described:
                return described[:200]
    except Exception:
        pass
    return f"HTTP {exc.code}"
