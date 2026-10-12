"""Self-service registration, email confirmation, and the login that uses them.

The route shapes and their orderings are ported from mem-mcp's ``gui.py``, and
the two orderings that look arbitrary are not:

* **Throttle before the registration flag.** Counting only *successful* signups
  would remove the rate limit from a still-mounted route the moment an operator
  turns registration off -- exactly when someone is probing for it.
* **A disabled method answers 404, not 403.** A 403 confirms the route exists
  and is merely closed; a 404 is indistinguishable from a path that was never
  mounted.

Three more properties this file is responsible for, each of which is a bug if
it regresses:

* **Registration never signs anybody in.** The account is unverified until the
  link is opened, so a session here would be a session for an account nobody
  has proved they own.
* **A failed confirmation email deletes the account it just made.** Leaving the
  row behind would strand a half-account that can never sign in and whose name
  its registrant cannot reclaim.
* **Nothing in a response is request-controlled text.** Every redirect target is
  built by this module; the ``verified=1``/``=0`` and ``google=`` outcomes are
  fixed strings, so no parameter a caller supplies reaches a Location header.
"""

import asyncio
import hmac
import logging
import secrets
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from asr_mcp.api.auth import get_session_user, login_user, verify_password
from asr_mcp.api.security import get_settings
from asr_mcp.auth import google_auth
from asr_mcp.auth.mailer import log_send_failure, send_mail, verification_email
from asr_mcp.auth.passwords import MIN_PASSWORD_CHARS
from asr_mcp.auth.store import (
    GOOGLE_PROVIDER,
    CredentialStore,
    allow_registration_attempt,
    credential_store,
    registration_enabled,
)
from asr_mcp.config.settings import Settings
from asr_mcp.api.schemas import RegisterRequest

logger = logging.getLogger("asr_mcp.api.auth_router")

router = APIRouter()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _store(request: Request, settings: Settings) -> CredentialStore:
    """A credential store, carrying the operator's htpasswd names with it.

    The parsed map is what makes "is this name taken" a pure in-process
    question: the file is read once at startup into ``app.state``, so
    registration never shells out to the ``htpasswd`` binary (which the
    container does not ship anyway).
    """
    users = getattr(request.app.state, "htpasswd_users", None) if request.app else None
    return credential_store(settings, htpasswd_users=users)


def _htpasswd_users(request: Request) -> dict:
    if request.app is None:
        return {}
    return getattr(request.app.state, "htpasswd_users", None) or {}


def _external_url(settings: Settings, request: Request) -> str:
    """The absolute base this deployment is served from.

    ``TRANSCRIBE_EXTERNAL_URL`` when set, otherwise derived from the request.
    The setting exists because deriving is wrong the moment a reverse proxy
    rewrites the Host header, and a confirmation link that names the wrong host
    is a link nobody can open.
    """
    base = str(settings.external_url or "").rstrip("/")
    if not base:
        base = str(request.base_url).rstrip("/")
    return base


def _public_base(settings: Settings, request: Request) -> str:
    """The absolute base INCLUDING the URL prefix, exactly once.

    Both spellings of ``TRANSCRIBE_EXTERNAL_URL`` are in real deployments and
    both are reasonable: ``https://host`` (the bare origin, prefix appended
    here) and ``https://host/asr-mcp`` (the page a user actually pastes into a
    browser, which is the whole public URL of the app). Appending the prefix
    unconditionally turns the second into ``/asr-mcp/asr-mcp/...``, and the
    symptom is a 404 on the one link that must work -- a confirmation link that
    404s looks exactly like a token that was refused.

    So the prefix is appended only when it is not already the tail of the base.
    Comparison is on a path-segment boundary, not a bare ``endswith``, so a
    prefix of ``/asr`` and a base of ``https://host/asr-mcp`` still combine to
    ``https://host/asr-mcp/asr`` rather than silently dropping the prefix.
    """
    base = _external_url(settings, request)
    prefix = str(settings.prefix or "").strip("/")
    if not prefix:
        return base
    if base == f"/{prefix}" or base.endswith(f"/{prefix}"):
        return base
    return f"{base}/{prefix}"


def _signup_client(request: Request) -> str:
    """The rightmost ``X-Forwarded-For`` hop, or the socket peer.

    Rightmost, because nginx *appends* the peer it saw: every entry to the left
    of the last one was supplied by the client and can be forged. Taking the
    leftmost hop -- the client's own claim about itself -- would let one
    attacker rotate their quota as often as they liked.
    """
    forwarded = request.headers.get("x-forwarded-for") or ""
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else ""


async def _require_registration(method: str, request: Request, settings: Settings) -> None:
    """Refuse a signup that should not be happening, in the order that matters."""
    if not allow_registration_attempt(_signup_client(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many signup attempts. Try again in a few minutes.",
        )
    if not registration_enabled(method, settings):
        raise HTTPException(status_code=404, detail="Not found")


def _verify_account(username: str, password: str, request: Request,
                    settings: Settings) -> tuple:
    """`(user_id, reason)` -- reason is "" on success, else why it failed.

    Two stores in a fixed order, so the login form and an HTTP Basic client
    cannot disagree about who can sign in:

    1. the credentials table, tried as the name exactly as typed AND lowercased
       (the table is keyed on the normalised username, so a user who types
       their own name with a capital letter must not get a second, empty
       vault);
    2. the htpasswd file, exactly as typed -- the operator's file is
       case-sensitive by nature and this app does not second-guess it.

    A refusal reason returned by the first store always stops the search, and a
    *missing* account does not: :meth:`check_login` answers ``("", "")`` for a
    name it does not know, which is indistinguishable from a wrong password.
    That is deliberate -- it means "no such user" cannot be used to enumerate
    accounts -- and it is also what lets the htpasswd file get its turn.
    """
    store = _store(request, settings)
    for candidate in (username, username.lower()):
        # Both spellings, because the table is keyed on the normalised
        # username: an operator's htpasswd file is case-sensitive and a
        # credential is not, so a user who types their own name capitalised
        # must not get a second, empty vault. And a reason from either spelling
        # stops the search -- an account that exists but cannot be used must be
        # told so, rather than being quietly authenticated by some other row
        # that happens to share the name.
        key, reason = store.check_login(candidate, password)
        if key:
            return key, ""
        if reason:
            return "", reason
    users = _htpasswd_users(request)
    stored = users.get(username)
    if stored and verify_password(stored, password):
        return username, ""
    return "", ""


# ----------------------------------------------------------------------
# What the login page may offer
# ----------------------------------------------------------------------

@router.get("/auth/methods")
async def auth_methods(request: Request, settings: Settings = Depends(get_settings)):
    """Which signup methods could actually work right now.

    Public, because it is what the login page needs BEFORE anybody is
    authenticated, and it carries nothing but three booleans and a number. A
    card is rendered only for a method that both is enabled and is configured,
    so a half-configured deployment never shows a button that starts a login it
    cannot finish.
    """
    return {
        "email": bool(registration_enabled("email", settings)),
        "google": bool(registration_enabled("google", settings)),
        "password_hint": MIN_PASSWORD_CHARS,
    }


# ----------------------------------------------------------------------
# Login / logout
# ----------------------------------------------------------------------

@router.post("/auth/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
):
    """Sign in. This is the form login.html posts to."""
    key, reason = _verify_account(username, password, request, settings)
    if not key:
        # 401 for "those are not a working pair" and 403 for "that account
        # exists and cannot be used". The second one has to be distinguishable,
        # because the person needs to be told to open their confirmation link
        # rather than to keep guessing at a password that is already right. It
        # is only reachable for a name that EXISTS, so it leaks nothing a wrong
        # password for a real account would not.
        status = 403 if reason else 401
        return JSONResponse(
            status_code=status,
            content={"success": False,
                     "error": reason or "Invalid username or password"},
        )
    # Cleared before the user is written, not after: a fresh session payload on
    # login means a cookie captured before it cannot be replayed as a logged-in
    # one.
    request.session.clear()
    login_user(request, key)
    logger.info("Login: %s", key)
    return JSONResponse({"success": True, "username": key,
                         "redirect": settings.prefix + "/gui"})


@router.post("/auth/login/session")
async def login_set_session(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
):
    """Login and set the session cookie (the form-POST redirect flow)."""
    key, reason = _verify_account(username, password, request, settings)
    if not key:
        return RedirectResponse(url=settings.prefix + "/login", status_code=302)
    request.session.clear()
    login_user(request, key)
    logger.info("Login (redirect flow): %s", key)
    return RedirectResponse(url=settings.prefix + "/gui", status_code=302)


@router.get("/auth/logout")
async def logout_get(request: Request, settings: Settings = Depends(get_settings)):
    from asr_mcp.api.auth import logout_user
    logout_user(request)
    return RedirectResponse(url=settings.prefix + "/login", status_code=302)


@router.post("/auth/logout")
async def logout_post(request: Request, settings: Settings = Depends(get_settings)):
    from asr_mcp.api.auth import logout_user
    logout_user(request)
    return JSONResponse({"success": True, "redirect": settings.prefix + "/login"})


@router.get("/user")
async def get_current_user_info(request: Request):
    """Who the session says you are.

    Reads the session only, deliberately: this is the endpoint the browser
    calls to render its own header, and the identity a request *acts* as is
    resolved by ``get_current_user`` (which also understands tokens). Mixing
    the two here would make an API-key caller's session-less request look
    anonymous on screen while writing to a real vault.
    """
    user = get_session_user(request)
    if user:
        return {"username": user, "authenticated": True}
    return {"username": "anonymous", "authenticated": False}


# ----------------------------------------------------------------------
# Email registration
# ----------------------------------------------------------------------

@router.post("/auth/register", status_code=201)
async def register(
    request: Request,
    payload: RegisterRequest,
    settings: Settings = Depends(get_settings),
):
    """Create an account and email a confirmation link. Never signs anybody in."""
    await _require_registration("email", request, settings)

    store = _store(request, settings)
    try:
        record = store.create(payload.username, payload.password, payload.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    try:
        token = store.issue_verification_token(record["user_id"])
    except ValueError as exc:
        # No address to send to, or a row that just vanished. Nothing was sent
        # and nothing can be, so the account is undone rather than stranded.
        store.delete(record["user_id"])
        raise HTTPException(status_code=400, detail=str(exc)) from None

    link = (f"{_public_base(settings, request)}"
            f"/api/auth/verify?token={quote(token, safe='')}")
    subject, body = verification_email(record["user_id"], link)
    try:
        await asyncio.to_thread(send_mail, record["email"], subject, body,
                                settings=settings)
    except Exception as exc:
        # Both halves matter: the row goes, so a failed mail leaves no
        # half-account; and the reason is logged by TYPE only, so neither the
        # address nor the live token reaches a log file.
        store.delete(record["user_id"])
        log_send_failure(exc)
        raise HTTPException(
            status_code=503,
            detail=("the account was not created because the confirmation email "
                    "could not be sent — nothing was saved, try again"),
        ) from None

    logger.info("Registration created (pending confirmation): %s", record["user_id"])
    return {
        "status": "ok",
        "user": record["user_id"],
        "email": record["email"],
        "verification_sent": True,
    }


@router.get("/auth/verify")
async def verify_email(request: Request, token: str = "",
                       settings: Settings = Depends(get_settings)):
    """Spend a confirmation link. A GET, because a mail client will not POST."""
    store = _store(request, settings)
    user_id = store.verify_email_token(token)
    if user_id:
        logger.info("Account confirmed: %s", user_id)
        return RedirectResponse(
            url=f"{settings.prefix}/login?verified=1", status_code=302)
    # One indistinguishable answer for a bad, spent or empty token. Naming which
    # one it was would tell someone probing with guesses whether a link ever
    # existed.
    return RedirectResponse(url=f"{settings.prefix}/login?verified=0", status_code=302)


# ----------------------------------------------------------------------
# Google sign-in
# ----------------------------------------------------------------------

def _google_callback_url(settings: Settings, request: Request) -> str:
    """The redirect URI registered with Google, verbatim.

    Derived from the deployment's external base URL plus the URL prefix, because
    a redirect URI that disagrees with the URL the app is actually served on
    fails at Google with an error page that names neither. ``_public_base`` is
    the same helper the confirmation link uses, and for the same reason: the two
    links have to agree about where this deployment lives, so they must not
    disagree about whether the prefix is already in the base.
    """
    return google_auth.callback_url(_public_base(settings, request))


def _google_signup(info: dict, request: Request, settings: Settings,
                   logger: logging.Logger):
    """First-time Google sign-in: create the account, or say why not.

    Returns the new user id, or None. Each refusal is a different problem and
    the caller is the only place that knows which, so they are logged here and
    answered there.

    The account key is the *address*, and there is no verification mail for it,
    which is the whole justification: Google has already verified that this
    account controls this address, so re-verifying it would prove nothing and
    cost a message that could not be delivered anyway.
    """
    address = info["email"]
    if not address or not info["email_verified"]:
        # Google says it will only return an email for a verified address, so an
        # unverified or empty one means the scope was not granted or the account
        # has no address. Either way there is no account key to build from.
        logger.info("google sign-in: no verified email address, refusing to create an account")
        return None
    if not registration_enabled(GOOGLE_PROVIDER, settings):
        logger.info("google sign-in: unknown account and registration is closed")
        return None
    store = _store(request, settings)
    if store.user_id_taken(address):
        # An account under this name already exists, so the right answer is to
        # sign in with its password rather than to open a second way into the
        # same account. Google has proved who the person is; it has not proved
        # which of their accounts they meant, and a subject link is the only
        # thing that can.
        logger.info(
            "google sign-in: %r is already an account; not opening a second way "
            "in -- sign in with that account's username and password", address,
        )
        return None
    store.link_google_identity(info["subject"], address, email=address,
                               name=info["name"])
    logger.info("signup: google account created %r", address)
    return address


@router.get("/auth/google/start")
async def google_start(request: Request, settings: Settings = Depends(get_settings)):
    """Redirect to Google's consent screen.

    Gated on the client being configured, and *not* on the registration flag --
    switching registration off has to stop new accounts without signing out the
    people who already have one.
    """
    if not google_auth.configured(settings):
        raise HTTPException(status_code=404, detail="Not found")
    state = secrets.token_urlsafe(24)
    # In the server-side session, not a cookie of our own: it is already
    # writable, it is HttpOnly, and it is cleared on the same request that
    # consumes it.
    request.session["oauth_state"] = state
    url = google_auth.authorization_url(state, _google_callback_url(settings, request),
                                        settings=settings)
    return RedirectResponse(url=url, status_code=302)


@router.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "",
                          error: str = "", error_description: str = "",
                          settings: Settings = Depends(get_settings)):
    """Finish a Google sign-in and create the account on first use.

    The `state` check is the CSRF defence for the whole flow: without it, a code
    minted for someone else's login could be posted here and signed in as them.
    It is compared with `compare_digest` and consumed either way, so a state can
    be replayed exactly once.
    """
    expected = str(request.session.get("oauth_state") or "")
    request.session["oauth_state"] = ""
    if not expected or not state or not hmac.compare_digest(str(state), expected):
        logger.info("google sign-in: state did not match")
        return RedirectResponse(url=f"{settings.prefix}/login?google=state", status_code=302)

    if error or not code:
        # Google refused, or the person dismissed the consent screen. Both are
        # normal outcomes, not failures to log loudly.
        return RedirectResponse(url=f"{settings.prefix}/login?google=declined", status_code=302)

    log = logging.getLogger("asr_mcp.api.auth_router")
    try:
        redirect_uri = _google_callback_url(settings, request)
        access_token = await asyncio.to_thread(google_auth.exchange_code, code,
                                               redirect_uri, settings=settings)
        info = await asyncio.to_thread(google_auth.fetch_userinfo, access_token)
    except google_auth.GoogleOAuthError as exc:
        # The exception's own text is free of code and token material and is the
        # one line an operator needs to tell "the code expired" from "Google is
        # unreachable".
        log.warning("google sign-in: %s", exc)
        return RedirectResponse(url=f"{settings.prefix}/login?google=failed", status_code=302)

    store = _store(request, settings)
    link = store.resolve_google_identity(info["subject"])
    if link:
        user_id = link["user_id"]
    else:
        user_id = _google_signup(info, request, settings, log)
        if not user_id:
            return RedirectResponse(url=f"{settings.prefix}/login?google=closed", status_code=302)

    # Cleared before the user is written, not after: a fresh session payload on
    # login means a cookie captured before it cannot be replayed as a logged-in
    # one.
    request.session.clear()
    login_user(request, user_id)
    log.info("google sign-in: user=%s", user_id)
    return RedirectResponse(url=f"{settings.prefix}/gui", status_code=302)
