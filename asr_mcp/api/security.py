import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.security")

DEFAULT_USER = "default"

SUPPORTED_AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".mkv", ".mp4",
}


def validate_upload_filename(filename: str) -> str:
    """Verify an uploaded file's extension is a supported audio/video format."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix or filename}")
    return filename


def lookup_db_token(token: str,
                    settings: Optional[Settings] = None) -> Optional[str]:
    """Return the user_id owning a database-issued API token, or None.

    Takes ``settings`` so the token rung and the static-key rung can honour the
    SAME configuration: without it this opened the process-global
    ``get_settings().db_path`` while the caller's ``settings`` was used for
    everything else, which is two different stores for one decision.
    """
    if not token:
        return None
    try:
        from asr_mcp.db.manager import DatabaseManager, TokenDB
        if settings is None:
            settings = get_settings()
        db = DatabaseManager(settings.db_path)
        return TokenDB(db).verify(token)
    except Exception as e:
        logger.debug("Token lookup failed: %s", e)
        return None


def resolve_api_key_user(x_api_key: str,
                         settings: Optional[Settings] = None) -> Optional[str]:
    """The user a raw API key belongs to, or None when it is not a usable key.

    ONE definition, used by the HTTP gate and the WebSocket handshake alike.
    A static ``TRANSCRIBE_API_KEYS`` value must resolve to the same synthetic
    user on both transports: when it did not, the same key read TWO different
    voiceprint libraries depending on which door the client came through
    (``user_<hash16>`` over HTTP, ``default`` over the WebSocket), so a client
    that matched its own speakers over HTTP lost them all the moment it opened
    a socket.

    ``None`` means "not a credential at all" -- an unknown key. It is
    deliberately distinct from a user id so callers can refuse rather than
    silently falling back to a shared identity.
    """
    if not x_api_key:
        return None
    if settings is None:
        settings = get_settings()
    token_user = lookup_db_token(x_api_key, settings)
    if token_user:
        return token_user
    if settings.api_keys and x_api_key in settings.api_key_set:
        h = hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
        return f"user_{h}"
    return None


def is_valid_api_key(x_api_key: str, settings: Optional[Settings] = None) -> bool:
    if not x_api_key:
        return False
    if settings is None:
        settings = get_settings()
    if settings.api_keys and x_api_key in settings.api_key_set:
        return True
    return lookup_db_token(x_api_key) is not None


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    from asr_mcp.api.auth import get_session_user
    if get_session_user(request):
        return None
    if x_api_key:
        if is_valid_api_key(x_api_key, settings):
            return x_api_key
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not settings.api_keys:
        return None
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def get_current_user(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    # Session-based auth (web UI) — check session cookie first
    from asr_mcp.api.auth import get_session_user
    session_user = get_session_user(request)
    if session_user:
        return session_user
    # API token auth (database-issued token, one per user)
    if x_api_key:
        key_user = resolve_api_key_user(x_api_key, settings)
        if key_user:
            return key_user
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not settings.api_keys:
        # While registration is off this is the documented open mode: one
        # shared `default` library. Once registration is on it is refused,
        # because an unauthenticated caller reading and writing one shared
        # voiceprint library is exactly what per-user separation exists to
        # prevent, and offering signup while leaving the back door open would
        # make the signup cosmetic.
        if settings.registration_enabled:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return DEFAULT_USER
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def is_admin_user(user: str, settings: Optional[Settings] = None) -> bool:
    """Whether ``user`` is the configured administrator.

    Plain equality against ``TRANSCRIBE_ADMIN_USER``, deliberately. A
    constant-time comparison would be the reflex for a password, but this is a
    username checked against a deployment constant: the answer is not a
    credential, and the timing signal is which of two configured names it is --
    which the caller could name anyway.

    An empty setting means **nobody** is admin. An operator who never set the
    variable has not opted into server-wide operations, so the default is the
    restrictive one rather than " everybody who can log in".
    """
    if settings is None:
        settings = get_settings()
    admin = str(settings.admin_user or "").strip()
    return bool(admin) and user == admin


async def require_admin(
    user: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> str:
    """The signed-in user, but only if they are the configured administrator.

    Everything this gates is **not** per-user. ``POST /shutdown`` stops a
    process other users are queueing jobs against; the admin user table reads
    every account's email and confirmation state; ``?purge=true`` deletes
    another user's voiceprints and recordings. Gating those on "whoever is
    logged in" would make them reachable by any account, so they require the one
    name in ``TRANSCRIBE_ADMIN_USER`` -- and when that is unset, nobody.

    The 403 names the variable so an operator who hit the wall learns what to
    set rather than concluding the route is missing.
    """
    if not is_admin_user(user, settings):
        raise HTTPException(
            status_code=403,
            detail=(
                "That operation is limited to the administrator. "
                "Set TRANSCRIBE_ADMIN_USER to the account that may run it."
            ),
        )
    return user


def validate_path_security(path: str, settings: Settings) -> Path:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    allowed_dirs = [settings.data_dir, Path("/tmp"), Path("/app/data")]
    resolved = p.resolve()
    if not any(str(resolved).startswith(str(d.resolve())) for d in allowed_dirs if d.exists()):
        raise HTTPException(status_code=403, detail="Path not in allowed directories")
    if p.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    return p
