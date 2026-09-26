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


def lookup_db_token(token: str) -> Optional[str]:
    """Return the user_id owning a database-issued API token, or None."""
    if not token:
        return None
    try:
        from asr_mcp.db.manager import DatabaseManager, TokenDB
        db = DatabaseManager(get_settings().db_path)
        return TokenDB(db).verify(token)
    except Exception as e:
        logger.debug("Token lookup failed: %s", e)
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
        token_user = lookup_db_token(x_api_key)
        if token_user:
            return token_user
        if settings.api_keys and x_api_key in settings.api_key_set:
            h = hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
            return f"user_{h}"
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not settings.api_keys:
        return DEFAULT_USER
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
