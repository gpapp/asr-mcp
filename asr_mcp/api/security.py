import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, Header, HTTPException

from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.security")

DEFAULT_USER = "default"


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    if not settings.api_keys:
        return None
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


async def get_current_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    if not settings.api_keys:
        return DEFAULT_USER
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    h = hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    return f"user_{h}"


def validate_path_security(path: str, settings: Settings) -> Path:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    allowed_dirs = [settings.data_dir, Path("/tmp"), Path("/app/data")]
    resolved = p.resolve()
    if not any(str(resolved).startswith(str(d.resolve())) for d in allowed_dirs if d.exists()):
        raise HTTPException(status_code=403, detail="Path not in allowed directories")
    if p.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    return p
