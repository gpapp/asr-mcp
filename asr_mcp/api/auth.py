import hashlib
import hmac
import logging
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import RedirectResponse

logger = logging.getLogger("asr_mcp.api.auth")

# Public paths that don't require authentication
PUBLIC_PATHS = {"/login", "/health", "/api/auth/login", "/api/user"}


def _get_prefix(request: Request) -> str:
    """Get the URL prefix from settings."""
    from asr_mcp.config.settings import get_settings
    return get_settings().prefix


def redirect_url(request: Request, path: str) -> str:
    """Prefix a path with the configured URL prefix."""
    prefix = _get_prefix(request)
    return prefix + path


def parse_htpasswd(path: str) -> dict[str, str]:
    """Parse an Apache htpasswd file into {username: hash} dict."""
    users: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        logger.warning("htpasswd file not found: %s", path)
        return users
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            username, password_hash = line.split(":", 1)
            users[username.strip()] = password_hash.strip()
    return users


def verify_password(stored_hash: str, password: str) -> bool:
    """Verify password against stored hash (bcrypt, apr1, sha256, or plain)."""
    if stored_hash.startswith("$2b$") or stored_hash.startswith("$2y$"):
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except ImportError:
            logger.error("bcrypt not installed")
            return False
    if stored_hash.startswith("$apr1$"):
        return _verify_apr1(stored_hash, password)
    if stored_hash.startswith("{SHA}"):
        encoded = hashlib.sha1(password.encode("utf-8")).digest()
        import base64
        return hmac.compare_digest(stored_hash[5:], base64.b64encode(encoded).decode())
    if stored_hash.startswith("{SSHA}"):
        import base64
        decoded = base64.b64decode(stored_hash[6:])
        salt = decoded[20:]
        check = hashlib.sha1(password.encode("utf-8") + salt).digest()
        return hmac.compare_digest(decoded[:20], check)
    return hmac.compare_digest(stored_hash, password)


def _verify_apr1(stored: str, password: str) -> bool:
    """Verify Apache MD5 (apr1) hash."""
    parts = stored.split("$")
    if len(parts) != 4:
        return False
    salt = parts[2]
    try:
        import subprocess
        result = subprocess.run(
            ["openssl", "passwd", "-apr1", "-salt", salt, password],
            capture_output=True, text=True, timeout=5,
        )
        return hmac.compare_digest(result.stdout.strip(), stored)
    except Exception:
        pass
    try:
        import crypt
        return hmac.compare_digest(crypt.crypt(password, stored), stored)
    except ImportError:
        return False


def get_session_user(request: Request) -> Optional[str]:
    """Get the logged-in username from the session, or None."""
    return request.scope.get("session", {}).get("user")


def require_auth(request: Request) -> Optional[RedirectResponse]:
    """Check if request is authenticated. Returns RedirectResponse to /login if not."""
    path = request.url.path or "/"
    # Normalize double slashes
    while "//" in path:
        path = path.replace("//", "/")

    # Allow public paths
    if path in PUBLIC_PATHS:
        return None
    if path.startswith("/static"):
        return None
    # Allow health checks and API user endpoint
    if path == "/health" or path.startswith("/api/user"):
        return None

    user = get_session_user(request)
    if user:
        return None

    return RedirectResponse(url=redirect_url(request, "/login"), status_code=302)


def login_user(request: Request, username: str) -> None:
    """Store username in session."""
    request.scope["session"]["user"] = username


def logout_user(request: Request) -> None:
    """Clear user from session."""
    if "session" in request.scope:
        request.scope["session"].pop("user", None)
