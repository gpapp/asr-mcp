import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

logger = logging.getLogger("asr_mcp.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from asr_mcp.config.settings import get_settings
    from asr_mcp.config.logging import setup_logging
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.core.model_loader import load_models
    from asr_mcp.core.model_state import state, executor as _executor
    import asr_mcp.core.model_state as ms
    from concurrent.futures import ThreadPoolExecutor
    from asr_mcp.speaker.service import SpeakerService
    from asr_mcp.sessions.manager import SessionManager
    from asr_mcp.voiceprint.service import VoiceprintService

    settings = get_settings()
    setup_logging(settings.log_dir, debug=False)

    logger.info("Starting asr-mcp server...")
    logger.info("CUDA device: %s", settings.cuda_device)
    logger.info("DB path: %s", settings.db_path)

    # Initialize database
    db_manager = DatabaseManager(settings.db_path)
    app.state.db_manager = db_manager

    # Initialize services
    app.state.speaker_service = SpeakerService(settings.data_dir, db_manager)
    app.state.session_manager = SessionManager(db_manager)
    app.state.voiceprint_service = VoiceprintService(settings.data_dir, db_manager)
    app.state.voiceprint_service.set_voices_dir(settings.voices_dir)

    # Initialize thread pool
    ms.executor = ThreadPoolExecutor(max_workers=2)

    # Load htpasswd users
    from asr_mcp.api.auth import parse_htpasswd
    app.state.htpasswd_users = parse_htpasswd(settings.htpasswd_path)
    logger.info("Loaded %d user(s) from htpasswd", len(app.state.htpasswd_users))

    # Load GPU models
    try:
        load_models(settings)
        app.state.voiceprint_service.set_embedding_session(state.embedding_session)
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        logger.warning("Server starting without GPU models")

    await app.state.speaker_service.initialize()
    await app.state.session_manager.initialize()
    await app.state.voiceprint_service.initialize()

    logger.info("asr-mcp server ready on %s:%d", settings.host, settings.port)

    yield

    # Shutdown
    logger.info("Shutting down...")
    state.clear_gpu_memory()
    db_manager.close()
    if ms.executor:
        ms.executor.shutdown(wait=False)
    logger.info("Shutdown complete")


app = FastAPI(
    title="ASR MCP Server",
    description="ASR transcription with diarization, voiceprint recognition, and MCP interface",
    version="2.0.0",
    lifespan=lifespan,
)

# Session middleware (must be first)
from asr_mcp.config.settings import get_settings as _init_settings
_init_secret = _init_settings().session_secret
app.add_middleware(
    SessionMiddleware,
    secret_key=_init_secret,
    session_cookie="asr_session",
    max_age=86400 * 7,
    same_site="lax",
    https_only=False,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware
from asr_mcp.api.middleware import apply_middleware
apply_middleware(app)

# Exception handlers
from asr_mcp.api.exceptions import register_exception_handlers
register_exception_handlers(app)

# API Router
from asr_mcp.api.router import api_router
app.include_router(api_router)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    from asr_mcp.api.auth import require_auth
    redirect = require_auth(request)
    if redirect:
        return redirect
    response = await call_next(request)
    return response


# Static files & templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _render(template_name: str) -> HTMLResponse:
    path = templates_dir / template_name
    if path.exists():
        content = path.read_text(encoding="utf-8")
        prefix = _init_settings().prefix
        content = content.replace("__PREFIX__", prefix)
        return HTMLResponse(content=content)
    return HTMLResponse(content=f"<h1>{template_name} not found</h1>")


@app.get("/", response_class=HTMLResponse)
async def root():
    prefix = _init_settings().prefix
    return RedirectResponse(url=prefix + "/gui", status_code=302)


@app.get("/health")
async def health():
    from asr_mcp.core.model_state import state
    from asr_mcp.config.settings import get_settings
    settings = get_settings()
    return {
        "status": "healthy" if state.is_ready else "loading",
        "model_status": "ready" if state.is_ready else "not_ready",
        "cuda_device": settings.cuda_device,
        "voiceprint_count": app.state.speaker_service._db.count() if hasattr(app.state, "speaker_service") else 0,
        "version": "2.0.0",
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _render("login.html")


@app.post("/api/auth/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    from asr_mcp.api.auth import verify_password, login_user

    users = app.state.htpasswd_users if hasattr(app.state, "htpasswd_users") else {}
    if not users:
        return JSONResponse({"success": False, "error": "No users configured. Create an htpasswd file."})

    stored = users.get(username)
    if not stored or not verify_password(stored, password):
        return JSONResponse({"success": False, "error": "Invalid username or password"})

    login_user(request, username)
    return JSONResponse({"success": True, "username": username, "redirect": _init_settings().prefix + "/gui"})


@app.post("/api/auth/login/session")
async def login_set_session(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Login and set session cookie (for form POST redirect flow)."""
    from asr_mcp.api.auth import verify_password, login_user

    users = app.state.htpasswd_users if hasattr(app.state, "htpasswd_users") else {}
    if not users:
        return RedirectResponse(url=_init_settings().prefix + "/login", status_code=302)

    stored = users.get(username)
    if not stored or not verify_password(stored, password):
        return _render("login.html")

    login_user(request, username)
    return RedirectResponse(url=_init_settings().prefix + "/gui", status_code=302)


@app.get("/api/auth/logout")
async def logout_get(request: Request):
    from asr_mcp.api.auth import logout_user
    logout_user(request)
    return RedirectResponse(url=_init_settings().prefix + "/login", status_code=302)


@app.post("/api/auth/logout")
async def logout_post(request: Request):
    from asr_mcp.api.auth import logout_user
    logout_user(request)
    return JSONResponse({"success": True, "redirect": _init_settings().prefix + "/login"})


@app.get("/gui", response_class=HTMLResponse)
async def gui(request: Request):
    return _render("index.html")


@app.get("/voices", response_class=HTMLResponse)
async def voices(request: Request):
    return _render("voices.html")


@app.get("/api/user")
async def get_user(request: Request):
    from asr_mcp.api.auth import get_session_user
    user = get_session_user(request)
    if user:
        return {"username": user, "authenticated": True}
    return {"username": "anonymous", "authenticated": False}


@app.post("/shutdown")
async def shutdown():
    import signal
    logger.info("Shutdown requested via API")
    asyncio.get_event_loop().call_later(1.0, lambda: signal.raise_signal(signal.SIGINT) if hasattr(signal, "raise_signal") else None)
    return {"status": "shutting down"}


if __name__ == "__main__":
    import uvicorn
    from asr_mcp.config.settings import get_settings as _get_settings
    settings = _get_settings()
    uvicorn.run(
        "asr_mcp.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
    )
