import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from asr_mcp import __version__

logger = logging.getLogger("asr_mcp.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from asr_mcp.config.settings import get_settings
    from asr_mcp.config.logging import setup_logging
    from asr_mcp.db.manager import DatabaseManager
    from asr_mcp.core.model_state import state, executor as _executor
    import asr_mcp.core.model_state as ms
    from concurrent.futures import ThreadPoolExecutor
    from asr_mcp.speaker.service import SpeakerService
    from asr_mcp.sessions.manager import SessionManager
    from asr_mcp.voiceprint.service import VoiceprintService
    from asr_mcp.core import job_state

    settings = get_settings()
    setup_logging(settings.log_dir, log_level=settings.log_level)

    logger.info("Starting asr-mcp server...")
    logger.info("CUDA device: %s", settings.cuda_device)
    logger.info("DB path: %s", settings.db_path)
    logger.info("Log level: %s", settings.log_level)
    logger.debug("Debug logging is active")

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

    await app.state.speaker_service.initialize()
    await app.state.session_manager.initialize()
    await app.state.voiceprint_service.initialize()

    # Report voiceprint-store drift (duplicate profiles, orphans, stale totals)
    # before any job can be attributed against it. The audit itself is read-only
    # (see voiceprint/audit.py for why most findings are the user's to decide),
    # but three of them -- snippets with no profile, profiles with no snippets,
    # snippets whose audio vanished -- are mechanical, and the only repair is to
    # rebuild the DERIVED profile row from the snippets. That is queued to disk
    # and drained by a background task below, because the embedding session is
    # deliberately not loaded at start.
    from asr_mcp.voiceprint.audit import run_startup_audit
    from asr_mcp.voiceprint import rebuild as vp_rebuild
    reports = run_startup_audit(app.state.voiceprint_service, logger)
    rebuild_queue = vp_rebuild.queue_path(settings.data_dir)
    try:
        queued = vp_rebuild.enqueue_from_reports(rebuild_queue, reports, logger)
    except Exception as e:
        logger.warning("Could not queue voiceprint rebuilds: %s", e)
        queued = 0
    if queued:
        logger.warning("Queued %d voiceprint profile rebuild(s) from the audit; "
                       "they will run in the background once the models load and "
                       "no job is active", queued)

    # Background TTL monitor — unload GPU models after N minutes idle (skip if job active)
    ttl_task = None
    if settings.model_ttl_minutes > 0 or queued:
        async def _ttl_monitor():
            interval = 60
            while True:
                await asyncio.sleep(interval)
                if state.any_loaded and settings.model_ttl_minutes > 0 \
                        and state.idle_seconds() > settings.model_ttl_minutes * 60:
                    if job_state.get_running() is None:
                        state.unload_models()
                    else:
                        logger.debug("TTL skip: job active")
                if queued or vp_rebuild.pending_entries(rebuild_queue):
                    await _drain_rebuild_queue()

        async def _drain_rebuild_queue():
            """Repair the profiles the audit flagged, off the request path.

            Only VAD and the ECAPA embedding model are needed
            (``ensure_diarize_ready``), never the ASR backend, and only when
            nothing else is running -- the models the rebuild loads are then
            unloaded again by the TTL branch above.
            """
            if job_state.get_running() is not None:
                return
            if not vp_rebuild.pending_entries(rebuild_queue):
                return
            loop = asyncio.get_running_loop()
            try:
                # ms.executor, not the name imported at the top of the lifespan:
                # the pool is (re)created below, after that import ran.
                await loop.run_in_executor(
                    ms.executor, state.ensure_diarize_ready)
                result = await loop.run_in_executor(
                    ms.executor, vp_rebuild.run_pending,
                    app.state.voiceprint_service, rebuild_queue, logger)
            except Exception as e:
                logger.warning("Voiceprint rebuild pass failed: %s", e)
                return
            if result.get("attempted") or result.get("failed"):
                logger.warning("Voiceprint rebuild pass: %s", result)

        ttl_task = asyncio.create_task(_ttl_monitor())
        logger.info("Model TTL monitor started (%d min)%s",
                    settings.model_ttl_minutes,
                    "; will also drain the voiceprint rebuild queue" if queued else "")

    logger.info("asr-mcp server ready on %s:%d", settings.host, settings.port)

    yield

    # Cancel TTL task
    if ttl_task:
        ttl_task.cancel()
        try:
            await ttl_task
        except asyncio.CancelledError:
            pass

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
    version=__version__,
    lifespan=lifespan,
)

# Auth middleware — added FIRST so it's innermost (runs AFTER SessionMiddleware)
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        from asr_mcp.api.auth import require_auth
        redirect = require_auth(request)
        if redirect:
            return redirect
        return await call_next(request)

app.add_middleware(AuthMiddleware)

# Session middleware
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

from asr_mcp.api.security import require_admin  # noqa: E402  (needs the router mounted)


# Static files & templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _render(template_name: str, active: str = "") -> HTMLResponse:
    path = templates_dir / template_name
    if path.exists():
        content = path.read_text(encoding="utf-8")
        nav_path = templates_dir / "_nav.html"
        if "__NAV__" in content and nav_path.exists():
            nav = nav_path.read_text(encoding="utf-8")
            for key in ("GUI", "LIVE", "VOICES", "TRANSCRIPTS", "SETTINGS"):
                nav = nav.replace(f"__ACT_{key}__",
                                  ' class="active"' if key.lower() == active else "")
            content = content.replace("__NAV__", nav)
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
    import shutil
    from asr_mcp.core.model_state import state
    from asr_mcp.core import job_state
    from asr_mcp.config.settings import get_settings
    settings = get_settings()

    def _voiceprint_total() -> int:
        """Profiles across every owner, not the ``default`` user's rows.

        A raw ``count()`` is user-scoped and defaults to ``default``, so it read
        0 on a store that actually held profiles under other owners -- a health
        endpoint reporting the store as empty is worse than no count at all.
        """
        try:
            db = getattr(app.state, "speaker_service", None)
            if db is None:
                return 0
            vp_db = db._db
            users = vp_db.user_ids()
            if not users:
                return 0
            return sum(vp_db.count(user_id=u) for u in users)
        except Exception:
            return 0

    ram = None
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = v.strip()
        total_kb = int(info["MemTotal"].split()[0])
        avail_kb = int(info["MemAvailable"].split()[0])
        total = total_kb * 1024
        used = (total_kb - avail_kb) * 1024
        ram = {"total": total, "used": used, "percent": round(used / total * 100, 1)}
    except Exception:
        ram = None

    def _gpu_query():
        import subprocess
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) < 5:
                return None
            name, util, mem_used, mem_total, temp = parts[:5]
            mem_used_i, mem_total_i = int(mem_used), int(mem_total)
            # nvidia-smi reports memory in MiB; expose bytes like ram/disk
            return {
                "name": name,
                "util": int(util),
                "mem_used": mem_used_i * 1048576,
                "mem_total": mem_total_i * 1048576,
                "mem_percent": round(mem_used_i / max(mem_total_i, 1) * 100, 1),
                "temperature": int(temp),
            }
        except Exception:
            return None

    gpu = await asyncio.to_thread(_gpu_query)

    disk = None
    try:
        d = shutil.disk_usage(settings.data_dir)
        disk = {"total": d.total, "used": d.used, "percent": round(d.used / d.total * 100, 1)}
    except Exception:
        disk = None

    asr_loaded = bool(state.backend is not None and state.backend.is_loaded)
    job = job_state.get_running()
    active_job = None
    if job is not None:
        m = job.meta()
        active_job = {
            "mode": m["mode"],
            "filename": m["filename"],
            "stage": m["stage"],
            "progress": m["progress"],
        }

    return {
        "status": "healthy" if state.is_ready else "loading",
        "model_status": "ready" if state.is_ready else "not_ready",
        "cuda_device": settings.cuda_device,
        # Counted across EVERY user: a raw count() is the `default` user's
        # rows, which read as 0 while the real store held profiles under other
        # owners -- a health endpoint that reports a store as empty is worse
        # than no count at all.
        "voiceprint_count": _voiceprint_total(),
        "version": __version__,
        "ram": ram,
        "gpu": gpu,
        "disk": disk,
        "models": {
            "asr": {
                "name": settings.asr_model,
                "backend": state.backend.name if state.backend else None,
                "loaded": asr_loaded,
            },
            "vad": state.vad_session is not None,
            "embedding": state.embedding_session is not None,
            "idle_seconds": round(state.idle_seconds(), 1) if state.any_loaded else None,
            "ttl_minutes": settings.model_ttl_minutes,
        },
        "active_job": active_job,
    }


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _render("login.html")


@app.get("/gui", response_class=HTMLResponse)
async def gui(request: Request):
    return _render("app.html", active="gui")


@app.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    return _render("app.html", active="live")


@app.get("/voices", response_class=HTMLResponse)
async def voices(request: Request):
    return _render("app.html", active="voices")


@app.get("/transcriptions", response_class=HTMLResponse)
async def transcriptions_page(request: Request):
    return _render("app.html", active="transcripts")


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return _render("app.html", active="settings")


@app.post("/shutdown")
async def shutdown(_: str = Depends(require_admin)):
    """Stop the process. Administrator only.

    This had no role check at all -- only the middleware, which any session or
    any valid API key satisfies. Stopping the server is not a per-user
    operation: it cancels other users' in-flight jobs, drops their GPU models
    mid-warm, and takes the queue away from whoever is next. That is one
    configured name, not "whoever logged in".
    """
    import signal
    logger.info("Shutdown requested via API by %r", _)
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
