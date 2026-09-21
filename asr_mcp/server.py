import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

# Static files & templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = templates_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>ASR MCP Server v2.0</h1><p>GPU-powered ASR with diarization and voiceprints.</p>")


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


@app.get("/gui", response_class=HTMLResponse)
async def gui():
    index_file = templates_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>GUI not available</h1>")


@app.get("/voices", response_class=HTMLResponse)
async def voices():
    voices_file = templates_dir / "voices.html"
    if voices_file.exists():
        return HTMLResponse(content=voices_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Voice manager not available</h1>")


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
