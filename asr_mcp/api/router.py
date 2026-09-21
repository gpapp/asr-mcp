from fastapi import APIRouter

from asr_mcp.api.asr_router import router as asr_router
from asr_mcp.api.speaker_router import router as speaker_router
from asr_mcp.api.mcp_router import router as mcp_router

api_router = APIRouter(prefix="/api")
api_router.include_router(asr_router)
api_router.include_router(speaker_router)
api_router.include_router(mcp_router)
