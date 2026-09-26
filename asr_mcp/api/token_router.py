import logging

from fastapi import APIRouter, Depends

from asr_mcp.api.security import get_current_user
from asr_mcp.config.settings import Settings, get_settings

logger = logging.getLogger("asr_mcp.api.token_router")
router = APIRouter(prefix="/token", tags=["Token"])


def _get_token_db(settings: Settings):
    from asr_mcp.db.manager import DatabaseManager, TokenDB
    db = DatabaseManager(settings.db_path)
    return TokenDB(db)


@router.get("")
async def token_status(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    info = _get_token_db(settings).get_info(user_id)
    if not info:
        return {"has_token": False, "user_id": user_id}
    return {"has_token": True, **info}


@router.post("")
async def generate_token(
    user_id: str = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    token = _get_token_db(settings).create(user_id)
    logger.info("API token issued for user %s", user_id)
    return {"token": token, "user_id": user_id}
