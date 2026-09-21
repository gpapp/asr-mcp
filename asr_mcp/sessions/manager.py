import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from asr_mcp.db.manager import DatabaseManager, SessionDB

logger = logging.getLogger("asr_mcp.sessions.manager")


class SessionManager:
    def __init__(self, db_manager: DatabaseManager, ttl_seconds: int = 3600):
        self._db = SessionDB(db_manager, ttl_seconds=ttl_seconds)

    async def initialize(self):
        expired = self._db.cleanup_expired()
        logger.info("SessionManager initialized — cleaned %d expired sessions", expired)

    def create_session(self, data: Optional[dict] = None) -> str:
        session_id = str(uuid.uuid4())
        self._db.create(session_id, data or {})
        logger.debug("Created session: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._db.get(session_id)

    def set_session_data(self, session_id: str, data: dict):
        self._db.set_data(session_id, data)

    def get_session_data(self, session_id: str) -> Optional[dict]:
        return self._db.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        deleted = self._db.delete(session_id)
        if deleted:
            logger.debug("Deleted session: %s", session_id)
        return deleted

    def cleanup_expired_sessions(self) -> int:
        count = self._db.cleanup_expired()
        if count:
            logger.info("Cleaned %d expired sessions", count)
        return count

    def list_sessions(self) -> list[dict]:
        return self._db.list_all()
