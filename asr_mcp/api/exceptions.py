import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("asr_mcp.api.exceptions")


class TranscriptionError(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TimeoutError(TranscriptionError):
    pass


class AudioValidationError(TranscriptionError):
    pass


class PathSecurityError(TranscriptionError):
    pass


async def transcription_error_handler(request: Request, exc: TranscriptionError):
    return JSONResponse(
        status_code=422,
        content={"error": exc.message, "details": exc.details},
    )


async def general_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)},
    )


def register_exception_handlers(app):
    app.add_exception_handler(TranscriptionError, transcription_error_handler)
    app.add_exception_handler(Exception, general_error_handler)
