import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("asr_mcp.api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        logger.info(
            "%s %s %d %.3fs",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response


def apply_middleware(app):
    app.add_middleware(RequestLoggingMiddleware)
