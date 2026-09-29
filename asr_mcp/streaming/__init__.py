"""Live streaming transcription.

``handle_ws_stream`` is imported lazily so that importing the package (or the
dependency-light :mod:`asr_mcp.streaming.turn_detector` module in tests) does
not pull in FastAPI/WebSocket machinery.
"""

__all__ = ["handle_ws_stream"]


def handle_ws_stream(websocket):
    from asr_mcp.streaming.handler import handle_ws_stream as _impl
    return _impl(websocket)
