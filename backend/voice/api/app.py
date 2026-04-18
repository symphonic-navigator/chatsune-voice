"""FastAPI app factory. Wires the transport layer to the engines.

Also installs:
- a request-id middleware that binds a UUID4 to structlog contextvars for every request;
- a catch-all exception handler that logs unhandled_error and returns HTTP 500
  without a traceback in the response body.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from voice.api import health as health_module
from voice.api import stt as stt_module
from voice.api import tts as tts_module
from voice.logging_setup import get_logger

log = get_logger(__name__)


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)
    app.include_router(tts_module.router)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def _http_error(_request, exc: HTTPException):
        # When detail is already a dict (e.g. {"error": "...", "mode": "..."}),
        # promote it to the top level so clients can read `response.json()["error"]`
        # directly rather than `response.json()["detail"]["error"]`.
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = {"error": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception):
        log.error(
            "unhandled_error",
            error_type=type(exc).__name__,
            message=str(exc),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error"},
        )

    # Static mount must come LAST so API routes take precedence.
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
