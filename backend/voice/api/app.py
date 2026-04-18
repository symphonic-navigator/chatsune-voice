"""FastAPI app factory. Wires the transport layer to the engines."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from voice.api import health as health_module
from voice.api import stt as stt_module
from voice.api import tts as tts_module


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)
    app.include_router(tts_module.router)

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

    return app
