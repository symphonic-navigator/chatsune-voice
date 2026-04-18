"""FastAPI app factory. Wires the transport layer to the engines."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from voice.api import health as health_module
from voice.api import stt as stt_module


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": str(exc.errors())},
        )

    return app
