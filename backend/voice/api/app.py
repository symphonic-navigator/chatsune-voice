"""FastAPI app factory. Wires the transport layer to the engines."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from voice.api import health as health_module


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    return app
