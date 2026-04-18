"""Entry point — wires components, preloads models, serves FastAPI via uvicorn.

A module-level `__getattr__` lazily builds the FastAPI `app` attribute when first
accessed (e.g. by `uvicorn voice.main:app`), so merely importing this module does
not trigger the full bootstrap.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

from voice.config import Settings
from voice.logging_setup import configure_logging, get_logger


def apply_hf_home(model_cache_dir: Path) -> None:
    """Copy our app-scoped cache path into HF_HOME before HF libraries are imported."""
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(model_cache_dir)


def build_registry(settings: Settings, *, tts_loader: Callable[[str], object]):
    from voice.engines.registry import TTSModelRegistry

    return TTSModelRegistry(
        enabled=settings.tts_enabled_modes,
        policy=settings.tts_vram_policy,
        loader=tts_loader,  # type: ignore[arg-type]
    )


def _default_tts_loader(settings: Settings) -> Callable[[str], object]:
    from voice.engines.qwen_tts import (
        QwenCustomVoiceModel,
        QwenVoiceDesignModel,
        load_qwen_tts,
    )

    def load(mode: str) -> object:
        if mode == "custom_voice":
            backend = load_qwen_tts(
                settings.tts_custom_voice_model,
                device=settings.tts_device,
                attention_impl=settings.tts_attention_impl,
            )
            return QwenCustomVoiceModel(backend=backend)
        if mode == "voice_design":
            backend = load_qwen_tts(
                settings.tts_voice_design_model,
                device=settings.tts_device,
                attention_impl=settings.tts_attention_impl,
            )
            return QwenVoiceDesignModel(backend=backend)
        raise ValueError(f"unknown TTS mode: {mode!r}")

    return load


def _default_stt(settings: Settings):
    from voice.engines.whisper import WhisperEngine, load_faster_whisper

    t0 = time.monotonic()
    backend = load_faster_whisper(
        settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        download_root=None,
    )
    engine = WhisperEngine(model_name=settings.stt_model, model=backend)
    engine.loaded = True  # type: ignore[attr-defined]
    get_logger(__name__).info(
        "stt_model_loaded",
        model=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        load_ms=int((time.monotonic() - t0) * 1000),
    )
    return engine


def _bootstrap(settings: Settings):
    """Synchronous bootstrap — safe to call from inside a running asyncio loop
    (e.g. uvicorn's importer when it resolves `voice.main:app`)."""
    from voice.api.app import build_app

    stt = _default_stt(settings) if settings.preload_at_startup else _LazySTT(settings)

    registry = build_registry(settings, tts_loader=_default_tts_loader(settings))
    if settings.preload_at_startup:
        registry.preload()

    app = build_app(stt=stt, registry=registry, settings=settings)
    return app, registry


class _LazySTT:
    """Stub that loads Whisper on first transcribe call."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._real = None
        self._lock = asyncio.Lock()
        self.model_name = settings.stt_model
        self.loaded = False

    async def transcribe(self, audio, **kwargs):
        async with self._lock:
            if self._real is None:
                self._real = _default_stt(self._settings)
                self.loaded = True
        return await self._real.transcribe(audio, **kwargs)

    async def aclose(self) -> None:
        if self._real is not None:
            await self._real.aclose()


def _lazy_app():
    settings = Settings()
    configure_logging(settings.log_level)
    apply_hf_home(settings.model_cache_dir)
    app, _registry = _bootstrap(settings)
    return app


def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info(
        "app_starting",
        stt_device=settings.stt_device,
        tts_device=settings.tts_device,
        tts_vram_policy=settings.tts_vram_policy,
        tts_enabled_modes=list(settings.tts_enabled_modes),
        preload_at_startup=settings.preload_at_startup,
    )

    apply_hf_home(settings.model_cache_dir)

    try:
        app, registry = _bootstrap(settings)
    except Exception as exc:
        log.error("startup_failed", error_type=type(exc).__name__, message=str(exc))
        sys.exit(2)

    import uvicorn

    config = uvicorn.Config(app=app, host="0.0.0.0", port=settings.app_port, log_config=None)
    server = uvicorn.Server(config)

    def _shutdown(_signum, _frame):
        server.should_exit = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.run()
    finally:
        asyncio.run(registry.aclose())
        log.info("shutdown", exit_code=0)


def __getattr__(name: str):
    if name == "app":
        _app = _lazy_app()
        globals()["app"] = _app
        return _app
    raise AttributeError(name)


if __name__ == "__main__":
    run()
