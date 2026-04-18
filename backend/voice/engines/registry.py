"""TTSModelRegistry — owns enabled modes, VRAM policy, and locks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from voice.engines.protocol import (
    ModeDisabledError,
    ModelLoadError,
    TTSMode,
    TTSModel,
)
from voice.logging_setup import get_logger

VRAMPolicy = Literal["keep_loaded", "swap"]

log = get_logger(__name__)


class TTSModelRegistry:
    def __init__(
        self,
        *,
        enabled: tuple[TTSMode, ...],
        policy: VRAMPolicy,
        loader: Callable[[TTSMode], TTSModel],
        on_evict: Callable[[TTSMode], None] | None = None,
    ) -> None:
        self._enabled: tuple[TTSMode, ...] = enabled
        self._policy: VRAMPolicy = policy
        self._loader = loader
        self._on_evict = on_evict
        self._locks: dict[TTSMode, asyncio.Lock] = {m: asyncio.Lock() for m in enabled}
        self._swap_lock = asyncio.Lock()
        self._loaded: dict[TTSMode, TTSModel] = {}

    @property
    def enabled_modes(self) -> tuple[TTSMode, ...]:
        return self._enabled

    @property
    def policy(self) -> VRAMPolicy:
        return self._policy

    def loaded_modes(self) -> tuple[TTSMode, ...]:
        return tuple(self._loaded.keys())

    def preload(self) -> None:
        """Load enabled models at start-up. No-op under 'swap' policy.

        Synchronous by design: preload runs during application bootstrap, and
        that bootstrap path may already be inside a running asyncio loop (for
        instance, uvicorn imports `voice.main:app` via __getattr__ while its
        own serve() loop is active — an `asyncio.run` there would blow up
        with "cannot be called from a running event loop"). At startup there
        is nothing to block anyway, so a plain synchronous loader call is
        both safer and simpler.
        """
        if self._policy == "swap":
            log.info("tts_registry_preload_skipped", reason="swap_policy")
            return
        for mode in self._enabled:
            self._load_sync(mode)

    def _load_sync(self, mode: TTSMode) -> None:
        try:
            model = self._loader(mode)
        except Exception as exc:
            raise ModelLoadError(mode, exc) from exc
        self._loaded[mode] = model
        log.info("tts_model_loaded", mode=mode)

    @asynccontextmanager
    async def acquire(self, mode: str) -> AsyncIterator[TTSModel]:
        if mode not in self._enabled:
            raise ModeDisabledError(mode)
        if self._policy == "keep_loaded":
            async with self._locks[mode]:  # type: ignore[index]
                if mode not in self._loaded:
                    await self._load_locked(mode)  # type: ignore[arg-type]
                yield self._loaded[mode]  # type: ignore[index]
        else:
            async with self._swap_lock:
                if mode not in self._loaded:
                    await self._evict_all()
                    await self._load_locked(mode)  # type: ignore[arg-type]
                yield self._loaded[mode]  # type: ignore[index]

    async def aclose(self) -> None:
        for mode, model in list(self._loaded.items()):
            try:
                await model.aclose()
            except Exception as exc:
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
        self._loaded.clear()

    async def _load_locked(self, mode: TTSMode) -> None:
        try:
            model = await asyncio.to_thread(self._loader, mode)
        except Exception as exc:
            raise ModelLoadError(mode, exc) from exc
        self._loaded[mode] = model
        log.info("tts_model_loaded", mode=mode)

    async def _evict_all(self) -> None:
        for mode, model in list(self._loaded.items()):
            try:
                await model.aclose()
            except Exception as exc:
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
            if self._on_evict is not None:
                self._on_evict(mode)
            log.info("tts_model_evicted", mode=mode)
        self._loaded.clear()
