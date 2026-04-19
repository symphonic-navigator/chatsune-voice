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
        always_resident_modes: frozenset[str] = frozenset(),
    ) -> None:
        self._enabled: tuple[TTSMode, ...] = enabled
        self._policy: VRAMPolicy = policy
        self._loader = loader
        self._on_evict = on_evict
        self._always_resident: frozenset[str] = always_resident_modes
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
        """Load enabled models at start-up.

        Under `keep_loaded` this loads every enabled mode.
        Under `swap` this loads only the always-resident modes; swappable
        modes are lazy-loaded on first request. Synchronous by design —
        safe to call from inside a running asyncio loop.
        """
        for mode in self._enabled:
            if self._policy == "keep_loaded" or mode in self._always_resident:
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
        # Always-resident modes and every mode under keep_loaded take the
        # per-mode lock and bypass the swap lock entirely.
        if self._policy == "keep_loaded" or mode in self._always_resident:
            async with self._locks[mode]:  # type: ignore[index]
                if mode not in self._loaded:
                    await self._load_locked(mode)  # type: ignore[arg-type]
                yield self._loaded[mode]  # type: ignore[index]
            return
        # Swappable mode under swap policy.
        async with self._swap_lock:
            if mode not in self._loaded:
                await self._evict_swappable()
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

    async def _evict_swappable(self) -> None:
        for mode, model in list(self._loaded.items()):
            if mode in self._always_resident:
                continue
            try:
                await model.aclose()
            except Exception as exc:
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
            del self._loaded[mode]
            if self._on_evict is not None:
                self._on_evict(mode)
            log.info("tts_model_evicted", mode=mode)
