"""Tests for voice.engines.registry — VRAM policy, locking, acquire semantics."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
import pytest

from tests.conftest import FakeTTSModel


def _loader_factory(counts: dict[str, int]) -> Callable[[str], FakeTTSModel]:
    def load(mode: str) -> FakeTTSModel:
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(22050, dtype=np.float32)
        return FakeTTSModel(mode=mode, samples=samples)
    return load


@pytest.mark.asyncio
async def test_keep_loaded_preload_loads_all_enabled():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design"),
        policy="keep_loaded",
        loader=_loader_factory(counts),
    )
    registry.preload()
    assert counts == {"custom_voice": 1, "voice_design": 1}


@pytest.mark.asyncio
async def test_keep_loaded_per_mode_lock_serialises_same_mode():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    order: list[str] = []
    registry = TTSModelRegistry(
        enabled=("custom_voice",),
        policy="keep_loaded",
        loader=_loader_factory(counts),
    )
    registry.preload()

    async def use(tag: str) -> None:
        async with registry.acquire("custom_voice") as _model:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.05)
            order.append(f"{tag}-out")

    await asyncio.gather(use("a"), use("b"))
    # One must fully finish before the other starts.
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


@pytest.mark.asyncio
async def test_keep_loaded_parallel_across_modes():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    order: list[str] = []
    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design"),
        policy="keep_loaded",
        loader=_loader_factory(counts),
    )
    registry.preload()

    async def use(mode: str, tag: str) -> None:
        async with registry.acquire(mode) as _model:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.05)
            order.append(f"{tag}-out")

    await asyncio.gather(
        use("custom_voice", "a"),
        use("voice_design", "b"),
    )
    # Both should be inside their critical sections before either exits.
    assert order.index("a-in") < order.index("b-out")
    assert order.index("b-in") < order.index("a-out")


@pytest.mark.asyncio
async def test_swap_evicts_and_reloads_on_mode_switch():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    evictions: list[str] = []
    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design"),
        policy="swap",
        loader=_loader_factory(counts),
        on_evict=lambda mode: evictions.append(mode),
    )
    registry.preload()   # no-op under swap
    assert counts == {}

    async with registry.acquire("custom_voice"):
        pass
    async with registry.acquire("voice_design"):
        pass
    async with registry.acquire("custom_voice"):
        pass

    assert counts["custom_voice"] == 2
    assert counts["voice_design"] == 1
    assert evictions == ["custom_voice", "voice_design"]


@pytest.mark.asyncio
async def test_swap_serialises_concurrent_requests():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    order: list[str] = []
    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design"),
        policy="swap",
        loader=_loader_factory(counts),
    )

    async def use(mode: str, tag: str) -> None:
        async with registry.acquire(mode):
            order.append(f"{tag}-in")
            await asyncio.sleep(0.05)
            order.append(f"{tag}-out")

    await asyncio.gather(
        use("custom_voice", "a"),
        use("voice_design", "b"),
    )
    # Swap serialises: one fully completes before the other starts.
    a_in, a_out = order.index("a-in"), order.index("a-out")
    b_in, b_out = order.index("b-in"), order.index("b-out")
    assert a_out < b_in or b_out < a_in


@pytest.mark.asyncio
async def test_mode_disabled_raises():
    from voice.engines.protocol import ModeDisabledError
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    registry = TTSModelRegistry(
        enabled=("custom_voice",),
        policy="keep_loaded",
        loader=_loader_factory(counts),
    )

    with pytest.raises(ModeDisabledError):
        async with registry.acquire("voice_design"):
            pass


@pytest.mark.asyncio
async def test_lazy_load_on_acquire_when_not_preloaded():
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    registry = TTSModelRegistry(
        enabled=("custom_voice",),
        policy="keep_loaded",
        loader=_loader_factory(counts),
    )
    # No preload.
    async with registry.acquire("custom_voice"):
        pass
    assert counts == {"custom_voice": 1}
    async with registry.acquire("custom_voice"):
        pass
    # Should not reload.
    assert counts == {"custom_voice": 1}


@pytest.mark.asyncio
async def test_loader_failure_propagates_as_model_load_error():
    from voice.engines.protocol import ModelLoadError
    from voice.engines.registry import TTSModelRegistry

    def failing_loader(mode: str):
        raise RuntimeError("boom")

    registry = TTSModelRegistry(
        enabled=("custom_voice",),
        policy="keep_loaded",
        loader=failing_loader,
    )

    with pytest.raises(ModelLoadError):
        async with registry.acquire("custom_voice"):
            pass


@pytest.mark.asyncio
async def test_swap_preloads_always_resident_only():
    """Under swap policy, preload loads always-resident modes immediately."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="swap",
        loader=load,
        always_resident_modes=frozenset({"clone"}),
    )
    registry.preload()
    assert counts == {"clone": 1}
    assert "clone" in registry.loaded_modes()


@pytest.mark.asyncio
async def test_swap_does_not_evict_always_resident():
    """A Qwen3 swap-switch leaves Chatterbox loaded."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    evictions: list[str] = []

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="swap",
        loader=load,
        on_evict=lambda mode: evictions.append(mode),
        always_resident_modes=frozenset({"clone"}),
    )
    registry.preload()  # loads clone
    async with registry.acquire("custom_voice"):
        pass
    async with registry.acquire("voice_design"):
        pass
    async with registry.acquire("clone"):
        pass

    assert counts == {"clone": 1, "custom_voice": 1, "voice_design": 1}
    assert "clone" not in evictions
    assert evictions == ["custom_voice"]


@pytest.mark.asyncio
async def test_keep_loaded_unchanged_with_always_resident():
    """keep_loaded still loads everything and evicts nothing."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    evictions: list[str] = []

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="keep_loaded",
        loader=load,
        on_evict=lambda mode: evictions.append(mode),
        always_resident_modes=frozenset({"clone"}),
    )
    registry.preload()

    async with registry.acquire("custom_voice"):
        pass
    async with registry.acquire("voice_design"):
        pass
    async with registry.acquire("clone"):
        pass

    assert counts == {"custom_voice": 1, "voice_design": 1, "clone": 1}
    assert evictions == []
