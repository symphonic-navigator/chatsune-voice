"""Tests for the application bootstrap in voice.main."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_apply_hf_home_sets_env_before_imports(tmp_path, monkeypatch):
    from voice.main import apply_hf_home

    monkeypatch.delenv("HF_HOME", raising=False)
    apply_hf_home(tmp_path)
    assert Path(os.environ["HF_HOME"]) == tmp_path


@pytest.mark.asyncio
async def test_build_registry_from_settings_uses_provided_loader():
    from voice.config import Settings
    from voice.main import build_registry

    called: list[str] = []

    class _FakeModel:
        mode = "custom_voice"
        sample_rate = 22050
        async def aclose(self): ...

    def loader(mode: str):
        called.append(mode)
        return _FakeModel()

    s = Settings(
        _env_file=None,
        tts_enabled_modes="custom_voice",
        tts_vram_policy="keep_loaded",
        preload_at_startup=False,
    )
    registry = build_registry(s, tts_loader=loader)
    assert registry.enabled_modes == ("custom_voice",)


@pytest.mark.asyncio
async def test_preload_success(tmp_path):
    from voice.config import Settings
    from voice.main import build_registry

    class _FakeModel:
        mode = "custom_voice"
        sample_rate = 22050
        async def aclose(self): ...

    def loader(mode):
        return _FakeModel()

    s = Settings(
        _env_file=None,
        tts_enabled_modes="custom_voice",
        tts_vram_policy="keep_loaded",
        preload_at_startup=True,
    )
    registry = build_registry(s, tts_loader=loader)
    registry.preload()
    assert "custom_voice" in registry.loaded_modes()


@pytest.mark.asyncio
async def test_preload_failure_raises_model_load_error():
    from voice.config import Settings
    from voice.engines.protocol import ModelLoadError
    from voice.main import build_registry

    def loader(mode):
        raise RuntimeError("boom")

    s = Settings(
        _env_file=None,
        tts_enabled_modes="custom_voice",
        tts_vram_policy="keep_loaded",
        preload_at_startup=True,
    )
    registry = build_registry(s, tts_loader=loader)
    with pytest.raises(ModelLoadError):
        registry.preload()


@pytest.mark.asyncio
async def test_bootstrap_with_all_three_modes_builds_registry():
    """Registry is built with clone mode and Chatterbox loader is registered."""
    from voice.config import Settings
    from voice.main import build_registry

    settings = Settings(_env_file=None, tts_enabled_modes="custom_voice,voice_design,clone")
    assert "clone" in settings.tts_enabled_modes

    calls: list[str] = []

    def fake_loader(mode: str):
        calls.append(mode)
        from tests.conftest import FakeTTSModel
        m = FakeTTSModel(mode=mode)
        m.always_resident = (mode == "clone")
        return m

    registry = build_registry(settings, tts_loader=fake_loader)
    assert registry.enabled_modes == ("custom_voice", "voice_design", "clone")
