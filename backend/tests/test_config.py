"""Tests for voice.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_defaults(monkeypatch):
    monkeypatch.delenv("CHATSUNE_VOICE_MODEL_CACHE_DIR", raising=False)
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert s.stt_model == "Systran/faster-whisper-large-v3-turbo"
    assert s.tts_custom_voice_model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert s.tts_voice_design_model == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert s.tts_enabled_modes == ("custom_voice", "voice_design")
    assert s.tts_vram_policy == "keep_loaded"
    assert s.tts_attention_impl == "sdpa"
    assert s.preload_at_startup is True
    assert s.device == "cuda"
    assert s.log_level == "info"
    assert s.app_port == 8000
    assert s.stt_max_audio_bytes == 25 * 1024 * 1024


def test_enabled_modes_parsing():
    from voice.config import Settings

    s = Settings(_env_file=None, tts_enabled_modes="custom_voice")
    assert s.tts_enabled_modes == ("custom_voice",)

    s = Settings(_env_file=None, tts_enabled_modes="voice_design,custom_voice")
    assert set(s.tts_enabled_modes) == {"custom_voice", "voice_design"}


def test_enabled_modes_cannot_be_empty():
    from voice.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_enabled_modes="")


def test_enabled_modes_rejects_unknown():
    from voice.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_enabled_modes="custom_voice,bogus")


def test_vram_policy_enum():
    from voice.config import Settings

    assert Settings(_env_file=None, tts_vram_policy="swap").tts_vram_policy == "swap"

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_vram_policy="lru")


def test_attention_impl_enum():
    from voice.config import Settings

    for val in ("sdpa", "flash_attention_2", "eager"):
        assert Settings(_env_file=None, tts_attention_impl=val).tts_attention_impl == val

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_attention_impl="xformers")


def test_log_level_enum():
    from voice.config import Settings

    for val in ("debug", "info", "warn", "error"):
        assert Settings(_env_file=None, log_level=val).log_level == val

    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="trace")


def test_model_cache_dir_env_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATSUNE_VOICE_MODEL_CACHE_DIR", str(tmp_path))
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert str(s.model_cache_dir) == str(tmp_path)


def test_enabled_modes_env_roundtrip_comma_separated(monkeypatch):
    """The env-var path must reach the comma-split validator untouched.

    pydantic-settings JSON-decodes complex-typed env values by default, which
    would choke on `custom_voice,voice_design` (invalid JSON). The NoDecode
    annotation suppresses that, letting our validator split commas instead.
    """
    monkeypatch.setenv("TTS_ENABLED_MODES", "custom_voice,voice_design")
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert set(s.tts_enabled_modes) == {"custom_voice", "voice_design"}


def test_enabled_modes_env_single_value(monkeypatch):
    monkeypatch.setenv("TTS_ENABLED_MODES", "custom_voice")
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert s.tts_enabled_modes == ("custom_voice",)


def test_enabled_modes_env_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("TTS_ENABLED_MODES", "custom_voice,bogus")
    from voice.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
