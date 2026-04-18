"""Application settings — pydantic-settings with fail-fast validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VRAMPolicy = Literal["keep_loaded", "swap"]
AttentionImpl = Literal["sdpa", "flash_attention_2", "eager"]
LogLevel = Literal["debug", "info", "warn", "error"]
TTSMode = Literal["custom_voice", "voice_design"]


class Settings(BaseSettings):
    """Validated configuration loaded from process environment and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_cache_dir: Path = Field(
        default=Path("/models"),
        alias="CHATSUNE_VOICE_MODEL_CACHE_DIR",
    )
    stt_model: str = "Systran/faster-whisper-large-v3-turbo"
    stt_max_audio_bytes: int = 25 * 1024 * 1024
    tts_custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_voice_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    # NoDecode suppresses pydantic-settings' default JSON-decoding of complex
    # types when the value comes from the environment. We want the raw string
    # (e.g. "custom_voice,voice_design") to reach _parse_enabled_modes intact
    # so it can split on commas rather than choke on invalid JSON.
    tts_enabled_modes: Annotated[
        tuple[TTSMode, ...], NoDecode
    ] = ("custom_voice", "voice_design")
    tts_vram_policy: VRAMPolicy = "keep_loaded"
    tts_attention_impl: AttentionImpl = "sdpa"
    preload_at_startup: bool = True
    device: str = "cuda"
    log_level: LogLevel = "info"
    app_port: int = 8000

    @field_validator("tts_enabled_modes", mode="before")
    @classmethod
    def _parse_enabled_modes(cls, value: Any) -> tuple[str, ...]:
        if value is None or value == "":
            raise ValueError("tts_enabled_modes must not be empty")
        if isinstance(value, str):
            parts = tuple(p.strip() for p in value.split(",") if p.strip())
            if not parts:
                raise ValueError("tts_enabled_modes must not be empty")
            return parts
        return tuple(value)

    @field_validator("tts_enabled_modes")
    @classmethod
    def _validate_mode_values(cls, value: tuple[str, ...]) -> tuple[TTSMode, ...]:
        allowed = {"custom_voice", "voice_design"}
        unknown = [m for m in value if m not in allowed]
        if unknown:
            raise ValueError(f"unknown TTS mode(s): {unknown!r}; allowed: {sorted(allowed)}")
        return value  # type: ignore[return-value]
