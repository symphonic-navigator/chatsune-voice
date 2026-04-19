"""Application settings — pydantic-settings with fail-fast validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VRAMPolicy = Literal["keep_loaded", "swap"]
AttentionImpl = Literal["sdpa", "flash_attention_2", "eager"]
LogLevel = Literal["debug", "info", "warn", "error"]
TTSMode = Literal["custom_voice", "voice_design", "clone"]
ChatterboxBackend = Literal["onnx", "torch"]
STTComputeType = Literal[
    "auto",
    "int8",
    "int8_float16",
    "int8_bfloat16",
    "int8_float32",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]


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
    stt_model: str = "h2oai/faster-whisper-large-v3-turbo"
    # CTranslate2 compute_type passed to WhisperModel at load time.
    # "auto" lets CT2 pick the best supported type for the detected device:
    # int8_float16 on CUDA with Tensor cores, int8 (or int8_float32 without
    # AVX-VNNI) on CPU. CPUs do not support int8_float16 at all — float16
    # computation is a GPU-only capability — so a hard-coded int8_float16
    # default fails on ROCm hosts that fall back to CPU for STT. Override
    # explicitly if you need to A/B a specific precision, e.g.
    # STT_COMPUTE_TYPE=float16 on CUDA or =int8 on CPU.
    stt_compute_type: STTComputeType = "auto"
    stt_max_audio_bytes: int = 25 * 1024 * 1024
    tts_custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_voice_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    chatterbox_model: str = "onnx-community/chatterbox-multilingual-ONNX"
    chatterbox_backend: ChatterboxBackend = "onnx"
    chatterbox_device: str = "cuda"
    chatterbox_max_reference_bytes: int = 10 * 1024 * 1024
    chatterbox_max_reference_seconds: float = 30.0
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
    # Device strings are split between STT and TTS because the two engines
    # reach the GPU through different libraries.
    #
    # stt_device → CTranslate2 (via faster-whisper). CT2's PyPI wheel is
    # CUDA-only: it has no ROCm build, so on an AMD host "cuda" errors out
    # with "CUDA driver version is insufficient". "auto" picks CUDA when a
    # NVIDIA driver is present and falls back to CPU otherwise — that is
    # the right default on ROCm hosts because Strix Halo's 16-core Zen 5
    # runs int8 Whisper Turbo near real-time on CPU.
    #
    # tts_device → PyTorch (via qwen-tts). ROCm-PyTorch exposes the CUDA
    # API via HIP, so "cuda" is correct on both NVIDIA and ROCm hosts.
    stt_device: str = "auto"
    tts_device: str = "cuda"
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
        allowed = {"custom_voice", "voice_design", "clone"}
        unknown = [m for m in value if m not in allowed]
        if unknown:
            raise ValueError(f"unknown TTS mode(s): {unknown!r}; allowed: {sorted(allowed)}")
        return value  # type: ignore[return-value]
