# chatsune-voice Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI REST service with a static HTML "tinker page" that exposes Whisper Turbo STT and Qwen3-TTS (CustomVoice + VoiceDesign) on the operator's machine, with clean architectural seams ready for Phase 2 (WebSocket tunnel).

**Architecture:** Python 3.12 + FastAPI + uvicorn. Transport-blind `engines/` package (Protocol-based) owns Whisper and Qwen3-TTS adapters plus a `TTSModelRegistry` that enforces the `TTS_ENABLED_MODES` / `TTS_VRAM_POLICY` configuration with `asyncio.Lock`-based serialisation. REST endpoints live in `api/` and call into engines via registry acquire contexts. Observability via structlog JSON. Two Dockerfiles (CUDA and ROCm) feed a single `compose.yml` with `cuda|rocm` profiles.

**Tech Stack:** FastAPI, Pydantic 2 / pydantic-settings, structlog, faster-whisper (CTranslate2), qwen-tts (PyPI), numpy, soundfile, uvicorn, python-multipart. Dev: pytest, pytest-asyncio, httpx, ruff. Package manager: uv.

**Source of truth:** `docs/superpowers/specs/2026-04-18-chatsune-voice-phase-1-design.md`. When this plan diverges from the spec, the spec wins; update the plan or raise the discrepancy.

**Conventions:**
- All code, comments, docstrings, log event names, and documentation in British English.
- Commit messages: imperative, free-form (no Conventional Commits prefix).
- After each task, commit on master. No branches in Phase 1.
- Tests never touch the real models; always use fakes/fixtures for engines.

---

## File Structure (what each file owns)

```
chatsune-voice/
├── backend/
│   ├── pyproject.toml                   # Project metadata + direct deps
│   ├── uv.lock                          # Resolved dependency pins
│   ├── voice/
│   │   ├── __init__.py                  # (empty)
│   │   ├── main.py                      # entry: builds app, env → HF_HOME, preload, uvicorn
│   │   ├── config.py                    # Settings (pydantic-settings), enum types
│   │   ├── logging_setup.py             # configure_logging() — structlog JSON
│   │   ├── audio.py                     # WAV streaming header + float32 → PCM16
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── protocol.py              # STTEngine/TTSModel Protocols + dataclasses + errors
│   │   │   ├── registry.py              # TTSModelRegistry: policy, locks, acquire()
│   │   │   ├── whisper.py               # WhisperEngine (faster-whisper wrapper)
│   │   │   └── qwen_tts.py              # QwenCustomVoiceModel + QwenVoiceDesignModel
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── app.py                   # FastAPI app factory + request-id middleware
│   │       ├── models.py                # Pydantic API request/response models
│   │       ├── stt.py                   # POST /v1/transcribe
│   │       ├── tts.py                   # POST /v1/speak (streaming)
│   │       └── health.py                # GET /healthz
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   ├── scripts/
│   │   └── prefetch_models.py
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py                  # FakeSTTEngine, FakeTTSModel, settings fixture
│       ├── test_config.py
│       ├── test_logging_setup.py
│       ├── test_audio.py
│       ├── test_registry.py
│       ├── test_whisper.py
│       ├── test_qwen_tts.py
│       ├── test_api_models.py
│       ├── test_health_api.py
│       ├── test_stt_api.py
│       ├── test_tts_api.py
│       ├── test_main_startup.py
│       └── test_integration_smoke.py
├── obsidian/                            # Obsidian vault (state gitignored)
│   └── .gitkeep
├── Dockerfile.cuda
├── Dockerfile.rocm
├── compose.yml
├── .env.example
├── .dockerignore
├── .gitignore                           # already exists — extend
├── .github/workflows/ci.yml             # uv sync + ruff + pytest
├── LICENSE                              # already exists (GPL-3.0)
└── README.md                            # already exists — rewrite
```

---

## Task 1: Initialise the backend package

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/voice/__init__.py`
- Create: `backend/voice/engines/__init__.py`
- Create: `backend/voice/api/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Modify: `.gitignore` (add Python + uv + models + obsidian state)
- Create: `backend/.python-version`

- [ ] **Step 1: Write `backend/pyproject.toml`**

```toml
[project]
name = "chatsune-voice"
version = "0.1.0"
description = "Voice homelab backend for chatsune (Whisper Turbo STT + Qwen3-TTS)"
authors = [{ name = "Chris (symphonic-navigator)" }]
license = { text = "GPL-3.0-or-later" }
readme = "../README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115,<0.116",
    "uvicorn[standard]>=0.32,<0.33",
    "pydantic>=2.9,<3",
    "pydantic-settings>=2.6,<3",
    "structlog>=24.4,<25",
    "faster-whisper>=1.0,<2",
    "qwen-tts>=0.1",
    "numpy>=1.26,<3",
    "soundfile>=0.12,<0.13",
    "python-multipart>=0.0.12,<0.1",
    "torch>=2.5,<3",
]

[dependency-groups]
dev = [
    "pytest>=8.3,<9",
    "pytest-asyncio>=0.24,<0.25",
    "httpx>=0.27,<0.28",
    "ruff>=0.7,<0.8",
]

[tool.uv]
package = true

[tool.hatch.build.targets.wheel]
packages = ["voice"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
```

- [ ] **Step 2: Create empty package markers**

Write empty files:
- `backend/voice/__init__.py` — `"""chatsune-voice package."""`
- `backend/voice/engines/__init__.py` — `"""Inference engine abstraction."""`
- `backend/voice/api/__init__.py` — `"""HTTP transport layer."""`
- `backend/tests/__init__.py` — empty
- `backend/tests/conftest.py` — `"""Shared pytest fixtures."""`
- `backend/.python-version` — `3.12`

- [ ] **Step 3: Extend `.gitignore`**

Append to the existing `/.gitignore`:

```gitignore

# Python / uv
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.venv/
.uv/

# Environment
.env
.env.local

# Model cache
/models/
/.model-cache/

# Obsidian local state (keep settings, ignore workspace state)
/obsidian/.obsidian/workspace*
/obsidian/.obsidian/cache
/obsidian/.trash/

# Build outputs
/dist/
/build/
```

- [ ] **Step 4: Run `uv sync` to create `uv.lock` and virtualenv**

Run:
```bash
cd /home/chris/workspace/chatsune-voice/backend
uv sync --dev
```
Expected: creates `backend/.venv/` and `backend/uv.lock`. No errors. Warnings about flash-attn etc. are fine.

- [ ] **Step 5: Verify pytest is callable**

Run:
```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest --collect-only
```
Expected: "no tests ran". Exit 5 is normal (pytest exit for "no tests collected").

- [ ] **Step 6: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/ .gitignore
git commit -m "Scaffold Python package with uv and FastAPI dependencies"
```

---

## Task 2: Create shared test fixtures

**Files:**
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: Write `backend/tests/conftest.py`**

```python
"""Shared pytest fixtures — fakes that substitute for real inference engines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


@dataclass
class FakeSTTEngine:
    """Deterministic, configurable replacement for WhisperEngine."""

    model_name: str = "fake-whisper"
    loaded: bool = True
    result_text: str = "hello world"
    result_language: str = "en"
    result_language_probability: float = 0.99
    result_duration: float = 1.0
    result_segments: list[dict[str, Any]] = field(default_factory=list)
    transcribe_delay: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def transcribe(
        self,
        audio,
        *,
        language: str | None = None,
        vad: bool = True,
    ):
        from voice.engines.protocol import TranscriptionResult, TranscriptionSegment

        self.calls.append({"audio_len": len(audio) if isinstance(audio, bytes) else -1,
                           "language": language, "vad": vad})
        if self.transcribe_delay:
            await asyncio.sleep(self.transcribe_delay)
        segs = [TranscriptionSegment(**s) for s in self.result_segments] or [
            TranscriptionSegment(start=0.0, end=self.result_duration, text=self.result_text)
        ]
        return TranscriptionResult(
            text=self.result_text,
            language=self.result_language,
            language_probability=self.result_language_probability,
            duration=self.result_duration,
            segments=segs,
        )

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeTTSModel:
    """Deterministic replacement for a Qwen3-TTS checkpoint."""

    mode: str = "custom_voice"
    sample_rate: int = 22050
    samples: np.ndarray | None = None
    stream_chunk_size: int = 4096
    generate_delay: float = 0.0
    raise_mid_stream_after: int | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        if self.samples is None:
            self.samples = np.linspace(-0.5, 0.5, num=22050, dtype=np.float32)

    async def stream(self, req) -> AsyncIterator[np.ndarray]:
        self.calls.append({
            "mode": req.mode,
            "text": req.text,
            "language": req.language,
            "speaker": req.speaker,
            "voice_prompt": req.voice_prompt,
            "instruct": req.instruct,
        })
        if self.generate_delay:
            await asyncio.sleep(self.generate_delay)
        assert self.samples is not None
        offset = 0
        chunks_emitted = 0
        while offset < len(self.samples):
            chunk = self.samples[offset:offset + self.stream_chunk_size]
            yield chunk
            chunks_emitted += 1
            offset += self.stream_chunk_size
            if (
                self.raise_mid_stream_after is not None
                and chunks_emitted >= self.raise_mid_stream_after
            ):
                raise RuntimeError("fake mid-stream error")

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_stt() -> FakeSTTEngine:
    return FakeSTTEngine()


@pytest.fixture
def fake_tts_custom() -> FakeTTSModel:
    return FakeTTSModel(mode="custom_voice")


@pytest.fixture
def fake_tts_design() -> FakeTTSModel:
    return FakeTTSModel(mode="voice_design")
```

- [ ] **Step 2: Verify fixtures import cleanly**

Run:
```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run python -c "from tests.conftest import FakeSTTEngine, FakeTTSModel; print('ok')"
```
Expected: `ImportError: cannot import name 'TranscriptionResult' from 'voice.engines.protocol'` — that's fine, it means the protocol module will be the next thing we build. The fixtures themselves import without syntax errors.

Actually run a simpler check:
```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run python -c "import ast; ast.parse(open('tests/conftest.py').read()); print('syntax ok')"
```
Expected: `syntax ok`.

- [ ] **Step 3: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/tests/conftest.py
git commit -m "Add FakeSTTEngine and FakeTTSModel test fixtures"
```

---

## Task 3: Configuration module

**Files:**
- Create: `backend/voice/config.py`
- Create: `backend/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_config.py -v
```
Expected: ModuleNotFoundError or ImportError — `voice.config` does not exist.

- [ ] **Step 3: Write the minimal Settings module**

Create `backend/voice/config.py`:

```python
"""Application settings — pydantic-settings with fail-fast validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    tts_enabled_modes: tuple[TTSMode, ...] = ("custom_voice", "voice_design")
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
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_config.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/config.py backend/tests/test_config.py
git commit -m "Add Settings with fail-fast env var validation"
```

---

## Task 4: Logging setup

**Files:**
- Create: `backend/voice/logging_setup.py`
- Create: `backend/tests/test_logging_setup.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_logging_setup.py`:

```python
"""Tests for voice.logging_setup."""

from __future__ import annotations

import json
import logging


def test_configure_logging_emits_json(capsys):
    from voice.logging_setup import configure_logging, get_logger

    configure_logging("info")
    log = get_logger("test")
    log.info("hello", foo="bar", count=3)

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["foo"] == "bar"
    assert record["count"] == 3
    assert record["level"] == "info"
    assert "timestamp" in record


def test_configure_logging_respects_level(capsys):
    from voice.logging_setup import configure_logging, get_logger

    configure_logging("warn")
    log = get_logger("test")
    log.info("suppressed")
    log.warning("shown")

    captured = capsys.readouterr()
    assert "suppressed" not in captured.out
    assert "shown" in captured.out


def test_stdlib_logger_is_captured(capsys):
    from voice.logging_setup import configure_logging

    configure_logging("info")
    logging.getLogger("external").info("from stdlib")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "from stdlib"
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_logging_setup.py -v
```
Expected: ImportError — module does not exist.

- [ ] **Step 3: Write logging_setup.py**

Create `backend/voice/logging_setup.py`:

```python
"""Structured JSON logging via structlog, with stdlib capture."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str = "info") -> None:
    """Configure structlog + stdlib logging to emit JSON lines to stdout.

    Two paths are wired up:
    1. `structlog.get_logger(...)` → PrintLoggerFactory → stdout (JSON).
    2. `logging.getLogger(...)` → StreamHandler with structlog ProcessorFormatter
       → stdout (also JSON).

    The second bridge is necessary because `logging.basicConfig` alone would emit
    stdlib log records as plain text; the ProcessorFormatter hands them through
    the same structlog rendering chain so every log line on stdout is JSON.
    """
    stdlib_level = _LEVELS.get(level, logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(stdlib_level),
        # Passing no file lets each PrintLogger instance resolve sys.stdout
        # at construction time, which plays nicely with pytest's capsys and
        # any other harness that swaps sys.stdout in/out dynamically.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(stdlib_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(stdlib_level)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
            ],
        )
    )
    root_logger.addHandler(handler)


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_logging_setup.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/logging_setup.py backend/tests/test_logging_setup.py
git commit -m "Add structlog JSON logging setup"
```

---

## Task 5: Audio helper — WAV streaming header + PCM16 conversion

**Files:**
- Create: `backend/voice/audio.py`
- Create: `backend/tests/test_audio.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_audio.py`:

```python
"""Tests for voice.audio."""

from __future__ import annotations

import struct

import numpy as np


def test_make_streaming_wav_header_layout():
    from voice.audio import make_streaming_wav_header

    hdr = make_streaming_wav_header(sample_rate=22050, channels=1)
    assert len(hdr) == 44
    assert hdr[0:4] == b"RIFF"
    # bytes 4..8 = total size - 8 = 0xFFFFFFFF - 8 (max)
    assert struct.unpack("<I", hdr[4:8])[0] == 0xFFFFFFFF - 8
    assert hdr[8:12] == b"WAVE"
    assert hdr[12:16] == b"fmt "
    assert struct.unpack("<I", hdr[16:20])[0] == 16          # fmt chunk size
    assert struct.unpack("<H", hdr[20:22])[0] == 1           # PCM format
    assert struct.unpack("<H", hdr[22:24])[0] == 1           # channels
    assert struct.unpack("<I", hdr[24:28])[0] == 22050       # sample rate
    assert struct.unpack("<I", hdr[28:32])[0] == 22050 * 2   # byte rate
    assert struct.unpack("<H", hdr[32:34])[0] == 2           # block align
    assert struct.unpack("<H", hdr[34:36])[0] == 16          # bits per sample
    assert hdr[36:40] == b"data"
    # bytes 40..44 = data chunk size = 0xFFFFFFFF (streaming marker)
    assert struct.unpack("<I", hdr[40:44])[0] == 0xFFFFFFFF


def test_float32_to_pcm16_clipping():
    from voice.audio import float32_to_pcm16

    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float32_to_pcm16(arr)
    # bytes: 5 samples * 2 bytes = 10
    assert len(pcm) == 10
    unpacked = struct.unpack("<5h", pcm)
    # -2.0 clipped to -1.0 → -32768; +2.0 clipped to +1.0 → +32767
    assert unpacked[0] == -32768
    assert unpacked[1] == -32768     # -1.0 → -32768 (i.e. int16 min)
    assert unpacked[2] == 0
    assert unpacked[3] == 32767
    assert unpacked[4] == 32767


def test_float32_to_pcm16_roundtrip_precision():
    from voice.audio import float32_to_pcm16

    arr = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    pcm = float32_to_pcm16(arr)
    ints = struct.unpack("<3h", pcm)
    assert abs(ints[0] - 16383) <= 1
    assert abs(ints[1] - (-8192)) <= 1
    assert abs(ints[2] - 4096) <= 1


def test_float32_to_pcm16_rejects_wrong_dtype():
    from voice.audio import float32_to_pcm16
    import pytest

    arr = np.array([0.0, 0.5], dtype=np.float64)
    with pytest.raises(TypeError):
        float32_to_pcm16(arr)
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_audio.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write audio.py**

Create `backend/voice/audio.py`:

```python
"""WAV streaming header + float32 → PCM16 sample conversion."""

from __future__ import annotations

import struct

import numpy as np

_STREAM_DATA_SIZE = 0xFFFFFFFF
_STREAM_RIFF_SIZE = _STREAM_DATA_SIZE - 8


def make_streaming_wav_header(*, sample_rate: int, channels: int = 1) -> bytes:
    """Produce a 44-byte RIFF/WAVE header advertising an open-ended PCM16 stream.

    The data chunk size is set to 0xFFFFFFFF, which browsers and media players
    accept as a "streaming / unknown length" marker. The audio payload written
    after this header is raw little-endian PCM16 samples.
    """
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    return (
        b"RIFF"
        + struct.pack("<I", _STREAM_RIFF_SIZE)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits_per_sample)
        + b"data"
        + struct.pack("<I", _STREAM_DATA_SIZE)
    )


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert a float32 numpy array in [-1, 1] to little-endian PCM16 bytes.

    int16 has an asymmetric range ([-32768, +32767]) so we scale negative and
    positive samples separately: negatives by 32768 and positives by 32767.
    This keeps full-scale silence-to-peak symmetric from the listener's point
    of view and makes -1.0 round-trip to INT16_MIN, matching the test.
    """
    if samples.dtype != np.float32:
        raise TypeError(f"expected float32, got {samples.dtype}")
    clipped = np.clip(samples, -1.0, 1.0)
    scaled = np.where(
        clipped < 0,
        np.round(clipped * 32768.0),
        np.round(clipped * 32767.0),
    ).astype(np.int16)
    return scaled.tobytes()
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_audio.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/audio.py backend/tests/test_audio.py
git commit -m "Add streaming WAV header helper and PCM16 sample converter"
```

---

## Task 6: Engine protocols and dataclasses

**Files:**
- Create: `backend/voice/engines/protocol.py`

Note: `engines/protocol.py` is pure types and dataclasses. No logic to unit-test; it gets exercised by every subsequent test.

- [ ] **Step 1: Write `engines/protocol.py`**

Create `backend/voice/engines/protocol.py`:

```python
"""Transport-agnostic protocols and data classes for STT and TTS engines.

No module in this file imports FastAPI, HTTP, or WebSockets — that is the
whole point. API handlers and future WebSocket adapters consume these types.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import BinaryIO, Literal, Protocol

import numpy as np

TTSMode = Literal["custom_voice", "voice_design"]


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[TranscriptionSegment]


class STTEngine(Protocol):
    model_name: str

    async def transcribe(
        self,
        audio: bytes | BinaryIO,
        *,
        language: str | None = None,
        vad: bool = True,
    ) -> TranscriptionResult: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class TTSRequest:
    mode: TTSMode
    text: str
    language: str
    speaker: str | None = None
    voice_prompt: str | None = None
    instruct: str | None = None


class TTSModel(Protocol):
    mode: TTSMode
    sample_rate: int

    def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]: ...

    async def aclose(self) -> None: ...


class EngineError(Exception):
    """Base class for engine-level failures surfaced to the transport layer."""


class ModeDisabledError(EngineError):
    def __init__(self, mode: str) -> None:
        super().__init__(f"TTS mode {mode!r} is not in TTS_ENABLED_MODES")
        self.mode = mode


class ModelLoadError(EngineError):
    def __init__(self, mode: str, underlying: BaseException) -> None:
        super().__init__(f"failed to load TTS model for mode {mode!r}: {underlying!r}")
        self.mode = mode
        self.underlying = underlying
```

- [ ] **Step 2: Verify it imports**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run python -c "from voice.engines.protocol import STTEngine, TTSModel, TTSRequest, TranscriptionResult, ModeDisabledError, ModelLoadError; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Re-run the previously written tests**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest -v
```
Expected: all prior tests still pass.

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/engines/protocol.py
git commit -m "Define STT and TTS engine Protocols with engine-level exceptions"
```

---

## Task 7: TTSModelRegistry

**Files:**
- Create: `backend/voice/engines/registry.py`
- Create: `backend/tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_registry.py`:

```python
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
    await registry.preload()
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
    await registry.preload()

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
    await registry.preload()

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
    await registry.preload()   # no-op under swap
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
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_registry.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write registry.py**

Create `backend/voice/engines/registry.py`:

```python
"""TTSModelRegistry — owns enabled modes, VRAM policy, and locks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from voice.engines.protocol import (
    ModelLoadError,
    ModeDisabledError,
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

    async def preload(self) -> None:
        """Load enabled models at start-up. No-op under 'swap' policy."""
        if self._policy == "swap":
            log.info("tts_registry_preload_skipped", reason="swap_policy")
            return
        for mode in self._enabled:
            await self._load_locked(mode)

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
            except Exception as exc:  # noqa: BLE001
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
        self._loaded.clear()

    async def _load_locked(self, mode: TTSMode) -> None:
        try:
            model = await asyncio.to_thread(self._loader, mode)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(mode, exc) from exc
        self._loaded[mode] = model
        log.info("tts_model_loaded", mode=mode)

    async def _evict_all(self) -> None:
        for mode, model in list(self._loaded.items()):
            try:
                await model.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
            if self._on_evict is not None:
                self._on_evict(mode)
            log.info("tts_model_evicted", mode=mode)
        self._loaded.clear()
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_registry.py -v
```
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/engines/registry.py backend/tests/test_registry.py
git commit -m "Add TTSModelRegistry with keep_loaded and swap VRAM policies"
```

---

## Task 8: Whisper engine adapter

**Files:**
- Create: `backend/voice/engines/whisper.py`
- Create: `backend/tests/test_whisper.py`

The real `faster_whisper.WhisperModel` is not loaded in tests. `WhisperEngine` depends on an injected factory so tests can substitute a fake.

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_whisper.py`:

```python
"""Tests for voice.engines.whisper — wrapper logic using a fake WhisperModel."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class _FakeInfo:
    language: str = "de"
    language_probability: float = 0.97
    duration: float = 1.5


@dataclass
class _FakeWhisperModel:
    segments: list[_FakeSegment] = field(default_factory=lambda: [_FakeSegment(0.0, 1.5, "hallo")])
    info: _FakeInfo = field(default_factory=_FakeInfo)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transcribe(self, audio, **kwargs):
        payload = audio.read() if hasattr(audio, "read") else audio
        self.calls.append({"audio_bytes": len(payload), "kwargs": kwargs})
        return iter(self.segments), self.info


@pytest.mark.asyncio
async def test_transcribe_returns_expected_result():
    from voice.engines.whisper import WhisperEngine

    fake = _FakeWhisperModel()
    engine = WhisperEngine(model_name="fake", model=fake)
    result = await engine.transcribe(b"\x00" * 100, language="de", vad=True)

    assert result.text == "hallo"
    assert result.language == "de"
    assert result.language_probability == pytest.approx(0.97)
    assert result.duration == pytest.approx(1.5)
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].text == "hallo"


@pytest.mark.asyncio
async def test_transcribe_joins_multiple_segments():
    from voice.engines.whisper import WhisperEngine

    fake = _FakeWhisperModel(segments=[
        _FakeSegment(0.0, 0.5, "Hallo"),
        _FakeSegment(0.5, 1.0, " Welt"),
    ])
    engine = WhisperEngine(model_name="fake", model=fake)
    result = await engine.transcribe(b"\x00" * 10)
    assert result.text == "Hallo Welt"
    assert len(result.segments) == 2


@pytest.mark.asyncio
async def test_transcribe_passes_language_and_vad():
    from voice.engines.whisper import WhisperEngine

    fake = _FakeWhisperModel()
    engine = WhisperEngine(model_name="fake", model=fake)
    await engine.transcribe(b"\x00", language="en", vad=False)

    assert fake.calls[-1]["kwargs"]["language"] == "en"
    assert fake.calls[-1]["kwargs"]["vad_filter"] is False


@pytest.mark.asyncio
async def test_transcribe_auto_detect_when_language_none():
    from voice.engines.whisper import WhisperEngine

    fake = _FakeWhisperModel()
    engine = WhisperEngine(model_name="fake", model=fake)
    await engine.transcribe(b"\x00", language=None)
    # language=None should be forwarded as "language" absent or None — either is fine,
    # but faster-whisper's convention is to omit the key or pass None.
    assert fake.calls[-1]["kwargs"].get("language") is None


@pytest.mark.asyncio
async def test_transcribe_accepts_binary_io():
    from voice.engines.whisper import WhisperEngine

    fake = _FakeWhisperModel()
    engine = WhisperEngine(model_name="fake", model=fake)
    buf = io.BytesIO(b"\x01\x02\x03")
    await engine.transcribe(buf)
    assert fake.calls[-1]["audio_bytes"] == 3


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    from voice.engines.whisper import WhisperEngine

    engine = WhisperEngine(model_name="fake", model=_FakeWhisperModel())
    await engine.aclose()
    await engine.aclose()
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_whisper.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write whisper.py**

Create `backend/voice/engines/whisper.py`:

```python
"""Whisper STT adapter wrapping faster-whisper.

The real `faster_whisper.WhisperModel` is injected at construction time so
tests can substitute a fake without loading the real model.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, BinaryIO, Protocol

from voice.engines.protocol import TranscriptionResult, TranscriptionSegment
from voice.logging_setup import get_logger

log = get_logger(__name__)


class _WhisperBackend(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> Any: ...


def load_faster_whisper(
    model_id: str,
    *,
    device: str = "cuda",
    compute_type: str = "float16",
    download_root: str | None = None,
) -> _WhisperBackend:
    """Factory: load a real faster-whisper model. Kept separate so tests skip it."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )


class WhisperEngine:
    def __init__(self, *, model_name: str, model: _WhisperBackend) -> None:
        self.model_name = model_name
        self._model = model
        self._lock = asyncio.Lock()
        self._closed = False

    async def transcribe(
        self,
        audio: bytes | BinaryIO,
        *,
        language: str | None = None,
        vad: bool = True,
    ) -> TranscriptionResult:
        if self._closed:
            raise RuntimeError("WhisperEngine is closed")

        buf: BinaryIO = audio if hasattr(audio, "read") else io.BytesIO(audio)  # type: ignore[assignment]

        async with self._lock:
            segments_iter, info = await asyncio.to_thread(
                self._model.transcribe,
                buf,
                language=language,
                vad_filter=vad,
            )
            collected = await asyncio.to_thread(list, segments_iter)

        segments = [
            TranscriptionSegment(start=s.start, end=s.end, text=s.text)
            for s in collected
        ]
        text = "".join(s.text for s in segments)
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0)),
            duration=float(getattr(info, "duration", 0.0)),
            segments=segments,
        )

    async def aclose(self) -> None:
        self._closed = True
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_whisper.py -v
```
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/engines/whisper.py backend/tests/test_whisper.py
git commit -m "Add WhisperEngine adapter over faster-whisper"
```

---

## Task 9: Qwen3-TTS engine adapters

**Files:**
- Create: `backend/voice/engines/qwen_tts.py`
- Create: `backend/tests/test_qwen_tts.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_qwen_tts.py`:

```python
"""Tests for voice.engines.qwen_tts — adapter logic using a fake Qwen3TTSModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from voice.engines.protocol import TTSRequest


@dataclass
class _FakeQwenModel:
    wavs: list[np.ndarray] = field(default_factory=lambda: [np.zeros(1000, dtype=np.float32)])
    sample_rate: int = 22050
    custom_voice_calls: list[dict[str, Any]] = field(default_factory=list)
    voice_design_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_custom_voice(self, *, text, language, speaker, instruct=None):
        self.custom_voice_calls.append({
            "text": text, "language": language, "speaker": speaker, "instruct": instruct,
        })
        return self.wavs, self.sample_rate

    def generate_voice_design(self, *, text, language, instruct=None):
        self.voice_design_calls.append({
            "text": text, "language": language, "instruct": instruct,
        })
        return self.wavs, self.sample_rate


@pytest.mark.asyncio
async def test_custom_voice_streams_samples_in_chunks():
    from voice.engines.qwen_tts import QwenCustomVoiceModel

    samples = np.linspace(-0.5, 0.5, num=10000, dtype=np.float32)
    fake = _FakeQwenModel(wavs=[samples], sample_rate=22050)
    model = QwenCustomVoiceModel(backend=fake, chunk_size=4096)

    req = TTSRequest(
        mode="custom_voice",
        text="hallo",
        language="German",
        speaker="Vivian",
        instruct="fröhlich",
    )
    chunks = [c async for c in model.stream(req)]

    assert sum(len(c) for c in chunks) == 10000
    assert all(c.dtype == np.float32 for c in chunks)
    assert fake.custom_voice_calls[-1]["speaker"] == "Vivian"
    assert fake.custom_voice_calls[-1]["instruct"] == "fröhlich"
    assert fake.custom_voice_calls[-1]["language"] == "German"


@pytest.mark.asyncio
async def test_voice_design_uses_voice_prompt_as_instruct():
    """For VoiceDesign, the model card puts the voice description into `instruct`."""
    from voice.engines.qwen_tts import QwenVoiceDesignModel

    samples = np.zeros(5000, dtype=np.float32)
    fake = _FakeQwenModel(wavs=[samples])
    model = QwenVoiceDesignModel(backend=fake)

    req = TTSRequest(
        mode="voice_design",
        text="hi",
        language="English",
        voice_prompt="warm low male voice",
        instruct="slowly",
    )
    chunks = [c async for c in model.stream(req)]
    assert sum(len(c) for c in chunks) == 5000
    # The adapter combines voice_prompt + instruct into the model's single `instruct` field.
    call = fake.voice_design_calls[-1]
    assert "warm low male voice" in call["instruct"]
    assert "slowly" in call["instruct"]


@pytest.mark.asyncio
async def test_voice_design_without_instruct_still_sends_voice_prompt():
    from voice.engines.qwen_tts import QwenVoiceDesignModel

    fake = _FakeQwenModel()
    model = QwenVoiceDesignModel(backend=fake)
    req = TTSRequest(
        mode="voice_design",
        text="hi",
        language="English",
        voice_prompt="raspy tenor",
    )
    _ = [c async for c in model.stream(req)]
    assert fake.voice_design_calls[-1]["instruct"] == "raspy tenor"


@pytest.mark.asyncio
async def test_custom_voice_without_instruct_passes_none():
    from voice.engines.qwen_tts import QwenCustomVoiceModel

    fake = _FakeQwenModel()
    model = QwenCustomVoiceModel(backend=fake)
    req = TTSRequest(
        mode="custom_voice",
        text="hi",
        language="English",
        speaker="Ryan",
    )
    _ = [c async for c in model.stream(req)]
    assert fake.custom_voice_calls[-1]["instruct"] is None


@pytest.mark.asyncio
async def test_mode_attribute_matches_class():
    from voice.engines.qwen_tts import QwenCustomVoiceModel, QwenVoiceDesignModel

    assert QwenCustomVoiceModel(backend=_FakeQwenModel()).mode == "custom_voice"
    assert QwenVoiceDesignModel(backend=_FakeQwenModel()).mode == "voice_design"


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    from voice.engines.qwen_tts import QwenCustomVoiceModel

    model = QwenCustomVoiceModel(backend=_FakeQwenModel())
    await model.aclose()
    await model.aclose()
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_qwen_tts.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write qwen_tts.py**

Create `backend/voice/engines/qwen_tts.py`:

```python
"""Qwen3-TTS adapters — one class per checkpoint (CustomVoice, VoiceDesign).

Both wrap a `backend` object that exposes the qwen-tts library's
`generate_custom_voice(...)` or `generate_voice_design(...)` method. The real
backend is loaded lazily; tests inject a fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from voice.engines.protocol import TTSMode, TTSRequest
from voice.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 4096


class _QwenBackend(Protocol):
    sample_rate: int
    def generate_custom_voice(self, **kwargs: Any) -> Any: ...
    def generate_voice_design(self, **kwargs: Any) -> Any: ...


def load_qwen_tts(
    model_id: str,
    *,
    device: str = "cuda",
    attention_impl: str = "sdpa",
) -> _QwenBackend:
    """Factory: load the real qwen-tts model. Kept separate so tests can skip."""
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attention_impl,
    )


class _QwenBase:
    mode: TTSMode
    sample_rate: int

    def __init__(self, *, backend: _QwenBackend, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._backend = backend
        self._chunk_size = chunk_size
        self.sample_rate = getattr(backend, "sample_rate", 22050)
        self._closed = False

    async def aclose(self) -> None:
        self._closed = True

    async def _chunked(self, samples: np.ndarray) -> AsyncIterator[np.ndarray]:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        offset = 0
        while offset < len(samples):
            yield samples[offset:offset + self._chunk_size]
            offset += self._chunk_size


class QwenCustomVoiceModel(_QwenBase):
    mode: TTSMode = "custom_voice"

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.speaker is None:
            raise ValueError("CustomVoice requires a speaker")

        def _generate() -> tuple[list[np.ndarray], int]:
            return self._backend.generate_custom_voice(
                text=req.text,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct,
            )

        wavs, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        samples = wavs[0] if isinstance(wavs, list) else wavs
        async for chunk in self._chunked(samples):
            yield chunk


class QwenVoiceDesignModel(_QwenBase):
    mode: TTSMode = "voice_design"

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.voice_prompt is None:
            raise ValueError("VoiceDesign requires a voice_prompt")

        # qwen-tts's voice-design API takes a single `instruct` that carries the voice
        # description; we concatenate voice_prompt (mandatory) and the caller's
        # optional instruct for speaking-style.
        combined = req.voice_prompt
        if req.instruct:
            combined = f"{combined}. {req.instruct}"

        def _generate() -> tuple[list[np.ndarray], int]:
            return self._backend.generate_voice_design(
                text=req.text,
                language=req.language,
                instruct=combined,
            )

        wavs, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        samples = wavs[0] if isinstance(wavs, list) else wavs
        async for chunk in self._chunked(samples):
            yield chunk
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_qwen_tts.py -v
```
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/engines/qwen_tts.py backend/tests/test_qwen_tts.py
git commit -m "Add QwenCustomVoiceModel and QwenVoiceDesignModel adapters"
```

---

## Task 10: Pydantic API request/response models

**Files:**
- Create: `backend/voice/api/models.py`
- Create: `backend/tests/test_api_models.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_api_models.py`:

```python
"""Tests for voice.api.models — discriminated unions, field validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, TypeAdapter


def test_custom_voice_request_roundtrip():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    data = {
        "mode": "custom_voice",
        "text": "Hallo",
        "language": "German",
        "speaker": "Vivian",
        "instruct": "fröhlich",
    }
    parsed = adapter.validate_python(data)
    assert parsed.mode == "custom_voice"
    assert parsed.speaker == "Vivian"
    assert parsed.instruct == "fröhlich"


def test_voice_design_request_roundtrip():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    data = {
        "mode": "voice_design",
        "text": "Hallo",
        "language": "German",
        "voice_prompt": "warme Stimme",
    }
    parsed = adapter.validate_python(data)
    assert parsed.mode == "voice_design"
    assert parsed.voice_prompt == "warme Stimme"
    assert parsed.instruct is None


def test_custom_voice_requires_speaker():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({"mode": "custom_voice", "text": "x", "language": "English"})


def test_voice_design_requires_voice_prompt():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({"mode": "voice_design", "text": "x", "language": "English"})


def test_unknown_speaker_rejected():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": "x", "language": "English", "speaker": "Bogus",
        })


def test_unknown_language_rejected():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": "x", "language": "Klingon", "speaker": "Vivian",
        })


def test_text_length_limit_enforced():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    long_text = "a" * 4001
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": long_text, "language": "English", "speaker": "Ryan",
        })


def test_transcribe_response_shape():
    from voice.api.models import TranscribeResponse, TranscribeResponseSegment

    resp = TranscribeResponse(
        text="hi",
        language="en",
        language_probability=0.9,
        duration=1.0,
        segments=[TranscribeResponseSegment(start=0.0, end=1.0, text="hi")],
    )
    dumped = resp.model_dump()
    assert dumped["text"] == "hi"
    assert dumped["segments"][0]["end"] == 1.0


def test_health_response_shape():
    from voice.api.models import HealthResponse, HealthSTTInfo, HealthTTSInfo

    resp = HealthResponse(
        status="ok",
        stt=HealthSTTInfo(model="m", loaded=True),
        tts=HealthTTSInfo(
            enabled_modes=["custom_voice"],
            vram_policy="keep_loaded",
            loaded_modes=["custom_voice"],
        ),
    )
    dumped = resp.model_dump()
    assert dumped["tts"]["enabled_modes"] == ["custom_voice"]
```

- [ ] **Step 2: Run tests — verify failure**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_api_models.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write api/models.py**

Create `backend/voice/api/models.py`:

```python
"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Language = Literal[
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

CustomVoiceSpeaker = Literal[
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


class SpeakCustomVoiceRequest(BaseModel):
    mode: Literal["custom_voice"]
    text: str = Field(min_length=1, max_length=4000)
    language: Language
    speaker: CustomVoiceSpeaker
    instruct: str | None = Field(default=None, max_length=500)


class SpeakVoiceDesignRequest(BaseModel):
    mode: Literal["voice_design"]
    text: str = Field(min_length=1, max_length=4000)
    language: Language
    voice_prompt: str = Field(min_length=1, max_length=1000)
    instruct: str | None = Field(default=None, max_length=500)


SpeakRequest = Annotated[
    SpeakCustomVoiceRequest | SpeakVoiceDesignRequest,
    Field(discriminator="mode"),
]


class TranscribeResponseSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[TranscribeResponseSegment]


class HealthSTTInfo(BaseModel):
    model: str
    loaded: bool


class HealthTTSInfo(BaseModel):
    enabled_modes: list[str]
    vram_policy: str
    loaded_modes: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    stt: HealthSTTInfo
    tts: HealthTTSInfo


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None
    request_id: str | None = None
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_api_models.py -v
```
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/api/models.py backend/tests/test_api_models.py
git commit -m "Add Pydantic API request and response models"
```

---

## Task 11: Health endpoint

**Files:**
- Create: `backend/voice/api/health.py`
- Create: `backend/voice/api/app.py` (minimal skeleton; extended in later tasks)
- Create: `backend/tests/test_health_api.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_health_api.py`:

```python
"""Tests for GET /healthz."""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport


class _StubRegistry:
    def __init__(self, enabled, policy, loaded):
        self._enabled = tuple(enabled)
        self._policy = policy
        self._loaded = tuple(loaded)

    @property
    def enabled_modes(self):
        return self._enabled

    @property
    def policy(self):
        return self._policy

    def loaded_modes(self):
        return self._loaded


class _StubSTT:
    model_name = "stub-whisper"
    loaded = True


@pytest.mark.asyncio
async def test_healthz_ok_when_everything_loaded():
    from voice.api.app import build_app

    app = build_app(
        stt=_StubSTT(),
        registry=_StubRegistry(["custom_voice", "voice_design"], "keep_loaded",
                               ["custom_voice", "voice_design"]),
        settings=None,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["stt"]["model"] == "stub-whisper"
    assert body["stt"]["loaded"] is True
    assert body["tts"]["vram_policy"] == "keep_loaded"
    assert set(body["tts"]["enabled_modes"]) == {"custom_voice", "voice_design"}


@pytest.mark.asyncio
async def test_healthz_503_when_stt_not_loaded():
    from voice.api.app import build_app

    stt = _StubSTT()
    stt.loaded = False
    app = build_app(
        stt=stt,
        registry=_StubRegistry(["custom_voice"], "keep_loaded", ["custom_voice"]),
        settings=None,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


@pytest.mark.asyncio
async def test_healthz_503_when_enabled_tts_not_loaded_under_keep_loaded():
    from voice.api.app import build_app

    app = build_app(
        stt=_StubSTT(),
        registry=_StubRegistry(["custom_voice", "voice_design"], "keep_loaded",
                               ["custom_voice"]),
        settings=None,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_healthz_ok_under_swap_even_if_no_models_loaded():
    from voice.api.app import build_app

    app = build_app(
        stt=_StubSTT(),
        registry=_StubRegistry(["custom_voice", "voice_design"], "swap", []),
        settings=None,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
```

- [ ] **Step 2: Write minimal `api/app.py` + `api/health.py`**

Create `backend/voice/api/app.py`:

```python
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
```

Create `backend/voice/api/health.py`:

```python
"""GET /healthz — service readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from voice.api.models import HealthResponse, HealthSTTInfo, HealthTTSInfo

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request, response: Response) -> HealthResponse:
    stt = request.app.state.stt
    registry = request.app.state.registry

    stt_loaded = bool(getattr(stt, "loaded", True))
    enabled_modes = list(registry.enabled_modes)
    loaded_modes = list(registry.loaded_modes())
    policy = registry.policy

    degraded = False
    if not stt_loaded:
        degraded = True
    if policy == "keep_loaded":
        if set(enabled_modes) - set(loaded_modes):
            degraded = True

    if degraded:
        response.status_code = 503
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        stt=HealthSTTInfo(model=stt.model_name, loaded=stt_loaded),
        tts=HealthTTSInfo(
            enabled_modes=enabled_modes,
            vram_policy=policy,
            loaded_modes=loaded_modes,
        ),
    )
```

- [ ] **Step 3: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_health_api.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/api/app.py backend/voice/api/health.py backend/tests/test_health_api.py
git commit -m "Add /healthz endpoint with STT and TTS readiness reporting"
```

---

## Task 12: STT endpoint

**Files:**
- Create: `backend/voice/api/stt.py`
- Modify: `backend/voice/api/app.py` (register router)
- Create: `backend/tests/test_stt_api.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_stt_api.py`:

```python
"""Tests for POST /v1/transcribe."""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport


class _Settings:
    stt_max_audio_bytes = 1000


@pytest.mark.asyncio
async def test_transcribe_happy_path(fake_stt):
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()
    app = build_app(stt=fake_stt, registry=_Reg(), settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("sample.wav", b"\x00\x01\x02", "audio/wav")}
        r = await client.post("/v1/transcribe", files=files, data={"language": "de"})

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == fake_stt.result_text
    assert body["language"] == fake_stt.result_language
    assert fake_stt.calls[-1]["language"] == "de"
    assert fake_stt.calls[-1]["vad"] is True


@pytest.mark.asyncio
async def test_transcribe_vad_false(fake_stt):
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()
    app = build_app(stt=fake_stt, registry=_Reg(), settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("sample.wav", b"\x00", "audio/wav")}
        r = await client.post("/v1/transcribe", files=files, data={"vad": "false"})

    assert r.status_code == 200
    assert fake_stt.calls[-1]["vad"] is False


@pytest.mark.asyncio
async def test_transcribe_overflow_413(fake_stt):
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()
    app = build_app(stt=fake_stt, registry=_Reg(), settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("big.wav", b"\x00" * 1500, "audio/wav")}
        r = await client.post("/v1/transcribe", files=files)

    assert r.status_code == 413
    assert r.json()["error"] == "audio_too_large"


@pytest.mark.asyncio
async def test_transcribe_empty_audio_400(fake_stt):
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()
    app = build_app(stt=fake_stt, registry=_Reg(), settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("empty.wav", b"", "audio/wav")}
        r = await client.post("/v1/transcribe", files=files)

    assert r.status_code == 400


@pytest.mark.asyncio
async def test_transcribe_auto_language_when_not_provided(fake_stt):
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()
    app = build_app(stt=fake_stt, registry=_Reg(), settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"audio": ("s.wav", b"\x00", "audio/wav")}
        r = await client.post("/v1/transcribe", files=files)

    assert r.status_code == 200
    assert fake_stt.calls[-1]["language"] is None
```

- [ ] **Step 2: Write `api/stt.py`**

Create `backend/voice/api/stt.py`:

```python
"""POST /v1/transcribe — single-file transcription with optional language hint."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from voice.api.models import TranscribeResponse, TranscribeResponseSegment
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    vad: bool = Form(default=True),
) -> TranscribeResponse:
    settings = request.app.state.settings
    stt = request.app.state.stt

    payload = await audio.read()
    if len(payload) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_audio", "message": "audio is empty"},
        )
    if settings is not None and len(payload) > settings.stt_max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "audio_too_large",
                "message": "audio exceeds STT_MAX_AUDIO_BYTES",
                "limit_bytes": settings.stt_max_audio_bytes,
            },
        )

    language_norm = (language or "").strip() or None

    log.info("transcribe_request", audio_bytes=len(payload), language_hint=language_norm)
    try:
        result = await stt.transcribe(payload, language=language_norm, vad=vad)
    except Exception as exc:  # noqa: BLE001
        log.error("transcribe_error", error_type=type(exc).__name__, message=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_server_error"},
        ) from exc

    log.info(
        "transcribe_complete",
        detected_language=result.language,
        duration_ms=int(result.duration * 1000),
    )

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
        segments=[
            TranscribeResponseSegment(start=s.start, end=s.end, text=s.text)
            for s in result.segments
        ],
    )
```

- [ ] **Step 3: Register router in `api/app.py`**

Edit `backend/voice/api/app.py` — change the imports and `include_router` calls:

```python
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
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_stt_api.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/api/stt.py backend/voice/api/app.py backend/tests/test_stt_api.py
git commit -m "Add POST /v1/transcribe endpoint with size limit and language hint"
```

---

## Task 13: TTS endpoint (streaming)

**Files:**
- Create: `backend/voice/api/tts.py`
- Modify: `backend/voice/api/app.py` (register router)
- Create: `backend/tests/test_tts_api.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_tts_api.py`:

```python
"""Tests for POST /v1/speak — discriminated union, streaming WAV response."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from tests.conftest import FakeTTSModel


class _Registry:
    def __init__(self, enabled, policy, models):
        self._enabled = tuple(enabled)
        self._policy = policy
        self._models = models

    @property
    def enabled_modes(self):
        return self._enabled

    @property
    def policy(self):
        return self._policy

    def loaded_modes(self):
        return tuple(self._models.keys())

    def acquire(self, mode):
        """Returns an async context manager that yields the model for `mode`.

        Matches the real TTSModelRegistry shape: the handler is responsible for
        the disabled-mode pre-check against `enabled_modes`, so acquire() only
        needs to yield an already-enabled model.
        """

        class _CM:
            def __init__(s, model):
                s._model = model
            async def __aenter__(s):
                return s._model
            async def __aexit__(s, *exc):
                return None

        return _CM(self._models[mode])


class _Settings:
    stt_max_audio_bytes = 10_000_000


@pytest.mark.asyncio
async def test_speak_custom_voice_happy_path():
    from voice.api.app import build_app

    samples = np.linspace(-0.5, 0.5, 8192, dtype=np.float32)
    fake = FakeTTSModel(mode="custom_voice", samples=samples, stream_chunk_size=4096)
    registry = _Registry(["custom_voice"], "keep_loaded", {"custom_voice": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {
            "mode": "custom_voice",
            "text": "hallo",
            "language": "German",
            "speaker": "Vivian",
            "instruct": "fröhlich",
        }
        r = await client.post("/v1/speak", json=body)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    data = r.content
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[40:44])[0] == 0xFFFFFFFF
    # WAV header (44) + PCM16 data (8192 samples * 2)
    assert len(data) == 44 + 8192 * 2
    assert fake.calls[-1]["speaker"] == "Vivian"
    assert fake.calls[-1]["instruct"] == "fröhlich"


@pytest.mark.asyncio
async def test_speak_voice_design_happy_path():
    from voice.api.app import build_app

    samples = np.zeros(4096, dtype=np.float32)
    fake = FakeTTSModel(mode="voice_design", samples=samples, stream_chunk_size=4096)
    registry = _Registry(["voice_design"], "keep_loaded", {"voice_design": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {
            "mode": "voice_design",
            "text": "hi",
            "language": "English",
            "voice_prompt": "warm low voice",
        }
        r = await client.post("/v1/speak", json=body)

    assert r.status_code == 200
    assert fake.calls[-1]["voice_prompt"] == "warm low voice"


@pytest.mark.asyncio
async def test_speak_mode_disabled_returns_403():
    from voice.api.app import build_app

    registry = _Registry(["voice_design"], "keep_loaded", {})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {"mode": "custom_voice", "text": "hi", "language": "English", "speaker": "Ryan"}
        r = await client.post("/v1/speak", json=body)

    assert r.status_code == 403
    assert r.json()["error"] == "mode_disabled"


@pytest.mark.asyncio
async def test_speak_invalid_body_returns_422():
    from voice.api.app import build_app

    registry = _Registry(["custom_voice"], "keep_loaded", {"custom_voice": FakeTTSModel()})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing speaker for custom_voice
        body = {"mode": "custom_voice", "text": "hi", "language": "English"}
        r = await client.post("/v1/speak", json=body)

    assert r.status_code == 422


@pytest.mark.asyncio
async def test_speak_mid_stream_error_closes_cleanly():
    from voice.api.app import build_app

    samples = np.zeros(12288, dtype=np.float32)
    fake = FakeTTSModel(
        mode="custom_voice",
        samples=samples,
        stream_chunk_size=4096,
        raise_mid_stream_after=1,
    )
    registry = _Registry(["custom_voice"], "keep_loaded", {"custom_voice": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = {"mode": "custom_voice", "text": "x", "language": "English", "speaker": "Ryan"}
        r = await client.post("/v1/speak", json=body)

    # Headers are committed before the stream begins, so status is still 200.
    assert r.status_code == 200
    # But response is truncated — at most header + one chunk.
    assert len(r.content) < 44 + 12288 * 2
```

- [ ] **Step 2: Write `api/tts.py`**

Create `backend/voice/api/tts.py`:

```python
"""POST /v1/speak — streaming WAV response, discriminated request union."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from voice.api.models import SpeakRequest
from voice.audio import float32_to_pcm16, make_streaming_wav_header
from voice.engines.protocol import TTSRequest
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


@router.post(
    "/speak",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "PCM16 WAV stream"},
        403: {"description": "mode disabled"},
        422: {"description": "validation error"},
    },
)
async def speak(request: Request, payload: SpeakRequest) -> StreamingResponse:
    registry = request.app.state.registry

    # Synchronously pre-check the enabled set so the 403 is raised before any
    # streaming response is committed. The registry's acquire() is an async
    # context manager; by the time __aenter__ runs, headers are already gone.
    if payload.mode not in registry.enabled_modes:
        raise HTTPException(
            status_code=403,
            detail={"error": "mode_disabled", "mode": payload.mode},
        )

    cm = registry.acquire(payload.mode)

    tts_req = TTSRequest(
        mode=payload.mode,
        text=payload.text,
        language=payload.language,
        speaker=getattr(payload, "speaker", None),
        voice_prompt=getattr(payload, "voice_prompt", None),
        instruct=payload.instruct,
    )

    log.info(
        "speak_request",
        mode=payload.mode,
        language=payload.language,
        text_len=len(payload.text),
        speaker=getattr(payload, "speaker", None),
        has_voice_prompt=getattr(payload, "voice_prompt", None) is not None,
        has_instruct=payload.instruct is not None,
    )

    async def body() -> AsyncIterator[bytes]:
        sample_rate = 22050
        first_chunk_time: float | None = None
        total_samples = 0
        start = time.monotonic()
        try:
            async with cm as model:
                sample_rate = model.sample_rate
                yield make_streaming_wav_header(sample_rate=sample_rate, channels=1)
                async for chunk in model.stream(tts_req):
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                        log.info(
                            "speak_stream_start",
                            mode=payload.mode,
                            time_to_first_chunk_ms=int((first_chunk_time - start) * 1000),
                        )
                    total_samples += len(chunk)
                    yield float32_to_pcm16(chunk)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "speak_error",
                mode=payload.mode,
                phase="during_stream" if first_chunk_time else "before_stream",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return
        log.info(
            "speak_stream_end",
            mode=payload.mode,
            total_samples=total_samples,
            total_ms=int((time.monotonic() - start) * 1000),
        )

    return StreamingResponse(body(), media_type="audio/wav")
```

- [ ] **Step 3: Register TTS router in `app.py`**

Edit `backend/voice/api/app.py` — add TTS import and include:

```python
"""FastAPI app factory. Wires the transport layer to the engines."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from voice.api import health as health_module
from voice.api import stt as stt_module
from voice.api import tts as tts_module


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)
    app.include_router(tts_module.router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": str(exc.errors())},
        )

    return app
```

- [ ] **Step 4: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_tts_api.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/api/tts.py backend/voice/api/app.py backend/tests/test_tts_api.py
git commit -m "Add POST /v1/speak streaming endpoint with discriminated union body"
```

---

## Task 14: Request-ID middleware and unhandled-error handler

**Files:**
- Modify: `backend/voice/api/app.py`

- [ ] **Step 1: Extend `api/app.py`**

Replace `backend/voice/api/app.py` with the full version:

```python
"""FastAPI app factory. Wires the transport layer to the engines.

Also installs:
- a request-id middleware that binds a UUID4 to structlog contextvars for every request;
- a catch-all exception handler that logs unhandled_error and returns HTTP 500
  without a traceback in the response body.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from voice.api import health as health_module
from voice.api import stt as stt_module
from voice.api import tts as tts_module
from voice.logging_setup import get_logger

log = get_logger(__name__)


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)
    app.include_router(tts_module.router)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception):
        log.error("unhandled_error", error_type=type(exc).__name__, message=str(exc),
                  exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error"},
        )

    return app
```

- [ ] **Step 2: Run all API tests to verify middleware doesn't break existing behaviour**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_health_api.py tests/test_stt_api.py tests/test_tts_api.py -v
```
Expected: all still pass.

- [ ] **Step 3: Add a dedicated middleware test**

Append to `backend/tests/test_health_api.py`:

```python


@pytest.mark.asyncio
async def test_request_id_header_echoed_and_generated():
    from voice.api.app import build_app

    class _Reg:
        enabled_modes = ()
        policy = "keep_loaded"
        def loaded_modes(self):
            return ()

    app = build_app(stt=_StubSTT(), registry=_Reg(), settings=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.get("/healthz")
        assert "x-request-id" in r1.headers
        assert len(r1.headers["x-request-id"]) > 0

        r2 = await client.get("/healthz", headers={"x-request-id": "test-123"})
        assert r2.headers["x-request-id"] == "test-123"
```

- [ ] **Step 4: Run the new test**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_health_api.py::test_request_id_header_echoed_and_generated -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/api/app.py backend/tests/test_health_api.py
git commit -m "Add request-id middleware and catch-all error handler"
```

---

## Task 15: Main entry point

**Files:**
- Create: `backend/voice/main.py`
- Create: `backend/tests/test_main_startup.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_main_startup.py`:

```python
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
    await registry.preload()
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
        await registry.preload()
```

- [ ] **Step 2: Write `main.py`**

Create `backend/voice/main.py`:

```python
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
                device=settings.device,
                attention_impl=settings.tts_attention_impl,
            )
            return QwenCustomVoiceModel(backend=backend)
        if mode == "voice_design":
            backend = load_qwen_tts(
                settings.tts_voice_design_model,
                device=settings.device,
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
        device=settings.device,
        download_root=None,
    )
    engine = WhisperEngine(model_name=settings.stt_model, model=backend)
    engine.loaded = True  # type: ignore[attr-defined]
    get_logger(__name__).info(
        "stt_model_loaded",
        model=settings.stt_model,
        load_ms=int((time.monotonic() - t0) * 1000),
    )
    return engine


async def _async_bootstrap(settings: Settings):
    from voice.api.app import build_app

    if settings.preload_at_startup:
        stt = _default_stt(settings)
    else:
        stt = _LazySTT(settings)

    registry = build_registry(settings, tts_loader=_default_tts_loader(settings))
    if settings.preload_at_startup:
        await registry.preload()

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
    app, _registry = asyncio.run(_async_bootstrap(settings))
    return app


def run() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)
    log.info(
        "app_starting",
        device=settings.device,
        tts_vram_policy=settings.tts_vram_policy,
        tts_enabled_modes=list(settings.tts_enabled_modes),
        preload_at_startup=settings.preload_at_startup,
    )

    apply_hf_home(settings.model_cache_dir)

    try:
        app, registry = asyncio.run(_async_bootstrap(settings))
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
```

- [ ] **Step 3: Run tests — verify pass**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest tests/test_main_startup.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/voice/main.py backend/tests/test_main_startup.py
git commit -m "Add main entry point with HF_HOME setup and lazy STT loader"
```

---

## Task 16: Static tinker page — HTML, JS, CSS

**Files:**
- Create: `backend/static/index.html`
- Create: `backend/static/app.js`
- Create: `backend/static/style.css`
- Modify: `backend/voice/api/app.py` (mount static directory)
- Create: `backend/tests/test_integration_smoke.py` (root / route test)

- [ ] **Step 1: Write `backend/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>chatsune-voice — Bastelstube</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>chatsune-voice</h1>
    <p class="tagline">Bastelstube — local STT + TTS experimentation</p>
  </header>

  <section id="stt">
    <h2>Speech-to-Text</h2>
    <div class="controls">
      <button id="stt-record">Record</button>
      <button id="stt-stop" disabled>Stop</button>
      <input type="file" id="stt-file" accept="audio/*">
      <label>Language: <select id="stt-language">
        <option value="">auto</option>
        <option value="de">de</option>
        <option value="en">en</option>
        <option value="fr">fr</option>
        <option value="it">it</option>
        <option value="es">es</option>
      </select></label>
      <label><input type="checkbox" id="stt-vad" checked> VAD</label>
    </div>
    <div class="status" id="stt-status">idle</div>
    <textarea id="stt-output" rows="6" placeholder="transcript appears here"></textarea>
    <div class="meta" id="stt-meta"></div>
  </section>

  <section id="tts">
    <h2>Text-to-Speech</h2>
    <div class="mode-tabs">
      <label><input type="radio" name="tts-mode" value="custom_voice" checked> CustomVoice</label>
      <label><input type="radio" name="tts-mode" value="voice_design"> VoiceDesign</label>
    </div>

    <div id="tts-custom-voice" class="tts-panel">
      <label>Text: <textarea id="cv-text" rows="3"></textarea></label>
      <label>Language: <select id="cv-language">
        <option>German</option><option>English</option><option>French</option>
        <option>Italian</option><option>Spanish</option><option>Portuguese</option>
        <option>Russian</option><option>Chinese</option><option>Japanese</option>
        <option>Korean</option><option>Auto</option>
      </select></label>
      <label>Speaker: <select id="cv-speaker">
        <option>Vivian</option><option>Serena</option><option>Uncle_Fu</option>
        <option>Dylan</option><option>Eric</option>
        <option>Ryan</option><option>Aiden</option>
        <option>Ono_Anna</option><option>Sohee</option>
      </select></label>
      <label>Instruct (optional): <input type="text" id="cv-instruct"></label>
      <button id="cv-speak">Speak</button>
    </div>

    <div id="tts-voice-design" class="tts-panel hidden">
      <label>Text: <textarea id="vd-text" rows="3"></textarea></label>
      <label>Language: <select id="vd-language">
        <option>German</option><option>English</option><option>French</option>
        <option>Italian</option><option>Spanish</option><option>Portuguese</option>
        <option>Russian</option><option>Chinese</option><option>Japanese</option>
        <option>Korean</option><option>Auto</option>
      </select></label>
      <label>Voice prompt: <textarea id="vd-voice-prompt" rows="3"
        placeholder="e.g. warme tiefe Männerstimme mit leichter Rauhigkeit"></textarea></label>
      <label>Instruct (optional): <input type="text" id="vd-instruct"></label>
      <button id="vd-speak">Speak</button>
    </div>

    <audio id="tts-audio" controls></audio>
    <a id="tts-download" href="#" download="speech.wav" class="hidden">Download WAV</a>
    <div class="status" id="tts-status">idle</div>
  </section>

  <section id="round-trip">
    <h2>Round-trip (record, transcribe, speak back)</h2>
    <button id="rt-run">Record + transcribe + speak</button>
    <div class="status" id="rt-status">idle</div>
  </section>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `backend/static/app.js`**

```javascript
"use strict";

const sttStatus = document.getElementById("stt-status");
const sttOutput = document.getElementById("stt-output");
const sttMeta = document.getElementById("stt-meta");
const ttsStatus = document.getElementById("tts-status");
const ttsAudio = document.getElementById("tts-audio");
const ttsDownload = document.getElementById("tts-download");

let mediaRecorder = null;
let recordedChunks = [];

document.getElementById("stt-record").addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => stream.getTracks().forEach(t => t.stop());
  mediaRecorder.start();
  sttStatus.textContent = "recording…";
  document.getElementById("stt-stop").disabled = false;
  document.getElementById("stt-record").disabled = true;
});

document.getElementById("stt-stop").addEventListener("click", async () => {
  mediaRecorder.stop();
  document.getElementById("stt-stop").disabled = true;
  document.getElementById("stt-record").disabled = false;
  sttStatus.textContent = "transcribing…";
  await new Promise(r => setTimeout(r, 100));
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  await transcribeBlob(blob);
});

document.getElementById("stt-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  sttStatus.textContent = "transcribing…";
  await transcribeBlob(file);
});

async function transcribeBlob(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  const lang = document.getElementById("stt-language").value;
  if (lang) fd.append("language", lang);
  fd.append("vad", document.getElementById("stt-vad").checked ? "true" : "false");
  const t0 = performance.now();
  const r = await fetch("/v1/transcribe", { method: "POST", body: fd });
  if (!r.ok) {
    sttStatus.textContent = "error: " + r.status;
    return;
  }
  const body = await r.json();
  const elapsed = Math.round(performance.now() - t0);
  sttOutput.value = body.text;
  sttMeta.textContent = `lang: ${body.language} (p=${body.language_probability.toFixed(2)}) · duration: ${body.duration.toFixed(2)}s · client rtt: ${elapsed}ms`;
  sttStatus.textContent = "done";
}

document.querySelectorAll('input[name="tts-mode"]').forEach(el => {
  el.addEventListener("change", () => {
    document.getElementById("tts-custom-voice").classList.toggle("hidden",
      el.value !== "custom_voice" || !el.checked);
    document.getElementById("tts-voice-design").classList.toggle("hidden",
      el.value !== "voice_design" || !el.checked);
  });
});

document.getElementById("cv-speak").addEventListener("click", async () => {
  await speak({
    mode: "custom_voice",
    text: document.getElementById("cv-text").value,
    language: document.getElementById("cv-language").value,
    speaker: document.getElementById("cv-speaker").value,
    instruct: document.getElementById("cv-instruct").value || null,
  });
});

document.getElementById("vd-speak").addEventListener("click", async () => {
  await speak({
    mode: "voice_design",
    text: document.getElementById("vd-text").value,
    language: document.getElementById("vd-language").value,
    voice_prompt: document.getElementById("vd-voice-prompt").value,
    instruct: document.getElementById("vd-instruct").value || null,
  });
});

async function speak(body) {
  ttsStatus.textContent = "synthesising…";
  ttsDownload.classList.add("hidden");
  const r = await fetch("/v1/speak", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    ttsStatus.textContent = "error: " + r.status;
    return;
  }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  ttsAudio.src = url;
  ttsAudio.play();
  ttsDownload.href = url;
  ttsDownload.classList.remove("hidden");
  ttsStatus.textContent = "playing";
}

document.getElementById("rt-run").addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recordedChunks.push(e.data); };
  document.getElementById("rt-status").textContent = "recording 3s…";
  mediaRecorder.start();
  await new Promise(r => setTimeout(r, 3000));
  mediaRecorder.stop();
  stream.getTracks().forEach(t => t.stop());
  await new Promise(r => setTimeout(r, 200));
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  document.getElementById("rt-status").textContent = "transcribing…";
  await transcribeBlob(blob);
  const mode = document.querySelector('input[name="tts-mode"]:checked').value;
  if (mode === "custom_voice") {
    document.getElementById("cv-text").value = sttOutput.value;
    document.getElementById("rt-status").textContent = "speaking back…";
    document.getElementById("cv-speak").click();
  } else {
    document.getElementById("vd-text").value = sttOutput.value;
    document.getElementById("rt-status").textContent = "speaking back…";
    document.getElementById("vd-speak").click();
  }
});
```

- [ ] **Step 3: Write `backend/static/style.css`**

```css
:root {
  --bg: #14151a;
  --fg: #e6e6e6;
  --muted: #8b8e99;
  --accent: #6ab0ff;
  --panel: #1d1f27;
  --border: #2a2d36;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--fg);
  padding: 2rem;
  max-width: 960px;
  margin-inline: auto;
}
header { border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
header h1 { margin: 0; }
header .tagline { color: var(--muted); margin: 0.25rem 0 1rem; }
section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}
section h2 { margin-top: 0; color: var(--accent); }
label { display: block; margin: 0.5rem 0; }
textarea, input[type="text"], select {
  background: #0f1017;
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
  font-size: 0.95rem;
  font-family: inherit;
  width: 100%;
}
textarea { resize: vertical; }
button {
  background: var(--accent);
  color: #0b0c10;
  border: 0;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-weight: 600;
  margin-right: 0.5rem;
}
button:disabled { opacity: 0.4; cursor: not-allowed; }
.controls { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-bottom: 0.5rem; }
.status { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
.meta { color: var(--muted); font-size: 0.8rem; margin-top: 0.4rem; }
.mode-tabs { margin-bottom: 0.75rem; }
.mode-tabs label { display: inline-block; margin-right: 1rem; }
.tts-panel { border-top: 1px dashed var(--border); padding-top: 0.75rem; }
.hidden { display: none; }
audio { width: 100%; margin-top: 0.75rem; }
a#tts-download { color: var(--accent); font-size: 0.85rem; }
```

- [ ] **Step 4: Mount static directory in `app.py`**

Replace `backend/voice/api/app.py` with the complete final version (supersedes the version from Task 14):

```python
"""FastAPI app factory. Wires the transport layer to the engines.

Also installs:
- a request-id middleware that binds a UUID4 to structlog contextvars for every request;
- a catch-all exception handler that logs unhandled_error and returns HTTP 500
  without a traceback in the response body;
- a static mount at / that serves the browser-based tinker page.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from voice.api import health as health_module
from voice.api import stt as stt_module
from voice.api import tts as tts_module
from voice.logging_setup import get_logger

log = get_logger(__name__)


def build_app(*, stt: Any, registry: Any, settings: Any) -> FastAPI:
    app = FastAPI(title="chatsune-voice", version="0.1.0")
    app.state.stt = stt
    app.state.registry = registry
    app.state.settings = settings

    app.include_router(health_module.router)
    app.include_router(stt_module.router)
    app.include_router(tts_module.router)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "message": str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception):
        log.error("unhandled_error", error_type=type(exc).__name__, message=str(exc),
                  exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error"},
        )

    # Static mount must come LAST so API routes take precedence.
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
```

- [ ] **Step 5: Write integration smoke test**

Create `backend/tests/test_integration_smoke.py`:

```python
"""End-to-end smoke test with fake engines."""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from tests.conftest import FakeTTSModel


class _Reg:
    enabled_modes = ()
    policy = "keep_loaded"
    def loaded_modes(self):
        return ()

    def acquire(self, mode):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_root_serves_index_html(fake_stt):
    from voice.api.app import build_app

    app = build_app(stt=fake_stt, registry=_Reg(), settings=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "chatsune-voice" in r.text


@pytest.mark.asyncio
async def test_healthz_under_integration(fake_stt):
    from voice.api.app import build_app

    app = build_app(stt=fake_stt, registry=_Reg(), settings=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
```

Note: `fake_stt` already has `loaded: bool = True` (added in Task 2) so no fixture changes are needed for the health endpoint test to work.

- [ ] **Step 6: Run all tests**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest -v
```
Expected: everything passes, including the two new smoke tests.

- [ ] **Step 7: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/static/ backend/voice/api/app.py backend/tests/test_integration_smoke.py
git commit -m "Add static tinker page and integration smoke test"
```

---

## Task 17: Prefetch script

**Files:**
- Create: `backend/scripts/__init__.py`
- Create: `backend/scripts/prefetch_models.py`

- [ ] **Step 1: Write `scripts/__init__.py`**

Empty file: `backend/scripts/__init__.py`

- [ ] **Step 2: Write `scripts/prefetch_models.py`**

```python
"""Download all three model checkpoints into the configured HF cache.

Run once before the first `docker compose up` to avoid long first-request delays.

    uv run python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
import time

from voice.config import Settings
from voice.logging_setup import configure_logging, get_logger
from voice.main import apply_hf_home


def _download(hf_id: str, log) -> None:
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    path = snapshot_download(repo_id=hf_id)
    log.info("model_downloaded", model=hf_id, path=path,
             elapsed_ms=int((time.monotonic() - t0) * 1000))


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)

    apply_hf_home(settings.model_cache_dir)

    ids = [settings.stt_model, settings.tts_custom_voice_model, settings.tts_voice_design_model]
    log.info("prefetch_starting", cache_dir=str(settings.model_cache_dir), models=ids)

    for hf_id in ids:
        try:
            _download(hf_id, log)
        except Exception as exc:  # noqa: BLE001
            log.error("prefetch_failed", model=hf_id, error_type=type(exc).__name__,
                      message=str(exc))
            return 1

    log.info("prefetch_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Syntax check**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run python -c "import ast; ast.parse(open('scripts/prefetch_models.py').read()); print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add backend/scripts/
git commit -m "Add prefetch_models.py to warm the Hugging Face cache"
```

---

## Task 18: Dockerfile.cuda

**Files:**
- Create: `Dockerfile.cuda`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

Create `.dockerignore`:

```
**/__pycache__
**/*.pyc
**/.pytest_cache
**/.ruff_cache
**/.venv
**/.uv
.git
.github
docs
obsidian
models
.model-cache
.env
.env.*
!.env.example
node_modules
.idea
.vscode
```

- [ ] **Step 2: Write `Dockerfile.cuda`**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency layer (cached unless pyproject or lock change)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source
COPY backend/ ./

# Optional flash-attn (enable with --build-arg INSTALL_FLASH_ATTN=1)
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      uv pip install flash-attn --no-build-isolation ; \
    fi

# Re-sync to install project itself
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Syntax check (Dockerfile linter optional; skip if not installed)**

```bash
cd /home/chris/workspace/chatsune-voice
test -f Dockerfile.cuda && echo "Dockerfile.cuda present"
```
Expected: `Dockerfile.cuda present`.

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add Dockerfile.cuda .dockerignore
git commit -m "Add CUDA Dockerfile with optional flash-attn build arg"
```

---

## Task 19: Dockerfile.rocm

**Files:**
- Create: `Dockerfile.rocm`

- [ ] **Step 1: Write `Dockerfile.rocm`**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM rocm/pytorch:rocm6.2.1_ubuntu22.04_py3.12_pytorch_2.5.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY backend/pyproject.toml backend/uv.lock ./
# The base image already ships a ROCm-built torch; do not replace it.
RUN uv sync --frozen --no-dev --no-install-project --no-install-package torch

COPY backend/ ./

RUN uv sync --frozen --no-dev --no-install-package torch

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add Dockerfile.rocm
git commit -m "Add ROCm Dockerfile preserving base image torch"
```

---

## Task 20: compose.yml and .env.example

**Files:**
- Create: `compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Write `compose.yml`**

```yaml
x-voice-common: &voice-common
  environment: &voice-env
    CHATSUNE_VOICE_MODEL_CACHE_DIR: /models
    STT_MODEL: ${STT_MODEL:-Systran/faster-whisper-large-v3-turbo}
    TTS_CUSTOM_VOICE_MODEL: ${TTS_CUSTOM_VOICE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
    TTS_VOICE_DESIGN_MODEL: ${TTS_VOICE_DESIGN_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}
    TTS_ENABLED_MODES: ${TTS_ENABLED_MODES:-custom_voice,voice_design}
    TTS_VRAM_POLICY: ${TTS_VRAM_POLICY:-keep_loaded}
    TTS_ATTENTION_IMPL: ${TTS_ATTENTION_IMPL:-sdpa}
    PRELOAD_AT_STARTUP: ${PRELOAD_AT_STARTUP:-true}
    DEVICE: ${DEVICE:-cuda}
    LOG_LEVEL: ${LOG_LEVEL:-info}
    STT_MAX_AUDIO_BYTES: ${STT_MAX_AUDIO_BYTES:-26214400}
    APP_PORT: ${APP_PORT:-8000}
  volumes:
    - ${MODEL_CACHE_DIR:-./models}:/models
  ports:
    - "${APP_PORT:-8000}:8000"
  restart: unless-stopped
  healthcheck:
    test: ["CMD-SHELL", "curl -fsS http://localhost:8000/healthz || exit 1"]
    interval: 15s
    timeout: 5s
    retries: 5
    start_period: 120s

services:
  voice-cuda:
    <<: *voice-common
    profiles: [cuda]
    image: ${APP_IMAGE:-ghcr.io/symphonic-navigator/chatsune-voice:cuda-latest}
    runtime: nvidia
    environment:
      <<: *voice-env
      NVIDIA_VISIBLE_DEVICES: all

  voice-rocm:
    <<: *voice-common
    profiles: [rocm]
    image: ${APP_IMAGE:-ghcr.io/symphonic-navigator/chatsune-voice:rocm-latest}
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    environment:
      <<: *voice-env
      HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-11.5.1}
```

- [ ] **Step 2: Write `.env.example`**

```env
# ===== Hardware profile =====
# Set one of: cuda (NVIDIA + nvidia-container-toolkit), rocm (AMD incl. Strix Halo).
# You MUST set this; otherwise `docker compose up` will not start any service.
COMPOSE_PROFILES=cuda

# Image tag. Published by GitHub Actions to both :cuda-latest and :rocm-latest.
# APP_IMAGE=ghcr.io/symphonic-navigator/chatsune-voice:cuda-latest

# ===== Paths =====
MODEL_CACHE_DIR=./models
APP_PORT=8000

# ===== Model selection (defaults are usually correct) =====
# STT_MODEL=Systran/faster-whisper-large-v3-turbo
# TTS_CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# TTS_VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# ===== Runtime behaviour =====
# Subset allowed, e.g. "custom_voice" for smaller GPUs.
TTS_ENABLED_MODES=custom_voice,voice_design

# keep_loaded: all enabled TTS models resident simultaneously (recommended >= 24 GB VRAM).
# swap:        one TTS model at a time; mode switches are serialised via an asyncio lock.
TTS_VRAM_POLICY=keep_loaded

# PyTorch attention: sdpa | flash_attention_2 | eager. flash_attention_2 needs --build-arg
# INSTALL_FLASH_ATTN=1 at image build time and is CUDA-only.
TTS_ATTENTION_IMPL=sdpa

# Load models at start-up (true) or lazily on first request (false).
PRELOAD_AT_STARTUP=true

# ===== ROCm-specific (only used when COMPOSE_PROFILES=rocm) =====
# Strix Halo iGPU (gfx1151) typically "11.5.1"; some discrete RDNA3 cards do not need this.
HSA_OVERRIDE_GFX_VERSION=11.5.1

# ===== Miscellaneous =====
LOG_LEVEL=info
STT_MAX_AUDIO_BYTES=26214400
```

- [ ] **Step 3: Validate compose syntax**

```bash
cd /home/chris/workspace/chatsune-voice
COMPOSE_PROFILES=cuda docker compose config >/dev/null
```
Expected: no errors (ignore warnings about env file missing; set `--env-file .env.example` if needed).

Alternatively:
```bash
cd /home/chris/workspace/chatsune-voice
docker compose --env-file .env.example config >/dev/null
```

- [ ] **Step 4: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add compose.yml .env.example
git commit -m "Add single compose.yml with cuda and rocm profiles"
```

---

## Task 21: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.4.x"

      - name: Set up Python
        run: uv python install 3.12

      - name: Sync dependencies
        run: uv sync --frozen --dev

      - name: Lint
        run: uv run ruff check .

      - name: Test
        run: uv run pytest -v
```

- [ ] **Step 2: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add .github/workflows/ci.yml
git commit -m "Add CI workflow running ruff and pytest on push and pull request"
```

---

## Task 22: README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# chatsune-voice

Voice homelab backend for [chatsune](https://github.com/symphonic-navigator). Runs Whisper Turbo for speech-to-text and Qwen3-TTS (both CustomVoice and VoiceDesign variants) for text-to-speech, exposed as a FastAPI REST service with a small browser-based tinker page.

## Current status

**Phase 1 — local experimentation tool.** The project runs on a single machine, accepts REST requests, and serves a static HTML page at `/`. Subsequent phases will extend it:

- **Phase 2** — WebSocket tunnel to the chatsune backend (analogous to `chatsune-ollama-sidecar`).
- **Phase 3** — prosumer-oriented packaging so end users can contribute their homelab compute to chatsune.

## Requirements

- GNU/Linux host with Docker and Docker Compose.
- One of:
  - **NVIDIA GPU** with `nvidia-container-toolkit` installed (CUDA 12.4+), or
  - **AMD GPU** supported by ROCm 6.2+ (including the Strix Halo iGPU).
- At least ~8 GB VRAM for `TTS_VRAM_POLICY=swap`, ~24 GB for `keep_loaded` with both TTS modes enabled.
- ~5 GB disk for the model cache.

## Quick start

```bash
cp .env.example .env
# Edit .env: set COMPOSE_PROFILES to `cuda` or `rocm`.
$EDITOR .env

docker compose up -d
```

Then open <http://localhost:8000> for the tinker page, or <http://localhost:8000/docs> for the OpenAPI schema.

On first start the container downloads the three Hugging Face checkpoints (~5 GB total) into `./models/`. Subsequent starts reuse that cache.

Optional: pre-warm the cache before the first container start:

```bash
cd backend
uv sync --dev
uv run python scripts/prefetch_models.py
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list with defaults. The most relevant knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `COMPOSE_PROFILES` | (must set) | `cuda` or `rocm`. Selects the corresponding service and base image. |
| `CHATSUNE_VOICE_MODEL_CACHE_DIR` | `/models` (inside container) | Single cache directory shared by all three models; mapped to `HF_HOME` at start-up. |
| `TTS_ENABLED_MODES` | `custom_voice,voice_design` | Which TTS modes the API exposes. A subset is allowed. |
| `TTS_VRAM_POLICY` | `keep_loaded` | `keep_loaded` keeps all enabled TTS models resident. `swap` keeps at most one resident at a time, serialising mode switches via an `asyncio.Lock`. |
| `TTS_ATTENTION_IMPL` | `sdpa` | `sdpa` (PyTorch default, works everywhere), `flash_attention_2` (CUDA-only, requires `INSTALL_FLASH_ATTN=1` build arg), or `eager`. |
| `PRELOAD_AT_STARTUP` | `true` | Load models during start-up. Set to `false` to defer loading until the first request. |
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | ROCm-only. `11.5.1` suits Strix Halo (gfx1151); discrete RDNA3 usually needs no override. |

## VRAM policy guidance

- **24 GB+ VRAM:** `keep_loaded` with both modes enabled. Requests of different modes run in parallel; same-mode requests serialise per model.
- **8–16 GB VRAM:** `swap`. Requests of different modes are serialised via a swap lock; expect a small startup cost on each mode switch.
- **< 8 GB VRAM:** enable only one mode (`TTS_ENABLED_MODES=custom_voice` or `voice_design`).

STT (Whisper Turbo) is always resident once loaded and does not participate in the TTS VRAM policy.

## API

Endpoints (full schema at `/docs`):

- `GET /healthz` — readiness. 200 when every enabled model is loaded (or loadable under `swap`); 503 otherwise.
- `POST /v1/transcribe` — multipart upload, fields `audio` (file), `language` (optional ISO code), `vad` (optional bool). Returns JSON with text, language, duration, and segments.
- `POST /v1/speak` — JSON body with discriminator `mode` (`custom_voice` or `voice_design`). Returns `audio/wav` as a chunked HTTP response.

## Development

```bash
cd backend
uv sync --dev
uv run uvicorn voice.main:app --reload
# In another shell:
uv run ruff check .
uv run pytest -v
```

Tests use in-memory fakes for Whisper and Qwen3-TTS; no GPU or network access is required.

## Licence

GPL-3.0-or-later. See `LICENCE`.
```

- [ ] **Step 2: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add README.md
git commit -m "Rewrite README with project description, quick start, and configuration table"
```

---

## Task 23: Obsidian vault skeleton

**Files:**
- Create: `obsidian/.gitkeep`
- Create: `obsidian/.obsidian/app.json`

- [ ] **Step 1: Create Obsidian vault directory**

```bash
cd /home/chris/workspace/chatsune-voice
mkdir -p obsidian/.obsidian
```

Create `obsidian/.gitkeep` (empty file).

Create `obsidian/.obsidian/app.json`:
```json
{}
```

The `.gitignore` written in Task 1 already excludes `obsidian/.obsidian/workspace*`, `obsidian/.obsidian/cache`, and `obsidian/.trash/`.

- [ ] **Step 2: Commit**

```bash
cd /home/chris/workspace/chatsune-voice
git add obsidian/
git commit -m "Seed empty Obsidian vault with placeholder app config"
```

---

## Task 24: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run pytest -v
```
Expected: all tests pass. If any fail, stop and investigate — do not commit further until green.

- [ ] **Step 2: Run ruff**

```bash
cd /home/chris/workspace/chatsune-voice/backend
uv run ruff check .
```
Expected: no errors.

- [ ] **Step 3: Validate compose**

```bash
cd /home/chris/workspace/chatsune-voice
docker compose --env-file .env.example config >/dev/null
```
Expected: no errors.

- [ ] **Step 4: Confirm structure**

```bash
cd /home/chris/workspace/chatsune-voice
ls -la
ls -la backend/
ls -la backend/voice/
ls -la backend/voice/engines/
ls -la backend/voice/api/
ls -la backend/static/
ls -la backend/tests/
```
Expected: every file listed in the "File Structure" section exists.

- [ ] **Step 5: Final commit marker**

```bash
cd /home/chris/workspace/chatsune-voice
git log --oneline | head -30
```
Expected: 24+ commits, starting from the initial commit through the Phase-1 implementation.

- [ ] **Step 6: Optional — smoke-test with real models on Chris's machine**

(Only on the target machine.)

```bash
cd /home/chris/workspace/chatsune-voice
docker compose build voice-rocm   # or voice-cuda
docker compose --env-file .env up -d
docker compose logs -f
```

Open <http://localhost:8000>, record a few seconds of speech, observe transcription, type some text and listen to the synthesised output in both CustomVoice and VoiceDesign modes. Check `/healthz` returns 200.

Phase 1 is complete.
