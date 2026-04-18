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
