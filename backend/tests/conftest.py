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
    always_resident: bool = False
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
            "reference_audio_len": len(req.reference_audio) if req.reference_audio else 0,
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "temperature": req.temperature,
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
