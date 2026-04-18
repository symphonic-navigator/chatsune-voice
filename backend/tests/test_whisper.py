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
