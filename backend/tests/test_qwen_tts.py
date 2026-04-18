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
