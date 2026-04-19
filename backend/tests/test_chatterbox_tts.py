"""Tests for voice.engines.chatterbox_tts — adapter logic and language mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from voice.engines.protocol import TTSRequest


@dataclass
class _FakeChatterboxBackend:
    samples: np.ndarray = field(
        default_factory=lambda: np.linspace(-0.3, 0.3, 8000, dtype=np.float32)
    )
    sample_rate: int = 24000
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        text: str,
        language: str,
        reference_audio: bytes,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
    ) -> tuple[np.ndarray, int]:
        self.calls.append({
            "text": text,
            "language": language,
            "reference_len": len(reference_audio),
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        })
        return self.samples, self.sample_rate


@pytest.mark.asyncio
async def test_clone_streams_samples_in_chunks():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    samples = np.linspace(-0.5, 0.5, 10000, dtype=np.float32)
    backend = _FakeChatterboxBackend(samples=samples, sample_rate=24000)
    model = ChatterboxCloneModel(backend=backend, chunk_size=4096)

    req = TTSRequest(
        mode="clone",
        text="Bonjour",
        language="French",
        reference_audio=b"fake-reference-wav",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )
    chunks = [c async for c in model.stream(req)]

    assert sum(len(c) for c in chunks) == 10000
    assert all(c.dtype == np.float32 for c in chunks)
    assert model.sample_rate == 24000
    call = backend.calls[-1]
    assert call["language"] == "fr"  # Mapped from "French"
    assert call["text"] == "Bonjour"
    assert call["reference_len"] == len(b"fake-reference-wav")
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


@pytest.mark.asyncio
async def test_clone_requires_reference_audio():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(mode="clone", text="hi", language="English")

    with pytest.raises(ValueError, match="reference_audio"):
        async for _ in model.stream(req):
            pass


@pytest.mark.asyncio
async def test_clone_rejects_language_auto():
    """Chatterbox requires a concrete language — 'Auto' is not supported."""
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(
        mode="clone",
        text="hi",
        language="Auto",
        reference_audio=b"ref",
    )
    with pytest.raises(ValueError, match="Auto"):
        async for _ in model.stream(req):
            pass


@pytest.mark.asyncio
async def test_clone_uses_defaults_when_params_are_none():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(
        mode="clone",
        text="hi",
        language="German",
        reference_audio=b"ref",
    )
    _ = [c async for c in model.stream(req)]
    call = backend.calls[-1]
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


def test_always_resident_is_true():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel
    assert ChatterboxCloneModel.always_resident is True


def test_language_mapping_covers_all_common_languages():
    from voice.engines.chatterbox_tts import language_to_iso639

    assert language_to_iso639("English") == "en"
    assert language_to_iso639("German") == "de"
    assert language_to_iso639("French") == "fr"
    assert language_to_iso639("Spanish") == "es"
    assert language_to_iso639("Italian") == "it"
    assert language_to_iso639("Portuguese") == "pt"
    assert language_to_iso639("Russian") == "ru"
    assert language_to_iso639("Korean") == "ko"


def test_language_mapping_rejects_auto():
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="Auto"):
        language_to_iso639("Auto")


def test_language_mapping_rejects_unknown():
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="unknown"):
        language_to_iso639("Klingon")


def test_language_mapping_rejects_japanese_and_chinese():
    """Phase 1 drops ja/zh because pkuseg/pykakasi are not installed."""
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="Japanese"):
        language_to_iso639("Japanese")
    with pytest.raises(ValueError, match="Chinese"):
        language_to_iso639("Chinese")


def test_torch_loader_raises_on_missing_chatterbox_package(monkeypatch):
    """If chatterbox is not installed, the loader surfaces a clear error."""
    import sys

    from voice.engines.chatterbox_tts import load_chatterbox_torch

    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", None)
    with pytest.raises((ImportError, AttributeError, TypeError)):
        load_chatterbox_torch("ResembleAI/chatterbox", device="cpu")


def test_onnx_loader_raises_on_missing_onnxruntime(monkeypatch):
    import sys

    from voice.engines.chatterbox_tts import load_chatterbox_onnx

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises((ImportError, AttributeError, TypeError)):
        load_chatterbox_onnx("onnx-community/chatterbox-multilingual-ONNX", device="cpu")
