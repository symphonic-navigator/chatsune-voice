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


def test_prepare_language_prepends_language_token():
    from voice.engines.chatterbox_tts import prepare_language

    assert prepare_language("Hallo", "de") == "[de]Hallo"
    assert prepare_language("Bonjour", "fr") == "[fr]Bonjour"


def test_prepare_language_korean_jamo_decomposition():
    """Hangul syllable '안' (0xC548) should decompose to ᄋ (0x110B) + ᅡ (0x1161) + ᆫ (0x11AB)."""
    from voice.engines.chatterbox_tts import prepare_language

    result = prepare_language("안", "ko")
    expected = "[ko]" + chr(0x110B) + chr(0x1161) + chr(0x11AB)
    assert result == expected


def test_prepare_language_korean_passthrough_for_non_hangul():
    """Non-Hangul characters within a ko text are preserved untouched after the prefix."""
    from voice.engines.chatterbox_tts import prepare_language

    assert prepare_language("Hello!", "ko") == "[ko]Hello!"


def test_prepare_language_korean_syllable_without_final():
    """Hangul syllable '가' (0xAC00, base case) has no final Jamo."""
    from voice.engines.chatterbox_tts import prepare_language

    result = prepare_language("가", "ko")
    # base=0, initial=0x1100, medial=0x1161, no final
    expected = "[ko]" + chr(0x1100) + chr(0x1161)
    assert result == expected


def test_repetition_penalty_processor_scales_positive_and_negative_correctly():
    """Positive scores at visited ids are divided by penalty; negative ones are multiplied."""
    from voice.engines.chatterbox_tts import repetition_penalty_processor

    # vocab_size=4, already-generated token ids are [1, 2]
    input_ids = np.array([[1, 2]], dtype=np.int64)
    scores = np.array([[1.0, 2.0, -3.0, 4.0]], dtype=np.float32)

    result = repetition_penalty_processor(input_ids, scores, penalty=2.0)

    # Index 0: not in input_ids, unchanged -> 1.0
    # Index 1: positive, in input_ids -> 2.0 / 2.0 = 1.0
    # Index 2: negative, in input_ids -> -3.0 * 2.0 = -6.0
    # Index 3: not in input_ids, unchanged -> 4.0
    np.testing.assert_allclose(result, np.array([[1.0, 1.0, -6.0, 4.0]], dtype=np.float32))


def test_repetition_penalty_processor_leaves_scores_unchanged_when_no_history():
    from voice.engines.chatterbox_tts import repetition_penalty_processor

    input_ids = np.zeros((1, 0), dtype=np.int64)
    scores = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)

    result = repetition_penalty_processor(input_ids, scores, penalty=1.2)

    np.testing.assert_allclose(result, scores)


def test_sample_next_token_zero_temperature_is_greedy():
    """temperature <= 0 falls back to deterministic argmax."""
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[1.0, 5.0, 2.0, 3.0]], dtype=np.float32)

    result = sample_next_token(logits, temperature=0.0)

    assert result.shape == (1, 1)
    assert result.dtype == np.int64
    assert result[0, 0] == 1  # index of max


def test_sample_next_token_positive_temperature_is_stochastic():
    """With temperature=1.0 and varied seeds, at least two distinct tokens appear."""
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[1.0, 1.1, 1.0, 1.05]], dtype=np.float32)

    seen: set[int] = set()
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = sample_next_token(logits, temperature=1.0, rng=rng)
        seen.add(int(result[0, 0]))

    assert len(seen) >= 2, f"Expected stochastic sampling, got single token {seen}"


def test_sample_next_token_returns_int64_shape_1_1():
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    result = sample_next_token(logits, temperature=0.5)

    assert result.shape == (1, 1)
    assert result.dtype == np.int64


def test_onnx_loader_downloads_from_onnx_subfolder(monkeypatch):
    """Each .onnx file is downloaded with subfolder='onnx' and the paired
    .onnx_data sidecar is fetched alongside."""
    from voice.engines import chatterbox_tts

    calls: list[dict] = []

    def fake_download(repo_id, filename, *, subfolder=None):
        calls.append({"filename": filename, "subfolder": subfolder})
        return f"/fake/{subfolder}/{filename}"

    class FakeSession:
        def __init__(self, path, providers=None):
            self.path = path
            self.providers = providers

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            return object()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    monkeypatch.setattr(
        "onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    chatterbox_tts.load_chatterbox_onnx("fake/repo", device="cpu")

    filenames = [c["filename"] for c in calls]
    subfolders = {c["subfolder"] for c in calls}

    for base in ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder"):
        assert f"{base}.onnx" in filenames
        assert f"{base}.onnx_data" in filenames
    assert subfolders == {"onnx"}
