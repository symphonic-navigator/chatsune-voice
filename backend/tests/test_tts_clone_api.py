"""Tests for POST /v1/speak/clone — multipart, validation, streaming."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from tests.conftest import FakeTTSModel


def _wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Build a valid in-memory WAV file for use as reference audio."""
    samples = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


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
    chatterbox_max_reference_bytes = 10 * 1024 * 1024
    chatterbox_max_reference_seconds = 30.0


@pytest.mark.asyncio
async def test_clone_happy_path():
    from voice.api.app import build_app

    samples = np.linspace(-0.2, 0.2, 8192, dtype=np.float32)
    fake = FakeTTSModel(mode="clone", samples=samples, stream_chunk_size=4096, sample_rate=24000)
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    ref = _wav_bytes(seconds=1.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={
                "text": "hallo",
                "language": "German",
                "exaggeration": "0.6",
                "cfg_weight": "0.4",
                "temperature": "0.7",
            },
            files={"reference_audio": ("ref.wav", ref, "audio/wav")},
        )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    data = r.content
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert len(data) == 44 + 8192 * 2
    call = fake.calls[-1]
    assert call["text"] == "hallo"
    assert call["language"] == "German"
    assert call["reference_audio_len"] == len(ref)
    assert call["exaggeration"] == 0.6
    assert call["cfg_weight"] == 0.4
    assert call["temperature"] == 0.7


@pytest.mark.asyncio
async def test_clone_default_params_are_applied():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone", stream_chunk_size=4096)
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "hi", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )

    assert r.status_code == 200
    call = fake.calls[-1]
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


@pytest.mark.asyncio
async def test_clone_mode_disabled_returns_403():
    from voice.api.app import build_app

    registry = _Registry(["custom_voice"], "keep_loaded", {})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 403
    assert r.json()["error"] == "mode_disabled"


@pytest.mark.asyncio
async def test_clone_missing_reference_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_empty_text_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_unparseable_reference_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", b"not-a-wav-file", "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_reference_audio"


@pytest.mark.asyncio
async def test_clone_reference_too_long_returns_422():
    from voice.api.app import build_app

    class _TightSettings(_Settings):
        chatterbox_max_reference_seconds = 0.5

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_TightSettings())

    long_ref = _wav_bytes(seconds=2.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", long_ref, "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "reference_audio_too_long"


@pytest.mark.asyncio
async def test_clone_reference_too_large_returns_413_or_422():
    from voice.api.app import build_app

    class _TightSettings(_Settings):
        chatterbox_max_reference_bytes = 100

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_TightSettings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(seconds=1.0), "audio/wav")},
        )
    assert r.status_code in (413, 422)


@pytest.mark.asyncio
async def test_clone_out_of_range_exaggeration_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English", "exaggeration": "5.0"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_language_auto_returns_422():
    """Chatterbox doesn't support auto-detection — 'Auto' must be rejected."""
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "Auto"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422
