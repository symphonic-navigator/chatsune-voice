"""Tests for POST /v1/speak — discriminated union, streaming WAV response."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

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
