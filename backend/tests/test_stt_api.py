"""Tests for POST /v1/transcribe."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


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
