"""Tests for GET /healthz."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


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
