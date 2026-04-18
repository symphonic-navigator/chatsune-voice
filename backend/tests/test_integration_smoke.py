"""End-to-end smoke test with fake engines."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


class _Reg:
    enabled_modes = ()
    policy = "keep_loaded"
    def loaded_modes(self):
        return ()

    def acquire(self, mode):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_root_serves_index_html(fake_stt):
    from voice.api.app import build_app

    app = build_app(stt=fake_stt, registry=_Reg(), settings=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "chatsune-voice" in r.text


@pytest.mark.asyncio
async def test_healthz_under_integration(fake_stt):
    from voice.api.app import build_app

    app = build_app(stt=fake_stt, registry=_Reg(), settings=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/healthz")
    assert r.status_code == 200
