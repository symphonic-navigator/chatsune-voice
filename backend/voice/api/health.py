"""GET /healthz — service readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from voice.api.models import HealthResponse, HealthSTTInfo, HealthTTSInfo

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request, response: Response) -> HealthResponse:
    stt = request.app.state.stt
    registry = request.app.state.registry

    stt_loaded = bool(getattr(stt, "loaded", True))
    enabled_modes = list(registry.enabled_modes)
    loaded_modes = list(registry.loaded_modes())
    policy = registry.policy

    degraded = False
    if not stt_loaded:
        degraded = True
    if policy == "keep_loaded" and set(enabled_modes) - set(loaded_modes):
        degraded = True

    if degraded:
        response.status_code = 503
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        stt=HealthSTTInfo(model=stt.model_name, loaded=stt_loaded),
        tts=HealthTTSInfo(
            enabled_modes=enabled_modes,
            vram_policy=policy,
            loaded_modes=loaded_modes,
        ),
    )
