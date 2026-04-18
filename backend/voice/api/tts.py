"""POST /v1/speak — streaming WAV response, discriminated request union."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from voice.api.models import SpeakRequest
from voice.audio import float32_to_pcm16, make_streaming_wav_header
from voice.engines.protocol import TTSRequest
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


@router.post(
    "/speak",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "PCM16 WAV stream"},
        403: {"description": "mode disabled"},
        422: {"description": "validation error"},
    },
)
async def speak(request: Request, payload: SpeakRequest) -> StreamingResponse:
    registry = request.app.state.registry

    # Synchronously pre-check the enabled set so the 403 is raised before any
    # streaming response is committed. The registry's acquire() is an async
    # context manager; by the time __aenter__ runs, headers are already gone.
    if payload.mode not in registry.enabled_modes:
        raise HTTPException(
            status_code=403,
            detail={"error": "mode_disabled", "mode": payload.mode},
        )

    cm = registry.acquire(payload.mode)

    tts_req = TTSRequest(
        mode=payload.mode,
        text=payload.text,
        language=payload.language,
        speaker=getattr(payload, "speaker", None),
        voice_prompt=getattr(payload, "voice_prompt", None),
        instruct=payload.instruct,
    )

    log.info(
        "speak_request",
        mode=payload.mode,
        language=payload.language,
        text_len=len(payload.text),
        speaker=getattr(payload, "speaker", None),
        has_voice_prompt=getattr(payload, "voice_prompt", None) is not None,
        has_instruct=payload.instruct is not None,
    )

    async def body() -> AsyncIterator[bytes]:
        sample_rate = 22050
        first_chunk_time: float | None = None
        total_samples = 0
        start = time.monotonic()
        try:
            async with cm as model:
                sample_rate = model.sample_rate
                yield make_streaming_wav_header(sample_rate=sample_rate, channels=1)
                async for chunk in model.stream(tts_req):
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                        log.info(
                            "speak_stream_start",
                            mode=payload.mode,
                            time_to_first_chunk_ms=int((first_chunk_time - start) * 1000),
                        )
                    total_samples += len(chunk)
                    yield float32_to_pcm16(chunk)
        except Exception as exc:
            log.error(
                "speak_error",
                mode=payload.mode,
                phase="during_stream" if first_chunk_time else "before_stream",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return
        log.info(
            "speak_stream_end",
            mode=payload.mode,
            total_samples=total_samples,
            total_ms=int((time.monotonic() - start) * 1000),
        )

    return StreamingResponse(body(), media_type="audio/wav")
