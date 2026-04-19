"""POST /v1/speak/clone — multipart reference-audio voice cloning via Chatterbox."""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from voice.api.models import Language
from voice.audio import float32_to_pcm16, make_streaming_wav_header
from voice.engines.protocol import TTSRequest
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


def _validate_and_probe_reference(
    audio_bytes: bytes, *, max_bytes: int, max_seconds: float
) -> float:
    """Return the clip duration if it decodes and fits within the limits.

    Raises HTTPException on any failure so the handler doesn't need to care
    about the specific error shape.
    """
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": "reference_audio_too_large", "bytes": len(audio_bytes)},
        )
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_reference_audio", "reason": str(exc)},
        ) from exc
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    if duration > max_seconds:
        raise HTTPException(
            status_code=422,
            detail={"error": "reference_audio_too_long", "seconds": duration},
        )
    return duration


@router.post(
    "/speak/clone",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "PCM16 WAV stream"},
        403: {"description": "mode disabled"},
        413: {"description": "reference audio too large"},
        422: {"description": "validation error"},
    },
)
async def speak_clone(
    request: Request,
    text: str = Form(min_length=1, max_length=4000),
    language: Language = Form(),  # noqa: B008 — FastAPI idiom for declaring a form field
    reference_audio: UploadFile = File(...),  # noqa: B008 — FastAPI idiom for declaring a multipart file field
    exaggeration: float = Form(0.5, ge=0.25, le=2.0),
    cfg_weight: float = Form(0.5, ge=0.0, le=1.0),
    temperature: float = Form(0.8, ge=0.05, le=2.0),
) -> StreamingResponse:
    settings = request.app.state.settings
    registry = request.app.state.registry

    if "clone" not in registry.enabled_modes:
        raise HTTPException(
            status_code=403,
            detail={"error": "mode_disabled", "mode": "clone"},
        )

    if language == "Auto":
        raise HTTPException(
            status_code=422,
            detail={"error": "language_auto_unsupported",
                    "message": "Chatterbox requires a concrete language"},
        )

    audio_bytes = await reference_audio.read()
    duration = _validate_and_probe_reference(
        audio_bytes,
        max_bytes=settings.chatterbox_max_reference_bytes,
        max_seconds=settings.chatterbox_max_reference_seconds,
    )

    log.info(
        "clone_request",
        language=language,
        text_len=len(text),
        reference_bytes=len(audio_bytes),
        reference_seconds=duration,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    cm = registry.acquire("clone")

    tts_req = TTSRequest(
        mode="clone",
        text=text,
        language=language,
        reference_audio=audio_bytes,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    async def body() -> AsyncIterator[bytes]:
        sample_rate = 24000
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
                            "clone_stream_start",
                            time_to_first_chunk_ms=int((first_chunk_time - start) * 1000),
                        )
                    total_samples += len(chunk)
                    yield float32_to_pcm16(chunk)
        except Exception as exc:
            log.error(
                "clone_error",
                phase="during_stream" if first_chunk_time else "before_stream",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return
        log.info(
            "clone_stream_end",
            total_samples=total_samples,
            total_ms=int((time.monotonic() - start) * 1000),
        )

    return StreamingResponse(body(), media_type="audio/wav")
