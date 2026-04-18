"""POST /v1/transcribe — single-file transcription with optional language hint."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from voice.api.models import TranscribeResponse, TranscribeResponseSegment
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),  # noqa: B008 — FastAPI idiom for declaring a multipart file field
    language: str | None = Form(default=None),
    vad: bool = Form(default=True),
) -> TranscribeResponse | JSONResponse:
    settings = request.app.state.settings
    stt = request.app.state.stt

    payload = await audio.read()
    if len(payload) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_audio", "message": "audio is empty"},
        )
    if settings is not None and len(payload) > settings.stt_max_audio_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": "audio_too_large",
                "message": "audio exceeds STT_MAX_AUDIO_BYTES",
                "limit_bytes": settings.stt_max_audio_bytes,
            },
        )

    language_norm = (language or "").strip() or None

    log.info("transcribe_request", audio_bytes=len(payload), language_hint=language_norm)
    try:
        result = await stt.transcribe(payload, language=language_norm, vad=vad)
    except Exception as exc:
        log.error("transcribe_error", error_type=type(exc).__name__, message=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error"},
        )

    log.info(
        "transcribe_complete",
        detected_language=result.language,
        duration_ms=int(result.duration * 1000),
    )

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
        segments=[
            TranscribeResponseSegment(start=s.start, end=s.end, text=s.text)
            for s in result.segments
        ],
    )
