"""Whisper STT adapter wrapping faster-whisper.

The real `faster_whisper.WhisperModel` is injected at construction time so
tests can substitute a fake without loading the real model.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, BinaryIO, Protocol

from voice.engines.protocol import TranscriptionResult, TranscriptionSegment
from voice.logging_setup import get_logger

log = get_logger(__name__)


class _WhisperBackend(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> Any: ...


def load_faster_whisper(
    model_id: str,
    *,
    device: str = "cuda",
    compute_type: str = "float16",
    download_root: str | None = None,
) -> _WhisperBackend:
    """Factory: load a real faster-whisper model. Kept separate so tests skip it."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )


class WhisperEngine:
    def __init__(self, *, model_name: str, model: _WhisperBackend) -> None:
        self.model_name = model_name
        self._model = model
        self._lock = asyncio.Lock()
        self._closed = False

    async def transcribe(
        self,
        audio: bytes | BinaryIO,
        *,
        language: str | None = None,
        vad: bool = True,
    ) -> TranscriptionResult:
        if self._closed:
            raise RuntimeError("WhisperEngine is closed")

        buf: BinaryIO = audio if hasattr(audio, "read") else io.BytesIO(audio)  # type: ignore[assignment]

        async with self._lock:
            segments_iter, info = await asyncio.to_thread(
                self._model.transcribe,
                buf,
                language=language,
                vad_filter=vad,
            )
            collected = await asyncio.to_thread(list, segments_iter)

        segments = [
            TranscriptionSegment(start=s.start, end=s.end, text=s.text)
            for s in collected
        ]
        text = "".join(s.text for s in segments)
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0)),
            duration=float(getattr(info, "duration", 0.0)),
            segments=segments,
        )

    async def aclose(self) -> None:
        self._closed = True
