"""Chatterbox Multilingual TTS adapter.

Exposes the TTSModel protocol by delegating to a backend object that handles
actual inference. Two loader functions live here (ONNX Runtime and Torch);
each one returns an object conforming to _ChatterboxBackend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from voice.engines.protocol import TTSMode, TTSRequest
from voice.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 4096

# Chatterbox uses ISO-639-1 codes. Our Language literal uses full English names.
# "Auto" is deliberately not mapped — Chatterbox needs a concrete language.
_LANGUAGE_MAP: dict[str, str] = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
}


def language_to_iso639(language: str) -> str:
    """Map a Language literal value to Chatterbox's ISO-639-1 code.

    Raises ValueError for "Auto" (unsupported) or unknown languages.
    """
    if language == "Auto":
        raise ValueError(
            "Chatterbox requires a concrete language; 'Auto' is not supported"
        )
    try:
        return _LANGUAGE_MAP[language]
    except KeyError as exc:
        raise ValueError(f"unknown language for Chatterbox: {language!r}") from exc


class _ChatterboxBackend(Protocol):
    sample_rate: int

    def generate(
        self,
        *,
        text: str,
        language: str,
        reference_audio: bytes,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
    ) -> tuple[np.ndarray, int]: ...


def load_chatterbox_torch(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Load the Chatterbox Multilingual model via the chatterbox-tts pip package.

    The `chatterbox-tts` PyPI package is NOT a declared dependency of this
    project — it conflicts with the transformers version pinned by qwen-tts.
    Users who want the Torch fallback install it themselves (see README).
    If it is not importable when this loader runs, a clean ImportError
    propagates to the registry's ModelLoadError.

    Returns a backend adapter that accepts reference_audio as bytes; it writes
    the bytes to a temporary file before calling Chatterbox's file-path API.
    """
    import tempfile
    from pathlib import Path

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-not-found]

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    class _TorchBackend:
        sample_rate: int = int(getattr(model, "sr", 24000))

        def generate(
            self,
            *,
            text: str,
            language: str,
            reference_audio: bytes,
            exaggeration: float,
            cfg_weight: float,
            temperature: float,
        ) -> tuple[np.ndarray, int]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(reference_audio)
                tmp_path = Path(tmp.name)
            try:
                wav = model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=str(tmp_path),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            return wav, self.sample_rate

    return _TorchBackend()


def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Factory for the ONNX Runtime-based Chatterbox backend. Implemented in Task 6."""
    raise NotImplementedError("ONNX backend loader — populated in Task 6")


class ChatterboxCloneModel:
    mode: TTSMode = "clone"
    always_resident: bool = True

    def __init__(
        self,
        *,
        backend: _ChatterboxBackend,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._backend = backend
        self._chunk_size = chunk_size
        self.sample_rate = getattr(backend, "sample_rate", 24000)
        self._closed = False

    async def aclose(self) -> None:
        self._closed = True

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.reference_audio is None or len(req.reference_audio) == 0:
            raise ValueError("Chatterbox requires reference_audio")

        language_id = language_to_iso639(req.language)
        exaggeration = req.exaggeration if req.exaggeration is not None else 0.5
        cfg_weight = req.cfg_weight if req.cfg_weight is not None else 0.5
        temperature = req.temperature if req.temperature is not None else 0.8

        def _generate() -> tuple[np.ndarray, int]:
            return self._backend.generate(
                text=req.text,
                language=language_id,
                reference_audio=req.reference_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        samples, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        offset = 0
        while offset < len(samples):
            yield samples[offset:offset + self._chunk_size]
            offset += self._chunk_size
