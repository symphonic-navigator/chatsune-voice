"""Qwen3-TTS adapters — one class per checkpoint (CustomVoice, VoiceDesign).

Both wrap a `backend` object that exposes the qwen-tts library's
`generate_custom_voice(...)` or `generate_voice_design(...)` method. The real
backend is loaded lazily; tests inject a fake.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from voice.engines.protocol import TTSMode, TTSRequest
from voice.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 4096


class _QwenBackend(Protocol):
    sample_rate: int
    def generate_custom_voice(self, **kwargs: Any) -> Any: ...
    def generate_voice_design(self, **kwargs: Any) -> Any: ...


def load_qwen_tts(
    model_id: str,
    *,
    device: str = "cuda",
    attention_impl: str = "sdpa",
) -> _QwenBackend:
    """Factory: load the real qwen-tts model. Kept separate so tests can skip."""
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attention_impl,
    )


class _QwenBase:
    mode: TTSMode
    sample_rate: int
    always_resident: bool = False

    def __init__(self, *, backend: _QwenBackend, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        self._backend = backend
        self._chunk_size = chunk_size
        self.sample_rate = getattr(backend, "sample_rate", 22050)
        self._closed = False

    async def aclose(self) -> None:
        self._closed = True

    async def _chunked(self, samples: np.ndarray) -> AsyncIterator[np.ndarray]:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        offset = 0
        while offset < len(samples):
            yield samples[offset:offset + self._chunk_size]
            offset += self._chunk_size


class QwenCustomVoiceModel(_QwenBase):
    mode: TTSMode = "custom_voice"

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.speaker is None:
            raise ValueError("CustomVoice requires a speaker")

        def _generate() -> tuple[list[np.ndarray], int]:
            return self._backend.generate_custom_voice(
                text=req.text,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct,
            )

        wavs, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        samples = wavs[0] if isinstance(wavs, list) else wavs
        async for chunk in self._chunked(samples):
            yield chunk


class QwenVoiceDesignModel(_QwenBase):
    mode: TTSMode = "voice_design"

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.voice_prompt is None:
            raise ValueError("VoiceDesign requires a voice_prompt")

        # qwen-tts's voice-design API takes a single `instruct` that carries the voice
        # description; we concatenate voice_prompt (mandatory) and the caller's
        # optional instruct for speaking-style.
        combined = req.voice_prompt
        if req.instruct:
            combined = f"{combined}. {req.instruct}"

        def _generate() -> tuple[list[np.ndarray], int]:
            return self._backend.generate_voice_design(
                text=req.text,
                language=req.language,
                instruct=combined,
            )

        wavs, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        samples = wavs[0] if isinstance(wavs, list) else wavs
        async for chunk in self._chunked(samples):
            yield chunk
