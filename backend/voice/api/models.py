"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Language = Literal[
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

CustomVoiceSpeaker = Literal[
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


class SpeakCustomVoiceRequest(BaseModel):
    mode: Literal["custom_voice"]
    text: str = Field(min_length=1, max_length=4000)
    language: Language
    speaker: CustomVoiceSpeaker
    instruct: str | None = Field(default=None, max_length=500)


class SpeakVoiceDesignRequest(BaseModel):
    mode: Literal["voice_design"]
    text: str = Field(min_length=1, max_length=4000)
    language: Language
    voice_prompt: str = Field(min_length=1, max_length=1000)
    instruct: str | None = Field(default=None, max_length=500)


SpeakRequest = Annotated[
    SpeakCustomVoiceRequest | SpeakVoiceDesignRequest,
    Field(discriminator="mode"),
]


class TranscribeResponseSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[TranscribeResponseSegment]


class HealthSTTInfo(BaseModel):
    model: str
    loaded: bool


class HealthTTSInfo(BaseModel):
    enabled_modes: list[str]
    vram_policy: str
    loaded_modes: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    stt: HealthSTTInfo
    tts: HealthTTSInfo


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None
    request_id: str | None = None
