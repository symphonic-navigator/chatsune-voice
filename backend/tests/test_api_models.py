"""Tests for voice.api.models — discriminated unions, field validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError, TypeAdapter


def test_custom_voice_request_roundtrip():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    data = {
        "mode": "custom_voice",
        "text": "Hallo",
        "language": "German",
        "speaker": "Vivian",
        "instruct": "fröhlich",
    }
    parsed = adapter.validate_python(data)
    assert parsed.mode == "custom_voice"
    assert parsed.speaker == "Vivian"
    assert parsed.instruct == "fröhlich"


def test_voice_design_request_roundtrip():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    data = {
        "mode": "voice_design",
        "text": "Hallo",
        "language": "German",
        "voice_prompt": "warme Stimme",
    }
    parsed = adapter.validate_python(data)
    assert parsed.mode == "voice_design"
    assert parsed.voice_prompt == "warme Stimme"
    assert parsed.instruct is None


def test_custom_voice_requires_speaker():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({"mode": "custom_voice", "text": "x", "language": "English"})


def test_voice_design_requires_voice_prompt():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({"mode": "voice_design", "text": "x", "language": "English"})


def test_unknown_speaker_rejected():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": "x", "language": "English", "speaker": "Bogus",
        })


def test_unknown_language_rejected():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": "x", "language": "Klingon", "speaker": "Vivian",
        })


def test_text_length_limit_enforced():
    from voice.api.models import SpeakRequest

    adapter = TypeAdapter(SpeakRequest)
    long_text = "a" * 4001
    with pytest.raises(ValidationError):
        adapter.validate_python({
            "mode": "custom_voice", "text": long_text, "language": "English", "speaker": "Ryan",
        })


def test_transcribe_response_shape():
    from voice.api.models import TranscribeResponse, TranscribeResponseSegment

    resp = TranscribeResponse(
        text="hi",
        language="en",
        language_probability=0.9,
        duration=1.0,
        segments=[TranscribeResponseSegment(start=0.0, end=1.0, text="hi")],
    )
    dumped = resp.model_dump()
    assert dumped["text"] == "hi"
    assert dumped["segments"][0]["end"] == 1.0


def test_health_response_shape():
    from voice.api.models import HealthResponse, HealthSTTInfo, HealthTTSInfo

    resp = HealthResponse(
        status="ok",
        stt=HealthSTTInfo(model="m", loaded=True),
        tts=HealthTTSInfo(
            enabled_modes=["custom_voice"],
            vram_policy="keep_loaded",
            loaded_modes=["custom_voice"],
        ),
    )
    dumped = resp.model_dump()
    assert dumped["tts"]["enabled_modes"] == ["custom_voice"]
