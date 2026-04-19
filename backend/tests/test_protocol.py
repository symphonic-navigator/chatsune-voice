"""Tests for voice.engines.protocol — type-level guarantees."""

from __future__ import annotations

import pytest

from voice.engines.protocol import TTSRequest


def test_ttsrequest_accepts_new_optional_fields():
    req = TTSRequest(
        mode="clone",
        text="hello",
        language="German",
        reference_audio=b"fake-wav-bytes",
        exaggeration=0.6,
        cfg_weight=0.4,
        temperature=0.9,
    )
    assert req.reference_audio == b"fake-wav-bytes"
    assert req.exaggeration == 0.6
    assert req.cfg_weight == 0.4
    assert req.temperature == 0.9


def test_ttsrequest_defaults_new_fields_to_none_for_existing_callers():
    req = TTSRequest(mode="custom_voice", text="x", language="English", speaker="Ryan")
    assert req.reference_audio is None
    assert req.exaggeration is None
    assert req.cfg_weight is None
    assert req.temperature is None


def test_ttsmode_literal_includes_clone():
    import typing
    from voice.engines.protocol import TTSMode
    assert "clone" in typing.get_args(TTSMode)
    assert "custom_voice" in typing.get_args(TTSMode)
    assert "voice_design" in typing.get_args(TTSMode)
