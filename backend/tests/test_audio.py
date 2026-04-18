"""Tests for voice.audio."""

from __future__ import annotations

import struct

import numpy as np


def test_make_streaming_wav_header_layout():
    from voice.audio import make_streaming_wav_header

    hdr = make_streaming_wav_header(sample_rate=22050, channels=1)
    assert len(hdr) == 44
    assert hdr[0:4] == b"RIFF"
    # bytes 4..8 = total size - 8 = 0xFFFFFFFF - 8 (max)
    assert struct.unpack("<I", hdr[4:8])[0] == 0xFFFFFFFF - 8
    assert hdr[8:12] == b"WAVE"
    assert hdr[12:16] == b"fmt "
    assert struct.unpack("<I", hdr[16:20])[0] == 16          # fmt chunk size
    assert struct.unpack("<H", hdr[20:22])[0] == 1           # PCM format
    assert struct.unpack("<H", hdr[22:24])[0] == 1           # channels
    assert struct.unpack("<I", hdr[24:28])[0] == 22050       # sample rate
    assert struct.unpack("<I", hdr[28:32])[0] == 22050 * 2   # byte rate
    assert struct.unpack("<H", hdr[32:34])[0] == 2           # block align
    assert struct.unpack("<H", hdr[34:36])[0] == 16          # bits per sample
    assert hdr[36:40] == b"data"
    # bytes 40..44 = data chunk size = 0xFFFFFFFF (streaming marker)
    assert struct.unpack("<I", hdr[40:44])[0] == 0xFFFFFFFF


def test_float32_to_pcm16_clipping():
    from voice.audio import float32_to_pcm16

    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float32_to_pcm16(arr)
    # bytes: 5 samples * 2 bytes = 10
    assert len(pcm) == 10
    unpacked = struct.unpack("<5h", pcm)
    # -2.0 clipped to -1.0 → -32768; +2.0 clipped to +1.0 → +32767
    assert unpacked[0] == -32768
    assert unpacked[1] == -32768     # -1.0 → -32768 (i.e. int16 min)
    assert unpacked[2] == 0
    assert unpacked[3] == 32767
    assert unpacked[4] == 32767


def test_float32_to_pcm16_roundtrip_precision():
    from voice.audio import float32_to_pcm16

    arr = np.array([0.5, -0.25, 0.125], dtype=np.float32)
    pcm = float32_to_pcm16(arr)
    ints = struct.unpack("<3h", pcm)
    assert abs(ints[0] - 16383) <= 1
    assert abs(ints[1] - (-8192)) <= 1
    assert abs(ints[2] - 4096) <= 1


def test_float32_to_pcm16_rejects_wrong_dtype():
    import pytest

    from voice.audio import float32_to_pcm16

    arr = np.array([0.0, 0.5], dtype=np.float64)
    with pytest.raises(TypeError):
        float32_to_pcm16(arr)
