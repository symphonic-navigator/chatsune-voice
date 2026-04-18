"""WAV streaming header + float32 → PCM16 sample conversion."""

from __future__ import annotations

import struct

import numpy as np

_STREAM_DATA_SIZE = 0xFFFFFFFF
_STREAM_RIFF_SIZE = _STREAM_DATA_SIZE - 8


def make_streaming_wav_header(*, sample_rate: int, channels: int = 1) -> bytes:
    """Produce a 44-byte RIFF/WAVE header advertising an open-ended PCM16 stream.

    The data chunk size is set to 0xFFFFFFFF, which browsers and media players
    accept as a "streaming / unknown length" marker. The audio payload written
    after this header is raw little-endian PCM16 samples.
    """
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    return (
        b"RIFF"
        + struct.pack("<I", _STREAM_RIFF_SIZE)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits_per_sample)
        + b"data"
        + struct.pack("<I", _STREAM_DATA_SIZE)
    )


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert a float32 numpy array in [-1, 1] to little-endian PCM16 bytes."""
    if samples.dtype != np.float32:
        raise TypeError(f"expected float32, got {samples.dtype}")
    clipped = np.clip(samples, -1.0, 1.0)
    # Scale: negative samples by 32768, positive by 32767 to match int16 asymmetry
    scaled = np.where(
        clipped < 0,
        np.round(clipped * 32768.0),
        np.round(clipped * 32767.0)
    ).astype(np.int16)
    return scaled.tobytes()
