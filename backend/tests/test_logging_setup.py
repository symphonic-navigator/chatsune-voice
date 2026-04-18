"""Tests for voice.logging_setup."""

from __future__ import annotations

import json
import logging


def test_configure_logging_emits_json(capsys):
    from voice.logging_setup import configure_logging, get_logger

    configure_logging("info")
    log = get_logger("test")
    log.info("hello", foo="bar", count=3)

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["foo"] == "bar"
    assert record["count"] == 3
    assert record["level"] == "info"
    assert "timestamp" in record


def test_configure_logging_respects_level(capsys):
    from voice.logging_setup import configure_logging, get_logger

    configure_logging("warn")
    log = get_logger("test")
    log.info("suppressed")
    log.warning("shown")

    captured = capsys.readouterr()
    assert "suppressed" not in captured.out
    assert "shown" in captured.out


def test_stdlib_logger_is_captured(capsys):
    from voice.logging_setup import configure_logging

    configure_logging("info")
    logging.getLogger("external").info("from stdlib")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "from stdlib"
