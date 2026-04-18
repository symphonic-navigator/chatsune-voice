"""Structured JSON logging via structlog, with stdlib capture."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str = "info") -> None:
    """Configure structlog + stdlib logging to emit JSON lines to stdout."""
    stdlib_level = _LEVELS.get(level, logging.INFO)

    # Configure structlog with PrintLoggerFactory for direct JSON output
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(stdlib_level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Set up stdlib logging to also output through structlog
    root_logger = logging.getLogger()
    root_logger.setLevel(stdlib_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a handler with ProcessorFormatter for JSON output from stdlib logs
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(stdlib_level)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
