"""Download all three model checkpoints into the configured HF cache.

Run once before the first `docker compose up` to avoid long first-request delays.

    uv run python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
import time

from voice.config import Settings
from voice.logging_setup import configure_logging, get_logger
from voice.main import apply_hf_home


def _download(hf_id: str, log) -> None:
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    path = snapshot_download(repo_id=hf_id)
    log.info("model_downloaded", model=hf_id, path=path,
             elapsed_ms=int((time.monotonic() - t0) * 1000))


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)

    apply_hf_home(settings.model_cache_dir)

    ids = [
        settings.stt_model,
        settings.tts_custom_voice_model,
        settings.tts_voice_design_model,
        settings.chatterbox_model,
    ]
    log.info("prefetch_starting", cache_dir=str(settings.model_cache_dir), models=ids)

    for hf_id in ids:
        try:
            _download(hf_id, log)
        except Exception as exc:
            log.error("prefetch_failed", model=hf_id, error_type=type(exc).__name__,
                      message=str(exc))
            return 1

    log.info("prefetch_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
