# chatsune-voice

Voice homelab backend for [chatsune](https://github.com/symphonic-navigator). Runs Whisper Turbo for speech-to-text and Qwen3-TTS (both CustomVoice and VoiceDesign variants) for text-to-speech, exposed as a FastAPI REST service with a small browser-based tinker page.

## Current status

**Phase 1 — local experimentation tool.** The project runs on a single machine, accepts REST requests, and serves a static HTML page at `/`. Subsequent phases will extend it:

- **Phase 2** — WebSocket tunnel to the chatsune backend (analogous to `chatsune-ollama-sidecar`).
- **Phase 3** — prosumer-oriented packaging so end users can contribute their homelab compute to chatsune.

## Requirements

- GNU/Linux host with Docker and Docker Compose.
- One of:
  - **NVIDIA GPU** with `nvidia-container-toolkit` installed (CUDA 12.4+), or
  - **AMD GPU** supported by ROCm 6.2+ (including the Strix Halo iGPU).
- At least ~8 GB VRAM for `TTS_VRAM_POLICY=swap`, ~24 GB for `keep_loaded` with both TTS modes enabled.
- ~5 GB disk for the model cache.

## Quick start

```bash
cp .env.example .env
# Edit .env: set COMPOSE_PROFILES to `cuda` or `rocm`.
$EDITOR .env

docker compose up -d
```

Then open <http://localhost:8000> for the tinker page, or <http://localhost:8000/docs> for the OpenAPI schema.

On first start the container downloads the three Hugging Face checkpoints (~5 GB total) into `./models/`. Subsequent starts reuse that cache.

Optional: pre-warm the cache before the first container start:

```bash
cd backend
uv sync --dev
uv run python scripts/prefetch_models.py
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list with defaults. The most relevant knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `COMPOSE_PROFILES` | (must set) | `cuda` or `rocm`. Selects the corresponding service and base image. |
| `CHATSUNE_VOICE_MODEL_CACHE_DIR` | `/models` (inside container) | Single cache directory shared by all three models; mapped to `HF_HOME` at start-up. |
| `TTS_ENABLED_MODES` | `custom_voice,voice_design` | Which TTS modes the API exposes. A subset is allowed. |
| `TTS_VRAM_POLICY` | `keep_loaded` | `keep_loaded` keeps all enabled TTS models resident. `swap` keeps at most one resident at a time, serialising mode switches via an `asyncio.Lock`. |
| `TTS_ATTENTION_IMPL` | `sdpa` | `sdpa` (PyTorch default, works everywhere), `flash_attention_2` (CUDA-only, requires `INSTALL_FLASH_ATTN=1` build arg), or `eager`. |
| `PRELOAD_AT_STARTUP` | `true` | Load models during start-up. Set to `false` to defer loading until the first request. |
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | ROCm-only. `11.5.1` suits Strix Halo (gfx1151); discrete RDNA3 usually needs no override. |

## VRAM policy guidance

- **24 GB+ VRAM:** `keep_loaded` with both modes enabled. Requests of different modes run in parallel; same-mode requests serialise per model.
- **8–16 GB VRAM:** `swap`. Requests of different modes are serialised via a swap lock; expect a small startup cost on each mode switch.
- **< 8 GB VRAM:** enable only one mode (`TTS_ENABLED_MODES=custom_voice` or `voice_design`).

STT (Whisper Turbo) is always resident once loaded and does not participate in the TTS VRAM policy.

## API

Endpoints (full schema at `/docs`):

- `GET /healthz` — readiness. 200 when every enabled model is loaded (or loadable under `swap`); 503 otherwise.
- `POST /v1/transcribe` — multipart upload, fields `audio` (file), `language` (optional ISO code), `vad` (optional bool). Returns JSON with text, language, duration, and segments.
- `POST /v1/speak` — JSON body with discriminator `mode` (`custom_voice` or `voice_design`). Returns `audio/wav` as a chunked HTTP response.

## Development

```bash
cd backend
uv sync --dev
uv run uvicorn voice.main:app --reload
# In another shell:
uv run ruff check .
uv run pytest -v
```

Tests use in-memory fakes for Whisper and Qwen3-TTS; no GPU or network access is required.

## Licence

GPL-3.0-or-later. See `LICENSE`.
