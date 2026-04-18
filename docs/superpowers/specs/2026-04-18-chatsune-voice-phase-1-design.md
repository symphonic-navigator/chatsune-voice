# chatsune-voice — Phase 1 design

**Date:** 2026-04-18
**Author:** Chris (symphonic-navigator), with Claude
**Status:** Approved for implementation planning

---

## 1. Intent

`chatsune-voice` provides STT (speech-to-text) and TTS (text-to-speech) inference, intended to be deployed on user-controlled homelab hardware and, in a later phase, tunnelled to the chatsune backend in the same way the `chatsune-ollama-sidecar` tunnels LLM inference.

Phase 1 — the scope of this document — is a **local experimentation tool**: a FastAPI REST service with a single static HTML "tinker page" that lets us exercise Whisper Turbo for STT and Qwen3-TTS (both CustomVoice and VoiceDesign variants) for TTS, running on a single user's machine.

The intent of Phase 1 is not to be production-ready. It is to:

- Confirm we can load and drive the chosen models correctly.
- Establish the architectural seams (engine abstraction, transport layer, configuration, observability) so that Phase 2 (WebSocket tunnel to the chatsune backend) and Phase 3 (prosumer packaging) are clean extensions rather than rewrites.
- Give the operator a browser-based test harness to record speech, transcribe it, synthesise replies, and hear the round-trip — fast iteration during exploration.

## 2. Non-goals (explicitly out of scope for Phase 1)

- WebSocket transport to the chatsune backend (Phase 2).
- Authentication, host-key handshake, TLS termination in-container (Phase 2/3, behind reverse proxy).
- Multi-user or multi-tenant concepts.
- Rate limiting, per-model inflight caps, backpressure queues.
- Prometheus or other metrics endpoints. (Carried to Phase 2 with specific metrics listed below.)
- Batch `speak` requests (an array of texts in one call).
- Voice cloning from reference audio (the CustomVoice checkpoint does not support it anyway).
- Partial / streaming STT (SSE or WebSocket-based).
- Browser end-to-end testing (Playwright, Selenium) — the test page is manually exercised.

## 3. Chosen models

| Role | Hugging Face model | Loader | Notes |
|---|---|---|---|
| STT | `h2oai/faster-whisper-large-v3-turbo` | `faster-whisper` (CTranslate2) | ~1 GB bf16; fast on both GPU and CPU; multilingual; tolerant of strong dialects (Viennese German, Québécois French, southern-London English). |
| TTS — CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `qwen-tts` (PyPI) | 9 preset speakers; free-text `instruct` field controls prosody/emotion. |
| TTS — VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `qwen-tts` (PyPI) | Free-text voice description (`voice_prompt`); optional `instruct` controls prosody. |

Output sample rate for both TTS variants is **22050 Hz** (the "12 Hz" in the model name refers to the token frame rate, not the waveform).

All three models are fetched from the Hugging Face Hub and share a single on-disk cache (`HF_HOME`).

## 4. High-level architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  HTTP client (browser, curl, Phase 2 WS adapter)                │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  api/ — FastAPI transport layer                                 │
│    /healthz, /v1/transcribe, /v1/speak, static/                 │
│    (request parsing, Pydantic validation, streaming responses)  │
└─────────────────────────────────────────────────────────────────┘
                           │   (engine-agnostic dataclasses)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  engines/ — inference layer (Protocol-based, transport-blind)   │
│    STTEngine  ─── whisper.py   (faster-whisper)                 │
│    TTSModel   ─── qwen_tts.py  (qwen-tts, one per checkpoint)   │
│    TTSModelRegistry — owns enabled modes, VRAM policy, locks    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                   GPU (CUDA or ROCm)
```

The key seam is the `engines/` package. It has no knowledge of HTTP, FastAPI, WebSockets, or any transport. In Phase 2 a parallel `transport/ws/` package will consume the same engines without changes to them.

## 5. Module layout

```
chatsune-voice/
├── backend/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── main.py              # entry point: builds app, wires components, starts uvicorn
│   │   ├── config.py            # pydantic-settings; fail-fast env validation
│   │   ├── logging_setup.py     # structlog JSON configuration
│   │   ├── audio.py             # WAV-streaming header + PCM16 chunk encoder
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── protocol.py      # STTEngine, TTSModel protocols; dataclasses
│   │   │   ├── registry.py      # TTSModelRegistry: modes, policy, locks
│   │   │   ├── whisper.py       # faster-whisper adapter
│   │   │   └── qwen_tts.py      # qwen-tts adapters (CustomVoice + VoiceDesign)
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── app.py           # FastAPI app factory; static mount; middleware
│   │       ├── models.py        # Pydantic request/response models
│   │       ├── stt.py           # POST /v1/transcribe
│   │       ├── tts.py           # POST /v1/speak
│   │       └── health.py        # GET /healthz
│   ├── static/
│   │   ├── index.html           # the tinker page
│   │   ├── app.js
│   │   └── style.css
│   ├── scripts/
│   │   └── prefetch_models.py   # optional cache warmer
│   └── tests/
│       ├── conftest.py
│       ├── test_config.py
│       ├── test_models_api.py
│       ├── test_registry.py
│       ├── test_stt_api.py
│       ├── test_tts_api.py
│       ├── test_audio.py
│       ├── test_main_startup.py
│       └── test_integration_smoke.py
├── obsidian/                    # project notes vault (state files gitignored)
├── Dockerfile.cuda
├── Dockerfile.rocm
├── compose.yml
├── .env.example
├── .gitignore
├── LICENSE                      # GPL-3.0
└── README.md
```

## 6. Configuration (environment variables)

All configuration is environment-driven and validated at start-up via `pydantic-settings`. Validation failures cause the process to exit with code 2 (fail-fast).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `CHATSUNE_VOICE_MODEL_CACHE_DIR` | no | `/models` (container) / `./.model-cache` (local) | Single cache directory for all three checkpoints. The app copies this value into the process environment as `HF_HOME` **before** importing `faster_whisper` or `qwen_tts`, which is early enough for the Hugging Face Hub client to pick it up. `prefetch_models.py` performs the same translation. Using an app-scoped name (instead of `HF_HOME` directly) keeps us insulated from future HF env-var renames. |
| `STT_MODEL` | no | `h2oai/faster-whisper-large-v3-turbo` | Hugging Face ID of the STT checkpoint. |
| `STT_MAX_AUDIO_BYTES` | no | `26214400` (25 MiB) | Hard limit on upload size. |
| `TTS_CUSTOM_VOICE_MODEL` | no | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Hugging Face ID of the CustomVoice checkpoint. |
| `TTS_VOICE_DESIGN_MODEL` | no | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Hugging Face ID of the VoiceDesign checkpoint. |
| `TTS_ENABLED_MODES` | no | `custom_voice,voice_design` | Comma-separated set of modes the API exposes. Values: `custom_voice`, `voice_design`. Empty set is invalid. |
| `TTS_VRAM_POLICY` | no | `keep_loaded` | `keep_loaded` or `swap`. See §9. |
| `TTS_ATTENTION_IMPL` | no | `sdpa` | `sdpa`, `flash_attention_2`, or `eager`. Passed straight to `from_pretrained`. |
| `PRELOAD_AT_STARTUP` | no | `true` | Whether enabled models are loaded during start-up. Ignored when `TTS_VRAM_POLICY=swap` (in which case at most one TTS model is ever loaded). |
| `DEVICE` | no | `cuda` | Torch device string. Under ROCm-PyTorch, `cuda` is correct (ROCm exposes the CUDA API). |
| `LOG_LEVEL` | no | `info` | `debug`, `info`, `warn`, `error`. |
| `APP_PORT` | no | `8000` | HTTP port inside the container. |
| `HSA_OVERRIDE_GFX_VERSION` | no | — | ROCm-only; set on Strix Halo to `11.5.1` (or `11.0.0` as fallback). Honoured by ROCm itself, not by our code. |

`TTS_ATTENTION_IMPL=flash_attention_2` is only valid on CUDA builds with the `flash-attn` package installed (opt-in via `INSTALL_FLASH_ATTN=1` Docker build arg).

## 7. REST API

Three endpoints. OpenAPI documentation is auto-generated by FastAPI at `/docs`.

### 7.1 `GET /healthz`

No input. Returns `application/json`:

```json
{
  "status": "ok",
  "stt": {
    "model": "h2oai/faster-whisper-large-v3-turbo",
    "loaded": true
  },
  "tts": {
    "enabled_modes": ["custom_voice", "voice_design"],
    "vram_policy": "keep_loaded",
    "loaded_modes": ["custom_voice", "voice_design"]
  }
}
```

HTTP 200 when all enabled models are either loaded or loadable on demand. HTTP 503 when at least one enabled model failed to load during startup (applies only with `PRELOAD_AT_STARTUP=true`) or the STT engine is unhealthy.

### 7.2 `POST /v1/transcribe`

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | file | yes | Audio in any format `faster-whisper` can decode (wav, mp3, ogg, webm, m4a, flac, …). |
| `language` | string | no | ISO 639-1 language code (e.g. `de`, `en`). Omit for auto-detection. |
| `vad` | bool | no | Default `true`. Runs faster-whisper's Voice Activity Detection to skip silence. |

**Response:** `application/json`

```json
{
  "text": "Grüß dich, wie geht's dir?",
  "language": "de",
  "language_probability": 0.99,
  "duration": 2.34,
  "segments": [
    { "start": 0.0, "end": 2.34, "text": "Grüß dich, wie geht's dir?" }
  ]
}
```

**Error responses:**

| Status | Body | Cause |
|---|---|---|
| 400 | `{"error": "invalid_audio", "message": "..."}` | Unparseable audio data. |
| 413 | `{"error": "audio_too_large", "limit_bytes": 26214400}` | Upload exceeds `STT_MAX_AUDIO_BYTES`. |
| 422 | Pydantic validation error | Malformed form fields. |
| 503 | `{"error": "stt_unavailable"}` | STT model not loaded. |

### 7.3 `POST /v1/speak`

**Request:** `application/json` — discriminated union on `mode`.

**CustomVoice variant:**

```json
{
  "mode": "custom_voice",
  "text": "Grüß dich!",
  "language": "German",
  "speaker": "Vivian",
  "instruct": "fröhlich und etwas schnell"
}
```

Required fields: `mode`, `text`, `language`, `speaker`. Optional: `instruct`.

Valid `speaker` values (Phase 1, from the CustomVoice model card): `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`.

**VoiceDesign variant:**

```json
{
  "mode": "voice_design",
  "text": "Grüß dich!",
  "language": "German",
  "voice_prompt": "Warme, tiefe Männerstimme mit leichter Rauhigkeit, entspannter Tonfall",
  "instruct": "langsam und betont"
}
```

Required fields: `mode`, `text`, `language`, `voice_prompt`. Optional: `instruct`.

Valid `language` values for both variants (from the model card): `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`, `Auto`.

Field length limits: `text` ≤ 4000, `voice_prompt` ≤ 1000, `instruct` ≤ 500 characters.

**Response:** `audio/wav` with `Transfer-Encoding: chunked`.

The response body begins with a 44-byte RIFF/WAVE header advertising sample rate 22050, 1 channel, 16-bit PCM, and a data size of `0xFFFFFFFF` (used here as a "streaming" marker; browsers accept this and begin playback immediately). PCM samples are written in frames of approximately 4096 samples (≈185 ms) each.

**Honest streaming semantics for Phase 1:** the `qwen-tts` library's public generation API returns the complete waveform as a single numpy array — it does not (yet) expose incremental sample production. Consequently, the effective time-to-first-audio is approximately the full inference time; we then stream the already-generated waveform over HTTP in chunks. The chunked response still helps downstream consumers that want to pipe audio incrementally (e.g. saving straight to disk, not waiting for a `Content-Length`), and it establishes the wire contract that Phase 2 will keep. True during-generation streaming — if achievable by reaching into qwen-tts internals or by a future library release — is a Phase-2 enhancement and does not require a client-visible change.

**Error responses:**

| Status | Body | Cause |
|---|---|---|
| 400 | `{"error": "invalid_request"}` | Malformed body. |
| 403 | `{"error": "mode_disabled", "mode": "..."}` | Mode not in `TTS_ENABLED_MODES`. |
| 422 | Pydantic validation error | Field-level validation failure. |
| 503 | `{"error": "tts_unavailable", "mode": "..."}` | Model failed to load. |

If an exception occurs **during** streaming, the HTTP status is already committed; the server logs the error (`speak_error` with `phase="during_stream"`) and closes the connection. Browsers receive truncated audio. This is an accepted limitation of HTTP streaming for Phase 1.

## 8. Engine abstraction

### 8.1 Protocols (`engines/protocol.py`)

```python
@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str

@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[TranscriptionSegment]

class STTEngine(Protocol):
    model_name: str
    async def transcribe(
        self,
        audio: bytes | BinaryIO,
        *,
        language: str | None = None,
        vad: bool = True,
    ) -> TranscriptionResult: ...
    async def aclose(self) -> None: ...


TTSMode = Literal["custom_voice", "voice_design"]

@dataclass(frozen=True)
class TTSRequest:
    mode: TTSMode
    text: str
    language: str
    speaker: str | None = None
    voice_prompt: str | None = None
    instruct: str | None = None

class TTSModel(Protocol):
    mode: TTSMode
    sample_rate: int
    def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]: ...
    async def aclose(self) -> None: ...
```

Structural (duck-typed) subtyping via `Protocol`. Tests substitute in-memory fakes without inheritance.

### 8.2 Engine implementations

`engines/whisper.py` wraps `faster_whisper.WhisperModel`. It calls the synchronous `model.transcribe(...)` inside `asyncio.to_thread` to avoid blocking the event loop.

`engines/qwen_tts.py` provides two `TTSModel` implementations that wrap the `qwen-tts` library's `Qwen3TTSModel.from_pretrained(...)` for each checkpoint. The `stream(req)` method runs the (synchronous) `generate_custom_voice` or `generate_voice_design` call in a worker thread via `asyncio.to_thread`, then yields fixed-size chunks of the returned numpy array as an async generator. See §7.3 for the consequences on time-to-first-audio. If the library gains true incremental generation in a later release, the adapter absorbs the change without touching the Protocol or the API handlers.

### 8.3 `TTSModelRegistry` (`engines/registry.py`)

Single point of ownership for:

- which TTS modes are enabled (`TTS_ENABLED_MODES`);
- how VRAM is managed across enabled modes (`TTS_VRAM_POLICY`);
- which models are currently resident;
- per-model `asyncio.Lock` instances plus a swap-wide lock used only in `swap` policy.

Public API:

```python
class TTSModelRegistry:
    async def preload(self) -> None: ...          # called at startup if PRELOAD_AT_STARTUP
    @asynccontextmanager
    async def acquire(self, mode: TTSMode) -> AsyncIterator[TTSModel]: ...
    async def aclose(self) -> None: ...
```

API handlers call `async with registry.acquire(mode) as model: ...` and never interact with locks directly.

Failure modes:

- Mode not in enabled set → `ModeDisabledError` → HTTP 403.
- Load failure during `preload()` → propagated to `main.py`, which exits with code 2.
- Load failure during lazy load → `ModelLoadError` → HTTP 503, lock is released in the context manager's `__aexit__`.

## 9. VRAM policies

Two policies, chosen via `TTS_VRAM_POLICY`:

### `keep_loaded` (default)

All enabled TTS models are kept resident simultaneously. STT model is always resident. Per-model `asyncio.Lock` serialises requests to the **same** model; requests to **different** models run in parallel.

Recommended for ≥ 24 GB VRAM (the two TTS checkpoints total ~7 GB in bf16, Whisper adds ~1 GB).

### `swap`

At most one TTS model is resident at a time. On a mode switch, the registry:

1. Acquires the swap-wide lock (blocks concurrent requests to any mode).
2. Evicts the currently resident model (`del model; torch.cuda.empty_cache()`).
3. Loads the requested model from disk cache.
4. Releases the swap lock after the request completes.

Requests that arrive while a swap is in flight are serialised behind the swap lock. No 429; they queue.

Recommended for 8–16 GB VRAM. Intentionally simple: no LRU, no heuristics, no speculative pre-loading. A smarter policy is a Phase 3 consideration when real traffic patterns are known.

### STT has no policy

Whisper Turbo is small (~1 GB in bf16) and, once loaded, stays resident for the lifetime of the process — there is no eviction path. A single `asyncio.Lock` inside the STT engine serialises transcription requests. `PRELOAD_AT_STARTUP` applies to STT as well: with `true`, Whisper is loaded during start-up; with `false`, it is loaded on the first transcribe request.

### Concurrency summary

| Concurrent request pair | `keep_loaded` | `swap` |
|---|---|---|
| STT + STT | serial (STT lock) | serial |
| TTS mode A + TTS mode A | serial (per-model lock) | serial (swap lock) |
| TTS mode A + TTS mode B | parallel | serial (swap lock) |
| STT + any TTS | parallel | parallel (the swap lock does not cover STT) |

## 10. Runtime & containers

### 10.1 Two Dockerfiles

`Dockerfile.cuda`: `FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. Installs `uv`, `uv sync --frozen --no-dev`, copies source, exposes 8000. Optional `INSTALL_FLASH_ATTN=1` build arg triggers `uv pip install flash-attn --no-build-isolation`.

`Dockerfile.rocm`: `FROM rocm/pytorch:rocm6.2.1_ubuntu22.04_py3.12_pytorch_2.5.1`. Similar, but `uv sync` uses `--no-install-package torch` because the base image already provides a ROCm-built PyTorch. No flash-attn on this path (ROCm SDPA is sufficient).

Both images run `uvicorn voice.main:app --host 0.0.0.0 --port 8000` as the default command.

### 10.2 One `compose.yml` with profiles

A single `compose.yml` defines two services, `voice-cuda` (profile `cuda`) and `voice-rocm` (profile `rocm`), sharing a common YAML anchor. The active service is selected by `COMPOSE_PROFILES` in `.env`. No service starts if no profile is selected — explicit, not silent.

Shared configuration:

- Volume: `${MODEL_CACHE_DIR:-./models}:/models`
- Port: `${APP_PORT:-8000}:8000`
- Health check: `curl -fsS http://localhost:8000/healthz` every 15 s, 120 s start period (first model load can take a minute or two).
- `restart: unless-stopped`.

CUDA-specific: `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`.

ROCm-specific: `devices: [/dev/kfd:/dev/kfd, /dev/dri:/dev/dri]`, `group_add: [video, render]`, `HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1}` (suitable for Strix Halo gfx1151; prosumer RDNA 3 discrete cards typically do not need this override).

### 10.3 `.env.example`

Ships with all configurable variables commented with their defaults and brief explanations. `COMPOSE_PROFILES` is uncommented and set to `cuda` by default (the commoner case), with `rocm` provided as a commented-out alternative.

### 10.4 Image publishing

Images are built and pushed by GitHub Actions to `ghcr.io/symphonic-navigator/chatsune-voice:cuda-latest` and `…:rocm-latest`. Tags follow the usual pattern (`:latest`, `:<git-sha>`, semver when released).

## 11. Observability

### 11.1 Logging

Structured JSON via `structlog`, one object per line to stdout. Standard fields on every record: `ts` (ISO 8601 UTC), `event`, `level`, `logger`, `request_id` (when inside a request scope).

A request-scope middleware generates a UUID per HTTP request and binds it to the structlog context.

Events emitted:

| Event | When | Key extras |
|---|---|---|
| `app_starting` | `main.py` entry | `device`, `tts_vram_policy`, `tts_enabled_modes`, `preload_at_startup` |
| `stt_model_loaded` | Whisper load complete | `model`, `load_ms` |
| `tts_model_loaded` | Qwen3-TTS checkpoint loaded | `mode`, `model`, `load_ms` |
| `tts_model_evicted` | Model unloaded under `swap` | `mode` |
| `transcribe_request` | request accepted | `audio_bytes`, `language_hint` |
| `transcribe_complete` | response sent | `detected_language`, `duration_ms`, `inference_ms` |
| `transcribe_error` | error during transcribe | `error_type`, `message` |
| `speak_request` | request accepted | `mode`, `language`, `text_len`, `speaker`, `has_voice_prompt`, `has_instruct` |
| `speak_stream_start` | first audio chunk written | `mode`, `time_to_first_chunk_ms` |
| `speak_stream_end` | stream completed | `mode`, `total_samples`, `total_ms` |
| `speak_error` | error before or during stream | `error_type`, `message`, `phase` |
| `unhandled_error` | uncaught exception | `error_type`, `traceback` (internal log only; not in response) |
| `shutdown` | SIGTERM / SIGINT | `exit_code` |

### 11.2 Error-handling surface

FastAPI exception handler catches anything not already mapped to an `HTTPException`, logs `unhandled_error` with the traceback, and returns HTTP 500 with body `{"error": "internal_server_error", "request_id": "..."}`. Tracebacks never reach the client.

### 11.3 Health check semantics

`/healthz` returns 200 if (a) the STT engine is usable and (b) every enabled TTS mode is either currently loaded (when `keep_loaded` and preload is on) or declared loadable (when `swap` or preload is off). It returns 503 otherwise.

## 12. Testing

pytest 8+, pytest-asyncio in auto mode, `httpx.AsyncClient` for FastAPI integration tests. No GPU required; no network access; no real model downloads.

| Test file | Focus |
|---|---|
| `test_config.py` | Fail-fast on missing / invalid env vars; combinations that are legal-but-odd (`swap` with `preload_at_startup=true` is legal, silently treated as "no preload"). |
| `test_models_api.py` | Pydantic discriminated-union round-trip; invalid / mismatched fields per variant; enum validation for `speaker` and `language`. |
| `test_registry.py` | `keep_loaded` lazy load, per-model locking, parallel-across-modes; `swap` eviction invokes the loader and emptying hook; ordered serialisation of concurrent mode switches; `ModeDisabledError`. |
| `test_stt_api.py` | `/v1/transcribe` happy path, invalid audio (400), overflow (413), language hint passed to engine, serialisation of concurrent STT calls. |
| `test_tts_api.py` | `/v1/speak` happy path per variant, WAV streaming header bytes exactly as expected, chunk count ≥ expected, `mode_disabled` (403), mid-stream engine failure closes cleanly. |
| `test_audio.py` | WAV header generator correctness; float32 → PCM16 clipping & rounding. |
| `test_main_startup.py` | Preload success populates registry; preload failure causes SystemExit(2). |
| `test_integration_smoke.py` | In-process TestClient: `/healthz` 200, `/v1/transcribe` end-to-end with fake STT, `/v1/speak` end-to-end with fake TTS, static root `/` returns HTML. |

CI: GitHub Actions on Linux. One job: `uv sync --frozen --dev`, `uv run ruff check`, `uv run pytest`. No GPU runner.

## 13. Front-end (the tinker page)

A single static HTML page with vanilla JavaScript and a short stylesheet, served by FastAPI via `StaticFiles(html=True)` mounted at `/`. API routes (`/v1/...`, `/healthz`, `/docs`) take precedence because they are defined before the mount.

Three sections:

1. **STT.** Microphone record button (`navigator.mediaDevices.getUserMedia` + `MediaRecorder`), file-upload fallback, language-hint dropdown (default `auto`), VAD toggle. Status line, transcript output area, meta line (detected language, duration, inference time).
2. **TTS** with a mode toggle (radio buttons: CustomVoice | VoiceDesign). Each panel has its own input fields (speaker picker for CustomVoice; voice-prompt textarea for VoiceDesign), a language dropdown, an optional `instruct` field, a speak button, and an audio player with a download link.
3. **Round-trip** (optional convenience block at the bottom): record audio, transcribe it, populate the currently selected TTS panel's `text` field, and synthesise — a one-button "record and hear yourself re-spoken" flow for lightweight fun / demos.

No build step, no framework, no package.json. Browsers assumed: current Chromium or Firefox.

## 14. README.md (structure)

Written in British English.

- Project description (one paragraph: what it is, what phase it is in).
- Requirements (GPU, Docker, disk).
- Quick start (`cp .env.example .env`, edit `COMPOSE_PROFILES`, `docker compose up`, visit `http://localhost:8000`).
- Configuration table (every env var, its default, its meaning).
- VRAM policies (when to choose which).
- API overview with a pointer to `/docs` for the full OpenAPI schema.
- Development (`uv sync`, `uv run uvicorn voice.main:app --reload`, `uv run pytest`, `uv run ruff check`).
- Roadmap: Phase 2 (WebSocket tunnel to chatsune backend), Phase 3 (prosumer packaging).
- Licence (GPL-3.0).

## 15. Dependencies

`pyproject.toml` declares:

**Runtime:**

- `fastapi ~= 0.115`
- `uvicorn[standard] ~= 0.32`
- `pydantic ~= 2.9`
- `pydantic-settings ~= 2.6`
- `structlog ~= 24.4`
- `faster-whisper ~= 1.0`
- `qwen-tts` (latest from PyPI; pinned exactly in `uv.lock`)
- `numpy` (transitive; explicit direct dependency to pin range)
- `soundfile ~= 0.12`
- `python-multipart ~= 0.0`
- `torch >= 2.5, < 3` (range; the concrete wheel comes from the base image)

**Dev:**

- `pytest ~= 8.3`
- `pytest-asyncio ~= 0.24`
- `httpx ~= 0.27`
- `ruff ~= 0.7`

Python 3.12 is the minimum supported version, matching the base images.

## 16. Phase 2 carry-overs

Noted here so they are not forgotten when we move to the next phase:

- WebSocket transport package (`voice/transport/ws/`), modelled on `chatsune-ollama-sidecar`'s `connection.py` + `dispatcher.py`. Consumes the same `engines/` layer unchanged.
- Handshake + host-key authentication (reuse the Sidecar's CSP/1 pattern, add voice-specific operations).
- Prometheus metrics endpoint with at minimum:
  - `inference_time_per_char` (TTS, histogram) — coarse but gives a clear tendency for tuning.
  - `realtime_factor` (STT, histogram, audio_duration / inference_duration).
  - `tts_vram_swaps_total` (counter, under `swap` policy).
- Request queue with bounded depth per model and backpressure semantics.
- Consider partial STT streaming via VAD-chunked transcription if Phase 2 use cases demand it.

## 17. Phase 3 summary

Phase 3 re-packages this project so a prosumer can run it on their own homelab and let the chatsune backend route STT/TTS compute through the WebSocket tunnel established in Phase 2. Phase 3 is mostly packaging, documentation, and any small policy refinements informed by Phase 2 telemetry. No architectural surprises are anticipated — this is the whole point of doing Phase 1's seams correctly.

## 18. Approval

This design was walked through section by section and approved by Chris on 2026-04-18. Any change from this point onward happens via updates to this document.
