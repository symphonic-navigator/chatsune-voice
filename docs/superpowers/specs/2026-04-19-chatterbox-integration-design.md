# Chatterbox Multilingual TTS integration

**Status:** design approved, pending implementation plan
**Date:** 2026-04-19
**Scope:** Phase 1 ("Bastelstube") — experimental integration of Chatterbox Multilingual as a third TTS engine alongside the existing Qwen3 custom-voice and voice-design modes.

## Motivation

Qwen3-TTS delivers excellent voice-design flexibility but measured RTF on the homelab sits at 12–15, which is far above the Phase 3 target of RTF 0.5 on Strix Halo (AMD Ryzen AI Max+ 395, unified 250 GB/s LPDDR5X) while co-locating an LLM workload. We need a faster workhorse for realtime synthesis. Chatterbox Multilingual (Resemble AI, MIT licence, ~0.5B parameters, zero-shot voice cloning, 23 languages including German, French, English) is the first candidate we evaluate.

Phase 1 is explicitly a Bastelstube: we integrate Chatterbox to play with the model on real hardware and measure the RTF baseline. Production features (voice library, Qwen3 → Chatterbox chaining, Opus output, WebSocket transport) are deliberately deferred to later specs.

## In scope

- A new engine adapter `ChatterboxCloneModel` exposing the existing `TTSModel` protocol.
- Primary backend: ONNX Q4 via `onnxruntime`. Fallback backend: bf16 PyTorch via `transformers`. Selected via env var.
- A third `TTSMode = "clone"` in the existing discriminated union.
- An `always_resident` flag on the `TTSModel` protocol so Chatterbox bypasses the Qwen3 swap lock — Chatterbox is the realtime primary path and must always be loaded when enabled.
- A new endpoint `POST /v1/speak/clone` accepting `multipart/form-data` (text + reference audio + Chatterbox-specific params).
- Tinker-page tab with drag-and-drop reference upload, Chatterbox parameter sliders, and in-page audio playback.
- Registry logic extension: `_evict_all` becomes `_evict_swappable`, exempting always-resident models.
- Configuration surface: Chatterbox model ID, backend choice (ONNX / Torch), device, reference-audio size and duration limits.
- Docker image updates: `onnxruntime-gpu` (CUDA) and `onnxruntime-rocm` (ROCm) added as conditional dependencies.
- Pre-fetch script includes the Chatterbox checkpoint.
- Test coverage for the new adapter, API endpoint, registry behaviour, and config validation using GPU-free fakes.
- README and `.env.example` updates.

## Out of scope (deferred to later specs)

- **Voice library**: server-side persistence of reference audio, `voice_id`-based API, CRUD endpoints.
- **Qwen3 → Chatterbox chaining**: generating a voice sample via Qwen3 VoiceDesign and automatically re-using it as a Chatterbox cloning reference.
- **Opus / Ogg output**: the design reserves `Accept`-header routing and `?format=` query parameters as future extensions, but all Phase 1 responses stream PCM16 WAV identically to `/v1/speak`.
- **WebSocket transport**, **gRPC**, **streaming-input** synthesis.
- **Strix Halo-specific tuning** beyond the Q4 backend choice. Baseline RTF measurement happens on the homelab; Strix Halo optimisation is a separate later investigation.
- **Changes to the existing Qwen3 adapters, Qwen3 API surface, or Qwen3 configuration.** The existing `/v1/speak` endpoint remains byte-identical to its current behaviour.

## Backward compatibility

- `/v1/speak` (JSON, two-mode discriminated union) is unchanged.
- The `TTSMode` literal gains `"clone"` as a third value; existing callers that only send `custom_voice` or `voice_design` continue to work without modification.
- The `TTS_ENABLED_MODES` default is extended to `custom_voice,voice_design,clone`. Deployments that want to disable Chatterbox set the env var explicitly.
- `TTS_VRAM_POLICY` semantics change only when `clone` is enabled and the policy is `swap`: the registry stops evicting always-resident models during a swap. Deployments without `clone` see no behavioural change.

## Architecture

### Engine layer

New file `backend/voice/engines/chatterbox_tts.py`, patterned on `qwen_tts.py`:

```python
class _ChatterboxBackend(Protocol):
    sample_rate: int
    def generate(
        self,
        *,
        text: str,
        language: str,
        reference_audio: bytes,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
    ) -> tuple[np.ndarray, int]: ...


def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend: ...
def load_chatterbox_torch(model_id: str, *, device: str) -> _ChatterboxBackend: ...


class ChatterboxCloneModel:
    mode: TTSMode = "clone"
    always_resident: bool = True
    sample_rate: int

    def __init__(self, *, backend: _ChatterboxBackend, chunk_size: int = 4096) -> None: ...
    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]: ...
    async def aclose(self) -> None: ...
```

The backend protocol hides the ONNX-vs-Torch choice from the adapter. The `_default_tts_loader` factory in `main.py` dispatches on `settings.chatterbox_backend`.

If ONNX loading fails (for example because `onnxruntime-rocm` lacks gfx1151 support for an op), the loader raises `ModelLoadError` with the underlying exception attached. There is **no silent fallback** to Torch — ROCm-related failures must be visible in the startup log so the operator can investigate.

### Protocol changes

`backend/voice/engines/protocol.py`:

- `TTSMode = Literal["custom_voice", "voice_design", "clone"]`
- `TTSRequest` gains four optional fields: `reference_audio: bytes | None`, `exaggeration: float | None`, `cfg_weight: float | None`, `temperature: float | None`. All four default to `None` to preserve the existing shape for Qwen3 callers.
- The `TTSModel` Protocol gains a class-level attribute `always_resident: bool = False`. The Qwen3 adapters inherit the default; `ChatterboxCloneModel` overrides to `True`.

### Registry changes

`TTSModelRegistry` in `backend/voice/engines/registry.py`:

- `_evict_all` is renamed to `_evict_swappable` and iterates only over loaded models whose `always_resident` attribute is `False`. Always-resident models remain loaded across swap events.
- `preload()` under `swap` policy: preloads **only** always-resident models (previously it preloaded nothing). This ensures Chatterbox is ready on first request even under `swap`, while Qwen3 modes still load on demand.
- The per-mode locks and the swap lock are unchanged. Always-resident modes acquire their per-mode lock without entering the swap lock, so they never contend with Qwen3 traffic.
- `loaded_modes()` continues to report every loaded mode, including always-resident ones.

### API layer

New router `backend/voice/api/tts_clone.py` mounted on the same FastAPI app:

- Endpoint: `POST /v1/speak/clone`
- Content type: `multipart/form-data`
- Response: `audio/wav`, streaming, with the same WAV header + PCM16 chunk layout as `/v1/speak`
- The endpoint shares audio helpers (`float32_to_pcm16`, `make_streaming_wav_header`) and logging conventions with `/v1/speak`

Request fields:

| Field | Type | Required | Default | Validation |
| --- | --- | --- | --- | --- |
| `text` | form string | yes | — | 1–4000 characters |
| `language` | form string | yes | — | must match the existing `Language` literal |
| `reference_audio` | file upload | yes | — | any format decodable by the server's libsndfile (WAV and FLAC are guaranteed; OGG / MP3 depend on the installed `libsndfile` version and are best-effort), ≤ `CHATTERBOX_MAX_REFERENCE_BYTES`, decoded duration ≤ `CHATTERBOX_MAX_REFERENCE_SECONDS` |
| `exaggeration` | form float | no | 0.5 | 0.25–2.0 |
| `cfg_weight` | form float | no | 0.5 | 0.0–1.0 |
| `temperature` | form float | no | 0.8 | 0.05–2.0 |

Error shapes match `/v1/speak`:

- Mode disabled (`clone` not in `TTS_ENABLED_MODES`): `403` with `{"error": "mode_disabled", "mode": "clone"}`
- Text empty or too long: `422` (FastAPI default validation shape)
- Reference audio missing: `422`
- Reference audio unparsable: `422` with `{"error": "invalid_reference_audio", "reason": ...}`
- Reference audio too large in bytes: `413` (FastAPI/Starlette default)
- Reference audio too long in duration: `422` with `{"error": "reference_audio_too_long", "seconds": ...}`
- Out-of-range numeric params: `422`
- Inference failure mid-stream: the stream is terminated; a structured `clone_error` log is emitted with a `phase` tag (`before_stream` or `during_stream`). No retry is attempted — identical to the Qwen3 pattern.

Logging events:
- `clone_request` — mode, language, text length, reference bytes, reference duration, numeric params
- `clone_stream_start` — time to first chunk
- `clone_stream_end` — total samples, total duration
- `clone_error` — phase, error type, message

### Configuration

`backend/voice/config.py` gains:

- `chatterbox_model: str = "onnx-community/chatterbox-multilingual-ONNX"` — the exact Q4 repo name is confirmed in the implementation plan.
- `chatterbox_backend: Literal["onnx", "torch"] = "onnx"`
- `chatterbox_device: str = "cuda"` — on ROCm hosts this resolves through HIP, analogous to `tts_device`.
- `chatterbox_max_reference_bytes: int = 10 * 1024 * 1024`
- `chatterbox_max_reference_seconds: float = 30.0`
- `tts_enabled_modes` default tuple is extended to `("custom_voice", "voice_design", "clone")`
- `_validate_mode_values` accepts `"clone"` in addition to the existing two modes

### Docker and dependencies

- `Dockerfile` (CUDA): install `onnxruntime-gpu` alongside existing torch.
- `Dockerfile.rocm`: install `onnxruntime-rocm`. Op coverage on gfx1151 is a known unknown at spec time; the implementation plan probes this on a Strix Halo host before the PR merges, and the fallback is `CHATTERBOX_BACKEND=torch` with a README note.
- `scripts/prefetch_models.py` adds the Chatterbox repo (both the ONNX variant and optionally the Torch variant depending on selected backend).

## Frontend — tinker page

The existing `backend/static/index.html` is vanilla HTML + CSS + JS with no build step. We extend it in the same style:

- A new tab labelled "Chatterbox (Voice Clone)".
- Form fields: `<textarea>` for text, `<select>` for language (identical options to the Qwen3 tabs), a drag-and-drop zone plus `<input type="file" accept="audio/*">` for the reference, a small `<audio controls>` preview of the selected reference, and three `<input type="range">` sliders with live numeric readouts for `exaggeration`, `cfg_weight`, and `temperature`.
- Submit assembles `FormData` and posts to `/v1/speak/clone`. The response `Blob` is rendered in an `<audio>` element via an object URL; the previous URL is revoked before replacement.
- Error handling: `4xx` surfaces `error.message` in a small toast row; `5xx` shows a generic message.
- No toolchain or framework is added.

## Testing

All tests remain GPU-free. Fakes substitute for both the ONNX and the Torch loader; real model runs are manual before merge.

New file `backend/tests/test_chatterbox_tts.py`:
- A `FakeChatterboxBackend` (deterministic sine wave keyed off text length, ignores reference audio contents but validates its presence).
- `stream()` produces `np.ndarray` chunks of the expected length.
- `ValueError` when `reference_audio` is missing.
- `always_resident` is `True`.

Extension to `backend/tests/test_registry.py`:
- Under `swap` policy with all three modes enabled: a Qwen3 mode switch evicts only the other Qwen3 mode; Chatterbox stays loaded.
- Under `keep_loaded` policy: behaviour unchanged — all three modes are loaded and none are evicted.
- Preload under `swap` policy loads always-resident modes and skips the rest.

New file `backend/tests/test_tts_clone_api.py`:
- Happy path: multipart upload with a valid WAV returns `200` and a streaming WAV body.
- Missing `reference_audio` returns `422`.
- Reference bytes exceed the limit: returns `413` (or `422`, whichever FastAPI emits for the configured limit — the test asserts the actual behaviour).
- Reference duration exceeds the limit: returns `422` with `reference_audio_too_long`.
- Reference audio that fails to decode returns `422` with `invalid_reference_audio`.
- Empty `text` returns `422`.
- Out-of-range `exaggeration`, `cfg_weight`, or `temperature` returns `422`.
- `clone` not in `TTS_ENABLED_MODES` returns `403`.

Extension to `backend/tests/test_config.py`:
- Default `tts_enabled_modes` contains `"clone"`.
- `chatterbox_backend` accepts `"onnx"` and `"torch"`; other values fail validation.
- Numeric limits (`chatterbox_max_reference_bytes`, `chatterbox_max_reference_seconds`) are parsed and validated.

Extension to `backend/tests/test_main_startup.py`:
- Bootstrap with all three modes enabled yields a registry holding all three; Chatterbox reports `always_resident=True`.

## Documentation

- `README.md`:
  - Configuration table extended with the new `CHATTERBOX_*` env vars.
  - VRAM-policy section gains one sentence explaining Chatterbox as the always-resident workhorse under `swap`.
  - New subsection "Cloning mode" with a curl example hitting `/v1/speak/clone`.
- `.env.example` extended with the new env vars and realistic placeholder values.

## Done criteria for Phase 1

1. `docker compose up` succeeds on both CUDA and ROCm hosts with all three modes enabled. The `/healthz` endpoint reports `ok`.
2. `curl -X POST /v1/speak/clone` with a 10-second reference WAV produces recognisably cloned audio in at least German, English, and French.
3. The tinker-page Chatterbox tab works in Firefox and Chromium: upload, slider adjustment, submit, playback.
4. RTF for Chatterbox is measured on the homelab and recorded in the README, so the Strix Halo gap is estimable.
5. `uv run pytest` is green. `uv run ruff check .` is clean.

## Open items deferred to the implementation plan

- Exact Hugging Face repository name for the Chatterbox Q4 ONNX variant (the implementation plan confirms this against the live hub).
- Whether `onnxruntime-rocm` provides the ops Chatterbox needs on gfx1151, and the required `onnxruntime-rocm` version. Probed during the implementation plan on whatever ROCm hardware is available; if the probe fails or Strix Halo access is not available in the plan phase, Phase 1 ships with `CHATTERBOX_BACKEND=torch` as the effective default on ROCm hosts and a documented README caveat, and real Strix Halo validation happens as a follow-up outside this spec.
- Whether Chatterbox's Python API can accept reference audio as `bytes` directly or requires a file path (in which case a temporary file per request is needed).
- Exact tokeniser / text-normalisation behaviour per language, and whether any pre-processing is needed beyond passing the text through.
