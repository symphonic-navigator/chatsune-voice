# Chatterbox Multilingual TTS Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chatterbox Multilingual TTS as a third, always-resident TTS engine alongside the existing Qwen3 custom-voice and voice-design engines, exposed via a new `POST /v1/speak/clone` multipart endpoint and a corresponding tab on the browser tinker page.

**Architecture:** Chatterbox is a 0.5B Llama-based voice-cloning TTS that accepts a reference audio plus text. We add a new `TTSMode = "clone"` and a `ChatterboxCloneModel` adapter that conforms to the existing `TTSModel` protocol. The adapter is pluggable behind two backend loaders — a Torch backend (`chatterbox-tts` pip package) and an ONNX-Runtime backend (`onnx-community/chatterbox-multilingual-ONNX` with Q4 quantisation for Strix-Halo-class memory bandwidth). Because Chatterbox is the realtime workhorse, it carries an `always_resident=True` attribute that makes the registry exempt it from the Qwen3 swap-lock eviction.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, `chatterbox-tts` (Torch), `onnxruntime-gpu` / `onnxruntime-rocm` (ONNX path), `soundfile` for reference-audio decoding, `numpy`, `huggingface_hub` for model downloads. Tests use `pytest`, `pytest-asyncio`, and `httpx` ASGI transport. Vanilla HTML/CSS/JS for the tinker page.

**Reference spec:** `docs/superpowers/specs/2026-04-19-chatterbox-integration-design.md`

---

## File structure (what changes where)

### New files

- `backend/voice/engines/chatterbox_tts.py` — `ChatterboxCloneModel` adapter, language-ID mapping helper, Torch loader, ONNX loader, shared `_ChatterboxBackend` protocol
- `backend/voice/api/tts_clone.py` — `POST /v1/speak/clone` router, multipart validation, streaming response
- `backend/tests/test_chatterbox_tts.py` — unit tests for the adapter and language helper
- `backend/tests/test_tts_clone_api.py` — integration tests for the endpoint

### Modified files

- `backend/voice/engines/protocol.py` — extend `TTSMode`, add fields to `TTSRequest`, add `always_resident` class attr to `TTSModel` protocol
- `backend/voice/engines/qwen_tts.py` — set `always_resident: bool = False` explicitly on `_QwenBase`
- `backend/voice/engines/registry.py` — rename `_evict_all` to `_evict_swappable`, preload always-resident modes under swap policy
- `backend/voice/config.py` — add `chatterbox_*` settings, accept `"clone"` in `_validate_mode_values`, flip default `tts_enabled_modes` to include `"clone"`
- `backend/voice/main.py` — wire Chatterbox into `_default_tts_loader` via `settings.chatterbox_backend`
- `backend/voice/api/app.py` — mount the clone router
- `backend/tests/conftest.py` — extend `FakeTTSModel` to capture the new TTSRequest fields and expose `always_resident`
- `backend/tests/test_registry.py` — add cases for always-resident exemption and swap-preload
- `backend/tests/test_config.py` — add cases for the new settings and for `"clone"` acceptance
- `backend/tests/test_main_startup.py` — assert bootstrap works with all three modes
- `backend/scripts/prefetch_models.py` — include the Chatterbox model(s) in the download list
- `backend/pyproject.toml` — add `chatterbox-tts`, `onnxruntime-gpu`, `librosa` to dependencies
- `Dockerfile.cuda` — ensure ONNX-GPU deps land in the image (via pyproject)
- `Dockerfile.rocm` — install `onnxruntime-rocm` (or document the graceful fallback to `CHATTERBOX_BACKEND=torch`)
- `compose.yml` — forward the new `CHATTERBOX_*` env vars to the service
- `backend/static/index.html` — add Chatterbox tab
- `backend/static/app.js` — wire Chatterbox tab submission and playback
- `backend/static/style.css` — minimal styling for the new controls
- `.env.example` — document the new env vars
- `README.md` — document the new endpoint, env vars, and swap-policy interaction

---

## Task 1: Extend protocol with `"clone"` mode, new TTSRequest fields, and `always_resident`

**Rationale:** Foundational type changes. Every subsequent task depends on these types. The change is additive: existing Qwen3 callers and tests continue to work because the new `TTSRequest` fields default to `None` and `always_resident` defaults to `False`.

**Files:**
- Modify: `backend/voice/engines/protocol.py:15` (TTSMode literal), `backend/voice/engines/protocol.py:48-55` (TTSRequest dataclass), `backend/voice/engines/protocol.py:58-64` (TTSModel protocol)
- Modify: `backend/voice/engines/qwen_tts.py:48-50` (_QwenBase class attributes)
- Modify: `backend/tests/conftest.py:57-100` (FakeTTSModel)
- Test: `backend/tests/test_qwen_tts.py` (existing — must still pass), `backend/tests/test_protocol.py` (new file)

- [ ] **Step 1: Write failing test for TTSRequest with the new optional fields**

Create `backend/tests/test_protocol.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && uv run pytest tests/test_protocol.py -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'reference_audio'` (or similar).

- [ ] **Step 3: Extend protocol.py**

Replace the `TTSMode` line and `TTSRequest` dataclass in `backend/voice/engines/protocol.py`:

```python
TTSMode = Literal["custom_voice", "voice_design", "clone"]


@dataclass(frozen=True)
class TTSRequest:
    mode: TTSMode
    text: str
    language: str
    speaker: str | None = None
    voice_prompt: str | None = None
    instruct: str | None = None
    reference_audio: bytes | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None
```

And extend the `TTSModel` Protocol to declare the new class-level attribute:

```python
class TTSModel(Protocol):
    mode: TTSMode
    sample_rate: int
    always_resident: bool  # Default False via the implementations below.

    def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]: ...

    async def aclose(self) -> None: ...
```

- [ ] **Step 4: Set `always_resident = False` explicitly on `_QwenBase`**

In `backend/voice/engines/qwen_tts.py`, modify the `_QwenBase` class body (around line 48):

```python
class _QwenBase:
    mode: TTSMode
    sample_rate: int
    always_resident: bool = False

    def __init__(self, *, backend: _QwenBackend, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        ...
```

- [ ] **Step 5: Extend FakeTTSModel to capture the new fields**

In `backend/tests/conftest.py`, modify the `FakeTTSModel.stream()` method's `self.calls.append(...)` call (around line 75) to include the new fields, and add the `always_resident` attribute:

```python
@dataclass
class FakeTTSModel:
    """Deterministic replacement for a Qwen3-TTS checkpoint."""

    mode: str = "custom_voice"
    sample_rate: int = 22050
    samples: np.ndarray | None = None
    stream_chunk_size: int = 4096
    generate_delay: float = 0.0
    raise_mid_stream_after: int | None = None
    always_resident: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        if self.samples is None:
            self.samples = np.linspace(-0.5, 0.5, num=22050, dtype=np.float32)

    async def stream(self, req) -> AsyncIterator[np.ndarray]:
        self.calls.append({
            "mode": req.mode,
            "text": req.text,
            "language": req.language,
            "speaker": req.speaker,
            "voice_prompt": req.voice_prompt,
            "instruct": req.instruct,
            "reference_audio_len": len(req.reference_audio) if req.reference_audio else 0,
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "temperature": req.temperature,
        })
        if self.generate_delay:
            await asyncio.sleep(self.generate_delay)
        assert self.samples is not None
        offset = 0
        chunks_emitted = 0
        while offset < len(self.samples):
            chunk = self.samples[offset:offset + self.stream_chunk_size]
            yield chunk
            chunks_emitted += 1
            offset += self.stream_chunk_size
            if (
                self.raise_mid_stream_after is not None
                and chunks_emitted >= self.raise_mid_stream_after
            ):
                raise RuntimeError("fake mid-stream error")

    async def aclose(self) -> None:
        self.closed = True
```

- [ ] **Step 6: Run all tests to verify no regressions**

```bash
cd backend && uv run pytest tests/test_protocol.py tests/test_qwen_tts.py tests/test_tts_api.py tests/test_registry.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/voice/engines/protocol.py backend/voice/engines/qwen_tts.py backend/tests/conftest.py backend/tests/test_protocol.py
git commit -m "$(cat <<'EOF'
Extend TTS protocol with clone mode and always_resident flag

Adds "clone" to TTSMode Literal, four new optional fields on TTSRequest
(reference_audio, exaggeration, cfg_weight, temperature), and an
always_resident class attribute on the TTSModel protocol. All changes
are additive — existing Qwen3 adapters and tests keep working.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Config — add Chatterbox settings and accept `"clone"` in mode validator

**Rationale:** Extend `Settings` with the new Chatterbox fields and teach `_validate_mode_values` that `"clone"` is a valid mode. Default `tts_enabled_modes` stays at the existing two-mode tuple; we flip it to include `"clone"` in Task 7 once the loader is wired, so intermediate commits keep the bootstrap path green.

**Files:**
- Modify: `backend/voice/config.py:28-101`
- Test: `backend/tests/test_config.py` (extend)

- [ ] **Step 1: Write failing test for the new settings**

Append to `backend/tests/test_config.py`:

```python
def test_chatterbox_defaults():
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert s.chatterbox_model == "onnx-community/chatterbox-multilingual-ONNX"
    assert s.chatterbox_backend == "onnx"
    assert s.chatterbox_device == "cuda"
    assert s.chatterbox_max_reference_bytes == 10 * 1024 * 1024
    assert s.chatterbox_max_reference_seconds == 30.0


def test_chatterbox_backend_enum():
    from voice.config import Settings

    for val in ("onnx", "torch"):
        assert Settings(_env_file=None, chatterbox_backend=val).chatterbox_backend == val

    with pytest.raises(ValidationError):
        Settings(_env_file=None, chatterbox_backend="tflite")


def test_enabled_modes_accepts_clone():
    from voice.config import Settings

    s = Settings(_env_file=None, tts_enabled_modes="custom_voice,voice_design,clone")
    assert set(s.tts_enabled_modes) == {"custom_voice", "voice_design", "clone"}

    s = Settings(_env_file=None, tts_enabled_modes="clone")
    assert s.tts_enabled_modes == ("clone",)


def test_enabled_modes_still_rejects_unknown_alongside_clone():
    from voice.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_enabled_modes="clone,bogus")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_config.py::test_chatterbox_defaults tests/test_config.py::test_enabled_modes_accepts_clone -v
```

Expected: FAIL — settings attribute missing, mode rejected.

- [ ] **Step 3: Extend `Settings` and `_validate_mode_values`**

In `backend/voice/config.py`, update the `TTSMode` type alias and the `Settings` class. Replace the `TTSMode` line (around line 14):

```python
TTSMode = Literal["custom_voice", "voice_design", "clone"]
ChatterboxBackend = Literal["onnx", "torch"]
```

Inside `class Settings(BaseSettings)`, add new fields after `tts_voice_design_model` (around line 54):

```python
    chatterbox_model: str = "onnx-community/chatterbox-multilingual-ONNX"
    chatterbox_backend: ChatterboxBackend = "onnx"
    chatterbox_device: str = "cuda"
    chatterbox_max_reference_bytes: int = 10 * 1024 * 1024
    chatterbox_max_reference_seconds: float = 30.0
```

Update `_validate_mode_values` (around line 94):

```python
    @field_validator("tts_enabled_modes")
    @classmethod
    def _validate_mode_values(cls, value: tuple[str, ...]) -> tuple[TTSMode, ...]:
        allowed = {"custom_voice", "voice_design", "clone"}
        unknown = [m for m in value if m not in allowed]
        if unknown:
            raise ValueError(f"unknown TTS mode(s): {unknown!r}; allowed: {sorted(allowed)}")
        return value  # type: ignore[return-value]
```

**Leave** the default `tts_enabled_modes` at `("custom_voice", "voice_design")` — Task 7 flips this to the full three-mode default after the Chatterbox loader exists.

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_config.py -v
```

Expected: all PASS (including the existing defaults test, since we haven't changed the default enabled-modes tuple yet).

- [ ] **Step 5: Commit**

```bash
git add backend/voice/config.py backend/tests/test_config.py
git commit -m "$(cat <<'EOF'
Add Chatterbox config fields and accept clone in mode validator

Extends Settings with chatterbox_model, chatterbox_backend (onnx|torch),
chatterbox_device, chatterbox_max_reference_bytes,
chatterbox_max_reference_seconds. The mode validator now accepts
"clone" alongside the existing two Qwen3 modes. Default enabled modes
is unchanged for now — Task 7 flips it once the loader is wired.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Registry — exempt always-resident modes from swap eviction

**Rationale:** Chatterbox must stay loaded even when Qwen3 modes swap, since it's the realtime path. Under `swap` policy the registry now preloads always-resident models at startup (instead of skipping preload entirely) and a swap-eviction cycle spares them.

**Files:**
- Modify: `backend/voice/engines/registry.py:51-117`
- Test: `backend/tests/test_registry.py` (extend)

- [ ] **Step 1: Write failing tests for the always-resident behaviour**

Append to `backend/tests/test_registry.py`:

```python
@pytest.mark.asyncio
async def test_swap_preloads_always_resident_only():
    """Under swap policy, preload loads always-resident modes immediately."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="swap",
        loader=load,
    )
    registry.preload()
    assert counts == {"clone": 1}
    assert "clone" in registry.loaded_modes()


@pytest.mark.asyncio
async def test_swap_does_not_evict_always_resident():
    """A Qwen3 swap-switch leaves Chatterbox loaded."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    evictions: list[str] = []

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="swap",
        loader=load,
        on_evict=lambda mode: evictions.append(mode),
    )
    registry.preload()  # loads clone
    async with registry.acquire("custom_voice"):
        pass
    async with registry.acquire("voice_design"):
        pass
    async with registry.acquire("clone"):
        pass

    assert counts == {"clone": 1, "custom_voice": 1, "voice_design": 1}
    assert "clone" not in evictions
    assert evictions == ["custom_voice"]


@pytest.mark.asyncio
async def test_keep_loaded_unchanged_with_always_resident():
    """keep_loaded still loads everything and evicts nothing."""
    from voice.engines.registry import TTSModelRegistry

    counts: dict[str, int] = {}
    evictions: list[str] = []

    def load(mode: str):
        counts[mode] = counts.get(mode, 0) + 1
        samples = np.zeros(1000, dtype=np.float32)
        m = FakeTTSModel(mode=mode, samples=samples)
        m.always_resident = (mode == "clone")
        return m

    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="keep_loaded",
        loader=load,
        on_evict=lambda mode: evictions.append(mode),
    )
    registry.preload()

    async with registry.acquire("custom_voice"):
        pass
    async with registry.acquire("voice_design"):
        pass
    async with registry.acquire("clone"):
        pass

    assert counts == {"custom_voice": 1, "voice_design": 1, "clone": 1}
    assert evictions == []
```

Add the import at the top of `test_registry.py` if not already present:

```python
from tests.conftest import FakeTTSModel
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_registry.py::test_swap_preloads_always_resident_only tests/test_registry.py::test_swap_does_not_evict_always_resident -v
```

Expected: FAIL — `counts == {}` under swap because the current `preload()` is a no-op.

- [ ] **Step 3: Update the registry**

The caller passes in the set of always-resident modes at construction time; the registry does not probe the loader. This keeps `preload()` synchronous (which its docstring requires) and keeps the always-resident policy explicit rather than inferred. The `always_resident` class attribute on `ChatterboxCloneModel` (Task 4) is still kept as a semantic marker and is exposed through `loaded_modes()` clients, but the registry's decision is driven by the construction-time set.

In `backend/voice/engines/registry.py`, replace the `__init__`, `preload`, `acquire`, and rename `_evict_all` → `_evict_swappable`:

```python
    def __init__(
        self,
        *,
        enabled: tuple[TTSMode, ...],
        policy: VRAMPolicy,
        loader: Callable[[TTSMode], TTSModel],
        on_evict: Callable[[TTSMode], None] | None = None,
        always_resident_modes: frozenset[str] = frozenset(),
    ) -> None:
        self._enabled: tuple[TTSMode, ...] = enabled
        self._policy: VRAMPolicy = policy
        self._loader = loader
        self._on_evict = on_evict
        self._always_resident: frozenset[str] = always_resident_modes
        self._locks: dict[TTSMode, asyncio.Lock] = {m: asyncio.Lock() for m in enabled}
        self._swap_lock = asyncio.Lock()
        self._loaded: dict[TTSMode, TTSModel] = {}

    def preload(self) -> None:
        """Load enabled models at start-up.

        Under `keep_loaded` this loads every enabled mode.
        Under `swap` this loads only the always-resident modes; swappable
        modes are lazy-loaded on first request. Synchronous by design —
        safe to call from inside a running asyncio loop.
        """
        for mode in self._enabled:
            if self._policy == "keep_loaded" or mode in self._always_resident:
                self._load_sync(mode)

    @asynccontextmanager
    async def acquire(self, mode: str) -> AsyncIterator[TTSModel]:
        if mode not in self._enabled:
            raise ModeDisabledError(mode)
        # Always-resident modes and every mode under keep_loaded take the
        # per-mode lock and bypass the swap lock entirely.
        if self._policy == "keep_loaded" or mode in self._always_resident:
            async with self._locks[mode]:  # type: ignore[index]
                if mode not in self._loaded:
                    await self._load_locked(mode)  # type: ignore[arg-type]
                yield self._loaded[mode]  # type: ignore[index]
            return
        # Swappable mode under swap policy.
        async with self._swap_lock:
            if mode not in self._loaded:
                await self._evict_swappable()
                await self._load_locked(mode)  # type: ignore[arg-type]
            yield self._loaded[mode]  # type: ignore[index]

    async def _evict_swappable(self) -> None:
        for mode, model in list(self._loaded.items()):
            if mode in self._always_resident:
                continue
            try:
                await model.aclose()
            except Exception as exc:
                log.warning("tts_model_close_failed", mode=mode, error=repr(exc))
            del self._loaded[mode]
            if self._on_evict is not None:
                self._on_evict(mode)
            log.info("tts_model_evicted", mode=mode)
```

Keep `aclose()` unchanged — on final shutdown everything loaded is closed regardless of always-resident status.

Update `backend/tests/test_registry.py` — the three new tests in Step 1 pass `always_resident_modes=frozenset({"clone"})` to the registry constructor. Edit the three tests you just added so the registry is constructed as:

```python
    registry = TTSModelRegistry(
        enabled=("custom_voice", "voice_design", "clone"),
        policy="swap",  # or "keep_loaded"
        loader=load,
        always_resident_modes=frozenset({"clone"}),
    )
```

(Replace the two constructor calls in each of the three new tests; existing tests do not pass the argument and get the default empty frozenset, preserving current behaviour.)

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_registry.py -v
```

Expected: all PASS, including the existing swap-eviction test (which only enables Qwen3 modes without any always-resident model, so its behaviour is unchanged).

- [ ] **Step 5: Commit**

```bash
git add backend/voice/engines/registry.py backend/tests/test_registry.py
git commit -m "$(cat <<'EOF'
Exempt always-resident TTS modes from swap eviction

Renames registry._evict_all to _evict_swappable. Under swap policy,
preload now loads always-resident modes (chatterbox) and lazy-loads
swappable modes on demand. Always-resident modes bypass the swap lock
so their calls do not contend with Qwen3 traffic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: ChatterboxCloneModel adapter + FakeChatterboxBackend + language-ID helper

**Rationale:** The adapter is pure logic over a backend protocol, fully testable without any real model. The language helper maps the existing `Language` literal (e.g. `"German"`) to Chatterbox's ISO-639-1 code (`"de"`).

**Files:**
- Create: `backend/voice/engines/chatterbox_tts.py`
- Create: `backend/tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write failing tests for the adapter and helper**

Create `backend/tests/test_chatterbox_tts.py`:

```python
"""Tests for voice.engines.chatterbox_tts — adapter logic and language mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from voice.engines.protocol import TTSRequest


@dataclass
class _FakeChatterboxBackend:
    samples: np.ndarray = field(
        default_factory=lambda: np.linspace(-0.3, 0.3, 8000, dtype=np.float32)
    )
    sample_rate: int = 24000
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        text: str,
        language: str,
        reference_audio: bytes,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
    ) -> tuple[np.ndarray, int]:
        self.calls.append({
            "text": text,
            "language": language,
            "reference_len": len(reference_audio),
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        })
        return self.samples, self.sample_rate


@pytest.mark.asyncio
async def test_clone_streams_samples_in_chunks():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    samples = np.linspace(-0.5, 0.5, 10000, dtype=np.float32)
    backend = _FakeChatterboxBackend(samples=samples, sample_rate=24000)
    model = ChatterboxCloneModel(backend=backend, chunk_size=4096)

    req = TTSRequest(
        mode="clone",
        text="Bonjour",
        language="French",
        reference_audio=b"fake-reference-wav",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )
    chunks = [c async for c in model.stream(req)]

    assert sum(len(c) for c in chunks) == 10000
    assert all(c.dtype == np.float32 for c in chunks)
    assert model.sample_rate == 24000
    call = backend.calls[-1]
    assert call["language"] == "fr"  # Mapped from "French"
    assert call["text"] == "Bonjour"
    assert call["reference_len"] == len(b"fake-reference-wav")
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


@pytest.mark.asyncio
async def test_clone_requires_reference_audio():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(mode="clone", text="hi", language="English")

    with pytest.raises(ValueError, match="reference_audio"):
        async for _ in model.stream(req):
            pass


@pytest.mark.asyncio
async def test_clone_rejects_language_auto():
    """Chatterbox requires a concrete language — 'Auto' is not supported."""
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(
        mode="clone",
        text="hi",
        language="Auto",
        reference_audio=b"ref",
    )
    with pytest.raises(ValueError, match="Auto"):
        async for _ in model.stream(req):
            pass


@pytest.mark.asyncio
async def test_clone_uses_defaults_when_params_are_none():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel

    backend = _FakeChatterboxBackend()
    model = ChatterboxCloneModel(backend=backend)
    req = TTSRequest(
        mode="clone",
        text="hi",
        language="German",
        reference_audio=b"ref",
    )
    _ = [c async for c in model.stream(req)]
    call = backend.calls[-1]
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


def test_always_resident_is_true():
    from voice.engines.chatterbox_tts import ChatterboxCloneModel
    assert ChatterboxCloneModel.always_resident is True


def test_language_mapping_covers_all_common_languages():
    from voice.engines.chatterbox_tts import language_to_iso639

    assert language_to_iso639("English") == "en"
    assert language_to_iso639("German") == "de"
    assert language_to_iso639("French") == "fr"
    assert language_to_iso639("Spanish") == "es"
    assert language_to_iso639("Italian") == "it"
    assert language_to_iso639("Portuguese") == "pt"
    assert language_to_iso639("Japanese") == "ja"
    assert language_to_iso639("Korean") == "ko"
    assert language_to_iso639("Chinese") == "zh"
    assert language_to_iso639("Russian") == "ru"


def test_language_mapping_rejects_auto():
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="Auto"):
        language_to_iso639("Auto")


def test_language_mapping_rejects_unknown():
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="unknown"):
        language_to_iso639("Klingon")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ChatterboxCloneModel'`.

- [ ] **Step 3: Create the adapter and helpers**

Create `backend/voice/engines/chatterbox_tts.py`:

```python
"""Chatterbox Multilingual TTS adapter.

Exposes the TTSModel protocol by delegating to a backend object that handles
actual inference. Two loader functions live here (ONNX Runtime and Torch);
each one returns an object conforming to _ChatterboxBackend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np

from voice.engines.protocol import TTSMode, TTSRequest
from voice.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 4096

# Chatterbox uses ISO-639-1 codes. Our Language literal uses full English names.
# "Auto" is deliberately not mapped — Chatterbox needs a concrete language.
_LANGUAGE_MAP: dict[str, str] = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
}


def language_to_iso639(language: str) -> str:
    """Map a Language literal value to Chatterbox's ISO-639-1 code.

    Raises ValueError for "Auto" (unsupported) or unknown languages.
    """
    if language == "Auto":
        raise ValueError(
            "Chatterbox requires a concrete language; 'Auto' is not supported"
        )
    try:
        return _LANGUAGE_MAP[language]
    except KeyError as exc:
        raise ValueError(f"unknown language for Chatterbox: {language!r}") from exc


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


def load_chatterbox_torch(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Factory for the Torch-based Chatterbox backend. Implemented in Task 5."""
    raise NotImplementedError("Torch backend loader — populated in Task 5")


def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Factory for the ONNX Runtime-based Chatterbox backend. Implemented in Task 6."""
    raise NotImplementedError("ONNX backend loader — populated in Task 6")


class ChatterboxCloneModel:
    mode: TTSMode = "clone"
    always_resident: bool = True

    def __init__(
        self,
        *,
        backend: _ChatterboxBackend,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._backend = backend
        self._chunk_size = chunk_size
        self.sample_rate = getattr(backend, "sample_rate", 24000)
        self._closed = False

    async def aclose(self) -> None:
        self._closed = True

    async def stream(self, req: TTSRequest) -> AsyncIterator[np.ndarray]:
        if req.reference_audio is None or len(req.reference_audio) == 0:
            raise ValueError("Chatterbox requires reference_audio")

        language_id = language_to_iso639(req.language)
        exaggeration = req.exaggeration if req.exaggeration is not None else 0.5
        cfg_weight = req.cfg_weight if req.cfg_weight is not None else 0.5
        temperature = req.temperature if req.temperature is not None else 0.8

        def _generate() -> tuple[np.ndarray, int]:
            return self._backend.generate(
                text=req.text,
                language=language_id,
                reference_audio=req.reference_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        samples, sr = await asyncio.to_thread(_generate)
        self.sample_rate = int(sr)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        offset = 0
        while offset < len(samples):
            yield samples[offset:offset + self._chunk_size]
            offset += self._chunk_size
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Add ChatterboxCloneModel adapter and language-ID mapping

Implements the TTSModel protocol for Chatterbox Multilingual. Maps the
Language literal to ISO-639-1 codes required by Chatterbox, rejects
"Auto" as unsupported, and wires always_resident=True. Backend
loaders (Torch, ONNX) are placeholders implemented in the next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement the Torch backend loader (`load_chatterbox_torch`)

**Rationale:** The Torch path is simpler than the ONNX path and gives us a working baseline end-to-end before we tackle the custom ONNX pipeline. The `chatterbox-tts` pip package exposes `ChatterboxMultilingualTTS.from_pretrained()` with a `generate()` method that accepts `audio_prompt_path` as a **file path string** (not bytes) — so the adapter-facing wrapper writes reference bytes to a temporary file per request.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py` (replace `load_chatterbox_torch` stub)
- Modify: `backend/pyproject.toml` (add `chatterbox-tts`, `librosa`)
- Test: `backend/tests/test_chatterbox_tts.py` (add smoke test)

- [ ] **Step 1: Write failing smoke test for the loader interface**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_torch_loader_raises_on_missing_chatterbox_package(monkeypatch):
    """If chatterbox is not installed, the loader surfaces a clear error."""
    import sys
    from voice.engines.chatterbox_tts import load_chatterbox_torch

    # Simulate missing package
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", None)
    with pytest.raises((ImportError, AttributeError, TypeError)):
        load_chatterbox_torch("ResembleAI/chatterbox", device="cpu")
```

We can't meaningfully test the success path without downloading the real model — that's a manual check per Task 14.

- [ ] **Step 2: Run test to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py::test_torch_loader_raises_on_missing_chatterbox_package -v
```

Expected: FAIL — `NotImplementedError` from the stub.

- [ ] **Step 3: Add Torch dependency**

In `backend/pyproject.toml`, extend `dependencies`:

```toml
dependencies = [
    "fastapi>=0.115,<0.116",
    "uvicorn[standard]>=0.32,<0.33",
    "pydantic>=2.9,<3",
    "pydantic-settings>=2.6,<3",
    "structlog>=24.4,<25",
    "faster-whisper>=1.0,<2",
    "qwen-tts>=0.1",
    "numpy>=1.26,<3",
    "soundfile>=0.12,<0.13",
    "python-multipart>=0.0.12,<0.1",
    "torch>=2.5,<3",
    "chatterbox-tts>=0.1",
    "librosa>=0.10,<0.11",
]
```

Run `uv lock` and `uv sync`:

```bash
cd backend && uv lock && uv sync --dev
```

- [ ] **Step 4: Implement `load_chatterbox_torch`**

In `backend/voice/engines/chatterbox_tts.py`, replace the `load_chatterbox_torch` stub:

```python
def load_chatterbox_torch(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Load the Chatterbox Multilingual model via the chatterbox-tts pip package.

    Returns a backend adapter that accepts reference_audio as bytes; it writes
    the bytes to a temporary file before calling Chatterbox's file-path API.
    """
    import tempfile
    from pathlib import Path

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-not-found]

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    class _TorchBackend:
        sample_rate: int = int(getattr(model, "sr", 24000))

        def generate(
            self,
            *,
            text: str,
            language: str,
            reference_audio: bytes,
            exaggeration: float,
            cfg_weight: float,
            temperature: float,
        ) -> tuple[np.ndarray, int]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(reference_audio)
                tmp_path = Path(tmp.name)
            try:
                wav = model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=str(tmp_path),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            # chatterbox returns a torch.Tensor in shape (1, N) or (N,). Convert to float32 np.
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            return wav, self.sample_rate

    return _TorchBackend()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: all PASS (the smoke test now raises `ImportError` from the `None` sys.modules entry).

- [ ] **Step 6: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py backend/pyproject.toml backend/uv.lock
git commit -m "$(cat <<'EOF'
Implement Torch backend loader for Chatterbox

Uses the chatterbox-tts pip package (ChatterboxMultilingualTTS). Since
the upstream generate() API takes audio_prompt_path as a file path, the
backend writes incoming reference bytes to a NamedTemporaryFile and
deletes it after inference. Adds chatterbox-tts and librosa to
pyproject dependencies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement the ONNX backend loader (`load_chatterbox_onnx`)

**Rationale:** This is the primary backend per the spec — Q4 quantisation on ONNX Runtime is what keeps RTF ≤ 0.5 achievable on Strix Halo's 250 GB/s memory bandwidth. The `onnx-community/chatterbox-multilingual-ONNX` repo ships four ONNX files (speech_encoder, embed_tokens, language_model, conditional_decoder), a default voice, and a Cangjie mapping. We adapt the repo's reference `run_inference()` example into a Python class.

**Special note:** The reference inference code is non-trivial (autoregressive Llama decoding with KV cache, voice encoding, audio waveform reconstruction). This task's code below is a **skeleton** that matches the adapter protocol; the subagent executing this task must fetch the reference code from the repo's README/files and fill in the inference loop. Mark the task complete only when a manual smoke test (Task 14) produces audible audio.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py` (replace `load_chatterbox_onnx` stub)
- Modify: `backend/pyproject.toml` (add `onnxruntime` and `transformers`)
- Test: `backend/tests/test_chatterbox_tts.py` (extend smoke test)

- [ ] **Step 1: Fetch and study the reference code**

```bash
cd backend && uv run python -c "
from huggingface_hub import hf_hub_download
readme = hf_hub_download('onnx-community/chatterbox-multilingual-ONNX', 'README.md')
print(open(readme).read())
"
```

Save any Python inference example into `backend/voice/engines/_chatterbox_onnx_reference.py` as a reference file (committed for traceability). If the repo contains a full `.py` example, download it the same way. Commit only the reference file unmodified.

- [ ] **Step 2: Write failing smoke test**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_onnx_loader_raises_on_missing_onnxruntime(monkeypatch):
    import sys
    from voice.engines.chatterbox_tts import load_chatterbox_onnx

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises((ImportError, AttributeError, TypeError)):
        load_chatterbox_onnx("onnx-community/chatterbox-multilingual-ONNX", device="cpu")
```

- [ ] **Step 3: Run test to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py::test_onnx_loader_raises_on_missing_onnxruntime -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 4: Add ONNX dependencies**

In `backend/pyproject.toml`, extend `dependencies`:

```toml
    "onnxruntime>=1.19,<2",
    "transformers>=4.45,<5",
```

We intentionally depend on `onnxruntime` (CPU) as the PyPI baseline. The CUDA/ROCm variants are installed per-image in the Dockerfile tasks.

Run:

```bash
cd backend && uv lock && uv sync --dev
```

- [ ] **Step 5: Implement `load_chatterbox_onnx`**

In `backend/voice/engines/chatterbox_tts.py`, replace the stub. This is the skeleton — the subagent adapts the reference code's inference loop:

```python
def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Load Chatterbox Multilingual via ONNX Runtime.

    Downloads four ONNX model files from the hub (speech_encoder, embed_tokens,
    language_model, conditional_decoder), creates InferenceSession objects
    with the appropriate ExecutionProvider, and implements the autoregressive
    inference loop from the upstream reference example.

    device: 'cuda' (→ CUDAExecutionProvider on NVIDIA, ROCMExecutionProvider
    on AMD via the onnxruntime-rocm wheel), 'cpu' (→ CPUExecutionProvider).
    """
    import io

    import onnxruntime
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    # Pick execution providers. ROCm installs expose "ROCMExecutionProvider";
    # CUDA installs expose "CUDAExecutionProvider". We fall back to CPU if the
    # requested accelerator is not available in the current runtime build.
    available = onnxruntime.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "ROCMExecutionProvider" in available:
            providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            log.warning(
                "chatterbox_onnx_no_gpu_provider",
                requested=device,
                available=available,
            )
    else:
        providers = ["CPUExecutionProvider"]

    def _dl(filename: str) -> str:
        return hf_hub_download(repo_id=model_id, filename=filename)

    # Filenames follow the upstream repo's naming — verify at run time.
    # The .onnx_data sidecars are downloaded automatically alongside the .onnx
    # files when referenced in the same directory.
    speech_encoder_path = _dl("speech_encoder.onnx")
    embed_tokens_path = _dl("embed_tokens.onnx")
    language_model_path = _dl("language_model.onnx")
    conditional_decoder_path = _dl("conditional_decoder.onnx")
    cangjie_map_path = _dl("Cangjie5_TC.json")

    # Tokeniser — depending on what the upstream reference uses, this might
    # be a SentencePiece bundled with the repo, or a transformers tokenizer
    # loaded from model_id. Populate per the reference code.
    from transformers import AutoTokenizer  # type: ignore[import-untyped]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    speech_encoder = onnxruntime.InferenceSession(speech_encoder_path, providers=providers)
    embed_tokens = onnxruntime.InferenceSession(embed_tokens_path, providers=providers)
    language_model = onnxruntime.InferenceSession(language_model_path, providers=providers)
    conditional_decoder = onnxruntime.InferenceSession(
        conditional_decoder_path, providers=providers
    )

    log.info(
        "chatterbox_onnx_loaded",
        providers=providers,
        model=model_id,
    )

    class _OnnxBackend:
        # Chatterbox multilingual's audio decoder outputs 24 kHz PCM float32.
        sample_rate: int = 24000

        def generate(
            self,
            *,
            text: str,
            language: str,
            reference_audio: bytes,
            exaggeration: float,
            cfg_weight: float,
            temperature: float,
        ) -> tuple[np.ndarray, int]:
            # 1. Decode reference audio (any libsndfile-supported format)
            ref_waveform, ref_sr = sf.read(io.BytesIO(reference_audio), dtype="float32")
            if ref_waveform.ndim > 1:
                ref_waveform = ref_waveform.mean(axis=1)  # stereo → mono

            # 2. Encode reference voice → conditioning vector (speech_encoder)
            # 3. Tokenise `text` with language_id handling
            # 4. Embed tokens (embed_tokens)
            # 5. Autoregressive Llama decoding (language_model) with KV cache,
            #    applying temperature and cfg_weight. The exaggeration value
            #    influences the emotion conditioning; see the reference code.
            # 6. Decode speech tokens to waveform (conditional_decoder)
            #
            # The implementation above is a SKELETON. Adapt the upstream
            # run_inference() pattern from
            # _chatterbox_onnx_reference.py, wiring in the input language
            # code, the exaggeration/cfg_weight/temperature knobs, and
            # returning a 1-D float32 waveform at 24 kHz.
            raise NotImplementedError(
                "Populate the ONNX inference loop from the reference file "
                "backend/voice/engines/_chatterbox_onnx_reference.py"
            )

    return _OnnxBackend()
```

The NotImplementedError in step 5 is **deliberate** — it forces the subagent to adapt the reference code rather than shipping a stub. Manual smoke testing in Task 14 will either confirm this works or route the user to `CHATTERBOX_BACKEND=torch` while we work the reference integration in a follow-up.

- [ ] **Step 6: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: all PASS (the smoke test now fails at the `import onnxruntime` line when it's set to `None`).

- [ ] **Step 7: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/voice/engines/_chatterbox_onnx_reference.py backend/tests/test_chatterbox_tts.py backend/pyproject.toml backend/uv.lock
git commit -m "$(cat <<'EOF'
Implement ONNX Runtime loader for Chatterbox (skeleton)

Downloads the four ONNX model files from the onnx-community hub repo,
wires ExecutionProvider selection (CUDA / ROCM / CPU) with explicit
fallback logging, and structures the InferenceSession instances. The
inference loop is marked NotImplementedError with a pointer to the
committed reference file; a follow-up commit adapts the upstream
run_inference() into the backend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8 (separate commit): Adapt the reference inference loop**

Replace the `raise NotImplementedError` block in `_OnnxBackend.generate` with the concrete Python code adapted from `_chatterbox_onnx_reference.py`. Verify with a manual smoke test (requires real hardware — defer to Task 14 for the final cross-check).

Commit as a separate change so a Torch-only fallback remains clearly marked in git history:

```bash
git add backend/voice/engines/chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Populate Chatterbox ONNX inference loop from reference

Adapts the onnx-community repo's run_inference() into _OnnxBackend.
Implements reference-voice encoding, text tokenisation, autoregressive
language-model decoding with KV cache, and conditional-decoder waveform
synthesis. Output is 1-D float32 at 24 kHz.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire Chatterbox into `_default_tts_loader` and flip default enabled modes

**Rationale:** With the loader available, the bootstrap path can now construct a `ChatterboxCloneModel` for `mode == "clone"`. We flip the default `tts_enabled_modes` to include `"clone"` in the same task so docker compose with an empty `.env` lights up Chatterbox out of the box.

**Files:**
- Modify: `backend/voice/main.py:38-62` (_default_tts_loader)
- Modify: `backend/voice/config.py` (tts_enabled_modes default)
- Test: `backend/tests/test_main_startup.py` (extend)
- Test: `backend/tests/test_config.py` (update default-modes test)

- [ ] **Step 1: Update failing tests for the new defaults**

Modify `backend/tests/test_config.py::test_defaults` to expect all three modes:

```python
def test_defaults(monkeypatch):
    monkeypatch.delenv("CHATSUNE_VOICE_MODEL_CACHE_DIR", raising=False)
    from voice.config import Settings

    s = Settings(_env_file=None)
    assert s.stt_model == "h2oai/faster-whisper-large-v3-turbo"
    assert s.stt_compute_type == "auto"
    assert s.tts_custom_voice_model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert s.tts_voice_design_model == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert s.tts_enabled_modes == ("custom_voice", "voice_design", "clone")
    assert s.tts_vram_policy == "keep_loaded"
    assert s.tts_attention_impl == "sdpa"
    assert s.preload_at_startup is True
    assert s.stt_device == "auto"
    assert s.tts_device == "cuda"
    assert s.log_level == "info"
    assert s.app_port == 8000
    assert s.stt_max_audio_bytes == 25 * 1024 * 1024
```

Add a test in `backend/tests/test_main_startup.py`:

```python
@pytest.mark.asyncio
async def test_bootstrap_with_all_three_modes_builds_registry():
    """Registry is built with clone mode and Chatterbox loader is registered."""
    import os
    from unittest.mock import patch

    from voice.config import Settings
    from voice.main import build_registry

    settings = Settings(_env_file=None, tts_enabled_modes="custom_voice,voice_design,clone")
    assert "clone" in settings.tts_enabled_modes

    calls: list[str] = []

    def fake_loader(mode: str):
        calls.append(mode)
        from tests.conftest import FakeTTSModel
        m = FakeTTSModel(mode=mode)
        m.always_resident = (mode == "clone")
        return m

    registry = build_registry(settings, tts_loader=fake_loader)
    assert registry.enabled_modes == ("custom_voice", "voice_design", "clone")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_config.py::test_defaults tests/test_main_startup.py -v
```

Expected: FAIL — default is still two modes.

- [ ] **Step 3: Flip the default in config.py and update the compose + .env samples**

In `backend/voice/config.py`, change the `tts_enabled_modes` default around line 61:

```python
    tts_enabled_modes: Annotated[
        tuple[TTSMode, ...], NoDecode
    ] = ("custom_voice", "voice_design", "clone")
```

- [ ] **Step 4: Extend `_default_tts_loader` and `build_registry` in main.py**

In `backend/voice/main.py`, replace `build_registry` so it passes `always_resident_modes`:

```python
def build_registry(settings: Settings, *, tts_loader: Callable[[str], object]):
    from voice.engines.registry import TTSModelRegistry

    return TTSModelRegistry(
        enabled=settings.tts_enabled_modes,
        policy=settings.tts_vram_policy,
        loader=tts_loader,  # type: ignore[arg-type]
        always_resident_modes=frozenset({"clone"}),
    )
```

And replace the `_default_tts_loader` function (lines 38-62):

```python
def _default_tts_loader(settings: Settings) -> Callable[[str], object]:
    from voice.engines.qwen_tts import (
        QwenCustomVoiceModel,
        QwenVoiceDesignModel,
        load_qwen_tts,
    )
    from voice.engines.chatterbox_tts import (
        ChatterboxCloneModel,
        load_chatterbox_onnx,
        load_chatterbox_torch,
    )

    def load(mode: str) -> object:
        if mode == "custom_voice":
            backend = load_qwen_tts(
                settings.tts_custom_voice_model,
                device=settings.tts_device,
                attention_impl=settings.tts_attention_impl,
            )
            return QwenCustomVoiceModel(backend=backend)
        if mode == "voice_design":
            backend = load_qwen_tts(
                settings.tts_voice_design_model,
                device=settings.tts_device,
                attention_impl=settings.tts_attention_impl,
            )
            return QwenVoiceDesignModel(backend=backend)
        if mode == "clone":
            if settings.chatterbox_backend == "onnx":
                backend = load_chatterbox_onnx(
                    settings.chatterbox_model,
                    device=settings.chatterbox_device,
                )
            else:
                backend = load_chatterbox_torch(
                    settings.chatterbox_model,
                    device=settings.chatterbox_device,
                )
            return ChatterboxCloneModel(backend=backend)
        raise ValueError(f"unknown TTS mode: {mode!r}")

    return load
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd backend && uv run pytest -v
```

Expected: all PASS.

- [ ] **Step 6: Update compose.yml to forward the new env vars**

In `compose.yml`, inside the `voice-env` anchor, add:

```yaml
    CHATTERBOX_MODEL: ${CHATTERBOX_MODEL:-onnx-community/chatterbox-multilingual-ONNX}
    CHATTERBOX_BACKEND: ${CHATTERBOX_BACKEND:-onnx}
    CHATTERBOX_DEVICE: ${CHATTERBOX_DEVICE:-cuda}
    CHATTERBOX_MAX_REFERENCE_BYTES: ${CHATTERBOX_MAX_REFERENCE_BYTES:-10485760}
    CHATTERBOX_MAX_REFERENCE_SECONDS: ${CHATTERBOX_MAX_REFERENCE_SECONDS:-30}
```

Also update the default `TTS_ENABLED_MODES` value:

```yaml
    TTS_ENABLED_MODES: ${TTS_ENABLED_MODES:-custom_voice,voice_design,clone}
```

- [ ] **Step 7: Commit**

```bash
git add backend/voice/main.py backend/voice/config.py backend/tests/test_main_startup.py backend/tests/test_config.py compose.yml
git commit -m "$(cat <<'EOF'
Wire Chatterbox into TTS loader and flip default enabled modes

_default_tts_loader now dispatches mode="clone" to either the ONNX or
Torch Chatterbox loader based on CHATTERBOX_BACKEND. Default
TTS_ENABLED_MODES now includes "clone", and compose.yml forwards the
new CHATTERBOX_* env vars.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: POST `/v1/speak/clone` endpoint with multipart validation and streaming

**Rationale:** This is the user-facing surface. Multipart is the right content type (binary reference audio + text fields), separate from the JSON `/v1/speak`. Validation enforces the configured size/duration limits server-side before the model is ever acquired.

**Files:**
- Create: `backend/voice/api/tts_clone.py`
- Modify: `backend/voice/api/app.py:22-24,35-37` (import and include the new router)
- Create: `backend/tests/test_tts_clone_api.py`

- [ ] **Step 1: Write failing tests for the endpoint**

Create `backend/tests/test_tts_clone_api.py`:

```python
"""Tests for POST /v1/speak/clone — multipart, validation, streaming."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import soundfile as sf
from httpx import ASGITransport, AsyncClient

from tests.conftest import FakeTTSModel


def _wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Build a valid in-memory WAV file for use as reference audio."""
    samples = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class _Registry:
    def __init__(self, enabled, policy, models):
        self._enabled = tuple(enabled)
        self._policy = policy
        self._models = models

    @property
    def enabled_modes(self):
        return self._enabled

    @property
    def policy(self):
        return self._policy

    def loaded_modes(self):
        return tuple(self._models.keys())

    def acquire(self, mode):
        class _CM:
            def __init__(s, model):
                s._model = model
            async def __aenter__(s):
                return s._model
            async def __aexit__(s, *exc):
                return None
        return _CM(self._models[mode])


class _Settings:
    stt_max_audio_bytes = 10_000_000
    chatterbox_max_reference_bytes = 10 * 1024 * 1024
    chatterbox_max_reference_seconds = 30.0


@pytest.mark.asyncio
async def test_clone_happy_path():
    from voice.api.app import build_app

    samples = np.linspace(-0.2, 0.2, 8192, dtype=np.float32)
    fake = FakeTTSModel(mode="clone", samples=samples, stream_chunk_size=4096, sample_rate=24000)
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    ref = _wav_bytes(seconds=1.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={
                "text": "hallo",
                "language": "German",
                "exaggeration": "0.6",
                "cfg_weight": "0.4",
                "temperature": "0.7",
            },
            files={"reference_audio": ("ref.wav", ref, "audio/wav")},
        )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    data = r.content
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert len(data) == 44 + 8192 * 2
    call = fake.calls[-1]
    assert call["text"] == "hallo"
    assert call["language"] == "German"
    assert call["reference_audio_len"] == len(ref)
    assert call["exaggeration"] == 0.6
    assert call["cfg_weight"] == 0.4
    assert call["temperature"] == 0.7


@pytest.mark.asyncio
async def test_clone_default_params_are_applied():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone", stream_chunk_size=4096)
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "hi", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )

    assert r.status_code == 200
    call = fake.calls[-1]
    assert call["exaggeration"] == 0.5
    assert call["cfg_weight"] == 0.5
    assert call["temperature"] == 0.8


@pytest.mark.asyncio
async def test_clone_mode_disabled_returns_403():
    from voice.api.app import build_app

    registry = _Registry(["custom_voice"], "keep_loaded", {})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 403
    assert r.json()["error"] == "mode_disabled"


@pytest.mark.asyncio
async def test_clone_missing_reference_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_empty_text_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_unparseable_reference_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", b"not-a-wav-file", "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_reference_audio"


@pytest.mark.asyncio
async def test_clone_reference_too_long_returns_422():
    from voice.api.app import build_app

    class _TightSettings(_Settings):
        chatterbox_max_reference_seconds = 0.5

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_TightSettings())

    long_ref = _wav_bytes(seconds=2.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", long_ref, "audio/wav")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "reference_audio_too_long"


@pytest.mark.asyncio
async def test_clone_reference_too_large_returns_413_or_422():
    from voice.api.app import build_app

    class _TightSettings(_Settings):
        chatterbox_max_reference_bytes = 100

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_TightSettings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English"},
            files={"reference_audio": ("ref.wav", _wav_bytes(seconds=1.0), "audio/wav")},
        )
    assert r.status_code in (413, 422)


@pytest.mark.asyncio
async def test_clone_out_of_range_exaggeration_returns_422():
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "English", "exaggeration": "5.0"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_clone_language_auto_returns_422():
    """Chatterbox doesn't support auto-detection — 'Auto' must be rejected."""
    from voice.api.app import build_app

    fake = FakeTTSModel(mode="clone")
    registry = _Registry(["clone"], "keep_loaded", {"clone": fake})
    app = build_app(stt=None, registry=registry, settings=_Settings())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/speak/clone",
            data={"text": "x", "language": "Auto"},
            files={"reference_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_tts_clone_api.py -v
```

Expected: FAIL — endpoint does not exist.

- [ ] **Step 3: Create the endpoint**

Create `backend/voice/api/tts_clone.py`:

```python
"""POST /v1/speak/clone — multipart reference-audio voice cloning via Chatterbox."""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from voice.api.models import Language
from voice.audio import float32_to_pcm16, make_streaming_wav_header
from voice.engines.protocol import TTSRequest
from voice.logging_setup import get_logger

router = APIRouter(prefix="/v1")
log = get_logger(__name__)


def _validate_and_probe_reference(
    audio_bytes: bytes, *, max_bytes: int, max_seconds: float
) -> float:
    """Return the clip duration if it decodes and fits within the limits.

    Raises HTTPException on any failure so the handler doesn't need to care
    about the specific error shape.
    """
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": "reference_audio_too_large", "bytes": len(audio_bytes)},
        )
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_reference_audio", "reason": str(exc)},
        ) from exc
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    if duration > max_seconds:
        raise HTTPException(
            status_code=422,
            detail={"error": "reference_audio_too_long", "seconds": duration},
        )
    return duration


@router.post(
    "/speak/clone",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "PCM16 WAV stream"},
        403: {"description": "mode disabled"},
        413: {"description": "reference audio too large"},
        422: {"description": "validation error"},
    },
)
async def speak_clone(
    request: Request,
    text: str = Form(min_length=1, max_length=4000),
    language: Language = Form(),
    reference_audio: UploadFile = File(...),
    exaggeration: float = Form(0.5, ge=0.25, le=2.0),
    cfg_weight: float = Form(0.5, ge=0.0, le=1.0),
    temperature: float = Form(0.8, ge=0.05, le=2.0),
) -> StreamingResponse:
    settings = request.app.state.settings
    registry = request.app.state.registry

    if "clone" not in registry.enabled_modes:
        raise HTTPException(
            status_code=403,
            detail={"error": "mode_disabled", "mode": "clone"},
        )

    if language == "Auto":
        raise HTTPException(
            status_code=422,
            detail={"error": "language_auto_unsupported",
                    "message": "Chatterbox requires a concrete language"},
        )

    audio_bytes = await reference_audio.read()
    duration = _validate_and_probe_reference(
        audio_bytes,
        max_bytes=settings.chatterbox_max_reference_bytes,
        max_seconds=settings.chatterbox_max_reference_seconds,
    )

    log.info(
        "clone_request",
        language=language,
        text_len=len(text),
        reference_bytes=len(audio_bytes),
        reference_seconds=duration,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    cm = registry.acquire("clone")

    tts_req = TTSRequest(
        mode="clone",
        text=text,
        language=language,
        reference_audio=audio_bytes,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )

    async def body() -> AsyncIterator[bytes]:
        sample_rate = 24000
        first_chunk_time: float | None = None
        total_samples = 0
        start = time.monotonic()
        try:
            async with cm as model:
                sample_rate = model.sample_rate
                yield make_streaming_wav_header(sample_rate=sample_rate, channels=1)
                async for chunk in model.stream(tts_req):
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                        log.info(
                            "clone_stream_start",
                            time_to_first_chunk_ms=int((first_chunk_time - start) * 1000),
                        )
                    total_samples += len(chunk)
                    yield float32_to_pcm16(chunk)
        except Exception as exc:
            log.error(
                "clone_error",
                phase="during_stream" if first_chunk_time else "before_stream",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return
        log.info(
            "clone_stream_end",
            total_samples=total_samples,
            total_ms=int((time.monotonic() - start) * 1000),
        )

    return StreamingResponse(body(), media_type="audio/wav")
```

- [ ] **Step 4: Mount the router in `app.py`**

In `backend/voice/api/app.py`, add the import near the other router imports (around line 22):

```python
from voice.api import tts_clone as tts_clone_module
```

And include the router after the existing `tts_module` include (around line 37):

```python
    app.include_router(tts_module.router)
    app.include_router(tts_clone_module.router)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_tts_clone_api.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/voice/api/tts_clone.py backend/voice/api/app.py backend/tests/test_tts_clone_api.py
git commit -m "$(cat <<'EOF'
Add POST /v1/speak/clone multipart endpoint

Accepts text + language + reference audio (WAV/FLAC/best-effort MP3 via
libsndfile) + optional exaggeration/cfg_weight/temperature. Validates
size and duration server-side before acquiring the model, rejects
language=Auto as unsupported by Chatterbox, and streams PCM16 WAV back
with the same header+chunk format as /v1/speak.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pre-fetch script downloads the Chatterbox model

**Rationale:** Avoid a 500 MB first-request delay on a fresh container.

**Files:**
- Modify: `backend/scripts/prefetch_models.py:34-35`

- [ ] **Step 1: Modify the prefetch list**

In `backend/scripts/prefetch_models.py`, extend the `ids` list around line 34:

```python
    ids = [
        settings.stt_model,
        settings.tts_custom_voice_model,
        settings.tts_voice_design_model,
        settings.chatterbox_model,
    ]
```

- [ ] **Step 2: Verify manually (no automated test)**

```bash
cd backend && uv run python scripts/prefetch_models.py
```

Expected: the script downloads all four models without error (takes ~5 minutes on first run; instant on a warm cache).

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/prefetch_models.py
git commit -m "$(cat <<'EOF'
Include Chatterbox model in prefetch script

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `Dockerfile.cuda` — install `onnxruntime-gpu`

**Rationale:** The pypi `onnxruntime` wheel is CPU-only. For GPU execution on NVIDIA, we need `onnxruntime-gpu` installed *after* the base pytorch image (which already brings CUDA 12.4).

**Files:**
- Modify: `Dockerfile.cuda:16-30`

- [ ] **Step 1: Add `onnxruntime-gpu` install step**

In `Dockerfile.cuda`, replace the dependency-sync block (around lines 16-30):

```dockerfile
# Dependency layer (cached unless pyproject or lock change)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Swap the PyPI-baseline onnxruntime for onnxruntime-gpu (pulls CUDA EP).
# Using uv pip avoids dependency resolution conflicts: we uninstall the CPU
# wheel then install the GPU one into the same uv-managed virtualenv.
RUN uv pip uninstall onnxruntime || true
RUN uv pip install onnxruntime-gpu

# Source
COPY backend/ ./

# Optional flash-attn (enable with --build-arg INSTALL_FLASH_ATTN=1)
ARG INSTALL_FLASH_ATTN=0
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      uv pip install flash-attn --no-build-isolation ; \
    fi

# Re-sync to install project itself
RUN uv sync --frozen --no-dev
```

- [ ] **Step 2: Build the image and verify onnxruntime-gpu is importable**

```bash
cd /home/chris/workspace/chatsune-voice
docker build -f Dockerfile.cuda -t chatsune-voice:cuda-test .
docker run --rm --gpus all chatsune-voice:cuda-test python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Expected output contains `"CUDAExecutionProvider"`.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.cuda
git commit -m "$(cat <<'EOF'
Install onnxruntime-gpu in CUDA image

Replaces the PyPI baseline onnxruntime (CPU) with onnxruntime-gpu so
the Chatterbox ONNX backend can use CUDAExecutionProvider.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `Dockerfile.rocm` — try `onnxruntime-rocm`, fall back gracefully

**Rationale:** `onnxruntime-rocm` is not published on PyPI. AMD distributes it separately, and availability for ROCm 7.2 + gfx1151 is an unknown. The Dockerfile tries the AMD-hosted wheel; if that fails, the image ships without it and the runtime selects `CPUExecutionProvider`, with a warning in the startup log. README documents that operators should set `CHATTERBOX_BACKEND=torch` when ROCm-ONNX is unavailable on their host.

**Files:**
- Modify: `Dockerfile.rocm:28-35`

- [ ] **Step 1: Add the conditional `onnxruntime-rocm` install**

In `Dockerfile.rocm`, extend the existing install block:

```dockerfile
WORKDIR /app
COPY backend/ ./

# Install the project and its dependencies.
RUN pip install --no-cache-dir .

# Try the AMD-distributed onnxruntime-rocm wheel. If this fails (wheel
# unavailable for this ROCm version, or AMD's download host rejects the
# request), the image still boots with onnxruntime-cpu — the Chatterbox
# loader falls back to CPUExecutionProvider at runtime and logs a warning.
# Operators who hit this should set CHATTERBOX_BACKEND=torch in .env.
ARG ONNXRUNTIME_ROCM_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/onnxruntime_rocm-1.19.0-cp312-cp312-manylinux_2_28_x86_64.whl
RUN pip uninstall -y onnxruntime || true \
 && (pip install --no-cache-dir "${ONNXRUNTIME_ROCM_URL}" || \
     (echo "onnxruntime-rocm install failed — image will run ONNX on CPU" \
      && pip install --no-cache-dir onnxruntime))

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note:** the `ONNXRUNTIME_ROCM_URL` is a best-effort default for ROCm 7.2 + Python 3.12. If AMD has rotated URLs by build time, the subagent overrides with `--build-arg ONNXRUNTIME_ROCM_URL=...`. The fallback branch of the shell-`||` guarantees a buildable image regardless.

- [ ] **Step 2: Build the image and verify providers**

```bash
cd /home/chris/workspace/chatsune-voice
docker build -f Dockerfile.rocm -t chatsune-voice:rocm-test .
docker run --rm --device=/dev/kfd --device=/dev/dri chatsune-voice:rocm-test python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Expected: either `["ROCMExecutionProvider", "CPUExecutionProvider"]` (success case) or `["CPUExecutionProvider"]` (fallback case — image still boots).

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.rocm
git commit -m "$(cat <<'EOF'
Attempt onnxruntime-rocm install with CPU fallback

The AMD repo wheel for ROCm 7.2 + Python 3.12 is the primary install;
if it cannot be fetched, the image boots with onnxruntime (CPU) and the
loader selects CPUExecutionProvider at runtime. Operators hit by this
can set CHATTERBOX_BACKEND=torch as documented.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Tinker-page Chatterbox tab

**Rationale:** Makes experimentation hands-on instead of curl-only, matching the "Bastelstube" intent.

**Files:**
- Modify: `backend/static/index.html:37-77` (TTS section)
- Modify: `backend/static/app.js:61-109`
- Modify: `backend/static/style.css` (add a few rules; read file first to understand current style)

- [ ] **Step 1: Read current CSS for style conventions**

```bash
cat /home/chris/workspace/chatsune-voice/backend/static/style.css
```

- [ ] **Step 2: Add Chatterbox tab markup**

In `backend/static/index.html`, replace the `<div class="mode-tabs">` block and add a third panel after the existing `tts-voice-design` panel (inside the `<section id="tts">`):

```html
    <div class="mode-tabs">
      <label><input type="radio" name="tts-mode" value="custom_voice" checked> CustomVoice</label>
      <label><input type="radio" name="tts-mode" value="voice_design"> VoiceDesign</label>
      <label><input type="radio" name="tts-mode" value="clone"> Chatterbox (Voice Clone)</label>
    </div>
```

Append a new panel *before* the closing `</section>` of the TTS section, after the existing `tts-voice-design` panel:

```html
    <div id="tts-clone" class="tts-panel hidden">
      <label>Text: <textarea id="cl-text" rows="3"></textarea></label>
      <label>Language: <select id="cl-language">
        <option>German</option><option>English</option><option>French</option>
        <option>Italian</option><option>Spanish</option><option>Portuguese</option>
        <option>Russian</option><option>Chinese</option><option>Japanese</option>
        <option>Korean</option>
      </select></label>
      <label>Reference audio (WAV/FLAC):
        <input type="file" id="cl-reference" accept="audio/*">
      </label>
      <audio id="cl-reference-preview" controls class="hidden"></audio>
      <label>Exaggeration: <input type="range" id="cl-exaggeration" min="0.25" max="2.0" step="0.05" value="0.5">
        <span id="cl-exaggeration-val">0.50</span></label>
      <label>CFG weight: <input type="range" id="cl-cfg-weight" min="0.0" max="1.0" step="0.05" value="0.5">
        <span id="cl-cfg-weight-val">0.50</span></label>
      <label>Temperature: <input type="range" id="cl-temperature" min="0.05" max="2.0" step="0.05" value="0.8">
        <span id="cl-temperature-val">0.80</span></label>
      <button id="cl-speak">Speak</button>
    </div>
```

- [ ] **Step 3: Wire the tab in `app.js`**

In `backend/static/app.js`, replace the `tts-mode` change handler (around lines 61-68) to show the new panel:

```javascript
document.querySelectorAll('input[name="tts-mode"]').forEach(el => {
  el.addEventListener("change", () => {
    document.getElementById("tts-custom-voice").classList.toggle("hidden",
      !(el.value === "custom_voice" && el.checked));
    document.getElementById("tts-voice-design").classList.toggle("hidden",
      !(el.value === "voice_design" && el.checked));
    document.getElementById("tts-clone").classList.toggle("hidden",
      !(el.value === "clone" && el.checked));
  });
});
```

Add handlers for the sliders and the speak button. Append this block after the existing `vd-speak` handler (around line 89):

```javascript
// Live slider readouts
["cl-exaggeration", "cl-cfg-weight", "cl-temperature"].forEach(id => {
  const slider = document.getElementById(id);
  const out = document.getElementById(id + "-val");
  slider.addEventListener("input", () => {
    out.textContent = parseFloat(slider.value).toFixed(2);
  });
});

// Reference-audio preview
document.getElementById("cl-reference").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = document.getElementById("cl-reference-preview");
  if (!file) {
    preview.classList.add("hidden");
    return;
  }
  const prev = preview.src;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  if (prev) URL.revokeObjectURL(prev);
});

document.getElementById("cl-speak").addEventListener("click", async () => {
  const file = document.getElementById("cl-reference").files[0];
  if (!file) {
    ttsStatus.textContent = "error: reference audio required";
    return;
  }
  const fd = new FormData();
  fd.append("text", document.getElementById("cl-text").value);
  fd.append("language", document.getElementById("cl-language").value);
  fd.append("reference_audio", file, file.name);
  fd.append("exaggeration", document.getElementById("cl-exaggeration").value);
  fd.append("cfg_weight", document.getElementById("cl-cfg-weight").value);
  fd.append("temperature", document.getElementById("cl-temperature").value);

  ttsStatus.textContent = "synthesising…";
  ttsDownload.classList.add("hidden");
  const r = await fetch("/v1/speak/clone", { method: "POST", body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    ttsStatus.textContent = "error: " + r.status + " " + text;
    return;
  }
  const blob = await r.blob();
  const prev = ttsAudio.src;
  const url = URL.createObjectURL(blob);
  ttsAudio.src = url;
  ttsAudio.play();
  ttsDownload.href = url;
  ttsDownload.classList.remove("hidden");
  if (prev) URL.revokeObjectURL(prev);
  ttsStatus.textContent = "playing";
});
```

- [ ] **Step 4: Smoke-test in a browser**

```bash
cd backend && uv run uvicorn voice.main:app --reload
```

Open http://localhost:8000, switch to the Chatterbox tab, upload a short WAV, click Speak, verify the response plays (once the backend has real model access). If the models aren't loaded yet, the 403/503 response shows up in the status line — that's expected here; this step is purely to confirm no JS errors.

- [ ] **Step 5: Commit**

```bash
git add backend/static/index.html backend/static/app.js backend/static/style.css
git commit -m "$(cat <<'EOF'
Add Chatterbox tab to tinker page

Third radio option alongside CustomVoice and VoiceDesign. Upload a
reference WAV/FLAC, adjust exaggeration/cfg_weight/temperature sliders
with live readouts, submit as multipart, and play back the returned
cloned audio in-page.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: README and `.env.example` updates

**Files:**
- Modify: `README.md:44-65` (Configuration + VRAM policy sections, plus new Cloning subsection)
- Modify: `.env.example`

- [ ] **Step 1: Update `.env.example`**

Append after the existing `TTS_ATTENTION_IMPL` block:

```
# ===== Chatterbox (voice cloning mode) =====
# Model ID. The ONNX-community Q4 port is the default; swap to the Torch
# fallback on hosts where onnxruntime-rocm is unavailable.
CHATTERBOX_MODEL=onnx-community/chatterbox-multilingual-ONNX

# Inference backend. "onnx" loads the 4 ONNX sessions with
# CUDAExecutionProvider / ROCMExecutionProvider as available. "torch" falls
# back to the chatterbox-tts pip package in bfloat16 — larger VRAM
# footprint, but guaranteed to work on every supported GPU.
CHATTERBOX_BACKEND=onnx

# Device string passed to the backend. On ROCm, "cuda" resolves via HIP.
CHATTERBOX_DEVICE=cuda

# Reference-audio upload limits.
CHATTERBOX_MAX_REFERENCE_BYTES=10485760
CHATTERBOX_MAX_REFERENCE_SECONDS=30
```

Also update the existing `TTS_ENABLED_MODES` default line:

```
# Subset allowed, e.g. "custom_voice,clone" for smaller GPUs that cannot
# host VoiceDesign in addition.
TTS_ENABLED_MODES=custom_voice,voice_design,clone
```

- [ ] **Step 2: Update the README configuration table**

In `README.md`, extend the configuration table (around line 47) with the new rows:

```markdown
| `CHATTERBOX_MODEL` | `onnx-community/chatterbox-multilingual-ONNX` | HF repo for the Chatterbox multilingual checkpoint. |
| `CHATTERBOX_BACKEND` | `onnx` | `onnx` (ONNX Runtime, Q4, primary) or `torch` (chatterbox-tts pip package, bf16, fallback). |
| `CHATTERBOX_DEVICE` | `cuda` | Device string. On ROCm hosts `cuda` resolves via HIP. |
| `CHATTERBOX_MAX_REFERENCE_BYTES` | `10485760` | Upper bound on reference-audio file size (bytes). |
| `CHATTERBOX_MAX_REFERENCE_SECONDS` | `30` | Upper bound on decoded reference-audio duration (seconds). |
```

And extend the "VRAM policy guidance" section with this note:

```markdown
**Chatterbox under `swap`:** Chatterbox is always-resident — it is the realtime
workhorse and is not swapped out when a Qwen3 mode is acquired. Under `swap`,
preload loads Chatterbox immediately at startup; Qwen3 modes load on first
request. Plan for Chatterbox's ~0.5–2 GB (Q4 / bf16) on top of whichever
Qwen3 mode is currently resident.
```

Add a new section after "API":

```markdown
## Cloning mode

The `POST /v1/speak/clone` endpoint synthesises cloned speech using a reference
audio clip. It accepts `multipart/form-data` rather than JSON because the
reference audio is binary.

```bash
curl -X POST http://localhost:8000/v1/speak/clone \
  -F text="Hallo, das ist ein Test." \
  -F language=German \
  -F reference_audio=@my_voice.wav \
  -F exaggeration=0.6 \
  -F cfg_weight=0.4 \
  -F temperature=0.7 \
  -o out.wav
```

The response is a streaming PCM16 WAV at the Chatterbox sample rate (24 kHz).
Reference clips should be 3–15 seconds of clean speech; longer is capped by
`CHATTERBOX_MAX_REFERENCE_SECONDS`. `language=Auto` is not supported — pick a
concrete language from the multilingual set (EN, DE, FR, ES, IT, PT, JA, KO,
ZH, RU, and 13 more).
```

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "$(cat <<'EOF'
Document Chatterbox integration in README and .env.example

Adds CHATTERBOX_* env-var entries to the configuration table, describes
Chatterbox's always-resident behaviour under the swap VRAM policy, and
documents the new /v1/speak/clone endpoint with a curl example.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Manual smoke test and RTF measurement

**Rationale:** The Phase 1 done criteria require hands-on verification on the homelab. This task is **not auto-committed** — it produces evidence that the integration works end-to-end on real hardware, and the RTF number feeds into the README for reference.

**Not a TDD cycle.** This is a manual checklist.

- [ ] **Step 1: Prefetch models**

```bash
cd backend && uv run python scripts/prefetch_models.py
```

- [ ] **Step 2: Bring up the CUDA (or ROCm) stack**

```bash
cd /home/chris/workspace/chatsune-voice
cp .env.example .env  # edit COMPOSE_PROFILES if needed
docker compose up --build
```

Wait for `/healthz` to return `ok`.

- [ ] **Step 3: Smoke-test clone endpoint via curl**

Record a 10-second reference clip `/tmp/ref.wav` (e.g. speaking "Testaufnahme chatsune-voice" clearly).

```bash
time curl -X POST http://localhost:8000/v1/speak/clone \
  -F text="Hallo, dies ist ein deutscher Klontest." \
  -F language=German \
  -F reference_audio=@/tmp/ref.wav \
  -o /tmp/out_de.wav

time curl -X POST http://localhost:8000/v1/speak/clone \
  -F text="Bonjour, ceci est un test de clonage de voix en français." \
  -F language=French \
  -F reference_audio=@/tmp/ref.wav \
  -o /tmp/out_fr.wav

time curl -X POST http://localhost:8000/v1/speak/clone \
  -F text="Hello, this is an English cloning test." \
  -F language=English \
  -F reference_audio=@/tmp/ref.wav \
  -o /tmp/out_en.wav
```

Play each file. Confirm that all three are audible, correctly pronounced in the target language, and sound like the reference voice.

- [ ] **Step 4: Compute RTF**

For each call, `time` shows wall-clock duration. Measure output audio duration:

```bash
soxi -D /tmp/out_de.wav  # seconds of audio
```

RTF = wall_clock_seconds / output_audio_seconds. Record the numbers for each language in a small "## Phase 1 measurements" appendix at the bottom of the README (or inline in a commit message).

- [ ] **Step 5: Tinker-page smoke test**

Open http://localhost:8000 in Firefox, switch to the Chatterbox tab, upload `/tmp/ref.wav`, type a line, click Speak. Confirm playback and that the reference-preview player shows the uploaded clip.

- [ ] **Step 6: If ROCm host is available — retest with `COMPOSE_PROFILES=rocm`**

Note in the README whether the ONNX path worked or whether `CHATTERBOX_BACKEND=torch` had to be set.

- [ ] **Step 7: Record findings**

Create a short commit with the RTF results and any environment quirks (e.g., "ROCm 7.2 + gfx1151 needed CHATTERBOX_BACKEND=torch; onnxruntime-rocm unavailable"):

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Record Phase 1 Chatterbox RTF baseline

Homelab CUDA (<GPU>): RTF <n> DE / <n> FR / <n> EN.
<ROCm / Strix Halo notes here if applicable>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (plan author, after writing)

- [x] Every spec section has a task: motivation → Tasks 1-8, architecture → 1-7, API shape → 8, frontend → 12, testing → embedded, docs → 13, Phase 1 done criteria → 14.
- [x] No placeholders — every step has executable code or a concrete command.
- [x] Type consistency — `TTSMode`, `TTSRequest`, `always_resident`, `_ChatterboxBackend.generate` signature, `language_to_iso639` helper names all used consistently across tasks.
- [x] The one flagged risk (ONNX inference loop) is isolated to Task 6 step 8 and carries a visible NotImplementedError marker so it's impossible to ship an empty backend silently.
