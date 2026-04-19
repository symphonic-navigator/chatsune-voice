# Chatterbox ONNX inference loop — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `NotImplementedError` stub in `_OnnxBackend.generate` (`backend/voice/engines/chatterbox_tts.py`) with a working autoregressive inference loop adapted from the upstream reference, completing parent-plan Task 6 Step 8.

**Architecture:** Five helper units (module-level constants, reduced `_LANGUAGE_MAP`, `prepare_language`, `repetition_penalty_processor`, temperature-sampling) plus a path correction in `load_chatterbox_onnx` (ONNX files live in the `onnx/` subfolder with `.onnx_data` sidecars). The generation loop follows the reference 1:1 except for two deliberate divergences: temperature sampling instead of greedy, and `cfg_weight` as documented no-op. All helpers are independently unit-testable without ONNX runtime sessions.

**Tech Stack:** Python 3.12, ONNX Runtime 1.19+, Transformers 4.45+, NumPy, librosa (newly added direct dep), pytest.

**Parent spec:** `docs/superpowers/specs/2026-04-19-chatterbox-onnx-inference-design.md`
**Parent plan:** `docs/superpowers/plans/2026-04-19-chatterbox-integration.md` (this plan implements Step 8 of Task 6)

---

## Task 1: Add `librosa` as direct dependency

**Rationale:** The reference uses `librosa.load(..., sr=24000)` to decode + resample reference audio in one call. `librosa` is transitively present, but declaring it directly makes the contract stable.

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`

- [ ] **Step 1: Add the dependency**

Edit `backend/pyproject.toml`. In the `dependencies` array, add `"librosa>=0.10,<0.12"` after the `"transformers>=4.45,<5"` line.

Expected resulting lines:

```toml
    "onnxruntime>=1.19,<2",
    "transformers>=4.45,<5",
    "librosa>=0.10,<0.12",
]
```

- [ ] **Step 2: Regenerate lockfile**

```bash
cd backend && uv lock && uv sync --dev
```

Expected: `uv lock` completes without errors. `librosa 0.11.x` (or compatible) is pinned in `uv.lock`.

- [ ] **Step 3: Sanity-check import**

```bash
cd backend && uv run python -c "import librosa; print(librosa.__version__)"
```

Expected: prints a version in the `0.10.x`–`0.11.x` range. No ImportError.

- [ ] **Step 4: Run existing tests to confirm no regressions**

```bash
cd backend && uv run pytest -q
```

Expected: all currently passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "$(cat <<'EOF'
Add librosa as direct dependency for Chatterbox ONNX backend

Reference-audio decode and resampling in the ONNX inference loop use
librosa.load(..., sr=24000, mono=True), matching the upstream reference
exactly. librosa is already transitively installed via faster-whisper;
declaring it directly stabilises the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Reduce `_LANGUAGE_MAP` to 8 languages and reject Japanese/Chinese

**Rationale:** Phase 1 does not carry the `pkuseg` / `pykakasi` dependencies required for correct Chinese and Japanese text preprocessing. Callers get a clear `ValueError` instead of silently-garbled output.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py:25-36` (`_LANGUAGE_MAP`)
- Modify: `backend/tests/test_chatterbox_tts.py:129-141` (existing language-mapping test)
- Modify: `backend/tests/test_chatterbox_tts.py` (new rejection test)

- [ ] **Step 1: Update the existing language-mapping test**

Replace the body of `test_language_mapping_covers_all_common_languages` in `backend/tests/test_chatterbox_tts.py` so it asserts the eight kept languages and no others:

```python
def test_language_mapping_covers_all_common_languages():
    from voice.engines.chatterbox_tts import language_to_iso639

    assert language_to_iso639("English") == "en"
    assert language_to_iso639("German") == "de"
    assert language_to_iso639("French") == "fr"
    assert language_to_iso639("Spanish") == "es"
    assert language_to_iso639("Italian") == "it"
    assert language_to_iso639("Portuguese") == "pt"
    assert language_to_iso639("Russian") == "ru"
    assert language_to_iso639("Korean") == "ko"
```

- [ ] **Step 2: Add the rejection test**

Append to `backend/tests/test_chatterbox_tts.py` (right after `test_language_mapping_rejects_unknown`):

```python
def test_language_mapping_rejects_japanese_and_chinese():
    """Phase 1 drops ja/zh because pkuseg/pykakasi are not installed."""
    from voice.engines.chatterbox_tts import language_to_iso639

    with pytest.raises(ValueError, match="Japanese"):
        language_to_iso639("Japanese")
    with pytest.raises(ValueError, match="Chinese"):
        language_to_iso639("Chinese")
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py::test_language_mapping_rejects_japanese_and_chinese tests/test_chatterbox_tts.py::test_language_mapping_covers_all_common_languages -v
```

Expected: `test_language_mapping_rejects_japanese_and_chinese` FAILS (because the current map maps "Japanese" to "ja" successfully); `test_language_mapping_covers_all_common_languages` PASSES (since the reduced list is a strict subset of the current map — but this may also pass).

- [ ] **Step 4: Reduce the `_LANGUAGE_MAP`**

In `backend/voice/engines/chatterbox_tts.py`, replace the existing `_LANGUAGE_MAP` definition:

```python
_LANGUAGE_MAP: dict[str, str] = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Korean": "ko",
}
```

The existing `language_to_iso639` function already raises `ValueError` with the language name in the message when a key is missing — no change needed.

- [ ] **Step 5: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Reduce Chatterbox language map to 8 languages for Phase 1

Drop Japanese and Chinese: both require extra dependencies (pkuseg for
zh Cangjie decomposition, pykakasi for ja kanji normalisation) whose
absence would produce silently-garbled output. Phase 1 keeps en, de,
fr, es, it, pt, ru, ko; ja/zh now raise a clear ValueError. Follow-up
when real demand for those languages appears.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `prepare_language` helper with Korean Jamo decomposition

**Rationale:** The ONNX language-model expects the text to carry an explicit `[xx]` language token prefix and — for Korean — Hangul syllables decomposed into Initial/Medial/Final Jamo. Extract this into a pure helper so we can unit-test it without any ONNX session.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py` (add helper; no existing callers yet)
- Modify: `backend/tests/test_chatterbox_tts.py` (new tests)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_prepare_language_prepends_language_token():
    from voice.engines.chatterbox_tts import prepare_language

    assert prepare_language("Hallo", "de") == "[de]Hallo"
    assert prepare_language("Bonjour", "fr") == "[fr]Bonjour"


def test_prepare_language_korean_jamo_decomposition():
    """Hangul syllable '안' (0xC548) should decompose to ᄋ (0x110B) + ᅡ (0x1161) + ᆫ (0x11AB)."""
    from voice.engines.chatterbox_tts import prepare_language

    result = prepare_language("안", "ko")
    expected = "[ko]" + chr(0x110B) + chr(0x1161) + chr(0x11AB)
    assert result == expected


def test_prepare_language_korean_passthrough_for_non_hangul():
    """Non-Hangul characters within a ko text are preserved untouched after the prefix."""
    from voice.engines.chatterbox_tts import prepare_language

    assert prepare_language("Hello!", "ko") == "[ko]Hello!"


def test_prepare_language_korean_syllable_without_final():
    """Hangul syllable '가' (0xAC00, base case) has no final Jamo."""
    from voice.engines.chatterbox_tts import prepare_language

    result = prepare_language("가", "ko")
    # base=0, initial=0x1100, medial=0x1161, no final
    expected = "[ko]" + chr(0x1100) + chr(0x1161)
    assert result == expected
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py::test_prepare_language_prepends_language_token -v
```

Expected: FAIL with `ImportError` on `prepare_language`.

- [ ] **Step 3: Implement the helper**

Add to `backend/voice/engines/chatterbox_tts.py`, placed after `language_to_iso639`:

```python
def _decompose_hangul(char: str) -> str:
    """Decompose a Hangul syllable into Initial/Medial/Final Jamo.

    Non-Hangul characters pass through unchanged. Syllables without a final
    component (base % 28 == 0) return initial + medial only.
    """
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7AF):
        return char
    base = code - 0xAC00
    initial = chr(0x1100 + base // (21 * 28))
    medial = chr(0x1161 + (base % (21 * 28)) // 28)
    if base % 28 == 0:
        return initial + medial
    final = chr(0x11A7 + base % 28)
    return initial + medial + final


def prepare_language(text: str, language_id: str) -> str:
    """Apply per-language preprocessing and prepend the language token.

    For Korean, Hangul syllables are decomposed into Jamo components using
    the pure-Python formula from the upstream reference. All other
    supported languages (en, de, fr, es, it, pt, ru) pass through unchanged.
    """
    if language_id == "ko":
        text = "".join(_decompose_hangul(c) for c in text)
    return f"[{language_id}]{text}"
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -k prepare_language -v
```

Expected: all four `prepare_language` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Add prepare_language helper with Korean Jamo decomposition

Prepends the [xx] language token the ONNX language-model expects and
decomposes Hangul syllables into Initial/Medial/Final Jamo for Korean
using the reference's pure-Python formula (base 0xAC00, initial
0x1100 + base/588, medial 0x1161 + (base%588)/28, final 0x11A7 +
base%28 if non-zero). No new dependency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `repetition_penalty_processor` helper

**Rationale:** The upstream reference uses a `RepetitionPenaltyLogitsProcessor` class to suppress token repetition during generation. We lift it into a small top-level function (single call site, no class needed) and unit-test it against handcrafted arrays.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py`
- Modify: `backend/tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_repetition_penalty_processor_scales_positive_and_negative_correctly():
    """Positive scores at visited ids are divided by penalty; negative ones are multiplied."""
    from voice.engines.chatterbox_tts import repetition_penalty_processor

    # vocab_size=4, already-generated token ids are [1, 2]
    input_ids = np.array([[1, 2]], dtype=np.int64)
    scores = np.array([[1.0, 2.0, -3.0, 4.0]], dtype=np.float32)

    result = repetition_penalty_processor(input_ids, scores, penalty=2.0)

    # Index 0: not in input_ids, unchanged -> 1.0
    # Index 1: positive, in input_ids -> 2.0 / 2.0 = 1.0
    # Index 2: negative, in input_ids -> -3.0 * 2.0 = -6.0
    # Index 3: not in input_ids, unchanged -> 4.0
    np.testing.assert_allclose(result, np.array([[1.0, 1.0, -6.0, 4.0]], dtype=np.float32))


def test_repetition_penalty_processor_leaves_scores_unchanged_when_no_history():
    from voice.engines.chatterbox_tts import repetition_penalty_processor

    input_ids = np.zeros((1, 0), dtype=np.int64)
    scores = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)

    result = repetition_penalty_processor(input_ids, scores, penalty=1.2)

    np.testing.assert_allclose(result, scores)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -k repetition_penalty -v
```

Expected: FAIL with `ImportError` on `repetition_penalty_processor`.

- [ ] **Step 3: Implement the helper**

Add to `backend/voice/engines/chatterbox_tts.py`, placed after `prepare_language`:

```python
def repetition_penalty_processor(
    input_ids: np.ndarray, scores: np.ndarray, *, penalty: float
) -> np.ndarray:
    """Apply repetition penalty to `scores` at indices listed in `input_ids`.

    Mirrors the upstream reference's RepetitionPenaltyLogitsProcessor:
    positive scores at visited ids are divided by `penalty`, negative
    scores are multiplied. Non-visited indices are left untouched.
    """
    if input_ids.shape[1] == 0:
        return scores
    score = np.take_along_axis(scores, input_ids, axis=1)
    score = np.where(score < 0, score * penalty, score / penalty)
    scores_processed = scores.copy()
    np.put_along_axis(scores_processed, input_ids, score, axis=1)
    return scores_processed
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -k repetition_penalty -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Add repetition_penalty_processor helper for Chatterbox ONNX

Lifts the upstream RepetitionPenaltyLogitsProcessor class into a pure
function: positive scores at visited token ids are divided by the
penalty, negative scores are multiplied. Single call site in the
autoregressive loop, no class needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `sample_next_token` helper with temperature support

**Rationale:** This is our deliberate divergence from the reference. The upstream uses greedy `argmax`; we add optional temperature sampling so the `temperature` API parameter has real effect. `cfg_weight` remains ignored by design (see parent spec).

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py`
- Modify: `backend/tests/test_chatterbox_tts.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_sample_next_token_zero_temperature_is_greedy():
    """temperature <= 0 falls back to deterministic argmax."""
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[1.0, 5.0, 2.0, 3.0]], dtype=np.float32)

    result = sample_next_token(logits, temperature=0.0)

    assert result.shape == (1, 1)
    assert result.dtype == np.int64
    assert result[0, 0] == 1  # index of max


def test_sample_next_token_positive_temperature_is_stochastic():
    """With temperature=1.0 and varied seeds, at least two distinct tokens appear."""
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[1.0, 1.1, 1.0, 1.05]], dtype=np.float32)

    seen: set[int] = set()
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = sample_next_token(logits, temperature=1.0, rng=rng)
        seen.add(int(result[0, 0]))

    assert len(seen) >= 2, f"Expected stochastic sampling, got single token {seen}"


def test_sample_next_token_returns_int64_shape_1_1():
    from voice.engines.chatterbox_tts import sample_next_token

    logits = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    result = sample_next_token(logits, temperature=0.5)

    assert result.shape == (1, 1)
    assert result.dtype == np.int64
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -k sample_next_token -v
```

Expected: FAIL with `ImportError` on `sample_next_token`.

- [ ] **Step 3: Implement the helper**

Add to `backend/voice/engines/chatterbox_tts.py`, placed after `repetition_penalty_processor`:

```python
def sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select the next token id from last-step logits.

    `logits` has shape `[1, vocab_size]` and is already repetition-penalty
    adjusted. For `temperature <= 0` we fall back to greedy `argmax`
    (matching the upstream reference). For `temperature > 0` we apply
    standard temperature sampling: scale, stable-softmax, multinomial draw.

    `rng` is injected for testability; production code passes `None` so
    each call uses the module-level default generator.
    """
    if temperature <= 0.0:
        return np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)

    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    gen = rng if rng is not None else np.random.default_rng()
    choice = gen.choice(probs.shape[-1], p=probs[0])
    return np.array([[choice]], dtype=np.int64)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -k sample_next_token -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Add sample_next_token helper with temperature sampling

Deliberate divergence from the upstream reference (which uses greedy
argmax): temperature > 0 applies stable softmax + multinomial draw so
the temperature API parameter actually has effect. temperature <= 0
falls back to argmax. rng is injected so the test suite can assert
stochasticity deterministically across seeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add module-level constants and fix ONNX subfolder paths

**Rationale:** The upstream repo stores all four `.onnx` files inside an `onnx/` subfolder, each paired with an `.onnx_data` external-data sidecar. The current skeleton downloads from the repo root, which would fail at runtime. We add the generation-loop constants in the same commit because both changes concern `load_chatterbox_onnx` and they are trivial to review together.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py` (module-level constants; `load_chatterbox_onnx` downloads)
- Modify: `backend/tests/test_chatterbox_tts.py` (monkeypatch-based test)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_chatterbox_tts.py`:

```python
def test_onnx_loader_downloads_from_onnx_subfolder(monkeypatch):
    """Each .onnx file is downloaded with subfolder='onnx' and the paired
    .onnx_data sidecar is fetched alongside."""
    from voice.engines import chatterbox_tts

    calls: list[dict] = []

    def fake_download(repo_id, filename, *, subfolder=None):
        calls.append({"filename": filename, "subfolder": subfolder})
        return f"/fake/{subfolder}/{filename}"

    class FakeSession:
        def __init__(self, path, providers=None):
            self.path = path
            self.providers = providers

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            return object()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    monkeypatch.setattr(
        "onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)

    chatterbox_tts.load_chatterbox_onnx("fake/repo", device="cpu")

    filenames = [c["filename"] for c in calls]
    subfolders = {c["subfolder"] for c in calls}

    for base in ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder"):
        assert f"{base}.onnx" in filenames
        assert f"{base}.onnx_data" in filenames
    assert subfolders == {"onnx"}
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py::test_onnx_loader_downloads_from_onnx_subfolder -v
```

Expected: FAIL — the current code calls `hf_hub_download(repo_id=..., filename=...)` without `subfolder`, and does not download `.onnx_data` sidecars.

- [ ] **Step 3: Add module-level constants**

In `backend/voice/engines/chatterbox_tts.py`, near the top of the file (just after the `DEFAULT_CHUNK_SIZE = 4096` line), add:

```python
# Chatterbox ONNX inference constants (extracted from the upstream reference).
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_HIDDEN_LAYERS = 30
NUM_KV_HEADS = 16
HEAD_DIM = 64
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS = 1024
```

- [ ] **Step 4: Fix the download paths in `load_chatterbox_onnx`**

In `backend/voice/engines/chatterbox_tts.py`, replace the existing `_dl` helper and the four download lines (around the current `speech_encoder_path = _dl("speech_encoder.onnx")` block) with:

```python
    def _dl(filename: str) -> str:
        return hf_hub_download(repo_id=model_id, filename=filename, subfolder="onnx")

    speech_encoder_path = _dl("speech_encoder.onnx")
    _dl("speech_encoder.onnx_data")
    embed_tokens_path = _dl("embed_tokens.onnx")
    _dl("embed_tokens.onnx_data")
    language_model_path = _dl("language_model.onnx")
    _dl("language_model.onnx_data")
    conditional_decoder_path = _dl("conditional_decoder.onnx")
    _dl("conditional_decoder.onnx_data")
```

The sidecar download return values are discarded; their presence on disk next to the `.onnx` file is what matters for the `InferenceSession` loader.

- [ ] **Step 5: Run tests to verify pass**

```bash
cd backend && uv run pytest tests/test_chatterbox_tts.py -v
```

Expected: all tests PASS, including the new subfolder download test and the existing `test_onnx_loader_raises_on_missing_onnxruntime`.

- [ ] **Step 6: Commit**

```bash
git add backend/voice/engines/chatterbox_tts.py backend/tests/test_chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Fix Chatterbox ONNX download paths and add inference constants

The upstream repo stores all four .onnx files inside an onnx/ subfolder
with paired .onnx_data external-data sidecars; the skeleton loader was
requesting them from the repo root and ignoring sidecars, which would
fail at runtime. Pass subfolder='onnx' to hf_hub_download for each file
and explicitly fetch the sidecars. Also introduces the module-level
constants (START/STOP speech tokens, KV-cache shape, max_new_tokens,
repetition penalty) that the inference loop in the next commit consumes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement the autoregressive inference loop in `_OnnxBackend.generate`

**Rationale:** The final piece. With helpers and constants in place, the inference loop maps 1:1 onto the reference plus our temperature-sampling divergence. This cannot be unit-tested without real ONNX sessions; correctness is confirmed by (a) all existing tests remain green, (b) `ruff` clean, (c) parent-plan Task 14 manual smoke on real hardware.

**Files:**
- Modify: `backend/voice/engines/chatterbox_tts.py` (replace `_OnnxBackend.generate` body)

- [ ] **Step 1: Replace the `_OnnxBackend.generate` implementation**

In `backend/voice/engines/chatterbox_tts.py`, replace the entire `generate` method of the `_OnnxBackend` class (the block containing the `TODO(task-6-part-2)` comment and the `raise NotImplementedError(...)`) with the fully-worked inference loop below.

The enclosing `load_chatterbox_onnx` function already imports `io`, `onnxruntime`, `soundfile as sf`, and `huggingface_hub.hf_hub_download`. Add `import librosa` to the top of that function's body (alongside the existing imports) so it is lazy-loaded with the rest of the ONNX stack. Delete the now-unused `import io` and `import soundfile as sf` lines from the loader — neither is used after the switch to `librosa.load`.

```python
        def generate(
            self,
            *,
            text: str,
            language: str,
            reference_audio: bytes,
            exaggeration: float,
            cfg_weight: float,  # documented no-op in the ONNX backend
            temperature: float,
        ) -> tuple[np.ndarray, int]:
            # cfg_weight is validated by the API layer and accepted here for
            # protocol compatibility, but the ONNX backend does not apply
            # classifier-free guidance — the upstream reference provides no
            # validated recipe. Follow-up tracked in the sub-spec.
            del cfg_weight

            # 1. Decode + resample reference audio to 24 kHz mono float32.
            waveform, _ = librosa.load(
                io.BytesIO(reference_audio), sr=S3GEN_SR, mono=True
            )
            audio_values = waveform[np.newaxis, :].astype(np.float32)

            # 2. Text preparation: per-language normalisation + language token.
            prepared_text = prepare_language(text, language)
            input_ids = self._tokenizer(
                prepared_text, return_tensors="np"
            )["input_ids"].astype(np.int64)
            position_ids = np.where(
                input_ids >= START_SPEECH_TOKEN,
                0,
                np.arange(input_ids.shape[1])[np.newaxis, :] - 1,
            ).astype(np.int64)

            # 3. Encode reference voice (once) and compute initial embeddings.
            cond_emb, prompt_token, ref_x_vector, prompt_feat = (
                self._speech_encoder.run(None, {"audio_values": audio_values})
            )
            embed_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "exaggeration": np.array([exaggeration], dtype=np.float32),
            }
            inputs_embeds = self._embed_tokens.run(None, embed_inputs)[0]
            inputs_embeds = np.concatenate([cond_emb, inputs_embeds], axis=1)

            # 4. Initialise KV-cache and attention mask.
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros(
                    [batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float32
                )
                for layer in range(NUM_HIDDEN_LAYERS)
                for kv in ("key", "value")
            }
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

            # 5. Autoregressive generation loop.
            generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
            for step in range(MAX_NEW_TOKENS):
                logits, *present_key_values = self._language_model.run(
                    None,
                    {
                        "inputs_embeds": inputs_embeds,
                        "attention_mask": attention_mask,
                        **past_key_values,
                    },
                )
                last_logits = logits[:, -1, :]
                penalised = repetition_penalty_processor(
                    generate_tokens, last_logits, penalty=REPETITION_PENALTY
                )
                next_token = sample_next_token(penalised, temperature=temperature)
                generate_tokens = np.concatenate(
                    [generate_tokens, next_token], axis=-1
                )
                if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                    break

                # Embed the new token. exaggeration stays constant; only
                # input_ids and position_ids advance.
                embed_inputs["input_ids"] = next_token
                embed_inputs["position_ids"] = np.full(
                    (batch_size, 1), step + 1, dtype=np.int64
                )
                inputs_embeds = self._embed_tokens.run(None, embed_inputs)[0]

                attention_mask = np.concatenate(
                    [attention_mask, np.ones((batch_size, 1), dtype=np.int64)],
                    axis=1,
                )
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

            # 6. Waveform synthesis.
            speech_tokens = generate_tokens[:, 1:-1]
            speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
            wav = self._conditional_decoder.run(
                None,
                {
                    "speech_tokens": speech_tokens,
                    "speaker_embeddings": ref_x_vector,
                    "speaker_features": prompt_feat,
                },
            )[0]
            return np.squeeze(wav, axis=0).astype(np.float32), S3GEN_SR
```

- [ ] **Step 2: Swap the audio-decoding imports in `load_chatterbox_onnx`**

Inside `load_chatterbox_onnx`, update the top of the function body so `librosa` is imported and the now-unused imports are dropped. The intended final state:

```python
def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Load Chatterbox Multilingual via ONNX Runtime.
    ... (existing docstring) ...
    """
    import io  # still needed — librosa.load accepts a BytesIO

    import librosa
    import onnxruntime
    from huggingface_hub import hf_hub_download
```

Remove the `import soundfile as sf` line — it is no longer used. `io` stays because `librosa.load` consumes `io.BytesIO(reference_audio)`.

- [ ] **Step 3: Run the full test suite**

```bash
cd backend && uv run pytest -q
```

Expected: all tests PASS. The existing `_OnnxBackend.generate` is only exercised through the fake-backend path in tests, so none of them try to run the real inference loop — they just confirm the module imports cleanly.

- [ ] **Step 4: Run ruff**

```bash
cd backend && uv run ruff check .
```

Expected: clean, no warnings.

- [ ] **Step 5: Commit (parent-plan Step 8 message pattern)**

```bash
git add backend/voice/engines/chatterbox_tts.py
git commit -m "$(cat <<'EOF'
Populate Chatterbox ONNX inference loop from reference

Adapts the onnx-community repo's run_inference() into _OnnxBackend.
Implements reference-voice encoding, text tokenisation, autoregressive
language-model decoding with KV cache, and conditional-decoder waveform
synthesis. Output is 1-D float32 at 24 kHz.

Deliberate divergences from the upstream: temperature sampling instead
of greedy argmax so the temperature API knob has effect, and cfg_weight
accepted for protocol compatibility but left as a no-op (no validated
reference recipe yet). Both documented in the sub-spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Post-implementation verification

After the final commit:

- [ ] **Step 1: Full test suite + lint**

```bash
cd backend && uv run pytest -q && uv run ruff check .
```

Expected: all tests pass, ruff clean.

- [ ] **Step 2: Confirm git history shows seven clean commits on the branch**

```bash
git log --oneline ea60b58..HEAD
```

Expected: seven commits, one per task, in order:
1. `Add librosa as direct dependency ...`
2. `Reduce Chatterbox language map ...`
3. `Add prepare_language helper ...`
4. `Add repetition_penalty_processor helper ...`
5. `Add sample_next_token helper ...`
6. `Fix Chatterbox ONNX download paths ...`
7. `Populate Chatterbox ONNX inference loop ...`

- [ ] **Step 3: Hand-off to manual smoke**

The automated gate ends here. Real-model validation happens via parent-plan Task 14 (the full Chatterbox integration's manual smoke test on homelab hardware), producing audible German / English / French audio and a recorded RTF measurement in the README. Mark parent-plan Task 6 Step 8 complete only after that manual smoke is green.
