# Chatterbox ONNX inference loop — design

**Status:** design approved, pending implementation plan
**Date:** 2026-04-19
**Scope:** Task 6 Step 8 of the Chatterbox integration — adapt the upstream reference inference loop into `_OnnxBackend.generate`, replacing the deliberate `NotImplementedError` skeleton.
**Parent spec:** `docs/superpowers/specs/2026-04-19-chatterbox-integration-design.md`

## Motivation

The Chatterbox integration ships with a loaded but inert ONNX backend: four `InferenceSession` objects, a tokeniser, and provider selection are wired, but `_OnnxBackend.generate` currently raises `NotImplementedError` with a pointer to `backend/voice/engines/_chatterbox_onnx_reference.py` (the `run_inference()` example copied from the upstream HF repo's README). This sub-spec locks in how we adapt that reference into production code, so the subagent executing the work does not re-decide scope-shaping questions mid-implementation.

Four design questions surfaced while reading the reference code. Each is resolved below with a short rationale so we can revisit the decision if Phase 3 (real-time multi-sentence workloads on Strix Halo) produces new constraints.

## Design decisions

### 1. Sampling strategy: temperature only, `cfg_weight` is a documented no-op

The upstream `run_inference()` example uses **greedy `argmax` decoding** and applies **neither** `temperature` nor classifier-free guidance — the reference simply does not show how to combine them with the exported ONNX graphs. Our `TTSRequest` protocol, unit tests, and tinker-page sliders however already expose both parameters.

**Decision:** Implement temperature sampling ourselves (standard pattern: `logits / temperature → softmax → np.random.choice`) and leave `cfg_weight` as a **documented no-op** inside the ONNX backend. The parameter is still validated at the API boundary (0.0–1.0) and threaded through the adapter, but the loop ignores it.

**Why:** Temperature is trivial and gives users the advertised knob. Genuine classifier-free guidance needs a second forward pass without the conditioning embedding, which doubles latency and has no validated reference recipe from the upstream repo — we would be inventing behaviour. Better to ship an honest no-op than unvalidated sampling.

**Revisit when:** An upstream reference integration for CFG appears, or when the Torch backend's measured CFG behaviour makes a quality gap against ONNX obvious.

### 2. Language coverage: 8 languages, no extra dependencies

The reference code branches into language-specific text normalisation: Chinese requires Cangjie decomposition (`pkuseg` + `Cangjie5_TC.json`), Japanese requires Kanji→Hiragana via `pykakasi`, Hebrew requires `dicta_onnx`, Korean needs only a pure-Python Jamo decomposition.

**Decision:** Phase-1 scope is **8 languages**: `en, de, fr, es, it, pt, ru, ko`. `ja` and `zh` are **removed** from `_LANGUAGE_MAP`; callers sending `"Japanese"` or `"Chinese"` get a `ValueError` from `language_to_iso639`. Korean is implemented with an in-module Jamo decomposer (~20 lines adapted from the reference), no new dependency.

**Why:** The parent spec's Phase-1 done-criterion only requires recognisable audio for German, English, and French. Adding `pkuseg` and `pykakasi` costs ~100 MB in the Docker image and brings their own model payloads — premature for a Bastelstube. Keeping the map small also prevents silent quality degradation if a user sends Japanese text and the tokeniser produces garbage tokens without the Kanji pre-processing step.

**Revisit when:** Any concrete user request for `ja`/`zh`/`he` lands. Extension is cheap: add the dep, extend `_LANGUAGE_MAP`, wire the language branch in `prepare_language()`.

### 3. Audio decode + resample: `librosa` as direct dependency

The reference uses `librosa.load(path, sr=24000)` which decodes (via `audioread`/`soundfile`) and resamples in one call. Our current skeleton uses `sf.read` directly and does not resample at all, so a 44.1 kHz reference would produce garbage conditioning.

**Decision:** Add `librosa>=0.10,<0.12` as a **direct** dependency and call `librosa.load(io.BytesIO(reference_audio), sr=24000, mono=True)`.

**Why:** `librosa` is already present transitively (via `faster-whisper`/`transformers`), so the image size is already paid. Matching the reference call exactly avoids subtle behavioural drift (different resampling filters produce different conditioning vectors). Deriving our own resample via `scipy.signal.resample_poly` is possible but costs three extra lines for zero gain.

**Revisit when:** Never, unless librosa disappears from the ecosystem.

### 4. Watermarking: deferred

The reference applies `perth.PerthImplicitWatermarker` by default; `apply_watermark=True` in its signature.

**Decision:** Phase 1 ships **without** watermarking. No `resemble-perth` dependency. A follow-up spec adds it behind a config flag when we move to public production use.

**Why:** Bastelstube runs on the homelab, not public-facing. The ethics argument for watermarking kicks in at production deployment (parent spec Phase 3), not during internal evaluation. We explicitly track this as a follow-up so it does not get lost.

**Revisit when:** Before any public-facing endpoint is exposed — e.g. when the voice library sub-spec lands, or when an external caller can reach the service.

## Architecture changes

Scope is confined to **one file** plus tests and `pyproject.toml`. Nothing in the API layer, registry, config, or frontend changes.

### `backend/voice/engines/chatterbox_tts.py`

Module-level additions:

```python
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_HIDDEN_LAYERS = 30
NUM_KV_HEADS = 16
HEAD_DIM = 64
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS = 1024
```

`MAX_NEW_TOKENS = 1024` corresponds to roughly 40 s of generated audio at the ~25 Hz Chatterbox speech-token rate. For texts near the 4000-character API limit this is short, but Phase 1 explicitly accepts that ceiling. The real-time path planned for later phases will inference sentence-by-sentence, so per-request token budgets become irrelevant there — we therefore deliberately do **not** expose `max_new_tokens` as an API parameter.

`_LANGUAGE_MAP` is reduced to eight entries:

```python
_LANGUAGE_MAP: dict[str, str] = {
    "English": "en", "German": "de", "French": "fr", "Spanish": "es",
    "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Korean": "ko",
}
```

A new helper `prepare_language(text: str, language_id: str) -> str`:

- For `language_id == "ko"`: decompose each Hangul syllable into initial/medial/final Jamo using the pure-Python formula from the reference (Hangul base `0xAC00`, initial `0x1100 + base // 588`, medial `0x1161 + (base % 588) // 28`, final `0x11A7 + base % 28` if non-zero).
- For all other languages: pass text through unchanged.
- Always prepend `f"[{language_id}]"`.

### `load_chatterbox_onnx` — path correction

The existing skeleton downloads `speech_encoder.onnx`, `embed_tokens.onnx`, `language_model.onnx`, `conditional_decoder.onnx` from the repo root. The actual repo layout (verified against the reference code at `_chatterbox_onnx_reference.py:364-371`) puts all four files inside an `onnx/` subfolder, each paired with a `.onnx_data` external-data sidecar.

Fix: pass `subfolder="onnx"` to each `hf_hub_download` call for both the `.onnx` and the `.onnx_data` files. Do not rely on `hf_hub_download` auto-fetching the sidecar — download it explicitly alongside each `.onnx`.

### `_OnnxBackend.generate` — inference flow

Replaces the current `NotImplementedError` block:

1. **Reference audio → `audio_values`:**
   `waveform, _ = librosa.load(io.BytesIO(reference_audio), sr=S3GEN_SR, mono=True)` then `audio_values = waveform[np.newaxis, :].astype(np.float32)`.

2. **Text preparation:**
   ```python
   text = prepare_language(text, language_id)
   input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
   position_ids = np.where(
       input_ids >= START_SPEECH_TOKEN, 0,
       np.arange(input_ids.shape[1])[None, :] - 1,
   ).astype(np.int64)
   ```

3. **Speech-encoder (once):** inputs `{"audio_values": audio_values}`, outputs `cond_emb, prompt_token, ref_x_vector, prompt_feat`.

4. **Embed-tokens initial call:** inputs `{"input_ids", "position_ids", "exaggeration": np.array([exaggeration], dtype=np.float32)}`, producing `inputs_embeds`. Concatenate: `inputs_embeds = np.concatenate([cond_emb, inputs_embeds], axis=1)`.

5. **Generation loop (up to `MAX_NEW_TOKENS`):**
   - Initial `past_key_values`: 60 zero tensors of shape `[1, NUM_KV_HEADS, 0, HEAD_DIM]`, keyed `past_key_values.{layer}.{key,value}`.
   - Initial `attention_mask`: `np.ones([1, seq_len], dtype=np.int64)` where `seq_len` is the length after `cond_emb` concatenation.
   - Per iteration:
     - `logits, *present_kv = language_model.run(None, {inputs_embeds, attention_mask, **past_kv})`
     - `logits = logits[:, -1, :]`
     - `logits = repetition_penalty_processor(generate_tokens, logits, penalty=REPETITION_PENALTY)`
     - Temperature sampling:
       ```python
       if temperature <= 0:
           next_token = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
       else:
           scaled = logits / temperature
           scaled -= scaled.max(axis=-1, keepdims=True)  # numerical stability
           probs = np.exp(scaled)
           probs /= probs.sum(axis=-1, keepdims=True)
           next_token = np.array(
               [[np.random.choice(probs.shape[-1], p=probs[0])]], dtype=np.int64
           )
       ```
     - `generate_tokens = np.concatenate([generate_tokens, next_token], axis=-1)`; break on `STOP_SPEECH_TOKEN`.
     - Re-embed the new token: reuse the `embed_tokens` input dict built in step 4, **mutate only `input_ids` and `position_ids`** for this iteration (the `exaggeration` array stays constant across the loop, matching the reference), then `embed_tokens.run(None, ...)` → fresh `inputs_embeds` of shape `[1, 1, hidden]`.
     - `attention_mask` extends by 1; `past_kv = present_kv`.

6. **Waveform synthesis:**
   ```python
   speech_tokens = generate_tokens[:, 1:-1]
   speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)
   wav = conditional_decoder.run(None, {
       "speech_tokens": speech_tokens,
       "speaker_embeddings": ref_x_vector,
       "speaker_features": prompt_feat,
   })[0]
   return np.squeeze(wav, axis=0).astype(np.float32), S3GEN_SR
   ```

The `RepetitionPenaltyLogitsProcessor` class from the reference is lifted into a small top-level function (no class needed — single call site). `cfg_weight` is accepted on the method signature but not referenced; the docstring states this explicitly.

### `backend/pyproject.toml`

Add `librosa>=0.10,<0.12` to the `dependencies` array. Run `uv lock && uv sync --dev`.

## Testing

Real ONNX inference cannot run in GPU-free unit tests (the model files are external). The inference loop itself is validated **manually** via parent-plan Task 14. Unit coverage stays focused on the deterministic building blocks:

Extensions in `backend/tests/test_chatterbox_tts.py`:

- **Update** `test_language_mapping_covers_all_common_languages` — assert against the eight remaining languages; remove the `Japanese` and `Chinese` assertions.
- **New** `test_language_mapping_rejects_japanese_and_chinese` — both raise `ValueError` from `language_to_iso639` with an informative message mentioning the language.
- **New** `test_prepare_language_prepends_language_token` — `prepare_language("Hallo", "de") == "[de]Hallo"`.
- **New** `test_prepare_language_korean_jamo_decomposition` — `prepare_language("안", "ko")` produces `"[ko]" + chr(0x110B) + chr(0x1161) + chr(0x11AB)` (ㅇ + ㅏ + ㄴ), verifying the decomposition formula on a single known syllable.
- **New** `test_prepare_language_korean_passthrough_for_non_hangul` — ASCII text under `ko` is not mutated beyond the language token.
- **New** `test_repetition_penalty_processor` — unit-tests the helper with hand-constructed arrays: positive scores at visited indices are divided by 1.2, negative ones are multiplied.
- **New** `test_temperature_sampling_zero_is_greedy` — given a fixed logits array, the sampling helper returns `argmax` when `temperature=0`.
- **New** `test_temperature_sampling_is_stochastic` — given a fixed logits array, multiple seeds produce different tokens at `temperature=1.0` (statistical; assert at least two distinct tokens across N≥20 seeds).

`test_onnx_loader_raises_on_missing_onnxruntime` stays — the ImportError surface is still covered.

ONNX `InferenceSession` objects are **not** mocked. A fake "session whose `.run()` returns handcrafted arrays in the expected shapes" would be brittle against upstream schema changes and rewards us nothing beyond false confidence.

## Deliverables

- `backend/voice/engines/chatterbox_tts.py` — module-level constants, reduced `_LANGUAGE_MAP`, new `prepare_language()` helper, path-corrected `load_chatterbox_onnx`, fully populated `_OnnxBackend.generate`.
- `backend/pyproject.toml` — `librosa` as direct dep.
- `backend/uv.lock` — regenerated.
- `backend/tests/test_chatterbox_tts.py` — new unit tests per the list above.
- **One commit** on `chatterbox-integration`, following parent-plan Step 8 conventions: separate from the skeleton commit, imperative message, `Co-Authored-By: Claude Opus 4.7 (1M context)` trailer.

Pre-commit gate: `cd backend && uv run pytest -q && uv run ruff check .` — must stay green.

## Out of scope

- **Classifier-free guidance** in the ONNX backend.
- **Japanese, Chinese, Hebrew** language support (returns `ValueError` from the language mapper until a follow-up adds the deps).
- **`max_new_tokens` as API parameter** — hardcoded constant; exposed config/parameter is deferred to the realtime phase that will inference sentence-by-sentence and make this irrelevant.
- **Watermarking** (`resemble-perth`) — deferred to the production-facing phase.
- **Mocked ONNX sessions** for integration tests — covered by manual smoke (Task 14) instead.
- **Optimisation work** (provider-specific flags, session options, batch-size tuning) — Phase 1 measures baseline RTF first, then optimises.

## Known follow-ups (tracked here so they do not get lost)

1. **Watermarking integration** with `resemble-perth` behind a config flag before any public endpoint.
2. **Classifier-free guidance** once a validated upstream reference or a cross-backend behavioural comparison is available.
3. **Sentence-by-sentence inference** in a later real-time spec — removes the `MAX_NEW_TOKENS` ceiling as a per-request concern.
4. **`ja`, `zh`, `he` support** driven by demand; low-effort extension once a real use case appears.

## Done criteria

1. `uv run pytest -q` green.
2. `uv run ruff check .` clean.
3. Manual smoke test (parent plan Task 14) produces audible German / English / French audio.
4. Single commit on `chatterbox-integration` merged with the Step 8 message pattern.
