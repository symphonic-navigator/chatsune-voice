"""Chatterbox Multilingual TTS adapter.

Exposes the TTSModel protocol by delegating to a backend object that handles
actual inference. Two loader functions live here (ONNX Runtime and Torch);
each one returns an object conforming to _ChatterboxBackend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from voice.engines.protocol import TTSMode, TTSRequest
from voice.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 4096

# Chatterbox ONNX inference constants (extracted from the upstream reference).
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
NUM_HIDDEN_LAYERS = 30
NUM_KV_HEADS = 16
HEAD_DIM = 64
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS = 1024

# Chatterbox uses ISO-639-1 codes. Our Language literal uses full English names.
# "Auto" is deliberately not mapped — Chatterbox needs a concrete language.
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
    """Load the Chatterbox Multilingual model via the chatterbox-tts pip package.

    The `chatterbox-tts` PyPI package is NOT a declared dependency of this
    project — it conflicts with the transformers version pinned by qwen-tts.
    Users who want the Torch fallback install it themselves (see README).
    If it is not importable when this loader runs, a clean ImportError
    propagates to the registry's ModelLoadError.

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
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            return wav, self.sample_rate

    return _TorchBackend()


def load_chatterbox_onnx(model_id: str, *, device: str) -> _ChatterboxBackend:
    """Load Chatterbox Multilingual via ONNX Runtime.

    Downloads four ONNX model files from the hub (speech_encoder, embed_tokens,
    language_model, conditional_decoder), creates InferenceSession objects
    with the appropriate ExecutionProvider, and implements the autoregressive
    inference loop from the upstream reference example.

    device: 'cuda' (-> CUDAExecutionProvider on NVIDIA, ROCMExecutionProvider
    on AMD via the onnxruntime-rocm wheel), 'cpu' (-> CPUExecutionProvider).
    """
    import io  # still needed — librosa.load accepts a BytesIO

    import librosa
    import onnxruntime
    from huggingface_hub import hf_hub_download

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
        return hf_hub_download(repo_id=model_id, filename=filename, subfolder="onnx")

    speech_encoder_path = _dl("speech_encoder.onnx")
    _dl("speech_encoder.onnx_data")
    embed_tokens_path = _dl("embed_tokens.onnx")
    _dl("embed_tokens.onnx_data")
    language_model_path = _dl("language_model.onnx")
    _dl("language_model.onnx_data")
    conditional_decoder_path = _dl("conditional_decoder.onnx")
    _dl("conditional_decoder.onnx_data")

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
        sample_rate: int = 24000

        def __init__(self) -> None:
            self._tokenizer = tokenizer
            self._speech_encoder = speech_encoder
            self._embed_tokens = embed_tokens
            self._language_model = language_model
            self._conditional_decoder = conditional_decoder

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

    return _OnnxBackend()


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
