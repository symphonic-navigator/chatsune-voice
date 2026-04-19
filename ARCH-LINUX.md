# Running chatsune-voice bare metal on Arch Linux

This document complements `README.md` and `compose.yml`. It targets developers
who want to run the backend directly on an Arch host — either to avoid the
Docker layer during active development, or to install the optional
`chatterbox-tts` Torch fallback (which conflicts with the pinned
`qwen-tts` dependency and therefore cannot live in the main virtualenv).

The canonical deployment is still `docker compose up`. Use this document only
if bare metal is required.

## Prerequisites

Arch packages (install once):

```fish
sudo pacman -S --needed base-devel git python uv \
                        libsndfile ffmpeg sox \
                        espeak-ng
```

- `libsndfile` — needed by `soundfile`.
- `ffmpeg` — needed by `librosa` for non-WAV inputs.
- `sox` — silences the noisy probe `qwen-tts` runs at import.
- `espeak-ng` — fallback phoneme backend for a few TTS stacks. Not strictly
  required for Chatterbox itself but cheap to have available.

For GPU inference (pick one):

- **NVIDIA**: `cuda` and a matching driver, OR rely on PyTorch/ONNXRuntime
  PyPI wheels that bundle their own CUDA runtime. The wheels are usually the
  path of least resistance.
- **AMD / Strix Halo (ROCm)**: install `rocm-hip-runtime` from the official
  repos or AUR. The base backend runs against PyTorch-ROCm built for
  `gfx1151`.

## Fast path — development venv (ONNX backend, any GPU)

The ONNX Chatterbox backend is the default and requires no extra wheels
beyond what `pyproject.toml` declares. On Arch this is the recommended setup.

```fish
git clone https://github.com/symphonic-navigator/chatsune-voice.git
cd chatsune-voice/backend
uv sync --dev
```

Run the server:

```fish
cp ../.env.example ../.env
# edit ../.env — leave CHATTERBOX_BACKEND unset (defaults to "onnx")
uv run uvicorn voice.main:app --reload --port 8002
```

The process reads `.env` from the repository root via `pydantic-settings`.
Point a smoke test at it:

```fish
curl -sS http://127.0.0.1:8002/v1/health | jq
```

### GPU providers on the ONNX backend

The loader auto-selects the strongest ONNX execution provider at startup.
On a fresh `uv sync`, the installed wheel is CPU-only because `onnxruntime`
on PyPI has no ROCm build. To get GPU acceleration, replace the wheel
*after* `uv sync`:

**NVIDIA.**

```fish
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

**AMD (Strix Halo, gfx1151).** AMD publishes a ROCm build only for CPython
3.12. If you are on Arch's default 3.13, install a side-by-side 3.12 first:

```fish
uv python install 3.12
cd backend
uv venv --python 3.12 --clear
uv sync --dev
uv pip uninstall onnxruntime
uv pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/onnxruntime_rocm-1.19.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

Verify the provider at runtime — the server logs
`chatterbox_onnx_loaded providers=[...]` during model load. If you see only
`CPUExecutionProvider`, the GPU wheel did not take and inference will run
on CPU.

For Strix Halo also export the GFX override before starting the server
(matches the value in the root `.env`):

```fish
set -x HSA_OVERRIDE_GFX_VERSION 11.5.1
```

## Torch backend — separate venv

The Torch backend exists as a fallback for historical reasons: before the
ONNX inference loop was adapted from upstream, clone requests against ONNX
returned empty WAVs. That path is now complete (`chatterbox_tts.py:279-383`),
so **most users no longer need this section**. Follow it only when you have
a specific reason to prefer the upstream PyTorch reference implementation.

### Why a separate venv

`chatterbox-tts` pins a `transformers` version that conflicts with the one
`qwen-tts` requires. They cannot share a venv. Running the full backend
(all three TTS modes) on bare metal is therefore not possible; the Torch
fallback venv supports **only** the `clone` mode.

### Setup

```fish
cd chatsune-voice/backend
uv venv --python 3.12 .venv-chatterbox
source .venv-chatterbox/bin/activate.fish
```

Install PyTorch first — the order matters. If you let `chatterbox-tts` pull
Torch itself, pip picks the default CUDA wheel regardless of your hardware.

**NVIDIA (default CUDA wheel).**

```fish
uv pip install torch torchaudio
```

**AMD / Strix Halo (ROCm 6.2 wheel).**

```fish
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch torchaudio
```

Then install Chatterbox plus the subset of backend dependencies needed to
run the server (everything from `pyproject.toml` *except* `qwen-tts` and
`onnxruntime`, which would reintroduce the conflict):

```fish
uv pip install chatterbox-tts \
    fastapi "uvicorn[standard]" pydantic pydantic-settings structlog \
    faster-whisper numpy soundfile python-multipart librosa
```

### Starting the server with the Torch venv

```fish
# in backend/ with .venv-chatterbox activated
set -x CHATTERBOX_BACKEND torch
set -x TTS_ENABLED_MODES clone
set -x HSA_OVERRIDE_GFX_VERSION 11.5.1   # Strix Halo only
uvicorn voice.main:app --host 0.0.0.0 --port 8002
```

Stop any running Docker Compose stack first — it binds the same port.

### Verifying the Torch load

The server logs `tts_model_loaded mode=clone` on successful startup.
A subsequent `/v1/speak/clone` request should stream a non-empty WAV.
If you see `clone_error phase=before_stream` in the log, check that
`chatterbox-tts` imported cleanly and that the HuggingFace cache is writable.

## systemd user unit (optional)

If you run bare metal regularly, a user-scope unit keeps the backend out of
your shell history. Example at `~/.config/systemd/user/chatsune-voice.service`:

```ini
[Unit]
Description=chatsune-voice backend (bare metal)
After=network-online.target

[Service]
WorkingDirectory=%h/workspace/chatsune-voice/backend
EnvironmentFile=%h/workspace/chatsune-voice/.env
ExecStart=%h/.local/bin/uv run uvicorn voice.main:app --host 0.0.0.0 --port 8002
Restart=on-failure

[Install]
WantedBy=default.target
```

Enable with `systemctl --user enable --now chatsune-voice`. `.env` is read
by the unit from the repo root and by the backend process from the same
file, so there is no duplication.

## Troubleshooting

- **`403 Forbidden` on `/v1/speak/clone`** — `clone` is not in
  `TTS_ENABLED_MODES`. Add it and restart the process.
- **`422 reference_audio_too_long`** — the reference clip exceeds
  `CHATTERBOX_MAX_REFERENCE_SECONDS` (30 s by default). The server does
  not trim; the README's 3 — 15 s range is a quality recommendation, not an
  automatic cap.
- **Empty WAV response, `clone_error phase=before_stream
  error_type=NotImplementedError`** — you are running an older backend
  revision where the ONNX loop was still stubbed. Pull the latest master.
- **ONNX runs on CPU despite GPU hardware** — the stock `onnxruntime` wheel
  is CPU-only. See the provider swap recipes above.
- **`pip install chatterbox-tts` downgrades `transformers` in the main
  venv** — you installed it into the wrong venv. Activate
  `.venv-chatterbox` first, or use `uv pip install --python
  .venv-chatterbox/bin/python ...` explicitly.
