# Local Interior Studio

Fully offline interior design tool that restyles room photos using a locally-runnable image editing model — no cloud API, no data leaving the laptop.

## Why Local?

- **Privacy** — room photos stay on your machine
- **Zero cost** — no per-image API fees
- **No internet required** — works on a plane, in a cabin, behind a strict proxy
- **Low latency** — no upload/download round-trip

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  UI Layer     │────▶│  Pipeline Core  │────▶│  Output       │
│  (Gradio)     │     │                 │     │  (Image +     │
│               │     │  ┌───────────┐ │     │   Save)       │
│  Upload photo │     │  │ Instruct- │ │     │               │
│  Pick style   │     │  │ Pix2Pix   │ │     │               │
│  View result  │     │  │ (local)   │ │ │     │               │
└──────────────┘     │  └───────────┘ │     └──────────────┘
                     │  ┌───────────┐ │
                     │  │ SAM       │ │
                     │  │ Segment   │ │
                     │  └───────────┘ │
                     │  ┌───────────┐ │
                     │  │ Presets   │ │
                     │  └───────────┘ │
                     └─────────────────┘
```

### Core components

| Component | Purpose | Implementation |
|---|---|---|
| **Image editing model** | Restyle room given a text prompt | InstructPix2Pix (SD 1.5, ~4 GB VRAM) via `pipeline/edit.py` |
| **Room segmentation** | Auto-detect walls, floor, furniture | Segment Anything (SAM) via `pipeline/segment.py` |
| **Style presets** | Curated prompt templates | 10 presets in `pipeline/presets.py` |
| **Image utilities** | Resize, pad, color-space, mask overlay | `utils/image.py` |
| **Web UI** | Upload → pick style → generate | Gradio Blocks via `app.py` |

### Model selection rationale

| Model | VRAM | Speed (1080 Ti) | Quality | Edit fidelity |
|---|---|---|---|---|
| InstructPix2Pix (SD 1.5) ✅ | ~4 GB | ~8 s/img | Good | High |
| SDXL Inpainting | ~8 GB | ~15 s/img | Excellent | Medium |
| FLUX.1-schnell + ControlNet | ~12 GB | ~20 s/img | Best | Medium |
| Stable Diffusion + ControlNet | ~6 GB | ~10 s/img | Good | High |

**Starting choice: InstructPix2Pix** — runs on 4 GB VRAM (most laptops), one-pass edit, no separate inpaint step.

## Minimum hardware

| Tier | GPU | VRAM | Expected perf |
|---|---|---|---|
| Minimum | GTX 1060 / M1 | 6 GB | SD 1.5 models, ~15 s/img |
| Recommended | RTX 3060 / M2 Pro | 8–12 GB | SDXL or FLUX, ~10 s/img |
| Comfortable | RTX 4070+ / M3 Max | 12+ GB | All models, <5 s/img |

CPU-only is possible via `torch.float32` on SD 1.5 but expect 30–60 s/img.

## Project structure

```
local-interior-studio/
├── README.md
├── Dockerfile              # GPU-enabled Docker image (PyTorch + CUDA)
├── docker-compose.yml      # One-command launch with GPU passthrough
├── .dockerignore
├── .env.example            # Environment variable template
├── requirements.txt
├── app.py                  # Gradio web UI entry point
├── pipeline/
│   ├── __init__.py
│   ├── edit.py             # InstructPix2Pix inference wrapper
│   ├── segment.py          # Room segmentation (SAM)
│   └── presets.py          # Style prompt templates (10 presets)
├── models/
│   └── README.md           # Download instructions (auto-download on first run)
├── output/                 # Generated images (mounted volume)
├── utils/
│   ├── __init__.py
│   └── image.py            # Resize, pad, color-space, mask overlay helpers
└── tests/
    ├── __init__.py
    ├── test_edit.py         # Unit tests for edit pipeline (mocked)
    └── test_segment.py      # Unit tests for segmentation (mocked)
```

## Quick start

### Prerequisites

- **Docker** (recommended) — or a local Python 3.10+ environment
- **NVIDIA Container Toolkit** (for GPU passthrough in Docker)
- **NVIDIA GPU** with CUDA support (~4 GB+ VRAM)

### Run with Docker Compose

```bash
docker compose up --build
# → Opens http://localhost:7860
```

Models are cached in the `./models` volume — downloaded once, reused across rebuilds.

### Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → Opens http://localhost:7860
```

### Environment variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `EDIT_MODEL_ID` | `timbrooks/instruct-pix2pix` | HuggingFace model ID |
| `SAM_CHECKPOINT` | `sam_vit_h_4b8939.pth` | SAM checkpoint filename |
| `DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` |
| `DEFAULT_STEPS` | `20` | Diffusion inference steps |
| `DEFAULT_GUIDANCE_SCALE` | `7.5` | Text prompt adherence |
| `DEFAULT_IMAGE_GUIDANCE_SCALE` | `1.5` | Edit strength (1.0–3.0) |
| `OUTPUT_DIR` | `output` | Where generated images are saved |
| `SERVER_NAME` | `0.0.0.0` | Gradio server bind address |
| `SERVER_PORT` | `7860` | Gradio server port |

## UI — Gradio interface

The entire user interaction flows through a **Gradio** web UI served at `http://localhost:7860`. No separate frontend needed — Gradio handles image upload, style selection, and result display in a single browser tab.

### Restyle tab (v0.1 — implemented)

Upload a room photo → pick a style preset or write a custom prompt → adjust edit strength → click **Generate** → view and download the restyled result.

| Control | Component | Range |
|---|---|---|
| Source photo | `gr.Image` (upload/clipboard) | — |
| Style preset | `gr.Dropdown` | 10 presets |
| Custom prompt | `gr.Textbox` | Free text |
| Negative prompt | `gr.Textbox` | Free text |
| Edit strength | `gr.Slider` | 1.0 – 3.0 (default 1.5) |
| Inference steps | `gr.Slider` | 10 – 50 (default 20) |
| Seed | `gr.Number` | −1 = random |

### Planned tabs

| Tab | Status | Workflow |
|---|---|---|
| **Restyle** | ✅ Implemented | Upload → pick style → generate |
| **Masked edit** | 🔲 Planned | Auto-segment → draw mask → restyle masked area only |
| **Batch** | 🔲 Planned | Upload folder → pick style → restyle all |
| **Upscale** | 🔲 Planned | 4× Real-ESRGAN upscaling for print quality |

### Available style presets

| Style | Description |
|---|---|
| Minimalist | Clean lines, neutral tones, and purposeful simplicity |
| Scandinavian | Light wood, white walls, and cozy hygge warmth |
| Industrial | Exposed brick, raw steel, and unfinished urban character |
| Japandi | Japanese wabi-sabi meets Scandinavian warmth and simplicity |
| Bohemian | Layered textures, global accents, and free-spirited warmth |
| Mid-Century Modern | Retro elegance with teak, tapered legs, and warm retro tones |
| Coastal | Breezy ocean palette with natural textures and nautical charm |
| Rustic | Reclaimed wood, stone, and cozy cabin warmth |
| Art Deco | Bold geometry, brass, velvet, and 1920s glamour |
| Modern Farmhouse | Shiplap, vintage charm, and contemporary comfort combined |

## Docker setup details

| File | Purpose |
|---|---|
| `Dockerfile` | PyTorch + CUDA base image, installs dependencies, sets entrypoint |
| `docker-compose.yml` | GPU passthrough, volume mounts (`models/`, `output/`), port 7860 |
| `.dockerignore` | Excludes `.venv`, `models/` weights, `output/`, `.git` from build context |

### docker-compose.yml

```yaml
version: "3.8"
services:
  studio:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Tests use mocked ML objects and run without GPU, model downloads, or internet access.

## Roadmap

- [x] **v0.1 — Proof of concept** — single-image restyle with InstructPix2Pix + Gradio UI (Restyle tab)
- [ ] **v0.2 — Masked editing** — SAM segmentation + user-drawn mask + inpainting (Masked edit tab)
- [ ] **v0.3 — Style presets** — curated prompt library with preview thumbnails
- [ ] **v0.4 — Multi-room batch** — process a folder of room photos with one style (Batch tab)
- [ ] **v0.5 — Upscale output** — Real-ESRGAN 4× upscaling for print-quality renders (Upscale tab)

## License

MIT
