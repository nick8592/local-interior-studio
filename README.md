# Local Interior Studio

Fully offline interior design tool that restyles room photos using a locally-runnable image editing model. No cloud APIs, no data leaving your machine.

## Key Features

- **Privacy First**: All image processing happens locally.
- **Zero Cost**: No per-image API fees.
- **Offline Capable**: Works without internet access.
- **Low Latency**: No upload or download round-trips.

## Quick Start

### Prerequisites
- **Docker** (recommended) or Python 3.10+
- **NVIDIA Container Toolkit** (for Docker GPU passthrough)
- **NVIDIA GPU** with CUDA support (~4 GB+ VRAM)

### Run with Docker
▶ `docker compose up --build`
> Opens http://localhost:7860. Models are cached in `./models` and reused across rebuilds.

### Run without Docker
▶ `python -m venv .venv && source .venv/bin/activate`
▶ `pip install -r requirements.txt`
▶ `python app.py`
> Opens http://localhost:7860.

## How It Works

Gradio UI  →  Pipeline (Edit / Inpaint / Segment)  →  Output

**Pipeline Components**
- **Edit Pipeline**: Uses InstructPix2Pix to restyle a room based on a text prompt.
- **Inpaint Pipeline**: Uses Stable Diffusion Inpainting to modify only the masked regions of an image.
- **Segment Pipeline**: Employs Segment Anything (SAM) to automatically detect and isolate objects.
- **Utilities**: Handles image resizing, padding, and mask extraction.

> Uses InstructPix2Pix (SD 1.5, ~4 GB VRAM) by default. See architecture docs for alternative model support.

## UI Overview

The tool uses a 3-tab workflow to move from global styling to precise object editing.

| Tab | Purpose | Key Controls | Status |
|---|---|---|---|
| **Restyle** | Global room style change | Style presets, Custom prompt, Edit strength | ✅ Implemented |
| **Auto-Segment** | Object selection + direct inpainting | SAM auto-segment, Multi-select canvas, Inpaint prompt | ✅ Implemented |
| **Masked Edit** | Manual brush-based inpainting | Brush mask, Inpaint prompt, Strength | ✅ Implemented |

The **Auto-Segment** tab features an interactive HTML5 canvas that allows users to hover over detected objects for info and click to select multiple regions. Selected objects are listed with color swatches and area percentages. Users can inpaint directly on the same page without switching tabs. The **Masked Edit** tab remains available for manual brush-drawn masks.

## Hardware Requirements

| Tier | GPU | VRAM | Expected Perf |
|---|---|---|---|
| Minimum | GTX 1060 / M1 | 6 GB | SD 1.5 models, ~15 s/img |
| Recommended | RTX 3060 / M2 Pro | 8–12 GB | SDXL or FLUX, ~10 s/img |
| Comfortable | RTX 4070+ / M3 Max | 12+ GB | All models, <5 s/img |

> CPU-only mode is available via `torch.float32` (SD 1.5), but expect 30–60 s/img.

## Configuration

<details>
<summary>View Environment Variables</summary>

Copy `.env.example` to `.env` and adjust as needed.

**Models**
- `EDIT_MODEL_ID`: HuggingFace model ID (default: `timbrooks/instruct-pix2pix`)
- `SAM_CHECKPOINT`: SAM checkpoint filename (default: `sam_vit_h_4b8939.pth`)
- `INPAINT_MODEL_ID`: HuggingFace inpaint model ID (default: `runwayml/stable-diffusion-inpainting`)

**Generation**
- `DEFAULT_STEPS`: Diffusion inference steps (default: `20`)
- `DEFAULT_GUIDANCE_SCALE`: Text prompt adherence (default: `7.5`)
- `DEFAULT_IMAGE_GUIDANCE_SCALE`: Edit strength (default: `1.5`)
- `DEFAULT_STRENGTH`: Inpaint strength (default: `1.0`)
- `INPAINT_STEPS`: Inpaint diffusion steps (default: `25`)
- `INPAINT_GUIDANCE_SCALE`: Inpaint text prompt adherence (default: `7.5`)

**Server**
- `DEVICE`: `cuda`, `mps`, `cpu`, or `auto` (default: `auto`)
- `OUTPUT_DIR`: Where generated images are saved (default: `output`)
- `SERVER_NAME`: Gradio server bind address (default: `0.0.0.0`)
- `SERVER_PORT`: Gradio server port (default: `7860`)
</details>

## Project Structure

<details>
<summary>View Directory Layout</summary>

```
local-interior-studio/
├── Dockerfile              # GPU-enabled image
├── docker-compose.yml      # GPU passthrough & volumes
├── .env.example            # Env template
├── requirements.txt
├── app.py                  # Gradio entry point
├── components/
│   └── instance_selector.py  # Interactive selection canvas
├── pipeline/
│   ├── edit.py             # InstructPix2Pix wrapper
│   ├── inpaint.py          # SD Inpainting wrapper
│   ├── segment.py          # SAM segmentation
│   └── presets.py          # Style prompt templates
├── models/                 # Model weights
├── examples/               # Example room photos
├── output/                 # Generated images
└── tests/                  # Unit tests (mocked)
```
</details>

## Style Presets

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

## Docker Setup

| File | Purpose |
|---|---|
| `Dockerfile` | PyTorch + CUDA base image, installs dependencies, sets entrypoint |
| `docker-compose.yml` | GPU passthrough, volume mounts (`models/`, `output/`), port 7860 |
| `.dockerignore` | Excludes `.venv`, `models/` weights, `output/`, `.git` from build context |

See `docker-compose.yml` for the full configuration.

## Testing

```bash
pip install pytest
pytest tests/ -v
```
Tests use mocked ML objects and run without GPU or internet access.

## Roadmap

- [x] **v0.1 — Proof of concept** — single-image restyle with InstructPix2Pix + Gradio UI (Restyle tab)
- [x] **v0.2 — Image editing (inpainting)** — SAM segmentation + user-drawn mask + Stable Diffusion Inpainting (Masked edit tab)
- [x] **v0.3 — Interactive auto-segment** — Interactive instance canvas with hover/select and mask transfer
- [x] **v0.4 — Direct inpaint in Auto-Segment** — Select objects on panoptic overlay, inpaint directly without tab-switch; selected-objects list with color swatches; Mask-Edit retained for manual brush workflow

## License

MIT
