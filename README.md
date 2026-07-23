# Local Interior Studio

Fully offline interior design tool that restyles room photos using a locally-runnable image editing model — no cloud API, no data leaving the laptop.

## Why Local?

- **Privacy** — room photos stay on your machine
- **Zero cost** — no per-image API fees
- **No internet required** — works on a plane, in a cabin, behind a strict proxy
- **Low latency** — no upload/download round-trip

## Architecture (planned)

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  UI Layer     │────▶│  Pipeline Core  │────▶│  Output       │
│  (Gradio/Web) │     │                 │     │  (Image +     │
│               │     │  ┌───────────┐ │     │   Mask Diff)  │
│  Upload photo │     │  │ Instruct- │ │     │               │
│  Pick style   │     │  │ Pix2Pix   │ │     │               │
│  Edit mask    │     │  │ (local)   │ │ │     │               │
│  View result  │     │  └───────────┘ │     │               │
└──────────────┘     └─────────────────┘     └──────────────┘
```

### Core components

| Component | Purpose | Candidate |
|---|---|---|
| **Image editing model** | Restyle room given a text prompt | InstructPix2Pix (SD-based, ~4 GB VRAM) |
| **Room segmentation** | Auto-detect walls, floor, furniture | Segment Anything (SAM) or OneFormer |
| **Style presets** | Curated prompt templates | Minimalist, Scandinavian, Industrial, Japandi, Bohemian … |
| **Mask editor** | Let user constrain which regions change | Gradio ImageEditor or simple canvas overlay |
| **Pipeline orchestrator** | Tie segmentation → masking → editing | Python script / simple API |

### Model selection rationale

| Model | VRAM | Speed (1080 Ti) | Quality | Edit fidelity |
|---|---|---|---|---|
| InstructPix2Pix (SD 1.5) | ~4 GB | ~8 s/img | Good | High |
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

## Project structure (planned)

```
local-interior-studio/
├── README.md
├── Dockerfile              # GPU-enabled Docker image (PyTorch + CUDA)
├── docker-compose.yml      # One-command launch with GPU passthrough
├── .dockerignore
├── requirements.txt
├── app.py                  # Gradio / web UI entry point
├── pipeline/
│   ├── __init__.py
│   ├── edit.py             # InstructPix2Pix inference wrapper
│   ├── segment.py          # Room segmentation (SAM / OneFormer)
│   └── presets.py          # Style prompt templates
├── models/
│   └── README.md           # Download instructions (auto-download on first run)
├── output/                 # Generated images (mounted volume)
├── utils/
│   ├── __init__.py
│   └── image.py            # Resize, pad, color-space helpers
└── tests/
    ├── test_edit.py
    └── test_segment.py
```

## Quick start (once implemented)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU passthrough)
- NVIDIA GPU with CUDA support (~4 GB+ VRAM)

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

## Docker setup details

| File | Purpose |
|---|---|
| `Dockerfile` | PyTorch + CUDA base image, installs dependencies, sets entrypoint |
| `docker-compose.yml` | GPU passthrough, volume mounts (`models/`, `output/`), port 7860 |
| `.dockerignore` | Excludes `.venv`, `models/` weights, `output/`, `.git` from build context |

### docker-compose.yml (planned shape)

```yaml
services:
  studio:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models    # cached model weights (~4 GB)
      - ./output:/app/output    # generated images
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## UI — Gradio interface

The entire user interaction flows through a **Gradio** web UI served at `http://localhost:7860`. No separate frontend needed — Gradio handles image upload, mask drawing, style selection, and result display in a single browser tab.

### Planned Gradio tabs

| Tab | Workflow | Gradio components |
|---|---|---|
| **Restyle** | Upload room photo → pick style / write prompt → generate | `Image` upload, `Dropdown` / `Textbox` for prompt, `Slider` for edit strength, `Image` output |
| **Masked edit** | Upload photo → auto-segment → draw mask → restyle masked area only | `ImageEditor` (brush mask), `Segmentation` overlay, `Image` output |
| **Batch** | Upload folder of photos → pick style → restyle all | `File` (multiple), `Dropdown`, `Gallery` output |
| **Upscale** | Upload or select a previous result → 4× upscale | `Image` input, `Image` output |

### Why Gradio

- **Zero frontend code** — pure Python, no HTML/CSS/JS needed
- **Built-in image editor** — `gr.ImageEditor` supports brush/eraser for mask drawing
- **Browser-based** — accessible from any device on the local network (phone, tablet)
- **Shareable** — `share=True` generates a temporary public link if needed
- **Fast to iterate** — hot-reload with `gr.reload()` during development

## Roadmap

- [ ] **v0.1 — Proof of concept** — single-image restyle with InstructPix2Pix + Gradio UI (Restyle tab)
- [ ] **v0.2 — Masked editing** — SAM segmentation + user-drawn mask + inpainting (Masked edit tab)
- [ ] **v0.3 — Style presets** — curated prompt library with preview thumbnails
- [ ] **v0.4 — Multi-room batch** — process a folder of room photos with one style (Batch tab)
- [ ] **v0.5 — Upscale output** — Real-ESRGAN 4× upscaling for print-quality renders (Upscale tab)

## License

MIT
