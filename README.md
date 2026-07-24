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
│  Draw mask    │     │  │ (local)   │ │     │               │
│  Select object│     │  └───────────┘ │     │               │
│  View result  │     │  ┌───────────┐ │     └──────────────┘
└──────────────┘     │  │ SD        │ │
                     │  │ Inpaint   │ │
                     │  └───────────┘ │
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
| **Inpainting model** | Restyle masked region only | Stable Diffusion Inpainting (~4 GB VRAM) via `pipeline/inpaint.py` |
| **Room segmentation** | Auto-detect and select individual objects in a room | Segment Anything (SAM) via `pipeline/segment.py` |
| **Style presets** | Curated prompt templates | 10 presets in `pipeline/presets.py` |
| **Image utilities** | Resize, pad, color-space, mask overlay, mask extraction, unpad, save | `utils/image.py` |
| **Web UI** | Upload → pick style → generate / select objects → inpaint | Gradio Blocks + interactive instance canvas (`gr.HTML` + iframe) via `app.py` |

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
├── components/
│   ├── __init__.py
│   └── instance_selector.py  # Interactive instance selection canvas (gr.HTML + iframe)
├── pipeline/
│   ├── __init__.py
│   ├── edit.py             # InstructPix2Pix inference wrapper
│   ├── inpaint.py          # Stable Diffusion Inpainting wrapper
│   ├── segment.py          # Room segmentation (SAM)
│   └── presets.py          # Style prompt templates (10 presets)
├── models/
│   └── README.md           # Download instructions (auto-download on first run)
├── examples/               # Example room photos (shown in Gradio UI)
├── output/                 # Generated images (mounted volume)
├── utils/
│   ├── __init__.py
│   └── image.py            # Resize, pad, unpad, color-space, mask overlay, mask extraction, save helpers
└── tests/
    ├── __init__.py
    ├── test_edit.py         # Unit tests for edit pipeline (mocked)
    ├── test_inpaint.py      # Unit tests for inpaint pipeline (mocked)
    ├── test_image.py        # Unit tests for mask utilities (mocked)
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
| `INPAINT_MODEL_ID` | `runwayml/stable-diffusion-inpainting` | HuggingFace inpaint model ID |
| `INPAINT_STEPS` | `25` | Inpaint diffusion steps |
| `INPAINT_GUIDANCE_SCALE` | `7.5` | Inpaint text prompt adherence |
| `DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` |
| `DEFAULT_STEPS` | `20` | Diffusion inference steps |
| `DEFAULT_GUIDANCE_SCALE` | `7.5` | Text prompt adherence |
| `DEFAULT_IMAGE_GUIDANCE_SCALE` | `1.5` | Edit strength (1.0–3.0) |
| `DEFAULT_STRENGTH` | `1.0` | Inpaint strength (0.1–1.0) |
| `OUTPUT_DIR` | `output` | Where generated images are saved |
| `SERVER_NAME` | `0.0.0.0` | Gradio server bind address |
| `SERVER_PORT` | `7860` | Gradio server port |

## UI — Gradio interface

The entire user interaction flows through a **Gradio** web UI served at `http://localhost:7860`. No separate frontend needed — Gradio handles image upload, style selection, and result display in a single browser tab.

### Masked Edit tab (v0.2 — implemented)

Upload a room photo → paint a mask on the areas you want to edit → describe what to fill the masked area with → adjust inpaint settings and click **Generate (Inpaint)** → view and download the result with only the masked region changed.

This tab is for **image editing only** (adding, removing, or replacing objects via inpainting). For **style changes** (restyling the entire room), use the Restyle tab. For **automatic object detection and mask selection**, use the Auto-Segment tab.

| Control | Component | Range |
|---|---|---|
| Source photo + mask | `gr.ImageEditor` (RGBA, brush) | — |
| Inpaint Prompt | `gr.Textbox` | Free text |
| Negative Prompt | `gr.Textbox` | Free text |
| Inpaint Strength | `gr.Slider` | 0.1 – 1.0 (default 1.0) |
| Guidance Scale | `gr.Slider` | 1.0 – 15.0 (default 7.5) |
| Inference Steps | `gr.Slider` | 10 – 50 (default 25) |
| Mask Dilation | `gr.Slider` | 0 – 30 px (default 10) |
| Seed | `gr.Number` | −1 = random |

### Auto-Segment tab (v0.3 — implemented)

Upload a room photo → click **Auto-Segment** → SAM detects all objects and renders them as interactive panoptic outlines on the image → **hover** over any instance to highlight it with a label tooltip (e.g. "Object 3 — 12.4% of image") → **click** to select (multi-select supported) → selected instances show a solid semi-transparent fill; unselected instances are dimmed → click **Confirm & Send → Masked Edit** to combine the selected masks and switch to the Masked Edit tab for inpainting.

This tab is for **object selection via SAM**. It replaces the previous "Auto-Segment" button that was inside the Masked Edit tab, providing a richer interactive experience with per-instance selection, hover feedback, and multi-select before sending masks to inpainting.

| Control | Component | Range |
|---|---|---|
| Source photo | `gr.ImageEditor` (upload/clipboard) | — |
| Auto-Segment | Button (SAM) | — |
| Instance overlay | `gr.HTML` + iframe (HTML5 Canvas, vanilla JS) | — |
| Confirm & Send | Button | → switches to Masked Edit tab |

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

### Tabs

| Tab | Status | Workflow |
|---|---|---|
| **Restyle** | ✅ Implemented | Upload → pick style → generate (style change only) |
| **Auto-Segment** | ✅ Implemented | Upload → SAM detects instances → hover/click to select objects → confirm → send mask to Masked Edit |
| **Masked Edit** | ✅ Implemented | Draw mask (or receive from Auto-Segment) → inpaint masked area (editing only, no style change) |

### InstanceSelector — interactive instance canvas (v0.3)

The Auto-Segment tab uses an **interactive HTML5 Canvas** rendered via `gr.HTML()` wrapped in an `<iframe srcdoc>`. This workaround is needed because Gradio 4.19.2 strips `<script>` tags from `innerHTML` — the iframe preserves the full JS runtime.

**Component behavior:**

| State | Visual |
|---|---|
| Initial (after segmentation) | Room photo with thin colored panoptic outlines around each detected instance (~20% opacity) |
| Hover (unselected) | Hovered instance: outline thickens, fill opacity rises to ~40%, tooltip appears ("Object N — X% of image") |
| Hover (already selected) | Slight brightness increase on the selected instance |
| Selected | Solid semi-transparent fill (~60% opacity) in the instance's assigned color |
| Unselected (while others selected) | Dimmed — instance outline stays but image behind is slightly grayed |

**Data flow:**

1. Backend (`segment_room_detailed()`) returns per-instance payload: `{ image_url, instances: [{ id, mask_rle, bbox, area, score }] }`
2. Frontend decodes RLE masks for instant client-side hit testing (no round-trip on hover)
3. Edge-detection outline canvases are built from decoded masks for panoptic-style outlines
4. User toggles selection → `localStorage` updated with selected IDs (e.g. `[0, 3]`)
5. Confirm button JS reads `localStorage` → passes value to Python handler → combines masks → switches to Masked Edit tab


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
services:
  app:
    build: .
    container_name: local-interior-studio
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - DEVICE=auto
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
- [x] **v0.2 — Image editing (inpainting)** — SAM segmentation + user-drawn mask + Stable Diffusion Inpainting (Masked edit tab — pure editing, no style change)
- [x] **v0.3 — Interactive auto-segment** — Extract Auto-Segment into its own tab with interactive instance canvas (hover-to-highlight, click-to-select, multi-select, confirm & send to Masked Edit)


## License

MIT
