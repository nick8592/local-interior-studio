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
├── requirements.txt
├── app.py                  # Gradio / web UI entry point
├── pipeline/
│   ├── __init__.py
│   ├── edit.py             # InstructPix2Pix inference wrapper
│   ├── segment.py          # Room segmentation (SAM / OneFormer)
│   └── presets.py          # Style prompt templates
├── models/
│   └── README.md           # Download instructions (auto-download on first run)
├── utils/
│   ├── __init__.py
│   └── image.py            # Resize, pad, color-space helpers
└── tests/
    ├── test_edit.py
    └── test_segment.py
```

## Quick start (once implemented)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → Opens http://localhost:7860
```

## Roadmap

- [ ] **v0.1 — Proof of concept** — single-image restyle with InstructPix2Pix + Gradio UI
- [ ] **v0.2 — Masked editing** — SAM segmentation + user-drawn mask + inpainting
- [ ] **v0.3 — Style presets** — curated prompt library with preview thumbnails
- [ ] **v0.4 — Multi-room batch** — process a folder of room photos with one style
- [ ] **v0.5 — Upscale output** — Real-ESRGAN 4× upscaling for print-quality renders

## License

MIT
