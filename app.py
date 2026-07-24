"""Local Interior Studio — Gradio web UI entry point.

Fully offline interior design tool that restyles room photos using
a locally-runnable InstructPix2Pix model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from pipeline.edit import edit_image, get_device, load_edit_model
from pipeline.inpaint import inpaint_image, load_inpaint_model
from pipeline.presets import get_preset, get_preset_names
from pipeline.segment import generate_mask, load_segmentation_model
from utils.image import (
    dilate_mask,
    extract_mask_from_editor,
    resize_for_model,
    save_output,
    to_rgb,
)

load_dotenv()

logger = logging.getLogger(__name__)

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ---------------------------------------------------------------------------
# Example images
# ---------------------------------------------------------------------------
EXAMPLES_DIR = Path(__file__).parent / "examples"


def _list_example_images() -> list[str]:
    """Return sorted list of example image paths, if the directory exists."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        str(p) for p in EXAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_pipeline = None
_inpaint_pipeline = None
_segment_predictor = None


def _ensure_model():
    """Lazily load the edit model on first request."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading model (first run)…")
        _pipeline = load_edit_model()
    return _pipeline


def _ensure_inpaint_model():
    """Lazily load the inpaint model on first request."""
    global _inpaint_pipeline
    if _inpaint_pipeline is None:
        logger.info("Loading inpaint model (first run)…")
        _inpaint_pipeline = load_inpaint_model()
    return _inpaint_pipeline


def _ensure_segment_model():
    """Lazily load the SAM segmentation model on first request."""
    global _segment_predictor
    if _segment_predictor is None:
        logger.info("Loading segmentation model (first run)…")
        _segment_predictor = load_segmentation_model()
    return _segment_predictor


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_preset_selected(preset_name: str):
    try:
        preset = get_preset(preset_name)
    except KeyError:
        return "", ""

    return preset["prompt"], preset["negative_prompt"]


def generate(
    source_image: Optional[Image.Image],
    preset_name: str,
    custom_prompt: str,
    negative_prompt: str,
    image_guidance_scale: float,
    num_steps: int,
    seed: int,
) -> tuple[Optional[Image.Image], str]:
    """Run the restyle pipeline and return the output image."""
    if source_image is None:
        raise gr.Error("Please upload a room photo first.")

    # Determine prompt: custom overrides preset
    prompt = custom_prompt.strip()
    if not prompt:
        try:
            preset = get_preset(preset_name)
            prompt = preset["prompt"]
            if not negative_prompt.strip():
                negative_prompt = preset["negative_prompt"]
        except KeyError:
            raise gr.Error("Select a style preset or write a custom prompt.")

    pipe = _ensure_model()

    # Preprocess
    source = to_rgb(source_image)
    original_size = source.size
    source, scale = resize_for_model(source, max_size=768)

    logger.info("Generating: prompt='%s…', steps=%d, img_gs=%.2f, seed=%d", prompt[:60], num_steps, image_guidance_scale, seed)

    result = edit_image(
        pipeline=pipe,
        image=source,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        image_guidance_scale=image_guidance_scale,
        seed=seed,
    )

    if scale < 1.0 and result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        logger.info("Upscaled output back to original size %s", original_size)

    # Save output
    saved_path = save_output(result, output_dir=OUTPUT_DIR)

    return result, f"Done — saved to `{saved_path}`"


def on_auto_segment(editor_value: dict | tuple | None) -> Optional[dict]:
    """Run SAM auto-segmentation and return an ImageEditor-compatible dict with the mask."""
    if editor_value is None:
        raise gr.Error("Please upload a room photo first.")

    # Extract background image from ImageEditor dict/tuple
    bg_arr = None
    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        if bg is not None:
            bg_arr = np.asarray(bg)
    elif isinstance(editor_value, tuple) and len(editor_value) >= 1:
        bg_arr = np.asarray(editor_value[0]) if editor_value[0] is not None else None

    if bg_arr is None:
        raise gr.Error("Could not extract source image from editor.")

    source = Image.fromarray(bg_arr)
    if source.mode != "RGB":
        source = source.convert("RGB")

    predictor = _ensure_segment_model()
    original_size = source.size  # (w, h)
    source, _ = resize_for_model(source, max_size=768)

    mask_arr = generate_mask(source, predictor, max_masks=5)
    mask_img = dilate_mask(Image.fromarray((mask_arr.astype(np.uint8) * 255)), kernel_size=10)

    # Resize mask back to original image dimensions
    if mask_img.size != original_size:
        mask_img = mask_img.resize(original_size, Image.Resampling.LANCZOS)

    # Restore source to original dimensions so the editor keeps full resolution
    if source.size != original_size:
        source = source.resize(original_size, Image.Resampling.LANCZOS)

    w, h = original_size
    bg_arr = np.array(source)
    mask_np = np.array(mask_img)

    # Build RGBA mask layer: white pixels where mask is active, transparent elsewhere.
    # This makes the mask visible as a white overlay in the ImageEditor.
    mask_layer = np.zeros((h, w, 4), dtype=np.uint8)
    mask_layer[..., :3] = 255           # white RGB everywhere in mask region
    mask_layer[..., 3] = mask_np        # alpha = mask intensity (0 or 255)

    return {"background": bg_arr, "layers": [mask_layer], "composite": bg_arr}


def generate_masked(
    editor_value: dict | tuple | None,
    inpaint_prompt: str,
    negative_prompt: str,
    inpaint_strength: float,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    dilate_kernel: int,
) -> tuple[Optional[Image.Image], str]:
    """Pure inpainting: fill the masked region using SD Inpainting."""
    if editor_value is None:
        raise gr.Error("Please upload a room photo and draw a mask first.")

    mask_img = extract_mask_from_editor(editor_value)

    if dilate_kernel > 0:
        mask_img = dilate_mask(mask_img, kernel_size=dilate_kernel)

    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")

    bg_arr = None
    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        if bg is not None:
            bg_arr = np.asarray(bg)
    elif isinstance(editor_value, tuple) and len(editor_value) >= 1:
        bg_arr = np.asarray(editor_value[0]) if editor_value[0] is not None else None

    if bg_arr is not None:
        source = Image.fromarray(bg_arr)
        if source.mode != "RGB":
            source = source.convert("RGB")
    else:
        raise gr.Error("Could not extract source image from editor.")

    original_size = source.size
    source, scale = resize_for_model(source, max_size=768)

    if mask_img.size != source.size:
        mask_img = mask_img.resize(source.size, Image.Resampling.LANCZOS)

    inpaint_pipe = _ensure_inpaint_model()

    fill_prompt = inpaint_prompt.strip() or "clean empty room"

    logger.info(
        "Inpaint: prompt='%s…', strength=%.2f, guidance=%.2f, steps=%d, seed=%d",
        fill_prompt[:60], inpaint_strength, guidance_scale, num_steps, seed,
    )

    result = inpaint_image(
        pipeline=inpaint_pipe,
        image=source,
        mask=mask_img,
        prompt=fill_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=inpaint_strength,
        seed=seed,
    )

    if scale < 1.0 and result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        logger.info("Upscaled output back to original size %s", original_size)

    saved_path = save_output(result, output_dir=OUTPUT_DIR, base_name="masked_edit")
    return result, f"Done — saved to `{saved_path}`"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    preset_names = get_preset_names()

    with gr.Blocks(
        title="Local Interior Studio",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🏠 Local Interior Studio\n"
            "Fully offline interior design — restyle your room photos with AI. "
            "No cloud, no data leaving your machine."
        )

        with gr.Tab("Restyle"):
            with gr.Row():
                # --- Input column ---
                with gr.Column(scale=1):
                    source_img = gr.Image(
                        type="pil",
                        label="Room Photo",
                        sources=["upload", "clipboard"],
                    )

                    _restyle_examples = _list_example_images()
                    if _restyle_examples:
                        gr.Examples(
                            examples=[[p] for p in _restyle_examples],
                            inputs=[source_img],
                            label="Example Room Photos",
                        )

                    preset_dd = gr.Dropdown(
                        choices=preset_names,
                        value=preset_names[0],
                        label="Style Preset",
                        filterable=True,
                    )

                    prompt_box = gr.Textbox(
                        label="Prompt",
                        placeholder="Select a preset or type a custom instruction…",
                        lines=3,
                    )

                    negative_box = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid (optional)…",
                        lines=2,
                    )

                    img_gs_slider = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="Edit Strength (image guidance scale)",
                        info="Lower = more faithful to original. Higher = more creative.",
                    )

                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps",
                        info="More steps = higher quality, slower.",
                    )

                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (−1 = random)",
                        precision=0,
                    )

                    generate_btn = gr.Button("✨ Generate", variant="primary")

                # --- Output column ---
                with gr.Column(scale=1):
                    output_img = gr.Image(type="pil", label="Result")
                    status_md = gr.Markdown("Ready")

        with gr.Tab("Masked Edit"):
            with gr.Row():
                # --- Input column ---
                with gr.Column(scale=1):
                    mask_editor = gr.ImageEditor(
                        type="numpy",
                        image_mode="RGBA",
                        brush=gr.Brush(default_size=25, colors=["#ffffff"]),
                        label="Room Photo + Mask",
                        sources=["upload", "clipboard"],
                    )

                    _masked_examples = _list_example_images()
                    if _masked_examples:
                        gr.Examples(
                            examples=[[p] for p in _masked_examples],
                            inputs=[mask_editor],
                            label="Example Room Photos",
                        )

                    auto_seg_btn = gr.Button("🔍 Auto-Segment (SAM)", variant="secondary")

                    gr.Markdown("### Inpaint settings")
                    inpaint_prompt_box = gr.Textbox(
                        label="Inpaint Prompt",
                        placeholder="What to fill the masked area with (e.g. 'clean empty room')…",
                        lines=2,
                    )

                    inpaint_negative_box = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="What to avoid in the inpainted area (optional)…",
                        lines=2,
                    )

                    inpaint_strength_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Inpaint Strength",
                        info="How much to change the masked area. 1.0 = full replacement.",
                    )

                    inpaint_guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely to follow the prompt. Higher = more faithful.",
                    )

                    mask_dilate_slider = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=10,
                        step=1,
                        label="Mask Dilation (px)",
                        info="Expand mask edges to avoid hard seams. 0 = no dilation.",
                    )

                    mask_steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Inference Steps",
                        info="More steps = higher quality, slower.",
                    )

                    mask_seed_input = gr.Number(
                        value=-1,
                        label="Seed (−1 = random)",
                        precision=0,
                    )

                    mask_generate_btn = gr.Button("✨ Generate (Inpaint)", variant="primary")

                # --- Output column ---
                with gr.Column(scale=1):
                    mask_output_img = gr.Image(type="pil", label="Result")
                    mask_status_md = gr.Markdown("Draw a mask → set inpaint prompt → generate.")

        # --- Events: Restyle tab ---
        preset_dd.change(
            fn=on_preset_selected,
            inputs=[preset_dd],
            outputs=[prompt_box, negative_box],
        )

        generate_btn.click(
            fn=generate,
            inputs=[
                source_img,
                preset_dd,
                prompt_box,
                negative_box,
                img_gs_slider,
                steps_slider,
                seed_input,
            ],
            outputs=[output_img, status_md],
        )

        # --- Events: Masked Edit tab ---
        auto_seg_btn.click(
            fn=on_auto_segment,
            inputs=[mask_editor],
            outputs=[mask_editor],
        )

        mask_generate_btn.click(
            fn=generate_masked,
            inputs=[
                mask_editor,
                inpaint_prompt_box,
                inpaint_negative_box,
                inpaint_strength_slider,
                inpaint_guidance_slider,
                mask_steps_slider,
                mask_seed_input,
                mask_dilate_slider,
            ],
            outputs=[mask_output_img, mask_status_md],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    demo = build_ui()
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
    )
