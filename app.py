"""Local Interior Studio — Gradio web UI entry point.

Fully offline interior design tool that restyles room photos using
a locally-runnable InstructPix2Pix model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from pipeline.edit import edit_image, get_device, load_edit_model
from pipeline.presets import get_preset, get_preset_names
from utils.image import resize_for_model, save_output, to_rgb

load_dotenv()

logger = logging.getLogger(__name__)

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_pipeline = None


def _ensure_model():
    """Lazily load the edit model on first request."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading model (first run)…")
        _pipeline = load_edit_model()
    return _pipeline


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_preset_selected(preset_name: str) -> dict:
    """Populate prompt fields when a style preset is chosen."""
    try:
        preset = get_preset(preset_name)
    except KeyError:
        return gr.update(value="", placeholder="Type a custom instruction…")

    return {
        prompt_box: gr.update(value=preset["prompt"]),
        negative_box: gr.update(value=preset["negative_prompt"]),
    }


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
    source, _ = resize_for_model(source, max_size=768)

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

    # Save output
    saved_path = save_output(result, output_dir=OUTPUT_DIR)

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

        # --- Events ---
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
