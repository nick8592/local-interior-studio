"""Local Interior Studio — Gradio web UI entry point.

Fully offline interior design tool that restyles room photos using
a locally-runnable InstructPix2Pix model.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from components.instance_selector import (
    encode_mask_rle,
    instance_color_hex,
    render_instance_selector_html,
    render_selected_objects_html,
)
from pipeline.edit import edit_image, get_device, load_edit_model
from pipeline.inpaint import inpaint_image, load_inpaint_model
from pipeline.presets import get_preset, get_preset_names
from pipeline.segment import load_segmentation_model, segment_room_detailed
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
# Model cache
# ---------------------------------------------------------------------------
_pipeline = None
_inpaint_pipeline = None
_segment_predictor = None
_active_model: str | None = None


def _reclaim_memory():
    """Force-release GPU and system memory after deleting a model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("Memory reclaimed (gc + CUDA cache cleared)")


def _swap_to_model(target: str):
    """Ensure only *target* model is loaded; delete all others.

    Valid targets: "edit", "inpaint", "segment".
    """
    global _pipeline, _inpaint_pipeline, _segment_predictor, _active_model

    if _active_model == target:
        return

    if target != "edit" and _pipeline is not None:
        logger.info("Deleting edit pipeline to free memory")
        del _pipeline
        _pipeline = None

    if target != "inpaint" and _inpaint_pipeline is not None:
        logger.info("Deleting inpaint pipeline to free memory")
        del _inpaint_pipeline
        _inpaint_pipeline = None

    if target != "segment" and _segment_predictor is not None:
        logger.info("Deleting SAM predictor to free memory")
        del _segment_predictor
        _segment_predictor = None

    _reclaim_memory()


def _ensure_model():
    """Load the edit model, deleting others if needed."""
    global _pipeline, _active_model
    _swap_to_model("edit")
    if _pipeline is None:
        logger.info("Loading edit model…")
        _pipeline = load_edit_model()
    _active_model = "edit"
    return _pipeline


def _ensure_inpaint_model():
    """Load the inpaint model, deleting others if needed."""
    global _inpaint_pipeline, _active_model
    _swap_to_model("inpaint")
    if _inpaint_pipeline is None:
        logger.info("Loading inpaint model…")
        _inpaint_pipeline = load_inpaint_model()
    _active_model = "inpaint"
    return _inpaint_pipeline


def _ensure_segment_model():
    """Load the SAM model, deleting others if needed."""
    global _segment_predictor, _active_model
    _swap_to_model("segment")
    if _segment_predictor is None:
        logger.info("Loading SAM model…")
        _segment_predictor = load_segmentation_model()
    _active_model = "segment"
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


def on_auto_segment_interactive(
    editor_value: dict | tuple | None,
) -> tuple[str, list[dict], str, str]:
    """Run SAM segmentation and return HTML canvas + instance state + empty selection list + reset selection JSON."""
    if editor_value is None:
        raise gr.Error("Please upload a room photo first.")

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
    source_resized, _ = resize_for_model(source, max_size=768)

    instances = segment_room_detailed(source_resized, predictor)

    if not instances:
        raise gr.Error("No objects detected in this image. Try a different photo.")

    canvas_source = source_resized
    if canvas_source.mode != "RGB":
        canvas_source = canvas_source.convert("RGB")
    buf = io.BytesIO()
    canvas_source.save(buf, format="JPEG", quality=85)
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    canvas_w, canvas_h = canvas_source.size
    original_w, original_h = original_size

    frontend_instances = []
    for inst in instances:
        frontend_instances.append({
            "id": inst["id"],
            "mask_rle": encode_mask_rle(inst["mask"]),
            "bbox": inst["bbox"],
            "area": int(inst["area"]),
            "score": inst["score"],
            "height": canvas_h,
            "width": canvas_w,
        })

    def _to_original_res(mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] != (original_h, original_w):
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            mask_img = mask_img.resize((original_w, original_h), Image.Resampling.LANCZOS)
            return np.array(mask_img) > 127
        return mask

    canvas_total_pixels = max(canvas_w * canvas_h, 1)
    total = len(instances)
    state_instances: list[dict] = []
    for idx, inst in enumerate(instances):
        area_canvas = int(inst["area"])
        area_pct = area_canvas / canvas_total_pixels * 100.0
        state_instances.append({
            "id": int(inst["id"]),
            "mask": _to_original_res(inst["mask"]),
            "colorHex": instance_color_hex(idx, total),
            "areaPct": area_pct,
        })

    html_str = render_instance_selector_html(image_data_url, frontend_instances)
    selected_html = render_selected_objects_html(state_instances, [])

    return html_str, state_instances, selected_html, "[]"


def on_refresh_selection(
    selection_json: str,
    instance_state: list[dict] | None,
) -> str:
    """Re-render the selected-objects HTML from the canvas's localStorage state."""
    if not instance_state:
        return render_selected_objects_html([], [])

    try:
        selected_ids = json.loads(selection_json)
    except (json.JSONDecodeError, TypeError):
        selected_ids = []

    return render_selected_objects_html(instance_state, selected_ids)


def on_auto_segment_inpaint(
    selection_json: str,
    instance_state: list[dict] | None,
    editor_value: dict | tuple | None,
    inpaint_prompt: str,
    negative_prompt: str,
    inpaint_strength: float,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    dilate_kernel: int,
) -> tuple[Optional[Image.Image], str, str]:
    """Combine selected instance masks, dilate, and run inpainting on the result."""
    if instance_state is None or not instance_state:
        raise gr.Error("Run Auto-Segment first to detect objects.")

    try:
        selected_ids = json.loads(selection_json)
    except (json.JSONDecodeError, TypeError):
        selected_ids = []

    if not selected_ids:
        raise gr.Error("Select at least one object on the canvas before inpainting.")

    if not inpaint_prompt.strip():
        raise gr.Error("Enter an inpaint prompt describing what to fill the region with.")

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

    original_size = source.size
    h, w = original_size[::-1]

    combined = np.zeros((h, w), dtype=bool)
    matched = 0
    for inst in instance_state:
        if inst["id"] in selected_ids:
            mask = inst["mask"]
            if mask.shape[:2] != (h, w):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
                mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
                mask = np.array(mask_img) > 127
            combined = np.logical_or(combined, mask)
            matched += 1

    if matched == 0:
        raise gr.Error("Selected ids did not match any segmented objects.")

    mask_for_dilation = Image.fromarray((combined.astype(np.uint8) * 255))
    if dilate_kernel > 0:
        mask_for_dilation = dilate_mask(mask_for_dilation, kernel_size=dilate_kernel)
    if mask_for_dilation.mode != "L":
        mask_for_dilation = mask_for_dilation.convert("L")

    source_resized, scale = resize_for_model(source, max_size=768)
    if mask_for_dilation.size != source_resized.size:
        mask_for_dilation = mask_for_dilation.resize(
            source_resized.size, Image.Resampling.LANCZOS
        )

    inpaint_pipe = _ensure_inpaint_model()

    fill_prompt = inpaint_prompt.strip()

    logger.info(
        "Auto-seg inpaint: prompt='%s…', strength=%.2f, guidance=%.2f, steps=%d, "
        "seed=%d, dilate=%d, objects=%d",
        fill_prompt[:60], inpaint_strength, guidance_scale, num_steps,
        seed, dilate_kernel, matched,
    )

    result = inpaint_image(
        pipeline=inpaint_pipe,
        image=source_resized,
        mask=mask_for_dilation,
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

    saved_path = save_output(
        result, output_dir=OUTPUT_DIR, base_name="auto_seg_inpaint"
    )

    selected_html = render_selected_objects_html(instance_state, selected_ids)
    status = (
        f"Done — saved to `{saved_path}` "
        f"(inpainted {matched} object{'s' if matched != 1 else ''})"
    )

    return result, status, selected_html


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

    _bridge_js = (
        "<script>\n"
        "window.addEventListener('message', function(e) {\n"
        "  if (e.data && e.data.source === 'instance-canvas' && "
        "typeof e.data.selection === 'string') {\n"
        "    var box = document.querySelector('#seg_selection_json textarea');\n"
        "    if (!box) box = document.querySelector("
        "'[data-testid*=\"seg_selection_json\"] textarea');\n"
        "    if (!box) box = document.querySelector("
        "'input[name=\"seg_selection_json\"]');\n"
        "    if (box) {\n"
        "      var nativeSetter = Object.getOwnPropertyDescriptor("
        "window.HTMLTextAreaElement.prototype, 'value').set;\n"
        "      nativeSetter.call(box, e.data.selection);\n"
        "      box.dispatchEvent(new Event('input', {bubbles:true}));\n"
        "    }\n"
        "  }\n"
        "  if (e.data && e.data.type === 'resize-iframe' && e.data.height) {\n"
        "    var iframes = document.querySelectorAll('iframe[id$=\"-iframe\"]');\n"
        "    iframes.forEach(function(f) {\n"
        "      try { f.style.height = e.data.height + 'px'; } catch(ex) {}\n"
        "    });\n"
        "  }\n"
        "});\n"
        "</script>"
    )

    with gr.Blocks(
        title="Local Interior Studio",
        theme=gr.themes.Soft(),
        head=_bridge_js,
    ) as demo:
        gr.Markdown(
            "# 🏠 Local Interior Studio\n"
            "Fully offline interior design — restyle your room photos with AI. "
            "No cloud, no data leaving your machine."
        )

        with gr.Tabs(selected="restyle") as tabs:
            with gr.Tab("Restyle", id="restyle"):
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

            with gr.Tab("Auto-Segment", id="auto-seg"):
                seg_state = gr.State(None)

                with gr.Row():
                    # --- Left column: input + selection + inpaint controls ---
                    with gr.Column(scale=1):
                        seg_editor = gr.ImageEditor(
                            type="numpy",
                            image_mode="RGBA",
                            brush=gr.Brush(default_size=25, colors=["#ffffff"]),
                            label="Room Photo",
                            sources=["upload", "clipboard"],
                        )

                        _seg_examples = _list_example_images()
                        if _seg_examples:
                            gr.Examples(
                                examples=[[p] for p in _seg_examples],
                                inputs=[seg_editor],
                                label="Example Room Photos",
                            )

                        seg_btn = gr.Button("🔍 Auto-Segment (SAM)", variant="secondary")

                        instance_html = gr.HTML(
                            value=(
                                "<div style=\"padding:14px;color:#888;font-style:italic;"
                                "text-align:center\">Upload a photo, then click Auto-Segment.</div>"
                            ),
                            label="Select Objects",
                        )

                        selected_objects_html = gr.HTML(
                            value=(
                                "<div style='padding:10px 12px;color:#888;font-style:italic'>"
                                "Click objects on the canvas above to select them.</div>"
                            ),
                            label="Selected Objects",
                        )

                        seg_selection_json = gr.Textbox(value="[]", visible=False, elem_id="seg_selection_json")

                        gr.Markdown("### Inpaint settings")
                        seg_inpaint_prompt_box = gr.Textbox(
                            label="Inpaint Prompt",
                            placeholder=(
                                "What to fill the selected region with "
                                "(e.g. 'clean empty room')…"
                            ),
                            lines=2,
                        )

                        seg_inpaint_negative_box = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the inpainted area (optional)…",
                            lines=2,
                        )

                        seg_inpaint_strength_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="Inpaint Strength",
                            info="How much to change the masked area. 1.0 = full replacement.",
                        )

                        seg_inpaint_guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                            info="How closely to follow the prompt. Higher = more faithful.",
                        )

                        seg_mask_dilate_slider = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Mask Dilation (px)",
                            info="Expand mask edges to avoid hard seams. 0 = no dilation.",
                        )

                        seg_mask_steps_slider = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=25,
                            step=1,
                            label="Inference Steps",
                            info="More steps = higher quality, slower.",
                        )

                        seg_inpaint_seed_input = gr.Number(
                            value=-1,
                            label="Seed (−1 = random)",
                            precision=0,
                        )

                        inpaint_btn = gr.Button(
                            "✨ Inpaint Selected", variant="primary"
                        )

                    # --- Right column: inpaint result ---
                    with gr.Column(scale=1):
                        inpaint_result_img = gr.Image(type="pil", label="Inpaint Result")
                        inpaint_status_md = gr.Markdown(
                            "Ready — segment, select objects, then click Inpaint Selected."
                        )

            with gr.Tab("Masked Edit", id="masked"):
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
                        mask_status_md = gr.Markdown(
                            "Draw a mask on the image, set inpaint prompt, then generate."
                        )

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

        seg_btn.click(
            fn=on_auto_segment_interactive,
            inputs=[seg_editor],
            outputs=[instance_html, seg_state, selected_objects_html, seg_selection_json],
        )

        seg_selection_json.change(
            fn=on_refresh_selection,
            inputs=[seg_selection_json, seg_state],
            outputs=[selected_objects_html],
        )

        inpaint_btn.click(
            fn=on_auto_segment_inpaint,
            inputs=[
                seg_selection_json,
                seg_state,
                seg_editor,
                seg_inpaint_prompt_box,
                seg_inpaint_negative_box,
                seg_inpaint_strength_slider,
                seg_inpaint_guidance_slider,
                seg_mask_steps_slider,
                seg_inpaint_seed_input,
                seg_mask_dilate_slider,
            ],
            outputs=[inpaint_result_img, inpaint_status_md, selected_objects_html],
        )

        # --- Events: Masked Edit tab ---
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
