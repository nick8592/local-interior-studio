"""Stable Diffusion Inpainting inference wrapper for masked room edits."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusionInpaintPipeline
from dotenv import load_dotenv
from PIL import Image

from pipeline.edit import _get_dtype, get_device

load_dotenv()

logger = logging.getLogger(__name__)

_INPAINT_MODEL_ID = os.getenv("INPAINT_MODEL_ID", "runwayml/stable-diffusion-inpainting")
_INPAINT_STEPS = int(os.getenv("INPAINT_STEPS", "25"))
_INPAINT_GUIDANCE_SCALE = float(os.getenv("INPAINT_GUIDANCE_SCALE", "7.5"))

_pipeline: Optional[StableDiffusionInpaintPipeline] = None


def load_inpaint_model(
    model_id: str = _INPAINT_MODEL_ID,
    device: str | None = None,
) -> StableDiffusionInpaintPipeline:
    """Load the Stable Diffusion Inpainting pipeline and cache it globally.

    Subsequent calls return the cached pipeline without re-downloading.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device string.  ``None`` → auto-detect.

    Returns:
        A ready-to-use ``StableDiffusionInpaintPipeline``.
    """
    global _pipeline

    if _pipeline is not None:
        logger.info("Returning cached Inpaint pipeline")
        return _pipeline

    target = device or get_device()
    dtype = _get_dtype(target)

    logger.info("Loading Inpaint model '%s' on %s (%s)", model_id, target, dtype)

    try:
        _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        _pipeline = _pipeline.to(target)
        _pipeline.set_progress_bar_config(disable=False)
    except MemoryError:
        logger.error("Out of GPU memory while loading %s", model_id)
        raise
    except Exception:
        logger.exception("Failed to load model '%s'", model_id)
        raise

    logger.info("Model loaded successfully")
    return _pipeline


def inpaint_image(
    pipeline: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = _INPAINT_STEPS,
    guidance_scale: float = _INPAINT_GUIDANCE_SCALE,
    seed: int = -1,
    strength: float = 1.0,
) -> Image.Image:
    """Run Stable Diffusion Inpainting on a masked region of a room photo.

    Args:
        pipeline: Pre-loaded Inpaint pipeline.
        image: Source room photo (RGB PIL Image).
        mask: Mask image (white = region to edit, black = region to keep).
        prompt: Text instruction describing the desired fill.
        negative_prompt: What to avoid in generation.
        num_inference_steps: Diffusion steps (more = higher quality, slower).
        guidance_scale: Text-prompt adherence strength.
        seed: Reproducibility seed.  ``-1`` = random each run.
        strength: How much to deviate from the masked source (0.0–1.0).

    Returns:
        Inpainted room photo as a PIL Image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    device = "cpu"
    if hasattr(pipeline, "device"):
        device = str(pipeline.device)

    generator: Optional[torch.Generator] = None
    if seed >= 0:
        device_type = "cpu"
        if "cuda" in device:
            device_type = "cuda"
        generator = torch.Generator(device=device_type).manual_seed(seed)

    logger.info(
        "Inpainting image: prompt='%s', steps=%d, guidance=%.2f, strength=%.2f, seed=%s",
        prompt,
        num_inference_steps,
        guidance_scale,
        strength,
        seed if seed >= 0 else "random",
    )

    w, h = image.size
    h = h // 8 * 8
    w = w // 8 * 8

    result = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        height=h,
        width=w,
        generator=generator,
    )

    output = result.images[0]
    logger.info("Inpaint complete — output size %s", output.size)
    return output
