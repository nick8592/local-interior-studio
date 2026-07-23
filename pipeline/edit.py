"""InstructPix2Pix inference wrapper for room photo restyling."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

_EDIT_MODEL_ID = os.getenv("EDIT_MODEL_ID", "timbrooks/instruct-pix2pix")
_DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
_DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5"))
_DEFAULT_IMAGE_GUIDANCE_SCALE = float(os.getenv("DEFAULT_IMAGE_GUIDANCE_SCALE", "1.5"))

_pipeline: Optional[StableDiffusionInstructPix2PixPipeline] = None


def get_device() -> str:
    """Return the best available compute device: 'cuda', 'mps', or 'cpu'.

    Respects the DEVICE environment variable when set to a concrete value.
    When DEVICE='auto' (default), probes hardware in order: CUDA > MPS > CPU.
    """
    env_device = os.getenv("DEVICE", "auto").lower()
    if env_device != "auto":
        return env_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str) -> torch.dtype:
    """Select float16 for CUDA/MPS, float32 for CPU."""
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def load_edit_model(
    model_id: str = _EDIT_MODEL_ID,
    device: str | None = None,
) -> StableDiffusionInstructPix2PixPipeline:
    """Load the InstructPix2Pix pipeline and cache it globally.

    Subsequent calls return the cached pipeline without re-downloading.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device string.  ``None`` → auto-detect.

    Returns:
        A ready-to-use ``StableDiffusionInstructPix2PixPipeline``.
    """
    global _pipeline

    if _pipeline is not None:
        logger.info("Returning cached InstructPix2Pix pipeline")
        return _pipeline

    target = device or get_device()
    dtype = _get_dtype(target)

    logger.info("Loading InstructPix2Pix model '%s' on %s (%s)", model_id, target, dtype)

    try:
        _pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
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


def edit_image(
    pipeline: StableDiffusionInstructPix2PixPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = _DEFAULT_STEPS,
    image_guidance_scale: float = _DEFAULT_IMAGE_GUIDANCE_SCALE,
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE,
    seed: int = -1,
) -> Image.Image:
    """Run InstructPix2Pix inference on a single room photo.

    Args:
        pipeline: Pre-loaded InstructPix2Pix pipeline.
        image: Source room photo (RGB PIL Image).
        prompt: Text instruction for restyling.
        negative_prompt: What to avoid in generation.
        num_inference_steps: Diffusion steps (more = higher quality, slower).
        image_guidance_scale: How much to preserve the original image (1.0–3.0).
        guidance_scale: Text-prompt adherence strength.
        seed: Reproducibility seed.  ``-1`` = random each run.

    Returns:
        Restyled room photo as a PIL Image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

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
        "Editing image: prompt='%s', steps=%d, img_guidance=%.2f, guidance=%.2f, seed=%s",
        prompt,
        num_inference_steps,
        image_guidance_scale,
        guidance_scale,
        seed if seed >= 0 else "random",
    )

    result = pipeline(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    output = result.images[0]
    logger.info("Edit complete — output size %s", output.size)
    return output
