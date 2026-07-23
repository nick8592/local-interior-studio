"""Image processing helpers for Local Interior Studio."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def resize_for_model(
    image: Image.Image,
    max_size: int = 768,
) -> tuple[Image.Image, float]:
    """Resize image so its longest side does not exceed ``max_size``.

    Aspect ratio is preserved.  If the image is already within bounds it is
    returned unchanged.

    Args:
        image: Source PIL Image.
        max_size: Maximum dimension in pixels.

    Returns:
        Tuple of (resized_image, scale_factor).
    """
    w, h = image.size
    longest = max(w, h)

    if longest <= max_size:
        return image, 1.0

    scale = max_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    logger.debug("Resized %dx%d → %dx%d (scale %.3f)", w, h, new_w, new_h, scale)
    return resized, scale


def pad_to_square(
    image: Image.Image,
    fill_color: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    """Pad an image to a square canvas, preserving the original content.

    Args:
        image: Source PIL Image.
        fill_color: Background fill value (0–255).

    Returns:
        Tuple of (square_image, pad_info) where pad_info can be passed to
        ``unpad_image()`` to reverse the operation.
    """
    w, h = image.size
    max_dim = max(w, h)

    pad_info: dict[str, Any] = {
        "top": (max_dim - h) // 2,
        "left": (max_dim - w) // 2,
        "original_size": (w, h),
    }

    if w == h:
        return image, pad_info

    canvas = Image.new(image.mode, (max_dim, max_dim), fill_color)
    canvas.paste(image, (pad_info["left"], pad_info["top"]))

    logger.debug("Padded %dx%d → %dx%d (top=%d, left=%d)", w, h, max_dim, max_dim, pad_info["top"], pad_info["left"])
    return canvas, pad_info


def unpad_image(image: Image.Image, pad_info: dict[str, Any]) -> Image.Image:
    """Reverse ``pad_to_square()`` using previously stored pad_info.

    Args:
        image: Square-padded PIL Image.
        pad_info: Dict returned by ``pad_to_square()``.

    Returns:
        Unpadded image at its original dimensions.
    """
    top = pad_info["top"]
    left = pad_info["left"]
    orig_w, orig_h = pad_info["original_size"]

    cropped = image.crop((left, top, left + orig_w, top + orig_h))
    return cropped


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL Image mode to RGB.

    Handles RGBA (drops alpha), grayscale, palette, and other modes.

    Args:
        image: Source PIL Image in any mode.

    Returns:
        RGB PIL Image.
    """
    if image.mode == "RGB":
        return image

    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg

    return image.convert("RGB")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a binary mask on an image for visualization.

    Args:
        image: Source PIL Image (will be converted to RGB).
        mask: Binary mask (H, W) with values 0/1 or 0/255.
        color: RGB color tuple for the overlay.
        alpha: Overlay opacity (0.0–1.0).

    Returns:
        PIL Image with mask overlay blended in.
    """
    rgb = to_rgb(image)
    img_arr = np.array(rgb, dtype=np.float32)

    # Normalize mask to 0/1
    mask_bool = mask.astype(bool) if mask.max() <= 1 else mask > 127

    overlay = np.zeros_like(img_arr)
    overlay[mask_bool] = list(color)

    blended = img_arr * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def save_output(
    image: Image.Image,
    output_dir: str = "output",
    base_name: str = "restyle",
) -> Path:
    """Save an image to the output directory with a timestamp filename.

    Creates ``output_dir`` if it does not exist.

    Args:
        image: PIL Image to save.
        output_dir: Target directory path.
        base_name: Prefix for the filename.

    Returns:
        Path to the saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{ts}.png"
    dest = out_path / filename

    image.save(str(dest), format="PNG")
    logger.info("Saved output to %s", dest)
    return dest


def extract_mask_from_editor(
    editor_value: dict | tuple | None,
) -> Image.Image:
    """Extract a binary edit mask from a Gradio ``ImageEditor`` output.

    Gradio ``ImageEditor`` returns either a dict ``{"background", "layers",
    "composite"}`` (current API) or a tuple ``(background, mask)`` (legacy
    API).  User-drawn strokes appear as RGBA ``layers`` where non-transparent
    pixels mark the edit region.

    Args:
        editor_value: Output value from ``gr.ImageEditor`` — dict, tuple, or
            ``None`` if the user has not interacted with the editor.

    Returns:
        Grayscale PIL Image in mode ``"L"``.  White (255) marks the edit
        region, black (0) marks the keep region.  When no strokes are
        present the returned mask is fully white (entire image is editable).
    """
    if editor_value is None:
        return Image.new("L", (512, 512), 255)

    if isinstance(editor_value, tuple):
        if len(editor_value) >= 2 and editor_value[1] is not None:
            mask_arr = np.asarray(editor_value[1])
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            binary = (mask_arr > 0).astype(np.uint8) * 255
            return Image.fromarray(binary, mode="L")
        bg = editor_value[0] if len(editor_value) >= 1 else None
        bg_arr = np.asarray(bg) if bg is not None else None
        if bg_arr is not None:
            h, w = bg_arr.shape[:2]
            return Image.new("L", (w, h), 255)
        return Image.new("L", (512, 512), 255)

    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        layers = editor_value.get("layers") or []

        bg_arr = np.asarray(bg) if bg is not None else None
        layer_arrs = [np.asarray(layer) for layer in layers if layer is not None]

        if bg_arr is not None:
            h, w = bg_arr.shape[:2]
        elif layer_arrs:
            h, w = layer_arrs[0].shape[:2]
        else:
            return Image.new("L", (512, 512), 255)

        if not layer_arrs:
            return Image.new("L", (w, h), 255)

        # alpha > 0 → edit region (user stroke).
        composite = np.zeros((h, w), dtype=np.uint8)
        for layer_arr in layer_arrs:
            if layer_arr.ndim == 2:
                layer_mask = (layer_arr > 0).astype(np.uint8)
            elif layer_arr.shape[-1] >= 4:
                alpha = layer_arr[..., 3]
                layer_mask = (alpha > 0).astype(np.uint8)
            else:
                layer_mask = (layer_arr.max(axis=-1) > 0).astype(np.uint8)
            composite = np.maximum(composite, layer_mask * 255)

        return Image.fromarray(composite, mode="L")

    return Image.new("L", (512, 512), 255)


def dilate_mask(
    mask: Image.Image,
    kernel_size: int = 10,
) -> Image.Image:
    """Expand a binary mask by ``kernel_size`` pixels via morphological dilation.

    An elliptical structuring element is used so the resulting boundary is
    softer than a square kernel — this gives inpainting models enough
    surrounding context to avoid hard seams at the mask edge.

    Args:
        mask: Binary PIL Image.  Non-zero pixels are treated as the active
            region.
        kernel_size: Side length of the elliptical kernel in pixels.  The
            effective expansion is roughly half this value per side.  Values
            below 1 are clamped to 1.

    Returns:
        Dilated PIL Image in mode ``"L"`` with values 0 or 255.
    """
    safe_kernel = max(1, int(kernel_size))
    mask_arr = np.asarray(mask.convert("L"))
    binary = (mask_arr > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_kernel, safe_kernel))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    active_before = int((binary > 0).sum())
    active_after = int((dilated > 0).sum())
    logger.debug(
        "Dilated mask by %dpx (active px: %d → %d)",
        safe_kernel, active_before, active_after,
    )
    return Image.fromarray(dilated, mode="L")
