"""SAM-based room segmentation for mask generation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

_SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
_MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

SAM_MODEL_TYPES: dict[str, str] = {
    "sam_vit_h_4b8939.pth": "vit_h",
    "sam_vit_l_0b3195.pth": "vit_l",
    "sam_vit_b_01ec64.pth": "vit_b",
}


def _get_device() -> str:
    """Return best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_model_type(checkpoint_name: str) -> str:
    """Map checkpoint filename to SAM registry type."""
    for pattern, model_type in SAM_MODEL_TYPES.items():
        if pattern in checkpoint_name:
            return model_type
    logger.warning("Unknown checkpoint '%s' — defaulting to vit_h", checkpoint_name)
    return "vit_h"


def download_sam_checkpoint(output_dir: str | None = None) -> Path:
    """Download the SAM ViT-H checkpoint if not already present.

    Args:
        output_dir: Directory to store the checkpoint.  Defaults to ``models/``.

    Returns:
        Path to the checkpoint file.
    """
    from segment_anything import sam_model_registry  # noqa: F401 — verify import

    dest_dir = Path(output_dir or _MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _SAM_CHECKPOINT

    if dest_path.exists():
        logger.info("SAM checkpoint already exists at %s", dest_path)
        return dest_path

    url = f"https://dl.fbaipublicfiles.com/segment_anything/{_SAM_CHECKPOINT}"
    logger.info("Downloading SAM checkpoint from %s → %s", url, dest_path)

    try:
        import urllib.request

        urllib.request.urlretrieve(url, str(dest_path))
        logger.info("Download complete (%.1f GB)", dest_path.stat().st_size / 1e9)
    except Exception:
        logger.exception("Failed to download SAM checkpoint")
        if dest_path.exists():
            dest_path.unlink()
        raise

    return dest_path


def load_segmentation_model(
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> "sam_model_registry":
    """Load SAM model and return a ``SamPredictor`` instance.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint.  ``None`` → auto-download.
        device: Target device.  ``None`` → auto-detect.

    Returns:
        A ``segment_anything.SamPredictor`` ready for inference.
    """
    from segment_anything import SamPredictor, sam_model_registry

    target = device or _get_device()

    if checkpoint_path is None:
        ckpt = Path(_MODELS_DIR) / _SAM_CHECKPOINT
        if not ckpt.exists():
            logger.info("Checkpoint not found — downloading")
            ckpt = download_sam_checkpoint()
        checkpoint_path = str(ckpt)

    model_type = _infer_model_type(Path(checkpoint_path).name)

    logger.info("Loading SAM '%s' from %s on %s", model_type, checkpoint_path, target)

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=target)

    predictor = SamPredictor(sam)
    logger.info("SAM predictor ready")
    return predictor


def segment_room(
    image: Image.Image,
    predictor: "SamPredictor",
    point_coords: list[tuple[int, int]] | None = None,
    point_labels: list[int] | None = None,
) -> list[np.ndarray]:
    """Segment a room image, returning binary masks for detected regions.

    Args:
        image: Room photo (PIL Image, any mode — converted to RGB internally).
        predictor: Pre-loaded ``SamPredictor``.
        point_coords: Optional click points ``[(x, y), ...]`` for prompt-based seg.
        point_labels: ``1`` = foreground, ``0`` = background for each point.

    Returns:
        List of binary masks (H, W) as ``np.ndarray`` (bool dtype).
    """
    from segment_anything import SamPredictor as _SP  # noqa: F811 — type hint

    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    predictor.set_image(img_np)

    masks: list[np.ndarray] = []

    if point_coords is not None:
        coords_np = np.array(point_coords, dtype=np.float32)
        labels_np = np.array(point_labels or [1] * len(point_coords), dtype=np.int32)
        result_masks, scores, _ = predictor.predict(
            point_coords=coords_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        # Pick highest-score mask
        best_idx = int(np.argmax(scores))
        masks.append(result_masks[best_idx])
    else:
        # Automatic mask generation
        from segment_anything import SamAutomaticMaskGenerator

        sam_model = predictor.model
        amg = SamAutomaticMaskGenerator(sam_model)
        amg_results = amg.generate(img_np)

        for entry in sorted(amg_results, key=lambda e: e["area"], reverse=True):
            masks.append(entry["segmentation"])

    logger.info("Segmentation produced %d mask(s)", len(masks))
    return masks


def segment_room_detailed(
    image: Image.Image,
    predictor: "SamPredictor",
    point_coords: list[tuple[int, int]] | None = None,
    point_labels: list[int] | None = None,
) -> list[dict]:
    """Segment a room image, returning per-instance metadata.

    Unlike :func:`segment_room`, this function preserves SAM's full metadata
    (bounding box, area, predicted IoU) for each detected instance, making it
    suitable for interactive instance-selection UIs.

    Args:
        image: Room photo (PIL Image, any mode — converted to RGB internally).
        predictor: Pre-loaded ``SamPredictor``.
        point_coords: Optional click points ``[(x, y), ...]`` for prompt-based seg.
        point_labels: ``1`` = foreground, ``0`` = background for each point.

    Returns:
        List of dicts, each containing:
        - ``id`` (int): Zero-based index sorted by area (largest first).
        - ``mask`` (np.ndarray): Binary mask (H, W), bool dtype.
        - ``bbox`` (list[float]): ``[x, y, w, h]`` bounding box.
        - ``area`` (int): Number of pixels in the mask.
        - ``score`` (float): SAM predicted IoU.
    """
    from segment_anything import SamPredictor as _SP  # noqa: F811 — type hint

    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    predictor.set_image(img_np)

    instances: list[dict] = []

    if point_coords is not None:
        coords_np = np.array(point_coords, dtype=np.float32)
        labels_np = np.array(point_labels or [1] * len(point_coords), dtype=np.int32)
        result_masks, scores, _ = predictor.predict(
            point_coords=coords_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        best_mask = result_masks[best_idx]
        instances.append({
            "id": 0,
            "mask": best_mask,
            "bbox": [0, 0, best_mask.shape[1], best_mask.shape[0]],
            "area": int(best_mask.sum()),
            "score": float(scores[best_idx]),
        })
    else:
        from segment_anything import SamAutomaticMaskGenerator

        sam_model = predictor.model
        amg = SamAutomaticMaskGenerator(sam_model)
        amg_results = amg.generate(img_np)

        for idx, entry in enumerate(
            sorted(amg_results, key=lambda e: e["area"], reverse=True)
        ):
            mask = entry["segmentation"]
            # SAM returns bbox as [x, y, w, h]
            bbox = entry.get("bbox", [0, 0, mask.shape[1], mask.shape[0]])
            instances.append({
                "id": idx,
                "mask": mask,
                "bbox": list(bbox),
                "area": int(entry["area"]),
                "score": float(entry.get("predicted_iou", 0.0)),
            })

    logger.info("Detailed segmentation produced %d instance(s)", len(instances))
    return instances


def generate_mask(
    image: Image.Image,
    predictor: "SamPredictor",
    max_masks: int = 5,
) -> np.ndarray:
    """Convenience function: segment a room and return a single combined mask.

    Args:
        image: Room photo.
        predictor: Pre-loaded SAM predictor.
        max_masks: Combine up to this many largest masks.

    Returns:
        Binary mask (H, W) as ``np.ndarray`` (bool dtype).
    """
    masks = segment_room(image, predictor)

    if not masks:
        h, w = image.size[::-1]
        logger.warning("No masks produced — returning empty mask")
        return np.zeros((h, w), dtype=bool)

    # Resize masks to match in case of size mismatch
    ref_h, ref_w = masks[0].shape[:2]
    combined = np.zeros((ref_h, ref_w), dtype=bool)

    for m in masks[:max_masks]:
        if m.shape[:2] != (ref_h, ref_w):
            m_img = Image.fromarray(m.astype(np.uint8) * 255).resize((ref_w, ref_h))
            m = np.array(m_img) > 127
        combined = np.logical_or(combined, m)

    logger.info("Combined %d mask(s) — %d px covered", min(len(masks), max_masks), combined.sum())
    return combined
