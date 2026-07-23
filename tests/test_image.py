"""Unit tests for utils.image module — mask utilities."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_background() -> np.ndarray:
    """64×64×3 RGB background array (Gradio ImageEditor background layer)."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:] = (128, 128, 128)
    return arr


@pytest.fixture
def sample_mask_image() -> Image.Image:
    """64×64 grayscale PIL Image with a 3×3 white dot in the center (rest black)."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[31:34, 31:34] = 255
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def white_mask() -> Image.Image:
    """64×64 fully-white PIL Image (mode ``"L"``)."""
    return Image.new("L", (64, 64), 255)


# ---------------------------------------------------------------------------
# extract_mask_from_editor tests
# ---------------------------------------------------------------------------

class TestExtractMaskFromEditor:
    def test_none_returns_white_mask(self) -> None:
        from utils.image import extract_mask_from_editor

        result = extract_mask_from_editor(None)

        assert result.mode == "L"
        arr = np.asarray(result)
        assert arr.min() == 255
        assert arr.max() == 255
        assert (arr == 255).all()

    def test_dict_with_layers(
        self, sample_background: np.ndarray
    ) -> None:
        from utils.image import extract_mask_from_editor

        # RGBA layer with a small square drawn in the top-left.
        # Drawn region: rows 10..20, cols 10..20 → alpha=255 (edit).
        # Rest of canvas: alpha=0 (keep).
        layer = np.zeros((64, 64, 4), dtype=np.uint8)
        layer[..., 3] = 0  # fully transparent by default
        layer[10:20, 10:20, 3] = 255  # draw a 10×10 square
        layer[10:20, 10:20, :3] = (255, 0, 0)  # arbitrary RGB fill

        editor_value = {
            "background": sample_background,
            "layers": [layer],
            "composite": None,
        }

        result = extract_mask_from_editor(editor_value)

        assert result.mode == "L"
        arr = np.asarray(result)

        # Drawn region (alpha > 0) → 255.
        assert (arr[10:20, 10:20] == 255).all()
        # Undrawn region (alpha == 0) → 0.
        assert (arr[:10, :] == 0).all()
        assert (arr[20:, :] == 0).all()
        assert (arr[:, :10] == 0).all()
        assert (arr[:, 20:] == 0).all()

    def test_empty_layers_returns_white(
        self, sample_background: np.ndarray
    ) -> None:
        from utils.image import extract_mask_from_editor

        editor_value = {
            "background": sample_background,
            "layers": [],
            "composite": None,
        }

        result = extract_mask_from_editor(editor_value)

        assert result.mode == "L"
        assert result.size == (64, 64)
        arr = np.asarray(result)
        assert (arr == 255).all()

    def test_tuple_format(self, sample_background: np.ndarray) -> None:
        from utils.image import extract_mask_from_editor

        # Legacy Gradio ImageEditor tuple: (background, mask).
        # The mask element is a 2-D array of 0/255 values.
        legacy_mask = np.zeros((64, 64), dtype=np.uint8)
        legacy_mask[5:15, 5:15] = 255  # drawn region

        result = extract_mask_from_editor((sample_background, legacy_mask))

        assert result.mode == "L"
        arr = np.asarray(result)
        assert (arr[5:15, 5:15] == 255).all()
        assert (arr[:5, :] == 0).all()
        assert (arr[15:, :] == 0).all()

    def test_output_is_mode_L(self, sample_background: np.ndarray) -> None:
        from utils.image import extract_mask_from_editor

        # None branch
        assert extract_mask_from_editor(None).mode == "L"

        # Dict branch
        layer = np.zeros((64, 64, 4), dtype=np.uint8)
        layer[0:5, 0:5, 3] = 255
        assert (
            extract_mask_from_editor(
                {"background": sample_background, "layers": [layer]}
            ).mode
            == "L"
        )

        # Dict with empty layers
        assert (
            extract_mask_from_editor(
                {"background": sample_background, "layers": []}
            ).mode
            == "L"
        )

        # Tuple branch
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[10:20, 10:20] = 255
        assert (
            extract_mask_from_editor((sample_background, mask_arr)).mode == "L"
        )


# ---------------------------------------------------------------------------
# dilate_mask tests
# ---------------------------------------------------------------------------

class TestDilateMask:
    def test_dilate_expands_mask(
        self, sample_mask_image: Image.Image
    ) -> None:
        from utils.image import dilate_mask

        before = np.asarray(sample_mask_image)
        active_before = int((before > 0).sum())
        # The 3×3 center dot has exactly 9 active pixels.
        assert active_before == 9

        result = dilate_mask(sample_mask_image, kernel_size=5)
        assert result.mode == "L"

        after = np.asarray(result)
        active_after = int((after > 0).sum())

        # The dilation with a 5×5 elliptical kernel must strictly enlarge
        # the active region around the original 3×3 dot.
        assert active_after > active_before

        # The original dot must still be fully inside the dilated mask.
        assert (after[31:34, 31:34] == 255).all()

    def test_zero_kernel_returns_same(
        self, sample_mask_image: Image.Image
    ) -> None:
        from utils.image import dilate_mask

        before = np.asarray(sample_mask_image)
        result = dilate_mask(sample_mask_image, kernel_size=0)

        assert result.mode == "L"
        after = np.asarray(result)

        # A 1×1 structuring element (kernel clamped to ≥1) is a no-op, so the
        # mask should be byte-identical to the input.
        np.testing.assert_array_equal(before, after)

    def test_output_is_mode_L(self, sample_mask_image: Image.Image) -> None:
        from utils.image import dilate_mask

        result = dilate_mask(sample_mask_image, kernel_size=3)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

        # Also verify the result is a strictly binary {0, 255} image.
        arr = np.asarray(result)
        unique = np.unique(arr)
        assert set(unique.tolist()).issubset({0, 255})

    def test_preserves_white_regions(self, white_mask: Image.Image) -> None:
        from utils.image import dilate_mask

        result = dilate_mask(white_mask, kernel_size=15)

        assert result.mode == "L"
        arr = np.asarray(result)

        # A fully-white mask dilated by anything stays fully white — dilation
        # can only grow the active region, never shrink it.
        assert (arr == 255).all()
