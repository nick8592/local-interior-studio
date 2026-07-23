"""Unit tests for pipeline.segment module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> Image.Image:
    """Small 64×64 RGB test image."""
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


@pytest.fixture
def fake_mask() -> np.ndarray:
    """Simple binary mask (top half True, bottom half False)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :] = True
    return mask


# ---------------------------------------------------------------------------
# Model type inference
# ---------------------------------------------------------------------------

class TestInferModelType:
    def test_vit_h(self) -> None:
        from pipeline.segment import _infer_model_type

        assert _infer_model_type("sam_vit_h_4b8939.pth") == "vit_h"

    def test_vit_l(self) -> None:
        from pipeline.segment import _infer_model_type

        assert _infer_model_type("sam_vit_l_0b3195.pth") == "vit_l"

    def test_vit_b(self) -> None:
        from pipeline.segment import _infer_model_type

        assert _infer_model_type("sam_vit_b_01ec64.pth") == "vit_b"

    def test_unknown_defaults_vit_h(self) -> None:
        from pipeline.segment import _infer_model_type

        assert _infer_model_type("custom_model.pth") == "vit_h"


# ---------------------------------------------------------------------------
# download_sam_checkpoint tests
# ---------------------------------------------------------------------------

class TestDownloadSAMCheckpoint:
    @patch("pipeline.segment.urllib.request.urlretrieve")
    @patch("pipeline.segment.Path")
    def test_downloads_if_missing(self, mock_path_cls: MagicMock, mock_urlretrieve: MagicMock) -> None:
        mock_dest = MagicMock()
        mock_dest.exists.return_value = False
        mock_path_cls.return_value = mock_dest
        mock_dest.__truediv__ = MagicMock(return_value=mock_dest)
        mock_dest.stat.return_value.st_size = int(2.4e9)

        # Patch the import check
        with patch.dict("sys.modules", {"segment_anything": MagicMock()}):
            from pipeline.segment import download_sam_checkpoint

            result = download_sam_checkpoint(output_dir="/tmp/test_models")

        mock_urlretrieve.assert_called_once()

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        """If the checkpoint already exists, no download occurs."""
        ckpt = tmp_path / "sam_vit_h_4b8939.pth"
        ckpt.write_bytes(b"fake_weights")

        with patch.dict("sys.modules", {"segment_anything": MagicMock()}):
            from pipeline.segment import download_sam_checkpoint

            with patch("pipeline.segment.urllib.request.urlretrieve") as mock_dl:
                result = download_sam_checkpoint(output_dir=str(tmp_path))
                mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# segment_room tests
# ---------------------------------------------------------------------------

class TestSegmentRoom:
    def test_point_prompt_returns_mask(self, sample_image: Image.Image, fake_mask: np.ndarray) -> None:
        from pipeline.segment import segment_room

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = (
            np.stack([fake_mask, ~fake_mask, fake_mask]),  # 3 masks
            np.array([0.9, 0.3, 0.1]),  # scores
            None,
        )

        masks = segment_room(
            image=sample_image,
            predictor=mock_predictor,
            point_coords=[(32, 32)],
            point_labels=[1],
        )

        mock_predictor.set_image.assert_called_once()
        assert len(masks) == 1
        # Should pick highest-score mask (index 0)
        np.testing.assert_array_equal(masks[0], fake_mask)

    def test_auto_mask_generation(self, sample_image: Image.Image, fake_mask: np.ndarray) -> None:
        from pipeline.segment import segment_room

        mock_predictor = MagicMock()
        mock_predictor.model = MagicMock()

        amg_results = [
            {"segmentation": fake_mask, "area": 2048},
            {"segmentation": ~fake_mask, "area": 1024},
        ]

        with patch("pipeline.segment.SamAutomaticMaskGenerator") as mock_amg_cls:
            mock_amg = MagicMock()
            mock_amg.return_value = amg_results
            mock_amg_cls.return_value = mock_amg

            masks = segment_room(image=sample_image, predictor=mock_predictor)

        assert len(masks) == 2

    def test_converts_non_rgb(self) -> None:
        from pipeline.segment import segment_room

        rgba_image = Image.new("RGBA", (64, 64))
        mock_predictor = MagicMock()
        mock_predictor.model = MagicMock()

        with patch("pipeline.segment.SamAutomaticMaskGenerator") as mock_amg_cls:
            mock_amg_cls.return_value = MagicMock(return_value=[])

            masks = segment_room(image=rgba_image, predictor=mock_predictor)

        # set_image should receive an RGB numpy array (3 channels)
        call_args = mock_predictor.set_image.call_args[0][0]
        assert call_args.shape[2] == 3


# ---------------------------------------------------------------------------
# generate_mask tests
# ---------------------------------------------------------------------------

class TestGenerateMask:
    def test_combines_masks(self, sample_image: Image.Image) -> None:
        from pipeline.segment import generate_mask

        mask_a = np.zeros((64, 64), dtype=bool)
        mask_a[:32, :] = True

        mask_b = np.zeros((64, 64), dtype=bool)
        mask_b[:, :32] = True

        with patch("pipeline.segment.segment_room") as mock_seg:
            mock_seg.return_value = [mask_a, mask_b]

            result = generate_mask(image=sample_image, predictor=MagicMock())

        # Combined: top half OR left half
        assert result.dtype == bool
        assert result.sum() > mask_a.sum()  # more pixels covered

    def test_empty_masks_returns_zeros(self, sample_image: Image.Image) -> None:
        from pipeline.segment import generate_mask

        with patch("pipeline.segment.segment_room") as mock_seg:
            mock_seg.return_value = []

            result = generate_mask(image=sample_image, predictor=MagicMock())

        assert result.shape == (64, 64)
        assert result.sum() == 0
