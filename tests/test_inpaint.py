"""Unit tests for pipeline.inpaint module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
def sample_mask() -> Image.Image:
    """Small 64×64 grayscale (L mode) test mask."""
    return Image.new("L", (64, 64), color=255)


@pytest.fixture(autouse=True)
def _reset_inpaint_cache():
    """Reset the cached inpaint pipeline before and after each test."""
    from pipeline import inpaint
    saved = inpaint._pipeline
    inpaint._pipeline = None
    yield
    inpaint._pipeline = saved


# ---------------------------------------------------------------------------
# load_inpaint_model tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestLoadInpaintModel:
    @patch("pipeline.inpaint.StableDiffusionInpaintPipeline.from_pretrained")
    def test_returns_pipeline(self, mock_fp: MagicMock) -> None:
        from pipeline import inpaint

        mock_pipeline = MagicMock()
        mock_fp.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.set_progress_bar_config.return_value = mock_pipeline

        result = inpaint.load_inpaint_model()

        mock_fp.assert_called_once()
        assert result is mock_pipeline

    @patch("pipeline.inpaint.StableDiffusionInpaintPipeline.from_pretrained")
    def test_caches_pipeline(self, mock_fp: MagicMock) -> None:
        from pipeline import inpaint

        mock_pipeline = MagicMock()
        mock_fp.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.set_progress_bar_config.return_value = mock_pipeline

        first = inpaint.load_inpaint_model()
        second = inpaint.load_inpaint_model()

        mock_fp.assert_called_once()
        assert first is second

    @patch("pipeline.inpaint.StableDiffusionInpaintPipeline.from_pretrained")
    def test_cuda_device(self, mock_fp: MagicMock) -> None:
        from pipeline import inpaint

        mock_pipeline = MagicMock()
        mock_fp.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.set_progress_bar_config.return_value = mock_pipeline

        # Simulate CUDA being available
        with patch("pipeline.inpaint.get_device", return_value="cuda"):
            inpaint.load_inpaint_model()

        mock_pipeline.to.assert_called_once_with("cuda")

    @patch("pipeline.inpaint.StableDiffusionInpaintPipeline.from_pretrained")
    def test_falls_to_cpu(self, mock_fp: MagicMock) -> None:
        from pipeline import inpaint

        mock_pipeline = MagicMock()
        mock_fp.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.set_progress_bar_config.return_value = mock_pipeline

        # Simulate CPU fallback
        with patch("pipeline.inpaint.get_device", return_value="cpu"):
            inpaint.load_inpaint_model()

        mock_pipeline.to.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# inpaint_image tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestInpaintImage:
    def test_inpaint_image_calls_pipeline(
        self,
        sample_image: Image.Image,
        sample_mask: Image.Image,
    ) -> None:
        from pipeline.inpaint import inpaint_image

        mock_pipeline = MagicMock()
        mock_output = Image.new("RGB", (64, 64), color=(200, 200, 200))
        mock_pipeline.return_value = MagicMock(images=[mock_output])

        result = inpaint_image(
            pipeline=mock_pipeline,
            image=sample_image,
            mask=sample_mask,
            prompt="Replace this wall with brick",
            num_inference_steps=15,
            guidance_scale=8.0,
            strength=0.9,
            seed=42,
        )

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["prompt"] == "Replace this wall with brick"
        assert call_kwargs["image"] is sample_image
        assert call_kwargs["mask_image"] is sample_mask
        assert call_kwargs["num_inference_steps"] == 15
        assert call_kwargs["guidance_scale"] == 8.0
        assert call_kwargs["strength"] == 0.9
        assert "generator" in call_kwargs
        assert result.size == (64, 64)

    def test_converts_rgba_to_rgb(self, sample_mask: Image.Image) -> None:
        from pipeline.inpaint import inpaint_image

        rgba_image = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[Image.new("RGB", (64, 64))])

        inpaint_image(
            pipeline=mock_pipeline,
            image=rgba_image,
            mask=sample_mask,
            prompt="test prompt",
        )

        # The image passed to the pipeline should be RGB
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["image"].mode == "RGB"

    def test_converts_mask_to_l_mode(self, sample_image: Image.Image) -> None:
        from pipeline.inpaint import inpaint_image

        rgb_mask = Image.new("RGB", (64, 64), color=(255, 255, 255))
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[sample_image])

        inpaint_image(
            pipeline=mock_pipeline,
            image=sample_image,
            mask=rgb_mask,
            prompt="test prompt",
        )

        # The mask passed to the pipeline should be in "L" mode
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["mask_image"].mode == "L"

    def test_negative_prompt_passed(
        self,
        sample_image: Image.Image,
        sample_mask: Image.Image,
    ) -> None:
        from pipeline.inpaint import inpaint_image

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[sample_image])

        inpaint_image(
            pipeline=mock_pipeline,
            image=sample_image,
            mask=sample_mask,
            prompt="test",
            negative_prompt="blurry, low quality",
        )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "blurry, low quality"

    def test_negative_prompt_none_when_empty(
        self,
        sample_image: Image.Image,
        sample_mask: Image.Image,
    ) -> None:
        from pipeline.inpaint import inpaint_image

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[sample_image])

        inpaint_image(
            pipeline=mock_pipeline,
            image=sample_image,
            mask=sample_mask,
            prompt="test",
            negative_prompt="",
        )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] is None


# ---------------------------------------------------------------------------
# Environment variable defaults
# ---------------------------------------------------------------------------

class TestEnvDefaults:
    def test_default_model_id(self) -> None:
        from pipeline.inpaint import _INPAINT_MODEL_ID

        assert _INPAINT_MODEL_ID == "runwayml/stable-diffusion-inpainting"

    def test_default_steps(self) -> None:
        from pipeline.inpaint import _INPAINT_STEPS

        assert _INPAINT_STEPS == 25

    def test_default_guidance_scale(self) -> None:
        from pipeline.inpaint import _INPAINT_GUIDANCE_SCALE

        assert _INPAINT_GUIDANCE_SCALE == 7.5
