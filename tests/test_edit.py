"""Unit tests for pipeline.edit module."""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# get_device tests
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_string(self) -> None:
        from pipeline.edit import get_device

        result = get_device()
        assert isinstance(result, str)
        assert result in ("cuda", "mps", "cpu")

    @patch.dict(os.environ, {"DEVICE": "cpu"})
    def test_respects_env_var(self) -> None:
        from pipeline.edit import get_device

        assert get_device() == "cpu"

    @patch("pipeline.edit.torch")
    def test_cuda_preferred(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = True

        # Force auto detection
        with patch.dict(os.environ, {"DEVICE": "auto"}):
            from pipeline.edit import get_device

            result = get_device()
            assert result == "cuda"

    @patch("pipeline.edit.torch")
    def test_falls_to_cpu(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False
        if hasattr(mock_torch.backends, "mps"):
            mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(os.environ, {"DEVICE": "auto"}):
            from pipeline.edit import get_device

            result = get_device()
            assert result == "cpu"


# ---------------------------------------------------------------------------
# edit_image tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestEditImage:
    def test_edit_image_calls_pipeline(self, sample_image: Image.Image) -> None:
        from pipeline.edit import edit_image

        mock_pipeline = MagicMock()
        mock_output = Image.new("RGB", (64, 64), color=(200, 200, 200))
        mock_pipeline.return_value = MagicMock(images=[mock_output])

        result = edit_image(
            pipeline=mock_pipeline,
            image=sample_image,
            prompt="Turn this room into a minimalist room",
            num_inference_steps=10,
            seed=42,
        )

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["prompt"] == "Turn this room into a minimalist room"
        assert call_kwargs["num_inference_steps"] == 10
        assert result.size == (64, 64)

    def test_edit_image_converts_rgba(self) -> None:
        from pipeline.edit import edit_image

        rgba_image = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[Image.new("RGB", (64, 64))])

        edit_image(
            pipeline=mock_pipeline,
            image=rgba_image,
            prompt="test prompt",
        )

        # The image passed to the pipeline should be RGB
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["image"].mode == "RGB"

    def test_negative_prompt_passed(self, sample_image: Image.Image) -> None:
        from pipeline.edit import edit_image

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[sample_image])

        edit_image(
            pipeline=mock_pipeline,
            image=sample_image,
            prompt="test",
            negative_prompt="blurry, low quality",
        )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] == "blurry, low quality"

    def test_negative_prompt_none_when_empty(self, sample_image: Image.Image) -> None:
        from pipeline.edit import edit_image

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = MagicMock(images=[sample_image])

        edit_image(
            pipeline=mock_pipeline,
            image=sample_image,
            prompt="test",
            negative_prompt="",
        )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["negative_prompt"] is None


# ---------------------------------------------------------------------------
# Environment variable defaults
# ---------------------------------------------------------------------------

class TestEnvDefaults:
    def test_default_steps(self) -> None:
        from pipeline.edit import _DEFAULT_STEPS

        assert _DEFAULT_STEPS == 20

    def test_default_guidance_scale(self) -> None:
        from pipeline.edit import _DEFAULT_GUIDANCE_SCALE

        assert _DEFAULT_GUIDANCE_SCALE == 7.5

    def test_default_image_guidance_scale(self) -> None:
        from pipeline.edit import _DEFAULT_IMAGE_GUIDANCE_SCALE

        assert _DEFAULT_IMAGE_GUIDANCE_SCALE == 1.5
