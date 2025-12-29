"""Tests for quantize module."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixel_snapper.config import Config, PixelSnapperError
from pixel_snapper.palette import Palette
from pixel_snapper.quantize import (
    kmeans_quantize,
    palette_quantize,
    quantize_image,
)


class TestQuantizeImage:
    """Tests for quantize_image function."""

    def test_reduces_colors(self, sample_image: Image.Image) -> None:
        """Should reduce number of unique colors."""
        config = Config(k_colors=4)
        result = quantize_image(sample_image, config)

        # Count unique colors
        arr = np.array(result)
        flat = arr.reshape(-1, 4)
        unique = np.unique(flat, axis=0)

        assert len(unique) <= 4

    def test_preserves_dimensions(self, sample_image: Image.Image) -> None:
        """Should preserve image dimensions."""
        config = Config(k_colors=8)
        result = quantize_image(sample_image, config)
        assert result.size == sample_image.size

    def test_preserves_mode(self, sample_image: Image.Image) -> None:
        """Should preserve RGBA mode."""
        config = Config(k_colors=8)
        result = quantize_image(sample_image, config)
        assert result.mode == "RGBA"

    def test_invalid_k_colors(self, sample_image: Image.Image) -> None:
        """Should reject k_colors <= 0."""
        config = Config(k_colors=0)
        with pytest.raises(PixelSnapperError, match="greater than 0"):
            quantize_image(sample_image, config)

    def test_transparent_pixels_unchanged(
        self, transparent_image: Image.Image
    ) -> None:
        """Should not modify transparent pixels."""
        config = Config(k_colors=4)
        result = quantize_image(transparent_image, config)

        arr = np.array(result)
        # Check corners are still transparent
        assert arr[0, 0, 3] == 0
        assert arr[0, -1, 3] == 0
        assert arr[-1, 0, 3] == 0
        assert arr[-1, -1, 3] == 0

    def test_deterministic(self, sample_image: Image.Image) -> None:
        """Should produce deterministic results with same seed."""
        config = Config(k_colors=4, k_seed=42)
        result1 = quantize_image(sample_image, config)
        result2 = quantize_image(sample_image, config)

        arr1 = np.array(result1)
        arr2 = np.array(result2)
        np.testing.assert_array_equal(arr1, arr2)


class TestKmeansQuantize:
    """Tests for kmeans_quantize function."""

    def test_empty_image(self) -> None:
        """Should handle fully transparent image."""
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        config = Config(k_colors=4)
        result = kmeans_quantize(img, config)
        assert result.size == img.size

    def test_single_color(self, solid_color_image: Image.Image) -> None:
        """Should handle single-color image."""
        config = Config(k_colors=4)
        result = kmeans_quantize(solid_color_image, config)

        arr = np.array(result)
        # All pixels should be the same color
        unique = np.unique(arr.reshape(-1, 4), axis=0)
        assert len(unique) == 1

    def test_convergence(self, sample_image: Image.Image) -> None:
        """Should converge with enough iterations."""
        config = Config(k_colors=4, max_kmeans_iterations=100)
        result = kmeans_quantize(sample_image, config)
        assert result is not None


class TestPaletteQuantize:
    """Tests for palette_quantize function."""

    def test_maps_to_palette_colors(self) -> None:
        """Should map all colors to palette colors."""
        img = Image.new("RGBA", (10, 10), (100, 100, 100, 255))
        palette = Palette(
            rgb=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            lab=[(53.23, 80.11, 67.22), (87.74, -86.18, 83.18), (32.30, 79.20, -107.86)],
        )
        result = palette_quantize(img, palette, "rgb")

        arr = np.array(result)
        unique_colors = set()
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                color = tuple(arr[y, x, :3])
                unique_colors.add(color)

        # All colors should be from palette
        for color in unique_colors:
            assert color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def test_rgb_space(self) -> None:
        """Should use RGB space when specified."""
        img = Image.new("RGBA", (10, 10), (254, 0, 0, 255))  # Almost red
        palette = Palette(
            rgb=[(255, 0, 0), (0, 0, 0)],
            lab=[(53.23, 80.11, 67.22), (0, 0, 0)],
        )
        result = palette_quantize(img, palette, "rgb")

        arr = np.array(result)
        # Should map to red (closest in RGB)
        assert tuple(arr[0, 0, :3]) == (255, 0, 0)

    def test_lab_space(self) -> None:
        """Should use LAB space when specified."""
        img = Image.new("RGBA", (10, 10), (128, 128, 128, 255))
        palette = Palette(
            rgb=[(255, 255, 255), (0, 0, 0)],
            lab=[(100, 0, 0), (0, 0, 0)],
        )
        result = palette_quantize(img, palette, "lab")
        # Result should be either black or white
        arr = np.array(result)
        color = tuple(arr[0, 0, :3])
        assert color in [(255, 255, 255), (0, 0, 0)]
