"""Tests for color module."""
from __future__ import annotations

import numpy as np
import pytest

from pixel_snapper.color import rgb_to_lab


class TestRgbToLab:
    """Tests for RGB to LAB conversion."""

    def test_black(self) -> None:
        """Black should convert to L=0."""
        rgb = np.array([[0, 0, 0]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (1, 3)
        assert abs(lab[0, 0]) < 1  # L should be ~0

    def test_white(self) -> None:
        """White should convert to L=100."""
        rgb = np.array([[255, 255, 255]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert abs(lab[0, 0] - 100) < 1  # L should be ~100

    def test_red(self) -> None:
        """Pure red should have positive a* value."""
        rgb = np.array([[255, 0, 0]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab[0, 1] > 0  # a* should be positive (red)

    def test_green(self) -> None:
        """Pure green should have negative a* value."""
        rgb = np.array([[0, 255, 0]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab[0, 1] < 0  # a* should be negative (green)

    def test_blue(self) -> None:
        """Pure blue should have negative b* value."""
        rgb = np.array([[0, 0, 255]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab[0, 2] < 0  # b* should be negative (blue)

    def test_yellow(self) -> None:
        """Pure yellow should have positive b* value."""
        rgb = np.array([[255, 255, 0]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab[0, 2] > 0  # b* should be positive (yellow)

    def test_batch_conversion(self) -> None:
        """Should handle batch conversion."""
        rgb = np.array([
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (5, 3)

    def test_gray_values(self) -> None:
        """Gray values should have near-zero a* and b*."""
        rgb = np.array([[128, 128, 128]], dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert abs(lab[0, 1]) < 1  # a* should be ~0
        assert abs(lab[0, 2]) < 1  # b* should be ~0

    def test_output_shape(self) -> None:
        """Output should have same first dimension as input."""
        rgb = np.array([[100, 150, 200]] * 10, dtype=np.float64)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (10, 3)
