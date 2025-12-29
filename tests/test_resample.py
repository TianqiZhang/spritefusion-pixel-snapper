"""Tests for resample module."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixel_snapper.config import PixelSnapperError
from pixel_snapper.resample import resample


class TestResample:
    """Tests for resample function."""

    def test_basic_resampling(self, sample_image: Image.Image) -> None:
        """Should resample image to grid dimensions."""
        cols = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        rows = [0, 8, 16, 24, 32, 40, 48, 56, 64]

        result = resample(sample_image, cols, rows)

        assert result.size == (8, 8)  # 8 cells x 8 cells

    def test_majority_vote(self) -> None:
        """Should select most common color in each cell."""
        # Create 4x4 image with 2x2 cells
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        arr = np.array(img)

        # Top-left cell: 3 red, 1 green
        arr[0, 0] = (255, 0, 0, 255)
        arr[0, 1] = (255, 0, 0, 255)
        arr[1, 0] = (255, 0, 0, 255)
        arr[1, 1] = (0, 255, 0, 255)

        img = Image.fromarray(arr, "RGBA")
        cols = [0, 2, 4]
        rows = [0, 2, 4]

        result = resample(img, cols, rows)
        result_arr = np.array(result)

        # Top-left cell should be red (majority)
        assert tuple(result_arr[0, 0, :3]) == (255, 0, 0)

    def test_preserves_alpha(self) -> None:
        """Should preserve alpha channel in output."""
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        cols = [0, 2, 4]
        rows = [0, 2, 4]

        result = resample(img, cols, rows)
        result_arr = np.array(result)

        assert result_arr[0, 0, 3] == 128

    def test_handles_transparent_cells(self) -> None:
        """Should handle cells with transparent pixels."""
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        arr = np.array(img)
        arr[0:2, 0:2] = (255, 0, 0, 255)  # Only top-left cell is opaque

        img = Image.fromarray(arr, "RGBA")
        cols = [0, 2, 4]
        rows = [0, 2, 4]

        result = resample(img, cols, rows)
        result_arr = np.array(result)

        # Top-left should be red, others transparent
        assert tuple(result_arr[0, 0]) == (255, 0, 0, 255)
        assert result_arr[0, 1, 3] == 0  # Transparent
        assert result_arr[1, 0, 3] == 0
        assert result_arr[1, 1, 3] == 0

    def test_tie_breaking(self) -> None:
        """Should break ties deterministically."""
        # Create 2x2 image with two colors each appearing once
        img = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
        arr = np.array(img)
        arr[0, 0] = (255, 0, 0, 255)
        arr[0, 1] = (0, 255, 0, 255)

        img = Image.fromarray(arr, "RGBA")
        cols = [0, 2]
        rows = [0, 1]

        result = resample(img, cols, rows)

        # Should pick one consistently (smaller packed value)
        assert result.size == (1, 1)

    def test_insufficient_cuts_error(self, sample_image: Image.Image) -> None:
        """Should raise error if insufficient cuts."""
        with pytest.raises(PixelSnapperError, match="Insufficient"):
            resample(sample_image, [0], [0, 64])

        with pytest.raises(PixelSnapperError, match="Insufficient"):
            resample(sample_image, [0, 64], [0])

    def test_single_cell(self, sample_image: Image.Image) -> None:
        """Should handle single-cell output."""
        cols = [0, 64]
        rows = [0, 64]

        result = resample(sample_image, cols, rows)
        assert result.size == (1, 1)

    def test_empty_cell_skipped(self) -> None:
        """Should skip cells with zero dimensions."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        # Cuts with zero-width cell
        cols = [0, 5, 5, 10]  # Cell from 5-5 has zero width
        rows = [0, 10]

        result = resample(img, cols, rows)
        # Should still produce output (skipping empty cells)
        assert result.mode == "RGBA"

    def test_clamping_coordinates(self) -> None:
        """Should clamp coordinates to image bounds."""
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        # Cuts that exceed image dimensions
        cols = [0, 5, 15]  # 15 > 10
        rows = [0, 5, 12]  # 12 > 10

        # Should not raise, should clamp
        result = resample(img, cols, rows)
        assert result.size == (2, 2)
