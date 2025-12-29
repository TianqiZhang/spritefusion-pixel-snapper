"""Tests for profile module."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixel_snapper.config import PixelSnapperError
from pixel_snapper.profile import compute_profiles


class TestComputeProfiles:
    """Tests for compute_profiles function."""

    def test_returns_two_lists(self, sample_image: Image.Image) -> None:
        """Should return column and row profiles."""
        col_prof, row_prof = compute_profiles(sample_image)
        assert isinstance(col_prof, list)
        assert isinstance(row_prof, list)

    def test_profile_lengths(self, sample_image: Image.Image) -> None:
        """Profile lengths should match image dimensions."""
        col_prof, row_prof = compute_profiles(sample_image)
        width, height = sample_image.size
        assert len(col_prof) == width
        assert len(row_prof) == height

    def test_minimum_size_error(self) -> None:
        """Should reject images smaller than 3x3."""
        img = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        with pytest.raises(PixelSnapperError, match="too small"):
            compute_profiles(img)

    def test_horizontal_edge_detection(self) -> None:
        """Should detect horizontal edges in column profile."""
        # Create image with vertical stripe
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        arr = np.array(img)
        arr[:, 5:] = (0, 0, 0, 255)  # Right half is black
        img = Image.fromarray(arr, "RGBA")

        col_prof, _ = compute_profiles(img)

        # Peak should be near the edge (column 5)
        peak_idx = col_prof.index(max(col_prof))
        assert 3 <= peak_idx <= 7  # Near the edge

    def test_vertical_edge_detection(self) -> None:
        """Should detect vertical edges in row profile."""
        # Create image with horizontal stripe
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        arr = np.array(img)
        arr[5:, :] = (0, 0, 0, 255)  # Bottom half is black
        img = Image.fromarray(arr, "RGBA")

        _, row_prof = compute_profiles(img)

        # Peak should be near the edge (row 5)
        peak_idx = row_prof.index(max(row_prof))
        assert 3 <= peak_idx <= 7  # Near the edge

    def test_solid_color_low_gradient(
        self, solid_color_image: Image.Image
    ) -> None:
        """Solid color image should have zero gradients."""
        col_prof, row_prof = compute_profiles(solid_color_image)

        # All values should be near zero (no edges)
        assert max(col_prof) == 0
        assert max(row_prof) == 0

    def test_transparent_pixels_handled(
        self, transparent_image: Image.Image
    ) -> None:
        """Should handle transparent pixels correctly."""
        col_prof, row_prof = compute_profiles(transparent_image)

        # Should not raise and should return valid profiles
        assert len(col_prof) == transparent_image.size[0]
        assert len(row_prof) == transparent_image.size[1]

    def test_checkerboard_pattern(self, sample_image: Image.Image) -> None:
        """Checkerboard should have high gradients at grid boundaries."""
        col_prof, row_prof = compute_profiles(sample_image)

        # Should have non-zero gradients (edges exist)
        assert max(col_prof) > 0
        assert max(row_prof) > 0

        # Gradients should be present at cell boundaries (every 8 pixels)
        # Check that some boundary positions have significant gradients
        cell_size = 8
        boundary_positions = [cell_size * i for i in range(1, 8)]

        col_boundary_sum = sum(col_prof[p] for p in boundary_positions if p < len(col_prof))
        row_boundary_sum = sum(row_prof[p] for p in boundary_positions if p < len(row_prof))

        # Boundaries should have higher gradient than average
        col_avg = sum(col_prof) / len(col_prof)
        row_avg = sum(row_prof) / len(row_prof)

        assert col_boundary_sum > col_avg * len(boundary_positions) * 0.5
        assert row_boundary_sum > row_avg * len(boundary_positions) * 0.5
