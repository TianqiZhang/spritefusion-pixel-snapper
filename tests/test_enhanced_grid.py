"""Tests for enhanced grid detection features."""
import pytest
import numpy as np
from PIL import Image

from pixel_snapper import Config
from pixel_snapper.grid import (
    estimate_step_size_autocorr,
    find_best_offset,
    walk_with_offset,
)
from pixel_snapper.scoring import (
    score_grid_uniformity,
    score_edge_alignment,
    select_best_grid,
    generate_uniform_cuts,
)


class TestAutocorrelation:
    """Tests for autocorrelation-based step size estimation."""

    def test_periodic_signal(self):
        """Test detection of clear periodic signal."""
        config = Config()
        # Create a periodic signal with period 10
        profile = [0.0] * 100
        for i in range(0, 100, 10):
            profile[i] = 100.0

        step, confidence = estimate_step_size_autocorr(profile, config)
        assert step is not None
        assert 9.0 <= step <= 11.0  # Allow some tolerance
        assert confidence >= 0.3

    def test_noisy_periodic_signal(self):
        """Test detection with added noise."""
        config = Config()
        np.random.seed(42)
        profile = [float(np.random.uniform(0, 10)) for _ in range(100)]
        # Add periodic peaks
        for i in range(0, 100, 8):
            profile[i] += 80.0

        step, confidence = estimate_step_size_autocorr(profile, config)
        assert step is not None
        assert 7.0 <= step <= 9.0

    def test_no_periodicity(self):
        """Test returns None for random signal."""
        config = Config()
        np.random.seed(42)
        profile = [float(np.random.uniform(0, 100)) for _ in range(100)]

        step, confidence = estimate_step_size_autocorr(profile, config)
        # Either no step found or low confidence
        assert step is None or confidence < 0.3

    def test_constant_signal(self):
        """Test handles constant signal."""
        config = Config()
        profile = [50.0] * 100

        step, confidence = estimate_step_size_autocorr(profile, config)
        assert step is None
        assert confidence == 0.0

    def test_short_profile(self):
        """Test handles very short profiles."""
        config = Config()
        profile = [10.0, 20.0, 10.0]

        step, confidence = estimate_step_size_autocorr(profile, config)
        assert step is None

    def test_empty_profile(self):
        """Test handles empty profile."""
        config = Config()
        step, confidence = estimate_step_size_autocorr([], config)
        assert step is None
        assert confidence == 0.0


class TestGridOffset:
    """Tests for grid offset detection."""

    def test_no_offset_needed(self):
        """Test when grid is already aligned at 0."""
        config = Config()
        # Peaks at 0, 10, 20, 30...
        profile = [0.0] * 100
        for i in range(0, 100, 10):
            profile[i] = 100.0

        offset = find_best_offset(profile, 10.0, 100, config)
        assert offset == 0

    def test_detects_offset(self):
        """Test detection of grid offset."""
        config = Config()
        # Peaks at 3, 13, 23, 33...
        profile = [0.0] * 100
        for i in range(3, 100, 10):
            profile[i] = 100.0

        offset = find_best_offset(profile, 10.0, 100, config)
        assert offset == 3

    def test_small_step_size(self):
        """Test with very small step size."""
        config = Config()
        profile = [10.0] * 100

        offset = find_best_offset(profile, 1.0, 100, config)
        assert offset == 0

    def test_empty_profile(self):
        """Test with empty profile."""
        config = Config()
        offset = find_best_offset([], 10.0, 100, config)
        assert offset == 0


class TestWalkWithOffset:
    """Tests for walking with offset."""

    def test_includes_offset_as_cut(self):
        """Test that offset is included in cuts."""
        config = Config()
        profile = [10.0] * 100

        cuts = walk_with_offset(profile, 20.0, 100, 5, config)
        assert 0 in cuts
        assert 5 in cuts
        assert 100 in cuts

    def test_zero_offset_same_as_walk(self):
        """Test that zero offset behaves like regular walk."""
        from pixel_snapper.grid import walk

        config = Config()
        profile = [10.0] * 100

        cuts_with_offset = walk_with_offset(profile, 20.0, 100, 0, config)
        cuts_normal = walk(profile, 20.0, 100, config)

        assert cuts_with_offset == cuts_normal

    def test_empty_profile_error(self):
        """Test raises error on empty profile."""
        from pixel_snapper.config import PixelSnapperError

        config = Config()
        with pytest.raises(PixelSnapperError):
            walk_with_offset([], 10.0, 100, 5, config)


class TestUniformityScoring:
    """Tests for grid uniformity scoring."""

    def test_uniform_cells_low_score(self):
        """Test that uniform cells give low variance score."""
        # Create 4x4 image with 2x2 grid of solid colors
        img = Image.new("RGBA", (4, 4))
        pixels = img.load()
        # Top-left: red
        pixels[0, 0] = pixels[1, 0] = pixels[0, 1] = pixels[1, 1] = (255, 0, 0, 255)
        # Top-right: green
        pixels[2, 0] = pixels[3, 0] = pixels[2, 1] = pixels[3, 1] = (0, 255, 0, 255)
        # Bottom-left: blue
        pixels[0, 2] = pixels[1, 2] = pixels[0, 3] = pixels[1, 3] = (0, 0, 255, 255)
        # Bottom-right: white
        pixels[2, 2] = pixels[3, 2] = pixels[2, 3] = pixels[3, 3] = (255, 255, 255, 255)

        col_cuts = [0, 2, 4]
        row_cuts = [0, 2, 4]

        score = score_grid_uniformity(img, col_cuts, row_cuts)
        assert score == 0.0  # Perfect uniformity

    def test_mixed_cells_higher_score(self):
        """Test that mixed color cells give higher variance score."""
        img = Image.new("RGBA", (4, 4))
        pixels = img.load()
        # Mix of colors in each cell
        for x in range(4):
            for y in range(4):
                pixels[x, y] = (x * 60, y * 60, 128, 255)

        col_cuts = [0, 2, 4]
        row_cuts = [0, 2, 4]

        score = score_grid_uniformity(img, col_cuts, row_cuts)
        assert score > 0.0  # Non-zero variance

    def test_invalid_grid_returns_inf(self):
        """Test that invalid grids return infinity."""
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 255))

        score = score_grid_uniformity(img, [0], [0])  # Single cut
        assert score == float("inf")

    def test_transparent_cells_handled(self):
        """Test that partially transparent images are handled correctly."""
        # Image with some opaque and some transparent cells
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        pixels = img.load()
        # Make top-left cell opaque and uniform
        pixels[0, 0] = pixels[1, 0] = pixels[0, 1] = pixels[1, 1] = (255, 0, 0, 255)

        col_cuts = [0, 2, 4]
        row_cuts = [0, 2, 4]

        score = score_grid_uniformity(img, col_cuts, row_cuts)
        # Should not be infinity since we have at least one opaque cell
        assert score != float("inf")
        # The opaque cell is uniform so score should be 0
        assert score == 0.0

    def test_fully_transparent_returns_inf(self):
        """Test that fully transparent image returns infinity (no valid grid)."""
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))  # All transparent
        col_cuts = [0, 2, 4]
        row_cuts = [0, 2, 4]

        score = score_grid_uniformity(img, col_cuts, row_cuts)
        # All-transparent has no meaningful variance to compute
        assert score == float("inf")


class TestEdgeAlignmentScoring:
    """Tests for edge alignment scoring."""

    def test_aligned_cuts_high_score(self):
        """Test that cuts aligned with peaks get high score."""
        profile = [0.0] * 100
        profile[10] = 100.0
        profile[20] = 100.0
        profile[30] = 100.0

        cuts = [0, 10, 20, 30, 100]
        score = score_edge_alignment(profile, cuts, 100)
        assert score == 100.0

    def test_misaligned_cuts_low_score(self):
        """Test that misaligned cuts get low score."""
        profile = [0.0] * 100
        profile[10] = 100.0
        profile[20] = 100.0
        profile[30] = 100.0

        cuts = [0, 15, 25, 35, 100]  # Misaligned
        score = score_edge_alignment(profile, cuts, 100)
        assert score == 0.0

    def test_empty_cuts(self):
        """Test handles empty or minimal cuts."""
        profile = [10.0] * 100
        assert score_edge_alignment(profile, [], 100) == 0.0
        assert score_edge_alignment(profile, [0], 100) == 0.0


class TestSelectBestGrid:
    """Tests for selecting best grid from candidates."""

    def test_selects_more_uniform_grid(self):
        """Test that more uniform grid is selected."""
        # Create image with clear 2x2 grid
        img = Image.new("RGBA", (8, 8))
        pixels = img.load()
        for x in range(4):
            for y in range(4):
                pixels[x, y] = (255, 0, 0, 255)
        for x in range(4, 8):
            for y in range(4):
                pixels[x, y] = (0, 255, 0, 255)
        for x in range(4):
            for y in range(4, 8):
                pixels[x, y] = (0, 0, 255, 255)
        for x in range(4, 8):
            for y in range(4, 8):
                pixels[x, y] = (255, 255, 0, 255)

        profile_x = [0.0] * 8
        profile_y = [0.0] * 8
        profile_x[4] = 100.0  # Edge at x=4
        profile_y[4] = 100.0  # Edge at y=4

        candidates = [
            ([0, 4, 8], [0, 4, 8], 4.0),  # Correct grid
            ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8], 2.0),  # Wrong grid
        ]

        col_cuts, row_cuts, step = select_best_grid(
            img, profile_x, profile_y, candidates, 8, 8
        )

        # Should select the 2x2 grid (step=4) as it has better uniformity
        assert step == 4.0
        assert col_cuts == [0, 4, 8]

    def test_single_candidate(self):
        """Test returns single candidate directly."""
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        candidates = [([0, 4, 8], [0, 4, 8], 4.0)]

        col_cuts, row_cuts, step = select_best_grid(
            img, [0.0] * 8, [0.0] * 8, candidates, 8, 8
        )

        assert col_cuts == [0, 4, 8]
        assert step == 4.0

    def test_empty_candidates(self):
        """Test handles empty candidate list."""
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))

        col_cuts, row_cuts, step = select_best_grid(
            img, [0.0] * 8, [0.0] * 8, [], 8, 8
        )

        # Should return fallback
        assert 0 in col_cuts
        assert 8 in col_cuts


class TestGenerateUniformCuts:
    """Tests for generating uniform grid cuts."""

    def test_basic_uniform_grid(self):
        """Test basic uniform grid generation."""
        cuts = generate_uniform_cuts(10.0, 100)
        assert cuts[0] == 0
        assert cuts[-1] == 100
        assert len(cuts) == 11  # 0, 10, 20, ..., 100

    def test_non_divisible_size(self):
        """Test when step doesn't divide evenly."""
        cuts = generate_uniform_cuts(15.0, 100)
        assert cuts[0] == 0
        assert cuts[-1] == 100

    def test_zero_step(self):
        """Test handles zero step size."""
        cuts = generate_uniform_cuts(0.0, 100)
        assert 0 in cuts
        assert 100 in cuts

    def test_zero_limit(self):
        """Test handles zero limit."""
        cuts = generate_uniform_cuts(10.0, 0)
        assert cuts == [0]


class TestIntegrationEnhancedDetection:
    """Integration tests for enhanced grid detection."""

    def test_process_with_enhanced_features(self):
        """Test processing with all enhanced features enabled."""
        from pixel_snapper import Config, process_image_bytes
        import io

        # Create a simple 32x32 image with 8x8 grid pattern
        img = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        pixels = img.load()
        for bx in range(4):
            for by in range(4):
                color = ((bx + by) % 2 * 128 + 64, 0, 0, 255)
                for x in range(bx * 8, (bx + 1) * 8):
                    for y in range(by * 8, (by + 1) * 8):
                        pixels[x, y] = color

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        config = Config(
            use_autocorrelation=True,
            detect_grid_offset=True,
            use_uniformity_scoring=True,
        )

        output = process_image_bytes(buf.getvalue(), config)
        assert len(output) > 0

    def test_disable_enhanced_features(self):
        """Test that enhanced features can be disabled."""
        from pixel_snapper import Config, process_image_bytes
        import io

        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        config = Config(
            use_autocorrelation=False,
            detect_grid_offset=False,
            use_uniformity_scoring=False,
        )

        output = process_image_bytes(buf.getvalue(), config)
        assert len(output) > 0
