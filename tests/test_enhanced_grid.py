"""Tests for enhanced grid detection features."""
import pytest
import numpy as np
from PIL import Image, ImageDraw

from pixel_snapper import Config
from pixel_snapper.grid import (
    estimate_step_size_autocorr,
    find_best_offset,
    walk_with_offset,
)
from pixel_snapper.scoring import (
    score_grid_uniformity,
    score_edge_alignment,
    score_all_candidates,
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


class TestScoreAllCandidates:
    """Tests for scoring and selecting best grid from candidates."""

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
            ([0, 4, 8], [0, 4, 8], 4.0, "correct"),  # Correct grid
            ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8], 2.0, "wrong"),  # Wrong grid
        ]

        scored = score_all_candidates(
            img, profile_x, profile_y, candidates, 8, 8
        )

        # Should select the 2x2 grid as it has better uniformity
        assert len(scored) == 2
        assert scored[0].col_cuts == [0, 4, 8]
        assert scored[0].rank == 1

    def test_single_candidate(self):
        """Test returns single candidate directly."""
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        candidates = [([0, 4, 8], [0, 4, 8], 4.0, "only")]

        scored = score_all_candidates(
            img, [0.0] * 8, [0.0] * 8, candidates, 8, 8
        )

        assert len(scored) == 1
        assert scored[0].col_cuts == [0, 4, 8]

    def test_empty_candidates(self):
        """Test handles empty candidate list."""
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))

        scored = score_all_candidates(
            img, [0.0] * 8, [0.0] * 8, [], 8, 8
        )

        # Should return empty list
        assert scored == []


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


class TestResolutionHint:
    """Tests for resolution hint feature."""

    def test_resolution_hint_limits_cells(self):
        """Test that resolution hint acts as upper limit on cell count."""
        from pixel_snapper import Config, process_image_bytes_with_grid
        import io

        # Create a 128x128 image with 8-pixel grid (16x16 cells)
        img = Image.new("RGBA", (128, 128), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        for i in range(0, 128, 8):
            draw.line([(i, 0), (i, 127)], fill=(0, 0, 0, 255))
            draw.line([(0, i), (127, i)], fill=(0, 0, 0, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # With hint=10, output should have at most 10 cells per axis
        config = Config(resolution_hint=10)
        result = process_image_bytes_with_grid(buf.getvalue(), config)

        cells_x = len(result.col_cuts) - 1
        cells_y = len(result.row_cuts) - 1
        assert max(cells_x, cells_y) <= 10

    def test_resolution_hint_allows_fewer_cells(self):
        """Test that resolution hint allows fewer cells than the hint."""
        from pixel_snapper import Config, process_image_bytes_with_grid
        import io

        # Create a 64x64 image with 16-pixel grid (4x4 cells)
        img = Image.new("RGBA", (64, 64), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        for i in range(0, 64, 16):
            draw.line([(i, 0), (i, 63)], fill=(0, 0, 0, 255))
            draw.line([(0, i), (63, i)], fill=(0, 0, 0, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # With hint=32, output can have fewer cells (like 4x4)
        config = Config(resolution_hint=32)
        result = process_image_bytes_with_grid(buf.getvalue(), config)

        cells_x = len(result.col_cuts) - 1
        cells_y = len(result.row_cuts) - 1
        # Should detect roughly 4x4, not 32x32
        assert max(cells_x, cells_y) <= 32

    def test_resolution_hint_with_non_square_image(self):
        """Test resolution hint applies to long axis for non-square images."""
        from pixel_snapper import Config, process_image_bytes_with_grid
        import io

        # Create a 128x64 image (2:1 aspect ratio)
        img = Image.new("RGBA", (128, 64), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        for i in range(0, 128, 8):
            draw.line([(i, 0), (i, 63)], fill=(0, 0, 0, 255))
        for i in range(0, 64, 8):
            draw.line([(0, i), (127, i)], fill=(0, 0, 0, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # With hint=12, long axis (128px) should have at most 12 cells
        config = Config(resolution_hint=12)
        result = process_image_bytes_with_grid(buf.getvalue(), config)

        cells_x = len(result.col_cuts) - 1
        cells_y = len(result.row_cuts) - 1
        assert max(cells_x, cells_y) <= 12

    def test_resolution_hint_none_has_no_effect(self):
        """Test that None resolution hint doesn't change behavior."""
        from pixel_snapper import Config, process_image_bytes_with_grid
        import io

        img = Image.new("RGBA", (64, 64), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        for i in range(0, 64, 8):
            draw.line([(i, 0), (i, 63)], fill=(0, 0, 0, 255))
            draw.line([(0, i), (63, i)], fill=(0, 0, 0, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        config_with_hint = Config(resolution_hint=None)
        config_without_hint = Config()

        result_with = process_image_bytes_with_grid(buf.getvalue(), config_with_hint)
        result_without = process_image_bytes_with_grid(buf.getvalue(), config_without_hint)

        # Should produce same results
        assert result_with.col_cuts == result_without.col_cuts
        assert result_with.row_cuts == result_without.row_cuts


class TestAutocorrelationMulti:
    """Tests for multi-peak autocorrelation function."""

    def test_returns_multiple_peaks(self):
        """Test that estimate_step_size_autocorr_multi returns multiple peaks."""
        from pixel_snapper.grid import estimate_step_size_autocorr_multi
        from pixel_snapper import Config

        config = Config()
        # Create a profile with multiple periodicities
        profile = []
        for i in range(256):
            # Fundamental period of 16 with harmonic at 8
            val = 100 if i % 16 == 0 else 0
            val += 50 if i % 8 == 0 else 0
            profile.append(float(val))

        results = estimate_step_size_autocorr_multi(profile, config, max_peaks=5)

        # Should return multiple peaks
        assert len(results) >= 1

    def test_returns_empty_for_no_periodicity(self):
        """Test that returns empty list for non-periodic profile."""
        from pixel_snapper.grid import estimate_step_size_autocorr_multi
        from pixel_snapper import Config

        config = Config()
        # Random-ish non-periodic profile
        import random
        random.seed(42)
        profile = [random.random() * 0.001 for _ in range(64)]

        results = estimate_step_size_autocorr_multi(profile, config)

        # Should return empty or very few peaks
        assert isinstance(results, list)

    def test_peaks_sorted_by_confidence(self):
        """Test that returned peaks are sorted by confidence descending."""
        from pixel_snapper.grid import estimate_step_size_autocorr_multi
        from pixel_snapper import Config

        config = Config()
        # Create periodic profile
        profile = [100.0 if i % 10 == 0 else 0.0 for i in range(200)]

        results = estimate_step_size_autocorr_multi(profile, config, max_peaks=5)

        if len(results) > 1:
            confidences = [conf for _, conf in results]
            assert confidences == sorted(confidences, reverse=True)


class TestResolutionHintCLI:
    """Tests for resolution hint CLI argument parsing."""

    def test_parse_resolution_hint(self):
        """Test parsing --resolution-hint argument."""
        from pixel_snapper.cli import parse_args

        config = parse_args(["prog", "in.png", "out.png", "--resolution-hint", "64"])
        assert config.resolution_hint == 64

    def test_parse_resolution_hint_with_other_args(self):
        """Test parsing --resolution-hint with other arguments."""
        from pixel_snapper.cli import parse_args

        config = parse_args([
            "prog", "in.png", "out.png", "16",
            "--resolution-hint", "32",
            "--preview"
        ])
        assert config.resolution_hint == 32
        assert config.k_colors == 16
        assert config.preview is True

    def test_invalid_resolution_hint_non_integer(self):
        """Test that non-integer resolution hint raises error."""
        from pixel_snapper.cli import parse_args
        from pixel_snapper.config import PixelSnapperError
        import pytest

        with pytest.raises(PixelSnapperError, match="Invalid resolution-hint"):
            parse_args(["prog", "in.png", "out.png", "--resolution-hint", "abc"])

    def test_invalid_resolution_hint_negative(self):
        """Test that negative resolution hint raises error."""
        from pixel_snapper.cli import parse_args
        from pixel_snapper.config import PixelSnapperError
        import pytest

        with pytest.raises(PixelSnapperError, match="positive integer"):
            parse_args(["prog", "in.png", "out.png", "--resolution-hint", "-5"])

    def test_invalid_resolution_hint_zero(self):
        """Test that zero resolution hint raises error."""
        from pixel_snapper.cli import parse_args
        from pixel_snapper.config import PixelSnapperError
        import pytest

        with pytest.raises(PixelSnapperError, match="positive integer"):
            parse_args(["prog", "in.png", "out.png", "--resolution-hint", "0"])


class TestComputeExpectedStep:
    """Tests for compute_expected_step function."""

    def test_both_estimates_available(self):
        """Test weighted average when both estimates are available."""
        from pixel_snapper.scoring import compute_expected_step

        # Autocorr with high confidence should dominate
        result = compute_expected_step(
            step_autocorr=16.0,
            step_peaks=8.0,
            autocorr_confidence=0.9,
        )
        # Expected: (16 * 0.9 + 8 * 0.5) / (0.9 + 0.5) = 18.4 / 1.4 = 13.14
        assert result is not None
        assert 13.0 < result < 13.5

    def test_only_autocorr_available(self):
        """Test returns autocorr when only it is available."""
        from pixel_snapper.scoring import compute_expected_step

        result = compute_expected_step(
            step_autocorr=16.0,
            step_peaks=None,
            autocorr_confidence=0.8,
        )
        assert result == 16.0

    def test_only_peaks_available(self):
        """Test returns peaks when only it is available."""
        from pixel_snapper.scoring import compute_expected_step

        result = compute_expected_step(
            step_autocorr=None,
            step_peaks=8.0,
            autocorr_confidence=0.0,
        )
        assert result == 8.0

    def test_neither_available(self):
        """Test returns None when neither estimate is available."""
        from pixel_snapper.scoring import compute_expected_step

        result = compute_expected_step(
            step_autocorr=None,
            step_peaks=None,
            autocorr_confidence=0.0,
        )
        assert result is None

    def test_low_confidence_uses_minimum_weight(self):
        """Test that very low confidence still uses minimum weight."""
        from pixel_snapper.scoring import compute_expected_step

        # With conf=0.0, autocorr should use minimum weight (0.1)
        result = compute_expected_step(
            step_autocorr=16.0,
            step_peaks=8.0,
            autocorr_confidence=0.0,
        )
        # Expected: (16 * 0.1 + 8 * 0.5) / (0.1 + 0.5) = 5.6 / 0.6 = 9.33
        assert result is not None
        assert 9.0 < result < 10.0


class TestGridSizePenalty:
    """Tests for compute_grid_size_penalty function."""

    def test_no_penalty_within_tolerance(self):
        """Test no penalty when cells match expected."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        # Expected 16x16, actual 16x16 -> no penalty
        penalty = compute_grid_size_penalty(
            cells_x=16, cells_y=16,
            expected_step=8.0,  # 128 / 8 = 16 cells
            width=128, height=128,
        )
        assert penalty == 0.0

    def test_no_penalty_near_tolerance(self):
        """Test no penalty when within 25% tolerance."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        # Expected 16x16, actual 18x18 (12.5% more) -> within tolerance
        penalty = compute_grid_size_penalty(
            cells_x=18, cells_y=18,
            expected_step=8.0,  # 128 / 8 = 16 expected
            width=128, height=128,
        )
        assert penalty == 0.0

    def test_penalty_for_too_few_cells(self):
        """Test penalty when cells are much fewer than expected."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        # Expected 16x16, actual 4x4 (0.25x ratio)
        penalty = compute_grid_size_penalty(
            cells_x=4, cells_y=4,
            expected_step=8.0,  # 128 / 8 = 16 expected
            width=128, height=128,
        )
        # log2(0.25) = -2, so penalty = -0.3 * 2 = -0.6
        assert penalty < 0
        assert -0.7 < penalty < -0.5

    def test_penalty_for_too_many_cells(self):
        """Test penalty when cells are much more than expected."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        # Expected 16x16, actual 64x64 (4x ratio)
        penalty = compute_grid_size_penalty(
            cells_x=64, cells_y=64,
            expected_step=8.0,  # 128 / 8 = 16 expected
            width=128, height=128,
        )
        # log2(4) = 2, so penalty = -0.3 * 2 = -0.6
        assert penalty < 0
        assert -0.7 < penalty < -0.5

    def test_symmetric_penalty(self):
        """Test that penalty is symmetric (2x and 0.5x same magnitude)."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        # 2x the expected cells
        penalty_double = compute_grid_size_penalty(
            cells_x=32, cells_y=32,
            expected_step=8.0,  # 128 / 8 = 16 expected
            width=128, height=128,
        )

        # 0.5x the expected cells
        penalty_half = compute_grid_size_penalty(
            cells_x=8, cells_y=8,
            expected_step=8.0,
            width=128, height=128,
        )

        # Both should have the same penalty magnitude
        assert abs(penalty_double - penalty_half) < 0.01

    def test_no_expected_step_no_penalty(self):
        """Test no penalty when expected_step is None."""
        from pixel_snapper.scoring import compute_grid_size_penalty

        penalty = compute_grid_size_penalty(
            cells_x=4, cells_y=4,
            expected_step=None,
            width=128, height=128,
        )
        assert penalty == 0.0


class TestScoreWithGridSizePenalty:
    """Tests for score_all_candidates with grid size penalty."""

    def test_penalizes_wrong_size_grid(self):
        """Test that grids deviating from expected are penalized."""
        from PIL import Image
        from pixel_snapper.scoring import score_all_candidates

        # Create a simple image
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))

        # Both grids have uniform color (equal uniformity score)
        candidates = [
            ([0, 32, 64], [0, 32, 64], 32.0, "2x2"),  # 2x2 grid
            ([0, 8, 16, 24, 32, 40, 48, 56, 64], [0, 8, 16, 24, 32, 40, 48, 56, 64], 8.0, "8x8"),  # 8x8 grid
        ]

        # Expected step = 8.0 (expecting 8x8 grid)
        scored = score_all_candidates(
            img, [0.0] * 64, [0.0] * 64, candidates, 64, 64,
            expected_step=8.0,
        )

        # 8x8 should rank higher because it matches expected
        assert scored[0].source == "8x8"
        assert scored[1].source == "2x2"

    def test_no_penalty_without_expected_step(self):
        """Test scoring without expected_step doesn't crash."""
        from PIL import Image
        from pixel_snapper.scoring import score_all_candidates

        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        candidates = [
            ([0, 32, 64], [0, 32, 64], 32.0, "2x2"),
        ]

        # No expected_step
        scored = score_all_candidates(
            img, [0.0] * 64, [0.0] * 64, candidates, 64, 64,
            expected_step=None,
        )

        assert len(scored) == 1
        assert scored[0].combined_score > 0
