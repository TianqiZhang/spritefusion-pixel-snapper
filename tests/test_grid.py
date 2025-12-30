"""Tests for grid module."""
from __future__ import annotations

import pytest

from pixel_snapper.config import Config, PixelSnapperError
from pixel_snapper.grid import (
    estimate_step_size,
    resolve_step_sizes,
    sanitize_cuts,
    snap_uniform_cuts,
    stabilize_both_axes,
    stabilize_cuts,
    walk_with_offset,
)


class TestEstimateStepSize:
    """Tests for estimate_step_size function."""

    def test_periodic_profile(self) -> None:
        """Should detect periodic pattern."""
        # Profile with peaks every 10 units
        profile = [0.0] * 100
        for i in range(0, 100, 10):
            profile[i] = 100.0

        config = Config()
        step = estimate_step_size(profile, config)

        assert step is not None
        assert abs(step - 10.0) < 2  # Should be close to 10

    def test_no_peaks(self) -> None:
        """Should return None if no peaks found."""
        profile = [1.0] * 100  # Flat profile
        config = Config()
        step = estimate_step_size(profile, config)
        assert step is None

    def test_empty_profile(self) -> None:
        """Should return None for empty profile."""
        config = Config()
        step = estimate_step_size([], config)
        assert step is None

    def test_single_peak(self) -> None:
        """Should return None for single peak."""
        profile = [0.0] * 100
        profile[50] = 100.0
        config = Config()
        step = estimate_step_size(profile, config)
        assert step is None


class TestResolveStepSizes:
    """Tests for resolve_step_sizes function."""

    def test_both_axes_detected(self) -> None:
        """Should average when both axes have similar steps."""
        config = Config()
        step_x, step_y = resolve_step_sizes(10.0, 12.0, 100, 100, config)
        assert step_x == step_y == 11.0

    def test_skewed_ratio(self) -> None:
        """Should use smaller step when ratio is too high."""
        config = Config(max_step_ratio=1.5)
        step_x, step_y = resolve_step_sizes(10.0, 20.0, 100, 100, config)
        assert step_x == step_y == 10.0

    def test_only_x_detected(self) -> None:
        """Should use X step for both when only X detected."""
        config = Config()
        step_x, step_y = resolve_step_sizes(10.0, None, 100, 100, config)
        assert step_x == step_y == 10.0

    def test_only_y_detected(self) -> None:
        """Should use Y step for both when only Y detected."""
        config = Config()
        step_x, step_y = resolve_step_sizes(None, 8.0, 100, 100, config)
        assert step_x == step_y == 8.0

    def test_neither_detected(self) -> None:
        """Should use fallback when neither detected."""
        config = Config(fallback_target_segments=64)
        step_x, step_y = resolve_step_sizes(None, None, 128, 128, config)
        assert step_x == step_y == 2.0  # 128/64


class TestWalk:
    """Tests for walk_with_offset function."""

    def test_basic_walk(self) -> None:
        """Should produce cuts at regular intervals."""
        profile = [10.0] * 100
        config = Config()
        cuts = walk_with_offset(profile, 10.0, 100, 0, config)

        assert cuts[0] == 0
        assert cuts[-1] == 100
        assert len(cuts) > 2

    def test_snaps_to_peaks(self) -> None:
        """Should snap to nearby peaks."""
        profile = [0.0] * 100
        profile[10] = 100.0  # Strong peak at 10
        profile[20] = 100.0  # Strong peak at 20

        config = Config(walker_strength_threshold=0.3)
        cuts = walk_with_offset(profile, 10.0, 30, 0, config)

        assert 10 in cuts
        assert 20 in cuts

    def test_empty_profile_error(self) -> None:
        """Should raise error for empty profile."""
        config = Config()
        with pytest.raises(PixelSnapperError, match="empty profile"):
            walk_with_offset([], 10.0, 100, 0, config)

    def test_includes_boundaries(self) -> None:
        """Should always include 0 and limit."""
        profile = [10.0] * 50
        config = Config()
        cuts = walk_with_offset(profile, 10.0, 50, 0, config)

        assert cuts[0] == 0
        assert cuts[-1] == 50


class TestSanitizeCuts:
    """Tests for sanitize_cuts function."""

    def test_adds_boundaries(self) -> None:
        """Should add 0 and limit if missing."""
        cuts = sanitize_cuts([10, 20, 30], 50)
        assert cuts[0] == 0
        assert cuts[-1] == 50

    def test_removes_duplicates(self) -> None:
        """Should remove duplicate values."""
        cuts = sanitize_cuts([0, 10, 10, 20, 50], 50)
        assert cuts.count(10) == 1

    def test_sorts_cuts(self) -> None:
        """Should sort cuts in order."""
        cuts = sanitize_cuts([30, 10, 20], 50)
        assert cuts == sorted(cuts)

    def test_clamps_to_limit(self) -> None:
        """Should clamp values exceeding limit."""
        cuts = sanitize_cuts([0, 60, 70], 50)
        assert max(cuts) == 50

    def test_zero_limit(self) -> None:
        """Should handle zero limit."""
        cuts = sanitize_cuts([0], 0)
        assert cuts == [0]


class TestSnapUniformCuts:
    """Tests for snap_uniform_cuts function."""

    def test_creates_uniform_grid(self) -> None:
        """Should create approximately uniform cuts."""
        profile = [10.0] * 100
        config = Config()
        cuts = snap_uniform_cuts(profile, 100, 10.0, config, 4)

        # Check roughly uniform spacing
        spacings = [cuts[i+1] - cuts[i] for i in range(len(cuts)-1)]
        avg_spacing = sum(spacings) / len(spacings)
        for s in spacings:
            assert abs(s - avg_spacing) < avg_spacing * 0.5

    def test_respects_min_required(self) -> None:
        """Should create at least min_required cuts."""
        profile = [10.0] * 100
        config = Config()
        cuts = snap_uniform_cuts(profile, 100, 50.0, config, 5)
        assert len(cuts) >= 5


class TestStabilizeCuts:
    """Tests for stabilize_cuts function."""

    def test_uses_sibling_step(self) -> None:
        """Should use sibling axis step when available."""
        profile = [10.0] * 100
        cuts = [0, 50, 100]  # Only 2 cells
        sibling_cuts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 10 cells
        config = Config(min_cuts_per_axis=4)

        result = stabilize_cuts(profile, cuts, 100, sibling_cuts, 100, config)
        assert len(result) > len(cuts)


class TestStabilizeBothAxes:
    """Tests for stabilize_both_axes function."""

    def test_returns_two_cut_lists(self) -> None:
        """Should return cuts for both axes."""
        profile_x = [10.0] * 100
        profile_y = [10.0] * 100
        col_cuts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        row_cuts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        config = Config()

        result_cols, result_rows = stabilize_both_axes(
            profile_x, profile_y, col_cuts, row_cuts, 100, 100, config
        )

        assert isinstance(result_cols, list)
        assert isinstance(result_rows, list)
        assert len(result_cols) >= 2
        assert len(result_rows) >= 2

    def test_fixes_skewed_ratio(self) -> None:
        """Should fix severely skewed aspect ratios."""
        profile_x = [10.0] * 100
        profile_y = [10.0] * 100
        col_cuts = [0, 50, 100]  # 2 cells (50px each)
        row_cuts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 10 cells (10px each)
        config = Config(max_step_ratio=1.8)

        result_cols, result_rows = stabilize_both_axes(
            profile_x, profile_y, col_cuts, row_cuts, 100, 100, config
        )

        # After stabilization, aspect ratio should be better
        col_step = 100 / (len(result_cols) - 1)
        row_step = 100 / (len(result_rows) - 1)
        ratio = max(col_step, row_step) / min(col_step, row_step)
        assert ratio <= config.max_step_ratio + 0.5
