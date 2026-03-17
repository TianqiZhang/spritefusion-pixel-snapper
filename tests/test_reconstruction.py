"""Tests for reconstruction-based grid detection."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from pixel_snapper import Config
from pixel_snapper.reconstruction import (
    _build_integral_images,
    _compute_reconstruction_error,
    _coarse_grid_search,
    _local_grid_correction,
    _make_uniform_cuts,
    _rect_query,
    _refine_grid_nelder_mead,
    _rgb_to_approx_lab,
    generate_reconstruction_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_image(
    cell_size: int,
    cells_x: int,
    cells_y: int,
    offset_x: int = 0,
    offset_y: int = 0,
    seed: int = 42,
) -> Image.Image:
    """Create a synthetic pixel art image with known grid parameters.

    Args:
        cell_size: Size of each grid cell in pixels.
        cells_x: Number of cells horizontally.
        cells_y: Number of cells vertically.
        offset_x: Horizontal margin/offset.
        offset_y: Vertical margin/offset.
        seed: Random seed for color generation.

    Returns:
        RGBA image with a perfect grid pattern.
    """
    width = cells_x * cell_size + offset_x
    height = cells_y * cell_size + offset_y
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # Fully opaque

    rng = np.random.RandomState(seed)

    for j in range(cells_y):
        for i in range(cells_x):
            color = rng.randint(30, 230, size=3)
            y0 = offset_y + j * cell_size
            y1 = y0 + cell_size
            x0 = offset_x + i * cell_size
            x1 = x0 + cell_size
            arr[y0:y1, x0:x1, :3] = color

    return Image.fromarray(arr, "RGBA")


def _make_grid_image_with_noise(
    cell_size: int,
    cells_x: int,
    cells_y: int,
    noise_level: float = 10.0,
    seed: int = 42,
) -> Image.Image:
    """Create a pixel art image with per-pixel noise simulating anti-aliasing.

    Args:
        cell_size: Cell size in pixels.
        cells_x: Horizontal cell count.
        cells_y: Vertical cell count.
        noise_level: Standard deviation of Gaussian noise to add.
        seed: Random seed.

    Returns:
        Noisy grid image.
    """
    img = _make_grid_image(cell_size, cells_x, cells_y, seed=seed)
    arr = np.array(img).astype(np.float64)
    rng = np.random.RandomState(seed + 1)
    noise = rng.normal(0, noise_level, arr[:, :, :3].shape)
    arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# Test integral images
# ---------------------------------------------------------------------------

class TestIntegralImages:
    def test_full_image_sum(self):
        """Integral image query over entire image should equal np.sum."""
        arr = np.array([
            [[10.0, 20.0, 30.0],
             [40.0, 50.0, 60.0]],
            [[70.0, 80.0, 90.0],
             [100.0, 110.0, 120.0]],
        ])
        mask = np.ones((2, 2), dtype=bool)
        isum, isq, icount = _build_integral_images(arr, mask)

        total_sum = _rect_query(isum, 0, 0, 2, 2)
        assert np.allclose(total_sum, arr.sum(axis=(0, 1)))

        total_count = _rect_query(icount, 0, 0, 2, 2)
        assert total_count == 4.0

    def test_sub_rect_sum(self):
        """Sub-rectangle query should match direct slicing."""
        arr = np.arange(48).reshape(4, 4, 3).astype(np.float64)
        mask = np.ones((4, 4), dtype=bool)
        isum, _, icount = _build_integral_images(arr, mask)

        # Query top-left 2x2
        result = _rect_query(isum, 0, 0, 2, 2)
        expected = arr[0:2, 0:2].sum(axis=(0, 1))
        assert np.allclose(result, expected)

        # Query bottom-right 2x2
        result = _rect_query(isum, 2, 2, 4, 4)
        expected = arr[2:4, 2:4].sum(axis=(0, 1))
        assert np.allclose(result, expected)

    def test_mask_excludes_transparent(self):
        """Masked pixels should not contribute to sums."""
        arr = np.ones((4, 4, 3), dtype=np.float64) * 100.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[0:2, 0:2] = True  # Only top-left 2x2 is opaque

        isum, _, icount = _build_integral_images(arr, mask)

        # Full image query should only count masked pixels
        total_count = _rect_query(icount, 0, 0, 4, 4)
        assert total_count == 4.0

        total_sum = _rect_query(isum, 0, 0, 4, 4)
        assert np.allclose(total_sum, np.array([400.0, 400.0, 400.0]))

    def test_single_pixel(self):
        """Single-pixel query should return that pixel's values."""
        arr = np.array([[[5.0, 10.0, 15.0]]]).astype(np.float64)
        mask = np.ones((1, 1), dtype=bool)
        isum, isq, icount = _build_integral_images(arr, mask)

        s = _rect_query(isum, 0, 0, 1, 1)
        assert np.allclose(s, [5.0, 10.0, 15.0])

        sq = _rect_query(isq, 0, 0, 1, 1)
        assert np.allclose(sq, [25.0, 100.0, 225.0])


# ---------------------------------------------------------------------------
# Test reconstruction error
# ---------------------------------------------------------------------------

class TestReconstructionError:
    def test_perfect_grid_has_minimal_error(self):
        """A grid with uniform cells should have only the parsimony term as error."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        col_cuts = [i * 8 for i in range(5)]
        row_cuts = [i * 8 for i in range(5)]

        error = _compute_reconstruction_error(
            isum, isq, icount, col_cuts, row_cuts, 32, 32
        )
        # Error should be small (only parsimony term, no within-cell variance)
        # A wrong grid should produce much higher error
        wrong_col = [0, 4, 12, 20, 28, 32]
        wrong_row = [0, 4, 12, 20, 28, 32]
        error_wrong = _compute_reconstruction_error(
            isum, isq, icount, wrong_col, wrong_row, 32, 32
        )
        assert error < error_wrong

    def test_wrong_grid_has_higher_error(self):
        """A misaligned grid should produce higher error than the correct one."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        # Correct grid
        correct_col = [0, 8, 16, 24, 32]
        correct_row = [0, 8, 16, 24, 32]
        error_correct = _compute_reconstruction_error(
            isum, isq, icount, correct_col, correct_row, 32, 32
        )

        # Wrong grid (shifted by 4)
        wrong_col = [0, 4, 12, 20, 28, 32]
        wrong_row = [0, 4, 12, 20, 28, 32]
        error_wrong = _compute_reconstruction_error(
            isum, isq, icount, wrong_col, wrong_row, 32, 32
        )

        assert error_correct < error_wrong

    def test_finer_wrong_grid_has_error(self):
        """A finer grid that doesn't align should have higher error than correct."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        correct_col = [0, 8, 16, 24, 32]
        correct_row = [0, 8, 16, 24, 32]
        error_correct = _compute_reconstruction_error(
            isum, isq, icount, correct_col, correct_row, 32, 32
        )

        # Wrong step size (step=5 doesn't align with cell boundaries)
        wrong_col = [0, 5, 10, 15, 20, 25, 30, 32]
        wrong_row = [0, 5, 10, 15, 20, 25, 30, 32]
        error_wrong = _compute_reconstruction_error(
            isum, isq, icount, wrong_col, wrong_row, 32, 32
        )

        assert error_correct < error_wrong

    def test_sub_multiple_step_has_higher_error(self):
        """Step=4 on an 8px grid should have higher error than step=8 due to parsimony."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        correct_col = [0, 8, 16, 24, 32]
        correct_row = [0, 8, 16, 24, 32]
        error_correct = _compute_reconstruction_error(
            isum, isq, icount, correct_col, correct_row, 32, 32
        )

        # Sub-multiple: step=4 (zero within-cell variance but more cells)
        fine_col = [i * 4 for i in range(9)]
        fine_row = [i * 4 for i in range(9)]
        error_fine = _compute_reconstruction_error(
            isum, isq, icount, fine_col, fine_row, 32, 32
        )

        assert error_correct < error_fine


# ---------------------------------------------------------------------------
# Test uniform cuts
# ---------------------------------------------------------------------------

class TestMakeUniformCuts:
    def test_basic_cuts(self):
        cuts = _make_uniform_cuts(8.0, 0.0, 32)
        assert cuts == [0, 8, 16, 24, 32]

    def test_with_offset(self):
        cuts = _make_uniform_cuts(8.0, 3.0, 35)
        assert cuts[0] == 0
        assert cuts[-1] == 35
        assert 3 in cuts
        assert 11 in cuts

    def test_includes_boundaries(self):
        cuts = _make_uniform_cuts(10.0, 0.0, 25)
        assert cuts[0] == 0
        assert cuts[-1] == 25


# ---------------------------------------------------------------------------
# Test coarse grid search
# ---------------------------------------------------------------------------

class TestCoarseGridSearch:
    def test_finds_correct_step_size(self):
        """Coarse search should identify the correct cell size."""
        img = _make_grid_image(8, 8, 8)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        results = _coarse_grid_search(
            isum, isq, icount, 64, 64,
            min_step=4, max_step=20,
        )

        assert len(results) > 0
        best_error, best_step, _, _ = results[0]
        assert best_step == pytest.approx(8.0, abs=0.5)

    def test_finds_correct_step_with_offset(self):
        """Should find correct step even with non-zero offset."""
        img = _make_grid_image(10, 5, 5, offset_x=3, offset_y=3)
        arr = np.array(img, dtype=np.uint8)
        w, h = img.size
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        results = _coarse_grid_search(
            isum, isq, icount, w, h,
            min_step=6, max_step=16,
        )

        best_error, best_step, _, _ = results[0]
        assert best_step == pytest.approx(10.0, abs=1.0)

    def test_respects_resolution_hint(self):
        """Should filter out step sizes that produce too many cells."""
        img = _make_grid_image(8, 8, 8)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        results = _coarse_grid_search(
            isum, isq, icount, 64, 64,
            min_step=4, max_step=20,
            resolution_hint=4,  # Only allow <=4 cells on long axis
        )

        for _, step, _, _ in results:
            cells = 64 / step
            assert cells <= 4.5

    def test_noisy_image_still_finds_step(self):
        """Should find correct step size even with noise."""
        img = _make_grid_image_with_noise(8, 8, 8, noise_level=15.0)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        results = _coarse_grid_search(
            isum, isq, icount, 64, 64,
            min_step=4, max_step=20,
        )

        best_step = results[0][1]
        assert best_step == pytest.approx(8.0, abs=0.5)


# ---------------------------------------------------------------------------
# Test refinement
# ---------------------------------------------------------------------------

class TestRefinement:
    def test_refine_improves_or_maintains_error(self):
        """Refinement should not make the error worse."""
        img = _make_grid_image(8, 8, 8)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        # Start slightly off
        initial_step = 7.5
        initial_ox, initial_oy = 1.0, 1.0

        col_cuts = _make_uniform_cuts(initial_step, initial_ox, 64)
        row_cuts = _make_uniform_cuts(initial_step, initial_oy, 64)
        initial_error = _compute_reconstruction_error(
            isum, isq, icount, col_cuts, row_cuts, 64, 64,
        )

        refined_error, r_step, r_ox, r_oy = _refine_grid_nelder_mead(
            isum, isq, icount, 64, 64,
            initial_step, initial_ox, initial_oy,
        )

        assert refined_error <= initial_error + 1e-6

    def test_refine_converges_to_correct_step(self):
        """Starting near the correct step, refinement should converge."""
        img = _make_grid_image(10, 6, 6)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        refined_error, r_step, r_ox, r_oy = _refine_grid_nelder_mead(
            isum, isq, icount, 60, 60,
            9.5, 0.5, 0.5,
        )

        assert r_step == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------------------
# Test local correction
# ---------------------------------------------------------------------------

class TestLocalCorrection:
    def test_corrects_shifted_boundary(self):
        """Local correction should fix a slightly misaligned cut."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        # Correct cuts
        correct_col = [0, 8, 16, 24, 32]
        row_cuts = [0, 8, 16, 24, 32]

        # Shift middle cut by 2 pixels
        shifted_col = [0, 8, 14, 24, 32]  # 16 -> 14

        adjusted = _local_grid_correction(
            isum, isq, icount,
            shifted_col, 32,
            search_radius=3,
            is_cols=True,
            other_cuts=row_cuts,
        )

        # Should correct back toward 16
        assert adjusted[2] == 16

    def test_preserves_correct_boundary(self):
        """Local correction should not move already-correct cuts."""
        img = _make_grid_image(8, 4, 4)
        arr = np.array(img, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        mask = arr[:, :, 3] > 0
        isum, isq, icount = _build_integral_images(lab, mask)

        correct_col = [0, 8, 16, 24, 32]
        row_cuts = [0, 8, 16, 24, 32]

        adjusted = _local_grid_correction(
            isum, isq, icount,
            correct_col[:], 32,
            search_radius=2,
            is_cols=True,
            other_cuts=row_cuts,
        )

        assert adjusted == correct_col


# ---------------------------------------------------------------------------
# Test full candidate generation
# ---------------------------------------------------------------------------

class TestGenerateReconstructionCandidates:
    def test_returns_valid_candidates(self):
        """Should return candidates in the correct format."""
        img = _make_grid_image(8, 8, 8)
        config = Config()

        candidates = generate_reconstruction_candidates(img, 64, 64, config)

        assert len(candidates) > 0
        for col_cuts, row_cuts, step, source in candidates:
            assert isinstance(col_cuts, list)
            assert isinstance(row_cuts, list)
            assert col_cuts[0] == 0
            assert row_cuts[0] == 0
            assert col_cuts[-1] == 64
            assert row_cuts[-1] == 64
            assert isinstance(step, float)
            assert "recon" in source

    def test_best_candidate_has_correct_step(self):
        """Best candidate should match the actual grid cell size."""
        img = _make_grid_image(10, 5, 5)
        config = Config()

        candidates = generate_reconstruction_candidates(img, 50, 50, config)

        assert len(candidates) > 0
        # First candidate should have step ≈ 10
        _, _, step, _ = candidates[0]
        assert step == pytest.approx(10.0, abs=1.0)

    def test_correct_cell_count(self):
        """Best candidate should produce the correct number of cells."""
        img = _make_grid_image(8, 8, 8)
        config = Config()

        candidates = generate_reconstruction_candidates(img, 64, 64, config)

        col_cuts, row_cuts, _, _ = candidates[0]
        cells_x = len(col_cuts) - 1
        cells_y = len(row_cuts) - 1
        assert cells_x == 8
        assert cells_y == 8

    def test_solid_color_image(self):
        """Should handle solid color images without crashing."""
        img = Image.new("RGBA", (32, 32), (128, 64, 32, 255))
        config = Config()

        candidates = generate_reconstruction_candidates(img, 32, 32, config)
        # Should return some candidates (all step sizes are equally good)
        assert isinstance(candidates, list)

    def test_transparent_image(self):
        """Should handle partially transparent images."""
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[8:24, 8:24] = [255, 128, 64, 255]
        img = Image.fromarray(arr, "RGBA")
        config = Config()

        candidates = generate_reconstruction_candidates(img, 32, 32, config)
        assert isinstance(candidates, list)

    def test_small_image(self):
        """Should return empty list for very small images."""
        img = Image.new("RGBA", (3, 3), (255, 0, 0, 255))
        config = Config()

        candidates = generate_reconstruction_candidates(img, 3, 3, config)
        assert candidates == []

    def test_respects_resolution_hint(self):
        """Should respect resolution_hint config."""
        img = _make_grid_image(8, 8, 8)
        config = Config(resolution_hint=4)

        candidates = generate_reconstruction_candidates(img, 64, 64, config)

        for col_cuts, row_cuts, _, _ in candidates:
            cells_x = len(col_cuts) - 1
            cells_y = len(row_cuts) - 1
            assert max(cells_x, cells_y) <= 4

    def test_noisy_image(self):
        """Should still find correct grid for noisy images."""
        img = _make_grid_image_with_noise(8, 8, 8, noise_level=20.0)
        config = Config()

        candidates = generate_reconstruction_candidates(img, 64, 64, config)

        assert len(candidates) > 0
        _, _, step, _ = candidates[0]
        assert step == pytest.approx(8.0, abs=1.0)

    def test_non_square_image(self):
        """Should handle non-square images correctly."""
        img = _make_grid_image(8, 10, 6)  # 80x48
        config = Config()

        candidates = generate_reconstruction_candidates(img, 80, 48, config)

        assert len(candidates) > 0
        col_cuts, row_cuts, step, _ = candidates[0]
        assert col_cuts[-1] == 80
        assert row_cuts[-1] == 48
        assert step == pytest.approx(8.0, abs=1.0)


# ---------------------------------------------------------------------------
# Test pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_reconstruction_candidates_appear_in_pipeline(self):
        """Reconstruction candidates should be included when running the full pipeline."""
        from pixel_snapper import process_image_bytes_with_grid

        img = _make_grid_image(8, 8, 8)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        config = Config(use_reconstruction=True, use_uniformity_scoring=True)
        result = process_image_bytes_with_grid(img_bytes, config)

        # Check that some recon candidates were scored
        recon_candidates = [
            c for c in result.scored_candidates
            if c.source.startswith("recon")
        ]
        assert len(recon_candidates) > 0

    def test_pipeline_produces_valid_output_with_reconstruction(self):
        """Full pipeline with reconstruction should produce valid output."""
        from pixel_snapper import process_image_bytes_with_grid

        img = _make_grid_image(10, 6, 6)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        config = Config(use_reconstruction=True)
        result = process_image_bytes_with_grid(img_bytes, config)

        output_img = Image.open(io.BytesIO(result.output_bytes))
        assert output_img.mode == "RGBA"
        assert output_img.size[0] > 0
        assert output_img.size[1] > 0

    def test_pipeline_works_without_reconstruction(self):
        """Pipeline should still work with reconstruction disabled."""
        from pixel_snapper import process_image_bytes_with_grid

        img = _make_grid_image(8, 8, 8)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        config = Config(use_reconstruction=False)
        result = process_image_bytes_with_grid(img_bytes, config)

        recon_candidates = [
            c for c in result.scored_candidates
            if c.source.startswith("recon")
        ]
        assert len(recon_candidates) == 0

        output_img = Image.open(io.BytesIO(result.output_bytes))
        assert output_img.size[0] > 0


class TestApproxLabConversion:
    def test_black_is_zero_luminance(self):
        arr = np.zeros((1, 1, 3), dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        assert lab[0, 0, 0] == pytest.approx(0.0)

    def test_white_has_high_luminance(self):
        arr = np.full((1, 1, 3), 255, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        assert lab[0, 0, 0] > 200.0

    def test_gray_has_zero_chroma(self):
        arr = np.full((1, 1, 3), 128, dtype=np.uint8)
        lab = _rgb_to_approx_lab(arr)
        # a and b channels should be ~0 for gray
        assert lab[0, 0, 1] == pytest.approx(0.0, abs=0.5)
        assert lab[0, 0, 2] == pytest.approx(0.0, abs=0.5)

    def test_handles_4_channel_input(self):
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[:, :, :3] = 100
        arr[:, :, 3] = 255
        lab = _rgb_to_approx_lab(arr)
        assert lab.shape == (2, 2, 3)
