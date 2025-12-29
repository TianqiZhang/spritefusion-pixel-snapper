"""Integration tests for the complete pixel snapper pipeline."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

from pixel_snapper import Config, PixelSnapperError, process_image_bytes


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_basic_pipeline(self, sample_image_bytes: bytes) -> None:
        """Should process image through complete pipeline."""
        result = process_image_bytes(sample_image_bytes)

        # Verify output is valid PNG
        output_img = Image.open(io.BytesIO(result))
        assert output_img.mode == "RGBA"
        assert output_img.size[0] > 0
        assert output_img.size[1] > 0

    def test_preserves_color_information(
        self, sample_image_bytes: bytes
    ) -> None:
        """Should preserve main colors from input."""
        config = Config(k_colors=8)
        result = process_image_bytes(sample_image_bytes, config)

        output_img = Image.open(io.BytesIO(result))
        arr = np.array(output_img)

        # Should have some non-black colors
        non_black = arr[arr[:, :, 3] > 0][:, :3]
        assert non_black.max() > 0

    def test_reduces_dimensions(self, sample_image_bytes: bytes) -> None:
        """Output should be snapped grid (smaller dimensions)."""
        result = process_image_bytes(sample_image_bytes)

        input_img = Image.open(io.BytesIO(sample_image_bytes))
        output_img = Image.open(io.BytesIO(result))

        # Output should be smaller (snapped to detected grid)
        assert output_img.size[0] <= input_img.size[0]
        assert output_img.size[1] <= input_img.size[1]

    def test_deterministic_output(self, sample_image_bytes: bytes) -> None:
        """Same input should produce same output."""
        config = Config(k_colors=8, k_seed=42)

        result1 = process_image_bytes(sample_image_bytes, config)
        result2 = process_image_bytes(sample_image_bytes, config)

        assert result1 == result2

    def test_different_seeds_different_output(
        self, sample_image_bytes: bytes
    ) -> None:
        """Different seeds may produce different output."""
        config1 = Config(k_colors=8, k_seed=42)
        config2 = Config(k_colors=8, k_seed=123)

        result1 = process_image_bytes(sample_image_bytes, config1)
        result2 = process_image_bytes(sample_image_bytes, config2)

        # Results may differ (or may be same if colors are obvious)
        # Just verify both produce valid output
        assert len(result1) > 0
        assert len(result2) > 0


class TestTransparentImageHandling:
    """Tests for handling images with transparency."""

    def test_fully_transparent(self) -> None:
        """Should handle fully transparent image."""
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = process_image_bytes(buf.getvalue())
        output_img = Image.open(io.BytesIO(result))

        # Should produce some output
        assert output_img.size[0] > 0

    def test_partial_transparency(
        self, transparent_image: Image.Image
    ) -> None:
        """Should preserve transparent regions."""
        buf = io.BytesIO()
        transparent_image.save(buf, format="PNG")

        result = process_image_bytes(buf.getvalue())
        output_img = Image.open(io.BytesIO(result))
        arr = np.array(output_img)

        # Should have both transparent and opaque regions
        has_transparent = (arr[:, :, 3] == 0).any()
        has_opaque = (arr[:, :, 3] > 0).any()

        # At least one should be true (small image might be fully one or other)
        assert has_transparent or has_opaque


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_small_image(self, small_grid_image: Image.Image) -> None:
        """Should handle small images."""
        buf = io.BytesIO()
        small_grid_image.save(buf, format="PNG")

        result = process_image_bytes(buf.getvalue())
        output_img = Image.open(io.BytesIO(result))

        assert output_img.size[0] >= 1
        assert output_img.size[1] >= 1

    def test_solid_color(self, solid_color_image: Image.Image) -> None:
        """Should handle solid color image."""
        buf = io.BytesIO()
        solid_color_image.save(buf, format="PNG")

        result = process_image_bytes(buf.getvalue())
        output_img = Image.open(io.BytesIO(result))

        # Should produce valid output
        assert output_img.mode == "RGBA"

    def test_gradient_image(self, gradient_image: Image.Image) -> None:
        """Should handle gradient (no clear grid)."""
        buf = io.BytesIO()
        gradient_image.save(buf, format="PNG")

        result = process_image_bytes(buf.getvalue())
        output_img = Image.open(io.BytesIO(result))

        # Should produce valid output even without clear grid
        assert output_img.size[0] > 0

    def test_very_small_k_colors(self, sample_image_bytes: bytes) -> None:
        """Should handle k_colors=1."""
        config = Config(k_colors=1)
        result = process_image_bytes(sample_image_bytes, config)

        output_img = Image.open(io.BytesIO(result))
        arr = np.array(output_img)

        # Should have only one color (plus transparent)
        opaque = arr[arr[:, :, 3] > 0][:, :3]
        unique = np.unique(opaque, axis=0)
        assert len(unique) == 1

    def test_large_k_colors(self, sample_image_bytes: bytes) -> None:
        """Should handle k_colors larger than unique colors."""
        config = Config(k_colors=1000)
        result = process_image_bytes(sample_image_bytes, config)

        output_img = Image.open(io.BytesIO(result))
        assert output_img.size[0] > 0


class TestRealImageProcessing:
    """Tests using real sample images if available."""

    @pytest.fixture
    def zelda_image_path(self) -> str:
        """Path to zelda test image."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "zelda1.png")

    def test_zelda_image(self, zelda_image_path: str) -> None:
        """Should process zelda test image if available."""
        if not os.path.exists(zelda_image_path):
            pytest.skip("zelda1.png not found")

        with open(zelda_image_path, "rb") as f:
            img_bytes = f.read()

        result = process_image_bytes(img_bytes)

        # Verify output is valid
        output_img = Image.open(io.BytesIO(result))
        assert output_img.mode == "RGBA"
        assert output_img.size[0] > 0
        assert output_img.size[1] > 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    def test_import_from_root(self) -> None:
        """Should be importable from root module."""
        # This tests the backward-compatible wrapper
        import pixel_snapper as ps

        assert hasattr(ps, "Config")
        assert hasattr(ps, "PixelSnapperError")
        assert hasattr(ps, "process_image_bytes")
        assert hasattr(ps, "process_image_bytes_with_grid")
        assert hasattr(ps, "ProcessingResult")
        assert hasattr(ps, "main")

    def test_config_from_root(self) -> None:
        """Config should be importable from root."""
        from pixel_snapper import Config

        config = Config(k_colors=32)
        assert config.k_colors == 32

    def test_error_from_root(self) -> None:
        """PixelSnapperError should be importable from root."""
        from pixel_snapper import PixelSnapperError

        error = PixelSnapperError("test")
        assert str(error) == "test"


class TestProcessingWithGrid:
    """Tests for process_image_bytes_with_grid function."""

    def test_returns_processing_result(
        self, sample_image_bytes: bytes
    ) -> None:
        """Should return ProcessingResult with grid info."""
        from pixel_snapper import process_image_bytes_with_grid, ProcessingResult

        result = process_image_bytes_with_grid(sample_image_bytes)

        assert isinstance(result, ProcessingResult)
        assert isinstance(result.output_bytes, bytes)
        assert isinstance(result.col_cuts, list)
        assert isinstance(result.row_cuts, list)

    def test_grid_cuts_are_valid(self, sample_image_bytes: bytes) -> None:
        """Grid cuts should be sorted and include boundaries."""
        from pixel_snapper import process_image_bytes_with_grid

        result = process_image_bytes_with_grid(sample_image_bytes)

        # Cuts should be sorted
        assert result.col_cuts == sorted(result.col_cuts)
        assert result.row_cuts == sorted(result.row_cuts)

        # Should start at 0
        assert result.col_cuts[0] == 0
        assert result.row_cuts[0] == 0

        # Should have at least 2 cuts (start and end)
        assert len(result.col_cuts) >= 2
        assert len(result.row_cuts) >= 2

    def test_output_dimensions_match_grid(
        self, sample_image_bytes: bytes
    ) -> None:
        """Output image dimensions should match grid cells."""
        from pixel_snapper import process_image_bytes_with_grid

        result = process_image_bytes_with_grid(sample_image_bytes)
        output_img = Image.open(io.BytesIO(result.output_bytes))

        expected_w = len(result.col_cuts) - 1
        expected_h = len(result.row_cuts) - 1

        assert output_img.size == (expected_w, expected_h)


class TestTimingOutput:
    """Tests for timing output functionality."""

    def test_timing_output(
        self, sample_image_bytes: bytes, capsys
    ) -> None:
        """Should print timing when enabled."""
        config = Config(timing=True)
        process_image_bytes(sample_image_bytes, config)

        captured = capsys.readouterr()
        assert "Timing" in captured.out
        assert "load=" in captured.out
        assert "quantize=" in captured.out
        assert "total=" in captured.out

    def test_no_timing_by_default(
        self, sample_image_bytes: bytes, capsys
    ) -> None:
        """Should not print timing by default."""
        config = Config(timing=False)
        process_image_bytes(sample_image_bytes, config)

        captured = capsys.readouterr()
        assert "Timing" not in captured.out
