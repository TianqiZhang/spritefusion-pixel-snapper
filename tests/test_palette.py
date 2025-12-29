"""Tests for palette module."""
from __future__ import annotations

import os
import tempfile

import pytest

from pixel_snapper.config import PixelSnapperError
from pixel_snapper.palette import Palette, load_palette, resolve_palette_path


class TestResolvePalettePath:
    """Tests for resolve_palette_path function."""

    def test_existing_file_path(self, tmp_path) -> None:
        """Should return path if file exists."""
        palette_file = tmp_path / "test.csv"
        palette_file.write_text("dummy")
        result = resolve_palette_path(str(palette_file))
        assert result == str(palette_file)

    def test_builtin_palette(self) -> None:
        """Should find built-in palettes by name."""
        # This assumes perler.csv exists in the colors directory
        try:
            path = resolve_palette_path("perler")
            assert path.endswith(".csv")
            assert os.path.exists(path)
        except PixelSnapperError:
            pytest.skip("perler palette not available")

    def test_missing_palette_error(self) -> None:
        """Should raise error for missing palette."""
        with pytest.raises(PixelSnapperError, match="Palette not found"):
            resolve_palette_path("nonexistent_palette_xyz")

    def test_adds_csv_extension(self) -> None:
        """Should add .csv extension if missing."""
        try:
            path = resolve_palette_path("perler")
            assert path.endswith(".csv")
        except PixelSnapperError:
            pytest.skip("perler palette not available")


class TestLoadPalette:
    """Tests for load_palette function."""

    def test_load_valid_palette(self, tmp_path) -> None:
        """Should load a valid palette CSV."""
        palette_file = tmp_path / "test.csv"
        # Format: id, name, symbol, R, G, B, hue, sat, light, L, a, b, source
        palette_file.write_text(
            "1,Red,R,255,0,0,0,1,0.5,53.23,80.11,67.22,test\n"
            "2,Green,G,0,255,0,120,1,0.5,87.74,-86.18,83.18,test\n"
        )
        palette = load_palette(str(palette_file))
        assert len(palette.rgb) == 2
        assert len(palette.lab) == 2
        assert palette.rgb[0] == (255, 0, 0)
        assert palette.rgb[1] == (0, 255, 0)

    def test_empty_palette_error(self, tmp_path) -> None:
        """Should raise error for empty palette."""
        palette_file = tmp_path / "empty.csv"
        palette_file.write_text("")
        with pytest.raises(PixelSnapperError, match="No colors found"):
            load_palette(str(palette_file))

    def test_invalid_row_error(self, tmp_path) -> None:
        """Should raise error for invalid row format."""
        palette_file = tmp_path / "invalid.csv"
        palette_file.write_text("1,Red,R\n")  # Too few columns
        with pytest.raises(PixelSnapperError, match="Invalid palette row"):
            load_palette(str(palette_file))

    def test_skips_empty_rows(self, tmp_path) -> None:
        """Should skip empty rows in CSV."""
        palette_file = tmp_path / "with_empty.csv"
        palette_file.write_text(
            "\n"
            "1,Red,R,255,0,0,0,1,0.5,53.23,80.11,67.22,test\n"
            "\n"
        )
        palette = load_palette(str(palette_file))
        assert len(palette.rgb) == 1


class TestPalette:
    """Tests for Palette dataclass."""

    def test_palette_structure(self) -> None:
        """Palette should hold RGB and LAB colors."""
        palette = Palette(
            rgb=[(255, 0, 0), (0, 255, 0)],
            lab=[(53.23, 80.11, 67.22), (87.74, -86.18, 83.18)],
        )
        assert len(palette.rgb) == 2
        assert len(palette.lab) == 2
        assert palette.rgb[0] == (255, 0, 0)
