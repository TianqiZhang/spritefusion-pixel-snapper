"""Tests for bead pattern rendering."""
from __future__ import annotations

from PIL import Image

from pixel_snapper.pattern import render_bead_pattern


def test_render_bead_pattern(tmp_path) -> None:
    """Should render a bead pattern image from a tiny grid."""
    palette_file = tmp_path / "test.csv"
    palette_file.write_text(
        "1,Red,R,255,0,0,0,1,0.5,53.23,80.11,67.22,test\n"
        "2,Green,G,0,255,0,120,1,0.5,87.74,-86.18,83.18,test\n"
    )

    grid = Image.new("RGB", (2, 2))
    pixels = grid.load()
    pixels[0, 0] = (255, 0, 0)
    pixels[1, 0] = (0, 255, 0)
    pixels[0, 1] = (255, 0, 0)
    pixels[1, 1] = (0, 255, 0)

    pattern = render_bead_pattern(
        grid,
        str(palette_file),
        title="Test Pattern",
        cell_size=10,
        major_every=1,
    )

    assert pattern.size[0] > grid.size[0]
    assert pattern.size[1] > grid.size[1]
