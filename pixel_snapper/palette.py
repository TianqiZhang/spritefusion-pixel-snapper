"""Palette loading and resolution utilities."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

from .config import PixelSnapperError


@dataclass
class Palette:
    """A color palette with RGB and LAB values."""

    rgb: List[Tuple[int, int, int]]
    lab: List[Tuple[float, float, float]]


def resolve_palette_path(palette_name: str) -> str:
    """Resolve a palette name to its file path.

    Args:
        palette_name: Palette name or path.

    Returns:
        Absolute path to the palette CSV file.

    Raises:
        PixelSnapperError: If palette is not found.
    """
    if os.path.exists(palette_name):
        return palette_name

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    colors_dir = os.path.join(base_dir, "colors")

    candidate = palette_name
    if not candidate.lower().endswith(".csv"):
        candidate = f"{candidate}.csv"
    candidate_path = os.path.join(colors_dir, candidate)

    if os.path.exists(candidate_path):
        return candidate_path

    available: List[str] = []
    if os.path.isdir(colors_dir):
        for name in os.listdir(colors_dir):
            if name.lower().endswith(".csv"):
                available.append(os.path.splitext(name)[0])
    available.sort()
    hint = f" Available palettes: {', '.join(available)}" if available else ""
    raise PixelSnapperError(f"Palette not found: {palette_name}.{hint}")


def load_palette(palette_name: str) -> Palette:
    """Load a palette from a CSV file.

    Args:
        palette_name: Palette name or path.

    Returns:
        Palette with RGB and LAB colors.

    Raises:
        PixelSnapperError: If palette cannot be loaded.
    """
    palette_path = resolve_palette_path(palette_name)
    colors: List[Tuple[int, int, int]] = []
    lab_colors: List[Tuple[float, float, float]] = []

    with open(palette_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                r = int(row[3])
                g = int(row[4])
                b = int(row[5])
                lab_l = float(row[9])
                lab_a = float(row[10])
                lab_b = float(row[11])
            except (IndexError, ValueError) as exc:
                raise PixelSnapperError(
                    f"Invalid palette row in {palette_path}: {row}"
                ) from exc
            colors.append((r, g, b))
            lab_colors.append((lab_l, lab_a, lab_b))

    if not colors:
        raise PixelSnapperError(f"No colors found in palette: {palette_path}")
    return Palette(rgb=colors, lab=lab_colors)
