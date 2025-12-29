"""Image resampling using majority-vote pixel selection."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from .config import PixelSnapperError


def resample(
    img: Image.Image, cols: Sequence[int], rows: Sequence[int]
) -> Image.Image:
    """Resample image to grid using majority-vote color selection.

    For each grid cell, the most common color is selected.
    Ties are broken by preferring smaller packed color values.

    Args:
        img: Input RGBA image.
        cols: Column cut positions.
        rows: Row cut positions.

    Returns:
        Resampled RGBA image with dimensions (len(cols)-1, len(rows)-1).

    Raises:
        PixelSnapperError: If insufficient grid cuts.
    """
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = Image.new("RGBA", (out_w, out_h))

    arr = np.array(img, dtype=np.uint8)
    height, width = arr.shape[:2]

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue

            # Clamp coordinates to image bounds
            ys_clamped = min(max(ys, 0), height)
            ye_clamped = min(max(ye, 0), height)
            xs_clamped = min(max(xs, 0), width)
            xe_clamped = min(max(xe, 0), width)

            block = arr[ys_clamped:ye_clamped, xs_clamped:xe_clamped]
            if block.size == 0:
                continue

            # Pack RGBA into single uint32 for efficient counting
            flat = block.reshape(-1, 4).astype(np.uint32)
            packed = (
                (flat[:, 0] << 24)
                | (flat[:, 1] << 16)
                | (flat[:, 2] << 8)
                | flat[:, 3]
            )

            # Find most common color (ties broken by smallest value)
            values, counts = np.unique(packed, return_counts=True)
            max_count = counts.max()
            best_value = values[counts == max_count][0]

            # Unpack back to RGBA
            r = int((best_value >> 24) & 0xFF)
            g = int((best_value >> 16) & 0xFF)
            b = int((best_value >> 8) & 0xFF)
            a = int(best_value & 0xFF)
            final_img.putpixel((x_i, y_i), (r, g, b, a))

    return final_img
