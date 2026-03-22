"""Image resampling strategies for pixel art grid cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .color import rgb_to_lab
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


def resample_palette_aware(
    img: Image.Image,
    cols: Sequence[int],
    rows: Sequence[int],
    palette_rgb: np.ndarray,
    palette_lab: np.ndarray,
) -> Image.Image:
    """Resample by computing the mean color per cell and snapping to the
    nearest palette color in LAB space.

    This combines the perceptual fidelity of mean-color resampling with
    the constraint that output only contains valid palette colors.

    Args:
        img: Input RGBA image (already palette-quantized).
        cols: Column cut positions.
        rows: Row cut positions.
        palette_rgb: (N, 3) array of palette RGB values.
        palette_lab: (N, 3) array of palette LAB values.
    """
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = Image.new("RGBA", (out_w, out_h))

    arr = np.array(img, dtype=np.float64)
    height, width = arr.shape[:2]

    # First pass: compute alpha-weighted mean RGB per cell
    cell_means = []  # (x_i, y_i, mean_rgb, mean_alpha)
    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue

            ys_c = min(max(ys, 0), height)
            ye_c = min(max(ye, 0), height)
            xs_c = min(max(xs, 0), width)
            xe_c = min(max(xe, 0), width)

            block = arr[ys_c:ye_c, xs_c:xe_c]
            if block.size == 0:
                continue

            alpha = block[:, :, 3:4] / 255.0
            total_alpha = np.sum(alpha)
            if total_alpha < 0.01:
                continue

            weighted_rgb = block[:, :, :3] * alpha
            mean_rgb = np.sum(weighted_rgb.reshape(-1, 3), axis=0) / total_alpha
            mean_alpha = int(np.clip(np.mean(block[:, :, 3]), 0, 255))
            cell_means.append((x_i, y_i, mean_rgb, mean_alpha))

    if not cell_means:
        return final_img

    # Batch LAB conversion and palette lookup
    all_rgb = np.array([cm[2] for cm in cell_means])
    all_lab = rgb_to_lab(all_rgb)
    # Squared distance is sufficient for argmin
    dists = np.sum((all_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    best_indices = np.argmin(dists, axis=1)

    for i, (x_i, y_i, _, mean_alpha) in enumerate(cell_means):
        idx = best_indices[i]
        r = int(palette_rgb[idx, 0])
        g = int(palette_rgb[idx, 1])
        b = int(palette_rgb[idx, 2])
        final_img.putpixel((x_i, y_i), (r, g, b, mean_alpha))

    return final_img


def resample_mean(
    img: Image.Image, cols: Sequence[int], rows: Sequence[int]
) -> Image.Image:
    """Resample image to grid using alpha-weighted mean color per cell.

    Averages all pixel colors within each cell, weighted by alpha.
    Produces smoother results that better preserve the overall visual
    impression, but may create blended colors not in the original palette.
    """
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = Image.new("RGBA", (out_w, out_h))

    arr = np.array(img, dtype=np.float64)
    height, width = arr.shape[:2]

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue

            ys_c = min(max(ys, 0), height)
            ye_c = min(max(ye, 0), height)
            xs_c = min(max(xs, 0), width)
            xe_c = min(max(xe, 0), width)

            block = arr[ys_c:ye_c, xs_c:xe_c]
            if block.size == 0:
                continue

            alpha = block[:, :, 3:4] / 255.0
            if np.sum(alpha) < 0.01:
                continue

            weighted_rgb = block[:, :, :3] * alpha
            mean_rgb = np.sum(weighted_rgb.reshape(-1, 3), axis=0) / np.sum(alpha)
            mean_alpha = np.mean(block[:, :, 3])

            r = int(np.clip(mean_rgb[0], 0, 255))
            g = int(np.clip(mean_rgb[1], 0, 255))
            b = int(np.clip(mean_rgb[2], 0, 255))
            a = int(np.clip(mean_alpha, 0, 255))
            final_img.putpixel((x_i, y_i), (r, g, b, a))

    return final_img


def resample_center(
    img: Image.Image, cols: Sequence[int], rows: Sequence[int]
) -> Image.Image:
    """Resample by picking the center pixel of each cell.

    In correctly-aligned pixel art grids, the center pixel is the
    original sprite pixel. Falls back to majority vote if center
    pixel is transparent.
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

            ys_c = min(max(ys, 0), height)
            ye_c = min(max(ye, 0), height)
            xs_c = min(max(xs, 0), width)
            xe_c = min(max(xe, 0), width)

            if ye_c <= ys_c or xe_c <= xs_c:
                continue

            cy = min((ys_c + ye_c) // 2, height - 1)
            cx = min((xs_c + xe_c) // 2, width - 1)
            pixel = arr[cy, cx]

            if pixel[3] == 0:
                block = arr[ys_c:ye_c, xs_c:xe_c]
                if block.size == 0:
                    continue
                flat = block.reshape(-1, 4).astype(np.uint32)
                packed = (
                    (flat[:, 0] << 24) | (flat[:, 1] << 16)
                    | (flat[:, 2] << 8) | flat[:, 3]
                )
                values, counts = np.unique(packed, return_counts=True)
                best = values[counts == counts.max()][0]
                pixel = np.array([
                    (best >> 24) & 0xFF, (best >> 16) & 0xFF,
                    (best >> 8) & 0xFF, best & 0xFF,
                ], dtype=np.uint8)

            final_img.putpixel((x_i, y_i), tuple(int(v) for v in pixel))

    return final_img


@dataclass
class FidelityResult:
    """Per-cell color fidelity metrics."""
    mean_delta_e: float
    median_delta_e: float
    max_delta_e: float
    p90_delta_e: float
    num_cells: int


def compute_fidelity(
    original: Image.Image,
    resampled: Image.Image,
    cols: Sequence[int],
    rows: Sequence[int],
    palette_rgb: np.ndarray | None = None,
    palette_lab: np.ndarray | None = None,
) -> FidelityResult:
    """Measure perceptual fidelity of resampled output vs Lanczos reference.

    Downscales the original image using Lanczos interpolation to the grid
    dimensions, then computes Delta-E (CIE76) between the resampled output
    and the reference for each cell.

    When palette_rgb/palette_lab are provided, the reference is also snapped
    to the palette so we measure end-to-end quality (both outputs constrained
    to the same palette).
    """
    grid_w = len(cols) - 1
    grid_h = len(rows) - 1

    if grid_w < 1 or grid_h < 1:
        return FidelityResult(
            mean_delta_e=float("inf"), median_delta_e=float("inf"),
            max_delta_e=float("inf"), p90_delta_e=float("inf"), num_cells=0,
        )

    reference = original.convert("RGBA").resize(
        (grid_w, grid_h), Image.LANCZOS
    )

    ref_arr = np.array(reference, dtype=np.float64)

    # If palette is provided, snap the reference to palette too (vectorized)
    if palette_lab is not None and palette_rgb is not None:
        opaque_mask = ref_arr[:, :, 3] >= 10
        if np.any(opaque_mask):
            rgb_flat = ref_arr[:, :, :3][opaque_mask]
            lab_flat = rgb_to_lab(rgb_flat)
            # Squared distance is sufficient for argmin
            dists = np.sum(
                (lab_flat[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2
            )
            nearest = np.argmin(dists, axis=1)
            ref_arr[:, :, :3][opaque_mask] = palette_rgb[nearest]

    res_arr = np.array(resampled, dtype=np.float64)

    h = min(ref_arr.shape[0], res_arr.shape[0])
    w = min(ref_arr.shape[1], res_arr.shape[1])
    ref_arr = ref_arr[:h, :w]
    res_arr = res_arr[:h, :w]

    ref_alpha = ref_arr[:, :, 3]
    res_alpha = res_arr[:, :, 3]
    opaque = (ref_alpha > 10) & (res_alpha > 10)

    if not np.any(opaque):
        return FidelityResult(
            mean_delta_e=float("inf"), median_delta_e=float("inf"),
            max_delta_e=float("inf"), p90_delta_e=float("inf"), num_cells=0,
        )

    ref_rgb = ref_arr[:, :, :3][opaque]
    res_rgb = res_arr[:, :, :3][opaque]

    ref_lab = rgb_to_lab(ref_rgb)
    res_lab = rgb_to_lab(res_rgb)

    delta_e = np.sqrt(np.sum((ref_lab - res_lab) ** 2, axis=1))

    return FidelityResult(
        mean_delta_e=float(np.mean(delta_e)),
        median_delta_e=float(np.median(delta_e)),
        max_delta_e=float(np.max(delta_e)),
        p90_delta_e=float(np.percentile(delta_e, 90)),
        num_cells=int(np.sum(opaque)),
    )
