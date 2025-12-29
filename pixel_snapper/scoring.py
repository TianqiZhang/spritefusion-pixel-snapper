"""Grid scoring and validation algorithms."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config


def score_grid_uniformity(
    img: Image.Image, col_cuts: List[int], row_cuts: List[int]
) -> float:
    """Score a grid by measuring color uniformity within cells.

    For true pixel art, each cell should contain a single uniform color.
    This function measures how uniform colors are within each cell.
    Lower scores indicate better grids (more uniform cells).

    Args:
        img: Input RGBA image.
        col_cuts: List of column cut positions.
        row_cuts: List of row cut positions.

    Returns:
        Average color variance across all cells. Lower is better.
        Returns infinity for invalid grids.
    """
    if len(col_cuts) < 2 or len(row_cuts) < 2:
        return float("inf")

    arr = np.array(img, dtype=np.float64)
    total_variance = 0.0
    cell_count = 0
    total_weight = 0.0

    for i in range(len(col_cuts) - 1):
        for j in range(len(row_cuts) - 1):
            x0, x1 = col_cuts[i], col_cuts[i + 1]
            y0, y1 = row_cuts[j], row_cuts[j + 1]

            if x1 <= x0 or y1 <= y0:
                continue

            cell = arr[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            # Get alpha channel to weight by opacity
            alpha = cell[:, :, 3] if cell.shape[2] == 4 else np.ones(cell.shape[:2])
            opaque_mask = alpha > 0

            if not np.any(opaque_mask):
                # Fully transparent cell - no variance contribution
                cell_count += 1
                continue

            # Compute variance for RGB channels on opaque pixels
            rgb = cell[:, :, :3]
            opaque_pixels = rgb[opaque_mask]

            if len(opaque_pixels) > 1:
                # Variance per channel, then average
                variance = np.var(opaque_pixels, axis=0).mean()
            else:
                variance = 0.0

            # Weight by cell size (larger cells matter more)
            cell_size = (x1 - x0) * (y1 - y0)
            weight = float(cell_size)

            total_variance += variance * weight
            total_weight += weight
            cell_count += 1

    if total_weight == 0:
        return float("inf")

    return total_variance / total_weight


def score_edge_alignment(
    profile: Sequence[float], cuts: List[int], limit: int
) -> float:
    """Score how well cuts align with gradient peaks.

    Measures the average gradient strength at cut positions.
    Higher scores indicate better alignment with edges.

    Args:
        profile: Gradient profile values.
        cuts: List of cut positions.
        limit: Maximum position.

    Returns:
        Average gradient value at cut positions. Higher is better.
    """
    if len(cuts) < 2 or not profile:
        return 0.0

    total_strength = 0.0
    count = 0

    # Skip first (0) and last (limit) cuts as they're always included
    for cut in cuts[1:-1]:
        if 0 <= cut < len(profile):
            total_strength += profile[cut]
            count += 1

    if count == 0:
        return 0.0

    return total_strength / count


def select_best_grid(
    img: Image.Image,
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    candidates: List[Tuple[List[int], List[int], float]],
    width: int,
    height: int,
    uniformity_weight: float = 1.0,
    edge_weight: float = 0.5,
) -> Tuple[List[int], List[int], float]:
    """Select the best grid from multiple candidates.

    Combines uniformity scoring (lower is better) with edge alignment
    scoring (higher is better) to pick the optimal grid.

    Args:
        img: Input RGBA image.
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        candidates: List of (col_cuts, row_cuts, step_size) tuples.
        width: Image width.
        height: Image height.
        uniformity_weight: Weight for uniformity score (lower is better).
        edge_weight: Weight for edge alignment score (higher is better).

    Returns:
        Best (col_cuts, row_cuts, step_size) tuple.
    """
    if not candidates:
        return [0, width], [0, height], float(width)

    if len(candidates) == 1:
        return candidates[0]

    best_score = float("-inf")
    best_candidate = candidates[0]

    # Compute scores for all candidates
    all_scores: List[Tuple[float, float, float]] = []
    for col_cuts, row_cuts, _ in candidates:
        uniformity = score_grid_uniformity(img, col_cuts, row_cuts)
        edge_x = score_edge_alignment(profile_x, col_cuts, width)
        edge_y = score_edge_alignment(profile_y, row_cuts, height)
        all_scores.append((uniformity, edge_x, edge_y))

    # Normalize scores for fair comparison
    uniformities = [s[0] for s in all_scores if s[0] < float("inf")]
    edge_xs = [s[1] for s in all_scores]
    edge_ys = [s[2] for s in all_scores]

    max_uniformity = max(uniformities) if uniformities else 1.0
    max_edge_x = max(edge_xs) if edge_xs else 1.0
    max_edge_y = max(edge_ys) if edge_ys else 1.0

    # Avoid division by zero
    max_uniformity = max(max_uniformity, 1e-10)
    max_edge_x = max(max_edge_x, 1e-10)
    max_edge_y = max(max_edge_y, 1e-10)

    for i, (col_cuts, row_cuts, step_size) in enumerate(candidates):
        uniformity, edge_x, edge_y = all_scores[i]

        # Normalize uniformity (invert so lower becomes higher score)
        if uniformity < float("inf"):
            norm_uniformity = 1.0 - (uniformity / max_uniformity)
        else:
            norm_uniformity = 0.0

        # Normalize edge scores
        norm_edge_x = edge_x / max_edge_x
        norm_edge_y = edge_y / max_edge_y
        norm_edge = (norm_edge_x + norm_edge_y) / 2.0

        # Combined score (higher is better)
        combined = (
            uniformity_weight * norm_uniformity + edge_weight * norm_edge
        )

        if combined > best_score:
            best_score = combined
            best_candidate = (col_cuts, row_cuts, step_size)

    return best_candidate
