"""Grid scoring and validation algorithms."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import Config

logger = logging.getLogger("pixel_snapper")


def compute_expected_step(
    step_autocorr: Optional[float],
    step_peaks: Optional[float],
    autocorr_confidence: float,
    default_weight: float = 0.5,
) -> Optional[float]:
    """Compute expected step size from detection methods.

    Uses weighted average of autocorrelation and peak-based estimates.
    Autocorrelation is weighted by its confidence score.

    Args:
        step_autocorr: Step size from autocorrelation (or None).
        step_peaks: Step size from peak detection (or None).
        autocorr_confidence: Confidence score for autocorr (0-1).
        default_weight: Default weight for peak-based estimate.

    Returns:
        Expected step size, or None if no estimates available.
    """
    estimates: List[float] = []
    weights: List[float] = []

    if step_autocorr is not None:
        estimates.append(step_autocorr)
        weights.append(max(autocorr_confidence, 0.1))  # Minimum weight

    if step_peaks is not None:
        estimates.append(step_peaks)
        weights.append(default_weight)

    if not estimates:
        return None

    # Weighted average
    total_weight = sum(weights)
    return sum(e * w for e, w in zip(estimates, weights)) / total_weight


def compute_grid_size_penalty(
    cells_x: int,
    cells_y: int,
    expected_step: Optional[float],
    width: int,
    height: int,
    tolerance: float = 0.25,
    penalty_weight: float = 0.3,
) -> float:
    """Compute penalty for deviating from expected grid size.

    Args:
        cells_x: Number of cells in X direction.
        cells_y: Number of cells in Y direction.
        expected_step: Expected step size from detection methods.
        width: Image width.
        height: Image height.
        tolerance: Deviation tolerance before penalty applies (e.g., 0.25 = 25%).
        penalty_weight: Weight of the penalty in final score.

    Returns:
        Penalty value (negative, to be added to score). Zero if within tolerance.
    """
    if expected_step is None or expected_step <= 0:
        return 0.0

    # Expected cell counts based on expected step
    expected_cells_x = width / expected_step
    expected_cells_y = height / expected_step

    # Compute deviation ratios
    if expected_cells_x > 0 and cells_x > 0:
        ratio_x = cells_x / expected_cells_x
    else:
        ratio_x = 1.0

    if expected_cells_y > 0 and cells_y > 0:
        ratio_y = cells_y / expected_cells_y
    else:
        ratio_y = 1.0

    # Average ratio (symmetric between axes)
    avg_ratio = (ratio_x + ratio_y) / 2.0

    # No penalty if within tolerance
    lower_bound = 1.0 - tolerance
    upper_bound = 1.0 + tolerance
    if lower_bound <= avg_ratio <= upper_bound:
        return 0.0

    # Logarithmic penalty for larger deviations (symmetric in log space)
    # log2(0.5) = -1, log2(2) = 1, so both 2x and 0.5x give same magnitude
    deviation = abs(math.log2(avg_ratio))
    penalty = -penalty_weight * deviation

    logger.debug(
        f"Grid size penalty: cells={cells_x}x{cells_y}, "
        f"expected={expected_cells_x:.1f}x{expected_cells_y:.1f}, "
        f"ratio={avg_ratio:.2f}, penalty={penalty:.3f}"
    )

    return penalty


@dataclass
class ScoredCandidate:
    """A grid candidate with its scores and metadata."""

    col_cuts: List[int]
    row_cuts: List[int]
    step_size: float
    source: str
    uniformity_score: float  # Raw variance (lower is better)
    edge_score: float  # Normalized edge alignment (higher is better)
    combined_score: float  # Final score (higher is better)
    rank: int  # 1 = best

    @property
    def cells_x(self) -> int:
        """Number of cells in X direction."""
        return len(self.col_cuts) - 1

    @property
    def cells_y(self) -> int:
        """Number of cells in Y direction."""
        return len(self.row_cuts) - 1

    @property
    def grid_size(self) -> str:
        """Grid size as string (e.g., '32x32')."""
        return f"{self.cells_x}x{self.cells_y}"


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


def score_all_candidates(
    img: Image.Image,
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    candidates: List[Tuple[List[int], List[int], float, str]],
    width: int,
    height: int,
    uniformity_weight: float = 1.0,
    edge_weight: float = 0.5,
    expected_step: Optional[float] = None,
    size_penalty_tolerance: float = 0.25,
    size_penalty_weight: float = 0.3,
) -> List[ScoredCandidate]:
    """Score all candidates and return them sorted by score.

    Args:
        img: Input RGBA image.
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        candidates: List of (col_cuts, row_cuts, step_size, source) tuples.
        width: Image width.
        height: Image height.
        uniformity_weight: Weight for uniformity score.
        edge_weight: Weight for edge alignment score.
        expected_step: Expected step size from detection methods (for size penalty).
        size_penalty_tolerance: Deviation tolerance before penalty applies.
        size_penalty_weight: Weight of size penalty in final score.

    Returns:
        List of ScoredCandidate objects, sorted by combined score (best first).
    """
    if not candidates:
        return []

    # Compute raw scores for all candidates
    raw_scores: List[Tuple[float, float, float]] = []
    for col_cuts, row_cuts, _, _ in candidates:
        uniformity = score_grid_uniformity(img, col_cuts, row_cuts)
        edge_x = score_edge_alignment(profile_x, col_cuts, width)
        edge_y = score_edge_alignment(profile_y, row_cuts, height)
        raw_scores.append((uniformity, edge_x, edge_y))

    # Normalize scores for fair comparison
    uniformities = [s[0] for s in raw_scores if s[0] < float("inf")]
    edge_xs = [s[1] for s in raw_scores]
    edge_ys = [s[2] for s in raw_scores]

    max_uniformity = max(uniformities) if uniformities else 1.0
    max_edge_x = max(edge_xs) if edge_xs else 1.0
    max_edge_y = max(edge_ys) if edge_ys else 1.0

    # Avoid division by zero
    max_uniformity = max(max_uniformity, 1e-10)
    max_edge_x = max(max_edge_x, 1e-10)
    max_edge_y = max(max_edge_y, 1e-10)

    # Build scored candidates
    scored: List[ScoredCandidate] = []
    for i, (col_cuts, row_cuts, step_size, source) in enumerate(candidates):
        uniformity, edge_x, edge_y = raw_scores[i]

        # Normalize uniformity (invert so lower becomes higher score)
        if uniformity < float("inf"):
            norm_uniformity = 1.0 - (uniformity / max_uniformity)
        else:
            norm_uniformity = 0.0

        # Normalize edge scores
        norm_edge_x = edge_x / max_edge_x
        norm_edge_y = edge_y / max_edge_y
        norm_edge = (norm_edge_x + norm_edge_y) / 2.0

        # Grid size penalty (penalizes deviation from expected size)
        cells_x = len(col_cuts) - 1
        cells_y = len(row_cuts) - 1
        size_penalty = compute_grid_size_penalty(
            cells_x, cells_y, expected_step, width, height,
            tolerance=size_penalty_tolerance,
            penalty_weight=size_penalty_weight,
        )

        # Combined score (higher is better)
        combined = uniformity_weight * norm_uniformity + edge_weight * norm_edge + size_penalty

        scored.append(
            ScoredCandidate(
                col_cuts=col_cuts,
                row_cuts=row_cuts,
                step_size=step_size,
                source=source,
                uniformity_score=uniformity,
                edge_score=norm_edge,
                combined_score=combined,
                rank=0,  # Will be set after sorting
            )
        )

    # Sort by combined score (descending - best first)
    scored.sort(key=lambda c: c.combined_score, reverse=True)

    # Assign ranks
    for i, candidate in enumerate(scored):
        candidate.rank = i + 1

    # Log all scored candidates
    logger.debug("Scoring candidates:")
    for c in scored:
        logger.debug(
            f"  [{c.rank}] {c.source} -> {c.grid_size}: "
            f"uniformity={c.uniformity_score:.2f}, "
            f"edge={c.edge_score:.3f}, combined={c.combined_score:.3f}"
        )

    return scored


