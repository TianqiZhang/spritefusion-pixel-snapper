"""Grid candidate generation for uniformity scoring.

This module handles generating and filtering candidate grids from multiple
detection methods (autocorrelation, peak-based, Hough, fixed steps).
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image

from .config import Config
from .grid import (
    estimate_step_size_autocorr_multi,
    resolve_step_sizes,
)
from .hough import detect_grid_hough

logger = logging.getLogger("pixel_snapper")

# Type alias for candidate tuples: (col_cuts, row_cuts, step_size, source_name)
Candidate = Tuple[List[int], List[int], float, str]

# Type alias for the grid cut detection function
DetectGridCutsFunc = Callable[
    [Sequence[float], Sequence[float], float, float, int, int, Config],
    Tuple[List[int], List[int]],
]


def generate_candidates(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    quantized_img: Image.Image,
    step_x_autocorr: Optional[float],
    step_y_autocorr: Optional[float],
    step_x_peaks: Optional[float],
    step_y_peaks: Optional[float],
    detect_grid_cuts: DetectGridCutsFunc,
) -> List[Candidate]:
    """Generate all grid candidates from various detection methods.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        quantized_img: Quantized image for Hough detection.
        step_x_autocorr: Autocorrelation step estimate for X axis (or None).
        step_y_autocorr: Autocorrelation step estimate for Y axis (or None).
        step_x_peaks: Peak-based step estimate for X axis (or None).
        step_y_peaks: Peak-based step estimate for Y axis (or None).
        detect_grid_cuts: Function to generate grid cuts from step sizes.

    Returns:
        List of candidate tuples (col_cuts, row_cuts, step_size, source).
    """
    candidates: List[Candidate] = []

    # Multi-peak autocorrelation candidates (when resolution hint is provided)
    if config.resolution_hint and config.use_autocorrelation:
        candidates.extend(
            _generate_multi_peak_autocorr_candidates(
                profile_x, profile_y, width, height, config, detect_grid_cuts
            )
        )
    else:
        # Single-peak autocorrelation candidate
        if step_x_autocorr is not None or step_y_autocorr is not None:
            candidates.append(
                _generate_single_autocorr_candidate(
                    profile_x, profile_y, width, height, config,
                    step_x_autocorr, step_y_autocorr, detect_grid_cuts
                )
            )

    # Peak-based candidate
    if step_x_peaks is not None or step_y_peaks is not None:
        candidates.append(
            _generate_peak_based_candidate(
                profile_x, profile_y, width, height, config,
                step_x_peaks, step_y_peaks, detect_grid_cuts
            )
        )

    # Hough transform candidate
    hough_candidate = _generate_hough_candidate(quantized_img, width)
    if hough_candidate is not None:
        candidates.append(hough_candidate)

    # Fixed step size candidates
    candidates.extend(
        _generate_fixed_step_candidates(
            profile_x, profile_y, width, height, config, detect_grid_cuts
        )
    )

    # Resolution hint candidate
    if config.resolution_hint:
        candidates.append(
            _generate_hint_candidate(
                profile_x, profile_y, width, height, config, detect_grid_cuts
            )
        )

    logger.debug(f"Generated {len(candidates)} candidates before dedup")
    return candidates


def _generate_multi_peak_autocorr_candidates(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    detect_grid_cuts: DetectGridCutsFunc,
) -> List[Candidate]:
    """Generate candidates from multiple autocorrelation peaks.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        detect_grid_cuts: Function to generate grid cuts.

    Returns:
        List of candidates from autocorrelation peaks.
    """
    candidates: List[Candidate] = []

    multi_peaks_x = estimate_step_size_autocorr_multi(profile_x, config)
    multi_peaks_y = estimate_step_size_autocorr_multi(profile_y, config)

    logger.debug(
        f"Multi-peak autocorr: X peaks={[(f'{s:.1f}', f'{c:.2f}') for s, c in multi_peaks_x]}, "
        f"Y peaks={[(f'{s:.1f}', f'{c:.2f}') for s, c in multi_peaks_y]}"
    )

    # Add candidates from each peak combination
    for step_x_ac, _ in multi_peaks_x:
        for step_y_ac, _ in multi_peaks_y:
            ac_step_x, ac_step_y = resolve_step_sizes(
                step_x_ac, step_y_ac, width, height, config
            )
            ac_col_cuts, ac_row_cuts = detect_grid_cuts(
                profile_x, profile_y, ac_step_x, ac_step_y,
                width, height, config
            )
            candidates.append((
                ac_col_cuts, ac_row_cuts, ac_step_x,
                f"autocorr({step_x_ac:.1f}x{step_y_ac:.1f})"
            ))

    # Add single-axis candidates for X peaks
    for step_x_ac, _ in multi_peaks_x:
        ac_step_x, ac_step_y = resolve_step_sizes(
            step_x_ac, None, width, height, config
        )
        ac_col_cuts, ac_row_cuts = detect_grid_cuts(
            profile_x, profile_y, ac_step_x, ac_step_y,
            width, height, config
        )
        candidates.append((
            ac_col_cuts, ac_row_cuts, ac_step_x,
            f"autocorr-x({step_x_ac:.1f})"
        ))

    # Add single-axis candidates for Y peaks
    for step_y_ac, _ in multi_peaks_y:
        ac_step_x, ac_step_y = resolve_step_sizes(
            None, step_y_ac, width, height, config
        )
        ac_col_cuts, ac_row_cuts = detect_grid_cuts(
            profile_x, profile_y, ac_step_x, ac_step_y,
            width, height, config
        )
        candidates.append((
            ac_col_cuts, ac_row_cuts, ac_step_x,
            f"autocorr-y({step_y_ac:.1f})"
        ))

    return candidates


def _generate_single_autocorr_candidate(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    step_x_autocorr: Optional[float],
    step_y_autocorr: Optional[float],
    detect_grid_cuts: DetectGridCutsFunc,
) -> Candidate:
    """Generate a candidate from single-peak autocorrelation.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        step_x_autocorr: Autocorrelation step for X axis.
        step_y_autocorr: Autocorrelation step for Y axis.
        detect_grid_cuts: Function to generate grid cuts.

    Returns:
        Single autocorrelation candidate.
    """
    ac_step_x, ac_step_y = resolve_step_sizes(
        step_x_autocorr, step_y_autocorr, width, height, config
    )
    ac_col_cuts, ac_row_cuts = detect_grid_cuts(
        profile_x, profile_y, ac_step_x, ac_step_y,
        width, height, config
    )
    return (ac_col_cuts, ac_row_cuts, ac_step_x, "autocorr")


def _generate_peak_based_candidate(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    step_x_peaks: Optional[float],
    step_y_peaks: Optional[float],
    detect_grid_cuts: DetectGridCutsFunc,
) -> Candidate:
    """Generate a candidate from peak-based detection.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        step_x_peaks: Peak-based step for X axis.
        step_y_peaks: Peak-based step for Y axis.
        detect_grid_cuts: Function to generate grid cuts.

    Returns:
        Peak-based candidate.
    """
    pk_step_x, pk_step_y = resolve_step_sizes(
        step_x_peaks, step_y_peaks, width, height, config
    )
    pk_col_cuts, pk_row_cuts = detect_grid_cuts(
        profile_x, profile_y, pk_step_x, pk_step_y, width, height, config
    )
    return (pk_col_cuts, pk_row_cuts, pk_step_x, "peak-based")


def _generate_hough_candidate(
    quantized_img: Image.Image,
    width: int,
) -> Optional[Candidate]:
    """Generate a candidate from Hough transform detection.

    Args:
        quantized_img: Quantized image for edge detection.
        width: Image width.

    Returns:
        Hough candidate or None if detection fails.
    """
    hough_result = detect_grid_hough(quantized_img)
    if hough_result is None:
        return None

    hough_col_cuts, hough_row_cuts = hough_result
    hough_step = width / max(len(hough_col_cuts) - 1, 1)
    return (hough_col_cuts, hough_row_cuts, hough_step, "hough")


def _generate_fixed_step_candidates(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    detect_grid_cuts: DetectGridCutsFunc,
) -> List[Candidate]:
    """Generate candidates from fixed step sizes.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        detect_grid_cuts: Function to generate grid cuts.

    Returns:
        List of fixed step candidates.
    """
    candidates: List[Candidate] = []
    min_dimension = min(width, height)

    for candidate_step in config.uniformity_candidate_steps:
        if candidate_step < min_dimension / 2:
            cand_col_cuts, cand_row_cuts = detect_grid_cuts(
                profile_x, profile_y,
                float(candidate_step), float(candidate_step),
                width, height, config
            )
            candidates.append((
                cand_col_cuts, cand_row_cuts, float(candidate_step),
                f"fixed({candidate_step})"
            ))

    return candidates


def _generate_hint_candidate(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    width: int,
    height: int,
    config: Config,
    detect_grid_cuts: DetectGridCutsFunc,
) -> Candidate:
    """Generate a candidate from resolution hint.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        width: Image width.
        height: Image height.
        config: Configuration options.
        detect_grid_cuts: Function to generate grid cuts.

    Returns:
        Resolution hint candidate.
    """
    long_axis = max(width, height)
    hint_step = long_axis / config.resolution_hint
    hint_col_cuts, hint_row_cuts = detect_grid_cuts(
        profile_x, profile_y, hint_step, hint_step, width, height, config
    )
    return (
        hint_col_cuts, hint_row_cuts, hint_step,
        f"hint({config.resolution_hint})"
    )


def deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """Remove duplicate candidates that produce the same grid.

    Args:
        candidates: List of candidate tuples.

    Returns:
        List of unique candidates (first occurrence kept).
    """
    unique_candidates: List[Candidate] = []
    seen_grids: set = set()

    for col_cuts, row_cuts, step, source in candidates:
        grid_key = (tuple(col_cuts), tuple(row_cuts))
        if grid_key not in seen_grids:
            seen_grids.add(grid_key)
            unique_candidates.append((col_cuts, row_cuts, step, source))

    logger.debug(f"After dedup: {len(unique_candidates)} unique candidates")
    return unique_candidates


def filter_by_resolution_hint(
    candidates: List[Candidate],
    resolution_hint: int,
) -> List[Candidate]:
    """Filter candidates that exceed the resolution hint.

    Args:
        candidates: List of candidate tuples.
        resolution_hint: Maximum cells allowed on the long axis.

    Returns:
        Filtered list of candidates.
    """
    filtered: List[Candidate] = []

    for col_cuts, row_cuts, step, source in candidates:
        cells_x = len(col_cuts) - 1
        cells_y = len(row_cuts) - 1
        if max(cells_x, cells_y) <= resolution_hint:
            filtered.append((col_cuts, row_cuts, step, source))
        else:
            logger.debug(
                f"  Filtered out: {source} -> {cells_x}x{cells_y} cells "
                f"(exceeds hint={resolution_hint})"
            )

    logger.debug(f"After hint filter: {len(filtered)} candidates")
    return filtered


def log_candidates(candidates: List[Candidate]) -> None:
    """Log all candidates for debugging.

    Args:
        candidates: List of candidate tuples.
    """
    for col_cuts, row_cuts, step, source in candidates:
        cells_x = len(col_cuts) - 1
        cells_y = len(row_cuts) - 1
        logger.debug(f"  Candidate: {source} -> {cells_x}x{cells_y} cells (step={step:.2f})")
