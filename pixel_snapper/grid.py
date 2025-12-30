"""Grid detection and stabilization algorithms."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, PixelSnapperError


def estimate_step_size_autocorr_multi(
    profile: Sequence[float], config: Config, max_peaks: int = 5
) -> List[Tuple[float, float]]:
    """Estimate grid cell sizes using autocorrelation, returning multiple candidates.

    Uses FFT-based autocorrelation to find periodic patterns in the gradient
    profile. Returns multiple peaks to provide more candidates for grid selection,
    useful when a resolution hint constrains the valid cell counts.

    Args:
        profile: Gradient profile values.
        config: Configuration options.
        max_peaks: Maximum number of peaks to return.

    Returns:
        List of (step_size, confidence) tuples, sorted by confidence descending.
    """
    if len(profile) < 8:
        return []

    arr = np.array(profile, dtype=np.float64)

    # Remove DC component (mean)
    arr = arr - arr.mean()

    # Handle zero-variance profiles
    if np.std(arr) < 1e-10:
        return []

    # Compute autocorrelation via FFT (much faster than naive O(n^2))
    n = len(arr)
    fft_size = 1 << (2 * n - 1).bit_length()
    fft = np.fft.fft(arr, n=fft_size)
    autocorr = np.fft.ifft(fft * np.conj(fft)).real[:n]

    # Normalize by zero-lag value
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        return []

    # Search for peaks after minimum lag
    min_lag = max(config.peak_distance_filter, 3)
    max_lag = n // 2

    if max_lag <= min_lag:
        return []

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) < 3:
        return []

    # Find all peaks in autocorrelation (local maxima above threshold)
    peaks: List[Tuple[int, float]] = []
    for i in range(1, len(search_region) - 1):
        if (
            search_region[i] > search_region[i - 1]
            and search_region[i] > search_region[i + 1]
            and search_region[i] > config.autocorr_min_confidence
        ):
            peaks.append((i + min_lag, search_region[i]))

    if not peaks:
        return []

    # Sort by confidence descending
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Refine peak positions with parabolic interpolation and return top N
    results: List[Tuple[float, float]] = []
    for lag, confidence in peaks[:max_peaks]:
        refined_lag = float(lag)
        if min_lag < lag < n - 1:
            y0 = autocorr[lag - 1]
            y1 = autocorr[lag]
            y2 = autocorr[lag + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if abs(denom) > 1e-10:
                offset = (y0 - y2) / denom
                refined_lag = lag + offset
        results.append((refined_lag, float(confidence)))

    return results


def estimate_step_size_autocorr(
    profile: Sequence[float], config: Config
) -> Tuple[Optional[float], float]:
    """Estimate grid cell size using autocorrelation.

    Uses FFT-based autocorrelation to find the dominant periodic pattern
    in the gradient profile. More robust to noise than peak-spacing.

    Args:
        profile: Gradient profile values.
        config: Configuration options.

    Returns:
        Tuple of (estimated step size or None, confidence score 0-1).
    """
    results = estimate_step_size_autocorr_multi(profile, config, max_peaks=1)
    if not results:
        return None, 0.0
    return results[0]


def find_best_offset(
    profile: Sequence[float], step_size: float, limit: int, config: Config
) -> int:
    """Find the grid offset that maximizes edge alignment.

    AI-generated pixel art often has margins or borders. This function
    tests different starting offsets and returns the one where grid
    lines best align with detected edges.

    Args:
        profile: Gradient profile values.
        step_size: Estimated step size.
        limit: Maximum position (image dimension).
        config: Configuration options.

    Returns:
        Best starting offset (0 to step_size-1).
    """
    if step_size <= 1 or not profile:
        return 0

    # Search within the step size (offsets 0 to step_size-1 are valid)
    # We search up to offset_search_fraction of the step, but at least
    # a reasonable range to catch common margin sizes
    min_search = min(int(step_size), 8)  # At least check first 8 pixels
    search_range = min(
        max(int(step_size * config.offset_search_fraction) + 1, min_search),
        int(step_size),
        limit // 4,
    )

    if search_range <= 1:
        return 0

    arr = np.array(profile, dtype=np.float64)
    best_score = -1.0
    best_offset = 0

    for offset in range(search_range):
        score = 0.0
        count = 0
        pos = float(offset)

        while pos < limit:
            # Sample gradient at expected grid positions
            idx = int(round(pos))
            if 0 <= idx < len(arr):
                # Weight by position certainty (higher weight near exact positions)
                frac = abs(pos - idx)
                weight = 1.0 - frac
                score += arr[idx] * weight
                count += 1
            pos += step_size

        # Normalize by number of samples
        if count > 0:
            score /= count

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def walk_with_offset(
    profile: Sequence[float],
    step_size: float,
    limit: int,
    offset: int,
    config: Config,
) -> List[int]:
    """Walk through profile placing grid cuts at edge peaks.

    Uses an elastic approach that snaps to strong edges within a search
    window around the expected position. Starts from a specified offset
    to handle images with margins.

    Args:
        profile: Gradient profile values.
        step_size: Expected step size.
        limit: Maximum position (image dimension).
        offset: Starting offset position.
        config: Configuration options.

    Returns:
        List of cut positions.

    Raises:
        PixelSnapperError: If profile is empty.
    """
    if not profile:
        raise PixelSnapperError("Cannot walk on empty profile")

    cuts = [0]  # Always include 0
    if offset > 0:
        cuts.append(offset)

    current_pos = float(offset)
    search_window = max(
        step_size * config.walker_search_window_ratio,
        config.walker_min_search_window,
    )
    mean_val = sum(profile) / float(len(profile))

    while current_pos < float(limit):
        target = current_pos + step_size
        if target >= float(limit):
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1.0))
        end_search = min(int(target + search_window), limit)

        if end_search <= start_search:
            current_pos = target
            continue

        # Find maximum gradient in search window
        max_val = -1.0
        max_idx = start_search
        for i in range(start_search, end_search):
            if profile[i] > max_val:
                max_val = profile[i]
                max_idx = i

        # Snap to edge if strong enough, otherwise use target
        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target

    return cuts


def estimate_step_size(
    profile: Sequence[float], config: Config
) -> Optional[float]:
    """Estimate grid cell size from a gradient profile.

    Finds peaks in the profile and calculates median spacing.

    Args:
        profile: Gradient profile values.
        config: Configuration options.

    Returns:
        Estimated step size, or None if no pattern found.
    """
    if not profile:
        return None

    max_val = max(profile)
    if max_val == 0.0:
        return None

    threshold = max_val * config.peak_threshold_multiplier
    peaks: List[int] = []

    for i in range(1, len(profile) - 1):
        if (
            profile[i] > threshold
            and profile[i] > profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)

    if len(peaks) < 2:
        return None

    # Filter peaks that are too close together
    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > (config.peak_distance_filter - 1):
            clean_peaks.append(p)

    if len(clean_peaks) < 2:
        return None

    # Calculate median spacing
    diffs = [
        float(clean_peaks[i + 1] - clean_peaks[i])
        for i in range(len(clean_peaks) - 1)
    ]
    diffs.sort()
    return diffs[len(diffs) // 2]


def resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    config: Config,
) -> Tuple[float, float]:
    """Resolve step sizes for both axes.

    Handles cases where one or both axes have no detected pattern.

    Args:
        step_x_opt: Detected X step size or None.
        step_y_opt: Detected Y step size or None.
        width: Image width.
        height: Image height.
        config: Configuration options.

    Returns:
        Tuple of (step_x, step_y).
    """
    if step_x_opt is not None and step_y_opt is not None:
        ratio = (
            step_x_opt / step_y_opt
            if step_x_opt > step_y_opt
            else step_y_opt / step_x_opt
        )
        if ratio > config.max_step_ratio:
            smaller = min(step_x_opt, step_y_opt)
            return smaller, smaller
        avg = (step_x_opt + step_y_opt) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback_step = max(
        min(width, height) / float(config.fallback_target_segments), 1.0
    )
    return fallback_step, fallback_step


def sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    """Sanitize cut list to ensure valid boundaries.

    Ensures cuts include 0 and limit, are sorted, and have no duplicates.

    Args:
        cuts: List of cut positions.
        limit: Maximum position.

    Returns:
        Sanitized list of cuts.
    """
    if limit == 0:
        return [0]

    has_zero = False
    has_limit = False
    sanitized = []

    for value in cuts:
        v = min(value, limit)
        if v == 0:
            has_zero = True
        if v == limit:
            has_limit = True
        sanitized.append(v)

    if not has_zero:
        sanitized.append(0)
    if not has_limit:
        sanitized.append(limit)

    sanitized.sort()

    # Remove duplicates while preserving order
    deduped: List[int] = []
    for v in sanitized:
        if not deduped or deduped[-1] != v:
            deduped.append(v)

    return deduped


def snap_uniform_cuts(
    profile: Sequence[float],
    limit: int,
    target_step: float,
    config: Config,
    min_required: int,
) -> List[int]:
    """Create uniform grid cuts snapped to nearby edges.

    Args:
        profile: Gradient profile values.
        limit: Maximum position.
        target_step: Target step size.
        config: Configuration options.
        min_required: Minimum required number of cuts.

    Returns:
        List of cut positions.
    """
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    if math.isfinite(target_step) and target_step > 0.0:
        desired_cells = int(round(limit / target_step))
    else:
        desired_cells = 0

    desired_cells = max(desired_cells, min_required - 1, 1)
    desired_cells = min(desired_cells, limit)

    cell_width = limit / float(desired_cells)
    search_window = max(
        cell_width * config.walker_search_window_ratio,
        config.walker_min_search_window,
    )
    mean_val = sum(profile) / float(len(profile)) if profile else 0.0

    cuts = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = int(math.floor(target - search_window))
        start = max(start, prev + 1, 0)
        end = int(math.ceil(target + search_window))
        end = min(end, limit - 1)

        if end < start:
            start = prev + 1
            end = start

        best_idx = min(start, len(profile) - 1) if profile else start
        best_val = -1.0
        last_index = min(end, len(profile) - 1)

        for i in range(start, last_index + 1):
            v = profile[i] if i < len(profile) else 0.0
            if v > best_val:
                best_val = v
                best_idx = i

        strength_threshold = mean_val * config.walker_strength_threshold
        if best_val < strength_threshold:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return sanitize_cuts(cuts, limit)


def stabilize_cuts(
    profile: Sequence[float],
    cuts: List[int],
    limit: int,
    sibling_cuts: Sequence[int],
    sibling_limit: int,
    config: Config,
) -> List[int]:
    """Stabilize cuts against a sibling axis.

    Args:
        profile: Gradient profile for this axis.
        cuts: Current cuts for this axis.
        limit: Maximum position for this axis.
        sibling_cuts: Cuts from the other axis.
        sibling_limit: Maximum position for other axis.
        config: Configuration options.

    Returns:
        Stabilized list of cuts.
    """
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = min(max(config.min_cuts_per_axis, 2), limit + 1)
    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)

    sibling_has_grid = (
        sibling_limit > 0
        and sibling_cells >= max(min_required - 1, 0)
        and sibling_cells > 0
    )

    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / float(axis_cells)
        sibling_step = sibling_limit / float(sibling_cells)
        step_ratio = axis_step / sibling_step
        steps_skewed = (
            step_ratio > config.max_step_ratio
            or step_ratio < 1.0 / config.max_step_ratio
        )

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    # Determine target step size
    if sibling_has_grid:
        target_step = sibling_limit / float(sibling_cells)
    elif config.fallback_target_segments > 1:
        target_step = limit / float(config.fallback_target_segments)
    elif axis_cells > 0:
        target_step = limit / float(axis_cells)
    else:
        target_step = float(limit)

    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    config: Config,
) -> Tuple[List[int], List[int]]:
    """Stabilize grid cuts for both axes.

    Performs two-pass stabilization to ensure consistent grid.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        raw_col_cuts: Initial column cuts.
        raw_row_cuts: Initial row cuts.
        width: Image width.
        height: Image height.
        config: Configuration options.

    Returns:
        Tuple of (column_cuts, row_cuts).
    """
    col_cuts_pass1 = stabilize_cuts(
        profile_x, raw_col_cuts[:], width, raw_row_cuts, height, config
    )
    row_cuts_pass1 = stabilize_cuts(
        profile_y, raw_row_cuts[:], height, raw_col_cuts, width, config
    )

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / float(col_cells)
    row_step = height / float(row_cells)

    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            final_col_cuts = snap_uniform_cuts(
                profile_x, width, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_col_cuts = col_cuts_pass1

        if row_step > target_step * 1.2:
            final_row_cuts = snap_uniform_cuts(
                profile_y, height, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_row_cuts = row_cuts_pass1

        return final_col_cuts, final_row_cuts

    return col_cuts_pass1, row_cuts_pass1
