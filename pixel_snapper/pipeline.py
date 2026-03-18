"""Core processing pipeline for pixel snapper."""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image

logger = logging.getLogger("pixel_snapper")

from .candidates import (
    Candidate,
    deduplicate_candidates,
    filter_by_resolution_hint,
    generate_candidates,
    log_candidates,
)
from .config import Config, PixelSnapperError, validate_image_dimensions
from .grid import (
    estimate_step_size,
    estimate_step_size_autocorr,
    find_best_offset,
    resolve_step_sizes,
    stabilize_both_axes,
    walk_with_offset,
)
from .profile import compute_profiles
from .quantize import quantize_image
from .resample import resample
from .scoring import ScoredCandidate, compute_expected_step, score_all_candidates
from .qwen import maybe_apply_qwen_edit


def _detect_grid_cuts(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    step_x: float,
    step_y: float,
    width: int,
    height: int,
    config: Config,
) -> Tuple[List[int], List[int]]:
    """Generate grid cuts with optional offset detection.

    Args:
        profile_x: Column gradient profile.
        profile_y: Row gradient profile.
        step_x: Step size for columns.
        step_y: Step size for rows.
        width: Image width.
        height: Image height.
        config: Configuration options.

    Returns:
        Tuple of (column_cuts, row_cuts).
    """
    if config.detect_grid_offset:
        offset_x = find_best_offset(profile_x, step_x, width, config)
        offset_y = find_best_offset(profile_y, step_y, height, config)
    else:
        offset_x = 0
        offset_y = 0

    col_cuts = walk_with_offset(profile_x, step_x, width, offset_x, config)
    row_cuts = walk_with_offset(profile_y, step_y, height, offset_y, config)
    return col_cuts, row_cuts


@dataclass
class ProcessingResult:
    """Result of image processing including grid information."""

    output_bytes: bytes
    col_cuts: List[int]
    row_cuts: List[int]
    scored_candidates: Optional[List[ScoredCandidate]] = None  # Top candidates with scores
    quantized_img: Optional[Image.Image] = None  # For preview rendering
    processed_input_bytes: Optional[bytes] = None  # Input after optional pre-processing

    def __post_init__(self):
        if self.scored_candidates is None:
            self.scored_candidates = []


def process_image_bytes(
    input_bytes: bytes, config: Optional[Config] = None
) -> bytes:
    """Process image bytes through the pixel snapping pipeline.

    Args:
        input_bytes: Input image as PNG/JPEG bytes.
        config: Configuration options. Uses defaults if None.

    Returns:
        Output PNG image bytes.
    """
    result = process_image_bytes_with_grid(input_bytes, config)
    return result.output_bytes


def process_image_bytes_with_grid(
    input_bytes: bytes, config: Optional[Config] = None
) -> ProcessingResult:
    """Process image bytes and return result with grid information.

    Uses enhanced grid detection with autocorrelation, offset detection,
    and uniformity scoring when enabled in config.

    Args:
        input_bytes: Input image as PNG/JPEG bytes.
        config: Configuration options. Uses defaults if None.

    Returns:
        ProcessingResult with output bytes and grid cuts.
    """
    config = config or Config()

    input_bytes = maybe_apply_qwen_edit(input_bytes, config)

    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    width, height = img.size
    validate_image_dimensions(width, height)
    t1 = time.perf_counter()

    quantized = quantize_image(img, config)
    t2 = time.perf_counter()

    profile_x, profile_y = compute_profiles(quantized)

    # Suppress edge artifacts by zeroing out border gradient values
    # (AI-generated images often have artifacts at edges)
    profile_border = 2
    if len(profile_x) > 2 * profile_border:
        for i in range(profile_border):
            profile_x[i] = 0.0
            profile_x[-(i + 1)] = 0.0
    if len(profile_y) > 2 * profile_border:
        for i in range(profile_border):
            profile_y[i] = 0.0
            profile_y[-(i + 1)] = 0.0

    t3 = time.perf_counter()

    # Enhanced step size estimation with autocorrelation
    if config.use_autocorrelation:
        step_x_autocorr, conf_x = estimate_step_size_autocorr(profile_x, config)
        step_y_autocorr, conf_y = estimate_step_size_autocorr(profile_y, config)
        logger.debug(
            f"Autocorrelation: step_x={step_x_autocorr:.2f} (conf={conf_x:.2f}), "
            f"step_y={step_y_autocorr:.2f} (conf={conf_y:.2f})"
            if step_x_autocorr and step_y_autocorr
            else f"Autocorrelation: step_x={step_x_autocorr}, step_y={step_y_autocorr}"
        )
    else:
        step_x_autocorr, conf_x = None, 0.0
        step_y_autocorr, conf_y = None, 0.0

    # Also get traditional peak-based estimates
    step_x_peaks = estimate_step_size(profile_x, config)
    step_y_peaks = estimate_step_size(profile_y, config)
    logger.debug(f"Peak-based: step_x={step_x_peaks}, step_y={step_y_peaks}")

    # Prefer autocorrelation if confident, else fall back to peaks
    step_x_opt = step_x_autocorr if conf_x >= config.autocorr_min_confidence else step_x_peaks
    step_y_opt = step_y_autocorr if conf_y >= config.autocorr_min_confidence else step_y_peaks

    step_x, step_y = resolve_step_sizes(
        step_x_opt, step_y_opt, width, height, config
    )
    logger.debug(f"Resolved step size: {step_x:.2f}x{step_y:.2f}")

    # Compute expected step size from detection methods (for scoring penalty)
    # Average X and Y expected steps, weighted by autocorr confidence
    avg_conf = (conf_x + conf_y) / 2.0
    expected_step_x = compute_expected_step(step_x_autocorr, step_x_peaks, conf_x)
    expected_step_y = compute_expected_step(step_y_autocorr, step_y_peaks, conf_y)
    if expected_step_x is not None and expected_step_y is not None:
        expected_step, _ = resolve_step_sizes(
            expected_step_x, expected_step_y, width, height, config
        )
    elif expected_step_x is not None:
        expected_step = expected_step_x
    elif expected_step_y is not None:
        expected_step = expected_step_y
    else:
        expected_step = None
    logger.debug(f"Expected step for scoring: {expected_step}")

    t4 = time.perf_counter()

    # Build candidate grids if uniformity scoring is enabled
    if config.use_uniformity_scoring:
        # Generate all candidates from various detection methods
        candidates_with_source = generate_candidates(
            profile_x=profile_x,
            profile_y=profile_y,
            width=width,
            height=height,
            config=config,
            quantized_img=quantized,
            step_x_autocorr=step_x_autocorr,
            step_y_autocorr=step_y_autocorr,
            step_x_peaks=step_x_peaks,
            step_y_peaks=step_y_peaks,
            detect_grid_cuts=_detect_grid_cuts,
            original_img=img,
        )

        # Deduplicate and filter candidates
        unique_candidates = deduplicate_candidates(candidates_with_source)
        if config.resolution_hint:
            unique_candidates = filter_by_resolution_hint(
                unique_candidates, config.resolution_hint
            )

        # Log all candidates before scoring
        log_candidates(unique_candidates)

        # Score all candidates and select best
        scored_candidates: List[ScoredCandidate] = []
        if unique_candidates:
            scored_candidates = score_all_candidates(
                quantized, profile_x, profile_y, unique_candidates, width, height,
                expected_step=expected_step,
            )
            if scored_candidates:
                best = scored_candidates[0]
                raw_col_cuts, raw_row_cuts = best.col_cuts, best.row_cuts
                logger.debug(f"Winner: {best.source} -> {best.grid_size} cells")
            else:
                raw_col_cuts, raw_row_cuts = _detect_grid_cuts(
                    profile_x, profile_y, step_x, step_y, width, height, config
                )
                logger.debug("No scored candidates, using fallback detection")
        else:
            raw_col_cuts, raw_row_cuts = _detect_grid_cuts(
                profile_x, profile_y, step_x, step_y, width, height, config
            )
            logger.debug("No candidates after filtering, using fallback detection")
    else:
        # Single detection path
        scored_candidates = []
        raw_col_cuts, raw_row_cuts = _detect_grid_cuts(
            profile_x, profile_y, step_x, step_y, width, height, config
        )
        logger.debug("Uniformity scoring disabled, using single detection path")
    t5 = time.perf_counter()

    col_cuts, row_cuts = stabilize_both_axes(
        profile_x,
        profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width,
        height,
        config,
    )
    t6 = time.perf_counter()

    output_img = resample(quantized, col_cuts, row_cuts)
    t7 = time.perf_counter()

    out_buf = io.BytesIO()
    output_img.save(out_buf, format="PNG")
    t8 = time.perf_counter()

    if config.timing:
        print(
            "Timing (s): "
            f"load={t1 - t0:.4f}, "
            f"quantize={t2 - t1:.4f}, "
            f"profiles={t3 - t2:.4f}, "
            f"step={t4 - t3:.4f}, "
            f"walk={t5 - t4:.4f}, "
            f"stabilize={t6 - t5:.4f}, "
            f"resample={t7 - t6:.4f}, "
            f"encode={t8 - t7:.4f}, "
            f"total={t8 - t0:.4f}"
        )

    return ProcessingResult(
        output_bytes=out_buf.getvalue(),
        col_cuts=col_cuts,
        row_cuts=row_cuts,
        scored_candidates=scored_candidates,
        quantized_img=quantized,
        processed_input_bytes=input_bytes,
    )
