"""Command-line interface for pixel snapper."""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

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
from .pattern import render_bead_pattern
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
    scored_candidates: List[ScoredCandidate] = None  # Top candidates with scores
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


def process_image(config: Config) -> None:
    """Process an image file.

    Args:
        config: Configuration with input/output paths.
    """
    print(f"Processing: {config.input_path}")
    with open(config.input_path, "rb") as f:
        img_bytes = f.read()

    result = process_image_bytes_with_grid(img_bytes, config)
    with open(config.output_path, "wb") as f:
        f.write(result.output_bytes)

    print(f"Saved to: {config.output_path}")

    if config.pattern_output and not config.palette:
        raise PixelSnapperError("pattern output requires --palette")

    if config.palette:
        pattern_path = config.pattern_output or _default_pattern_path(
            config.output_path, config.pattern_format
        )
        output_img = Image.open(io.BytesIO(result.output_bytes)).convert("RGB")
        pattern_img = render_bead_pattern(
            output_img,
            config.palette,
            title=os.path.basename(config.input_path),
        )
        pattern_img.save(
            pattern_path,
            format=config.pattern_format.upper(),
            dpi=(300, 300),
        )
        print(f"Saved pattern to: {pattern_path}")
    if config.preview:
        preview_input_bytes = result.processed_input_bytes or img_bytes
        # Use candidate preview if we have scored candidates and quantized image
        if result.scored_candidates and result.quantized_img:
            preview_candidates(
                preview_input_bytes,
                result.output_bytes,
                result.scored_candidates,
                result.col_cuts,
                result.row_cuts,
                result.quantized_img,
            )
        else:
            # Fall back to simple side-by-side preview
            preview_side_by_side(
                preview_input_bytes, result.output_bytes, result.col_cuts, result.row_cuts
            )


def _default_pattern_path(output_path: str, pattern_format: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}_pattern.{pattern_format}"


def preview_side_by_side(
    input_bytes: bytes,
    output_bytes: bytes,
    col_cuts: List[int],
    row_cuts: List[int],
    scale: int = 4,
    grid_color: Tuple[int, int, int, int] = (255, 0, 0, 180),
) -> None:
    """Display input and output images side by side with grid overlay.

    Args:
        input_bytes: Original image bytes.
        output_bytes: Processed image bytes.
        col_cuts: Column cut positions for grid lines.
        row_cuts: Row cut positions for grid lines.
        scale: Scale factor for enlarging both images.
        grid_color: RGBA color for grid lines.
    """
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    in_w, in_h = input_img.size
    out_w, out_h = output_img.size

    # Calculate scale factor to make images reasonably sized
    # Target at least 400px on shortest side, or use provided scale
    min_dimension = min(in_w, in_h)
    if min_dimension * scale < 400:
        scale = max(scale, 400 // min_dimension + 1)

    # Scale up input image (nearest neighbor to keep pixel edges sharp)
    scaled_in_w, scaled_in_h = in_w * scale, in_h * scale
    scaled_input = input_img.resize((scaled_in_w, scaled_in_h), resample=Image.NEAREST)

    # Scale up output image to match input dimensions then apply scale
    scaled_output = output_img.resize((scaled_in_w, scaled_in_h), resample=Image.NEAREST)

    # Draw grid lines on the scaled input image
    input_with_grid = draw_grid_overlay(
        scaled_input, col_cuts, row_cuts, scale, grid_color
    )

    # Draw grid lines on the scaled output (1px per output pixel)
    output_with_grid = draw_output_grid(
        scaled_output, out_w, out_h, scale, grid_color
    )

    # Create side-by-side preview with a small gap
    gap = 4
    preview_img = Image.new("RGBA", (scaled_in_w * 2 + gap, scaled_in_h), (40, 40, 40, 255))
    preview_img.paste(input_with_grid, (0, 0))
    preview_img.paste(output_with_grid, (scaled_in_w + gap, 0))
    preview_img.show(title="Pixel Snapper Preview")


def draw_grid_overlay(
    img: Image.Image,
    col_cuts: List[int],
    row_cuts: List[int],
    scale: int,
    color: Tuple[int, int, int, int],
) -> Image.Image:
    """Draw grid lines on an image.

    Args:
        img: Image to draw on (will be modified).
        col_cuts: Column positions (in original coordinates).
        row_cuts: Row positions (in original coordinates).
        scale: Scale factor applied to the image.
        color: RGBA color for grid lines.

    Returns:
        Image with grid overlay.
    """
    result = img.copy()
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    # Draw vertical lines at column cuts
    for col in col_cuts:
        x = col * scale
        if 0 <= x < width:
            draw.line([(x, 0), (x, height - 1)], fill=color, width=1)

    # Draw horizontal lines at row cuts
    for row in row_cuts:
        y = row * scale
        if 0 <= y < height:
            draw.line([(0, y), (width - 1, y)], fill=color, width=1)

    return result


def draw_output_grid(
    img: Image.Image,
    grid_w: int,
    grid_h: int,
    scale: int,
    color: Tuple[int, int, int, int],
) -> Image.Image:
    """Draw uniform grid lines on the output image.

    Args:
        img: Image to draw on.
        grid_w: Number of grid columns.
        grid_h: Number of grid rows.
        scale: Scale factor applied to the image.
        color: RGBA color for grid lines.

    Returns:
        Image with grid overlay.
    """
    result = img.copy()
    draw = ImageDraw.Draw(result, "RGBA")
    width, height = result.size

    # Use floating point to avoid cumulative rounding errors
    cell_w = width / grid_w
    cell_h = height / grid_h

    # Draw vertical lines
    for i in range(grid_w + 1):
        x = round(i * cell_w)
        if 0 <= x < width:
            draw.line([(x, 0), (x, height - 1)], fill=color, width=1)
        elif x == width:
            # Draw at last pixel
            draw.line([(width - 1, 0), (width - 1, height - 1)], fill=color, width=1)

    # Draw horizontal lines
    for i in range(grid_h + 1):
        y = round(i * cell_h)
        if 0 <= y < height:
            draw.line([(0, y), (width - 1, y)], fill=color, width=1)
        elif y == height:
            # Draw at last pixel
            draw.line([(0, height - 1), (width - 1, height - 1)], fill=color, width=1)

    return result


def preview_candidates(
    input_bytes: bytes,
    output_bytes: bytes,
    candidates: List[ScoredCandidate],
    winner_col_cuts: List[int],
    winner_row_cuts: List[int],
    quantized_img: Image.Image,
    scale: int = 4,
    max_candidates: int = 5,
) -> None:
    """Display top candidates in two rows: inputs with grids, then outputs.

    Row 1: Input images with grid overlays for each candidate
    Row 2: Resampled outputs for each candidate

    Args:
        input_bytes: Original image bytes.
        output_bytes: Processed (winner) image bytes.
        candidates: List of scored candidates, sorted by score (best first).
        winner_col_cuts: Column cuts used for the winner (post-stabilization).
        winner_row_cuts: Row cuts used for the winner (post-stabilization).
        quantized_img: Quantized image used for resampling candidates.
        scale: Scale factor for enlarging images.
        max_candidates: Maximum number of candidates to display.
    """
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    in_w, in_h = input_img.size
    out_w, out_h = output_img.size

    # Calculate scale factor to make images reasonably sized
    min_dimension = min(in_w, in_h)
    if min_dimension * scale < 250:
        scale = max(scale, 250 // min_dimension + 1)

    # Limit scale for very small images to avoid huge previews
    max_size = 350
    if in_w * scale > max_size or in_h * scale > max_size:
        scale = min(max_size // in_w, max_size // in_h, scale)
        scale = max(scale, 1)

    scaled_w, scaled_h = in_w * scale, in_h * scale

    # Prepare candidate list
    display_candidates = list(candidates[:max_candidates])

    # Colors for different candidates
    colors = [
        (0, 255, 0, 200),    # Green - winner
        (255, 165, 0, 180),  # Orange - #2
        (255, 0, 255, 180),  # Magenta - #3
        (0, 255, 255, 180),  # Cyan - #4
        (255, 255, 0, 180),  # Yellow - #5
    ]

    # Layout constants
    label_height = 40
    row_gap = 8  # Gap between rows
    col_gap = 6  # Gap between columns

    # Number of columns: candidates + final output
    num_cols = len(display_candidates) + 1

    # Calculate total dimensions
    total_width = num_cols * scaled_w + (num_cols - 1) * col_gap
    total_height = 2 * scaled_h + row_gap + label_height

    # Create preview canvas
    preview = Image.new("RGBA", (total_width, total_height), (30, 30, 30, 255))
    draw = ImageDraw.Draw(preview)

    # Try to get a font for labels
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_small = ImageFont.truetype("arial.ttf", 9)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 9)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

    # Row 1: Input images with grid overlays
    # Row 2: Resampled outputs
    for idx, candidate in enumerate(display_candidates):
        col_x = idx * (scaled_w + col_gap)
        color = colors[idx % len(colors)]
        is_winner = idx == 0

        # Row 1: Input with grid overlay
        scaled_input = input_img.resize((scaled_w, scaled_h), resample=Image.NEAREST)
        img_with_grid = draw_grid_overlay(
            scaled_input, candidate.col_cuts, candidate.row_cuts, scale, color
        )
        preview.paste(img_with_grid, (col_x, 0))

        # Row 2: Resampled output
        candidate_output = resample(quantized_img, candidate.col_cuts, candidate.row_cuts)
        scaled_candidate_output = candidate_output.resize(
            (scaled_w, scaled_h), resample=Image.NEAREST
        )
        preview.paste(scaled_candidate_output, (col_x, scaled_h + row_gap))

        # Label below row 2
        label_y = 2 * scaled_h + row_gap + 4

        if is_winner:
            line1 = f"#1 WINNER: {candidate.source}"
            text_color = (0, 255, 0)
        else:
            line1 = f"#{candidate.rank}: {candidate.source}"
            text_color = (200, 200, 200)

        line2 = f"{candidate.grid_size} | {candidate.combined_score:.3f}"

        draw.text((col_x + 2, label_y), line1, fill=text_color, font=font)
        draw.text((col_x + 2, label_y + 13), line2, fill=(150, 150, 150), font=font_small)

    # Final column: Stabilized winner
    col_x = len(display_candidates) * (scaled_w + col_gap)

    # Row 1: Input with stabilized grid
    scaled_input = input_img.resize((scaled_w, scaled_h), resample=Image.NEAREST)
    img_with_grid = draw_grid_overlay(
        scaled_input, winner_col_cuts, winner_row_cuts, scale, (255, 255, 255, 200)
    )
    preview.paste(img_with_grid, (col_x, 0))

    # Row 2: Final output
    scaled_output = output_img.resize((scaled_w, scaled_h), resample=Image.NEAREST)
    preview.paste(scaled_output, (col_x, scaled_h + row_gap))

    # Label for final output
    label_y = 2 * scaled_h + row_gap + 4
    draw.text((col_x + 2, label_y), "FINAL (stabilized)", fill=(255, 255, 255), font=font)
    draw.text((col_x + 2, label_y + 13), f"{out_w}x{out_h}", fill=(150, 150, 150), font=font_small)

    # Add row labels on the left side
    draw.text((3, 3), "INPUT", fill=(255, 255, 255, 200), font=font_small)
    draw.text((3, scaled_h + row_gap + 3), "OUTPUT", fill=(255, 255, 255, 200), font=font_small)

    preview.show(title="Pixel Snapper - Top Candidates")


def parse_args(argv: Sequence[str]) -> Config:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments (including program name).

    Returns:
        Configured Config instance.

    Raises:
        PixelSnapperError: If arguments are invalid.
    """
    args = list(argv[1:])
    config = Config()
    debug = False
    positional: List[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--preview":
            config.preview = True
            i += 1
        elif arg == "--timing":
            config.timing = True
            i += 1
        elif arg == "--debug":
            debug = True
            i += 1
        elif arg == "--palette":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            config.palette = args[i + 1]
            i += 2
        elif arg == "--pattern-out":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            config.pattern_output = args[i + 1]
            i += 2
        elif arg == "--pattern-format":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            config.pattern_format = args[i + 1].lower()
            i += 2
        elif arg == "--palette-space":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            config.palette_space = args[i + 1].lower()
            i += 2
        elif arg == "--resolution-hint":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            try:
                resolution_hint = int(args[i + 1])
                if resolution_hint <= 0:
                    raise PixelSnapperError(
                        "resolution-hint must be a positive integer"
                    )
            except ValueError:
                raise PixelSnapperError(
                    f"Invalid resolution-hint value: '{args[i + 1]}'"
                )
            config.resolution_hint = resolution_hint
            i += 2
        elif arg == "--qwen":
            config.qwen_enabled = True
            i += 1
        elif arg.startswith("--qwen-"):
            raise PixelSnapperError(
                f"Unsupported Qwen option '{arg}'. Use Config defaults or code overrides."
            )
        else:
            positional.append(arg)
            i += 1

    if config.palette_space not in ("rgb", "lab"):
        raise PixelSnapperError("palette-space must be 'rgb' or 'lab'")
    if config.pattern_format not in ("pdf", "png"):
        raise PixelSnapperError("pattern-format must be 'pdf' or 'png'")

    if len(positional) < 2:
        raise PixelSnapperError(_usage_message())

    config.input_path = positional[0]
    config.output_path = positional[1]

    if len(positional) >= 3:
        try:
            k = int(positional[2])
            if k > 0:
                config.k_colors = k
            else:
                print(
                    f"Warning: invalid k_colors '{positional[2]}', "
                    f"falling back to default ({config.k_colors})"
                )
        except ValueError:
            print(
                f"Warning: invalid k_colors '{positional[2]}', "
                f"falling back to default ({config.k_colors})"
            )

    if len(positional) > 3:
        raise PixelSnapperError(_usage_message())

    # Enable debug logging if requested
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s: %(message)s"
        )
        logging.getLogger("pixel_snapper").setLevel(logging.DEBUG)

    return config


def _usage_message() -> str:
    """Return usage message string."""
    return (
        "Usage: python pixel_snapper.py input.png output.png [k_colors] "
        "[--palette NAME] [--pattern-out PATH] [--pattern-format pdf|png] "
        "[--palette-space rgb|lab] [--resolution-hint N] [--preview] "
        "[--timing] [--debug] [--qwen]"
    )


def main(argv: Sequence[str]) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config = parse_args(argv)
        process_image(config)
        return 0
    except PixelSnapperError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 1
