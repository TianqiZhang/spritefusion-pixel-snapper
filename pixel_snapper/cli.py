"""Command-line interface for pixel snapper."""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .config import Config, PixelSnapperError, validate_image_dimensions
from .grid import (
    estimate_step_size,
    resolve_step_sizes,
    stabilize_both_axes,
    walk,
)
from .profile import compute_profiles
from .quantize import quantize_image
from .resample import resample


@dataclass
class ProcessingResult:
    """Result of image processing including grid information."""

    output_bytes: bytes
    col_cuts: List[int]
    row_cuts: List[int]


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

    Args:
        input_bytes: Input image as PNG/JPEG bytes.
        config: Configuration options. Uses defaults if None.

    Returns:
        ProcessingResult with output bytes and grid cuts.
    """
    config = config or Config()

    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    width, height = img.size
    validate_image_dimensions(width, height)
    t1 = time.perf_counter()

    quantized = quantize_image(img, config)
    t2 = time.perf_counter()

    profile_x, profile_y = compute_profiles(quantized)
    t3 = time.perf_counter()

    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(
        step_x_opt, step_y_opt, width, height, config
    )
    t4 = time.perf_counter()

    raw_col_cuts = walk(profile_x, step_x, width, config)
    raw_row_cuts = walk(profile_y, step_y, height, config)
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
    if config.preview:
        preview_side_by_side(
            img_bytes, result.output_bytes, result.col_cuts, result.row_cuts
        )


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
    preview = False
    timing = False
    palette: Optional[str] = None
    palette_space = "lab"
    positional: List[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--preview":
            preview = True
            i += 1
        elif arg == "--timing":
            timing = True
            i += 1
        elif arg == "--palette":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            palette = args[i + 1]
            i += 2
        elif arg == "--palette-space":
            if i + 1 >= len(args):
                raise PixelSnapperError(_usage_message())
            palette_space = args[i + 1].lower()
            i += 2
        else:
            positional.append(arg)
            i += 1

    if palette_space not in ("rgb", "lab"):
        raise PixelSnapperError("palette-space must be 'rgb' or 'lab'")

    if len(positional) < 2:
        raise PixelSnapperError(_usage_message())

    config = Config(
        input_path=positional[0],
        output_path=positional[1],
        preview=preview,
        timing=timing,
        palette=palette,
        palette_space=palette_space,
    )

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

    return config


def _usage_message() -> str:
    """Return usage message string."""
    return (
        "Usage: python pixel_snapper.py input.png output.png [k_colors] "
        "[--palette NAME] [--palette-space rgb|lab] [--preview] [--timing]"
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
