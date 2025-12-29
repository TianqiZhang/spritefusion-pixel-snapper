"""Command-line interface for pixel snapper."""
from __future__ import annotations

import io
import sys
import time
from typing import List, Optional, Sequence

from PIL import Image

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

    return out_buf.getvalue()


def process_image(config: Config) -> None:
    """Process an image file.

    Args:
        config: Configuration with input/output paths.
    """
    print(f"Processing: {config.input_path}")
    with open(config.input_path, "rb") as f:
        img_bytes = f.read()

    output_bytes = process_image_bytes(img_bytes, config)
    with open(config.output_path, "wb") as f:
        f.write(output_bytes)

    print(f"Saved to: {config.output_path}")
    if config.preview:
        preview_side_by_side(img_bytes, output_bytes)


def preview_side_by_side(input_bytes: bytes, output_bytes: bytes) -> None:
    """Display input and output images side by side.

    Args:
        input_bytes: Original image bytes.
        output_bytes: Processed image bytes.
    """
    input_img = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    in_w, in_h = input_img.size
    scaled_output = output_img.resize((in_w, in_h), resample=Image.NEAREST)

    preview_img = Image.new("RGBA", (in_w * 2, in_h))
    preview_img.paste(input_img, (0, 0))
    preview_img.paste(scaled_output, (in_w, 0))
    preview_img.show(title="Pixel Snapper Preview")


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
