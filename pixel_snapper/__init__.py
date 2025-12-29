"""Pixel Snapper - Convert messy pixel art to perfect grids.

This package provides tools for converting AI-generated or imperfect
pixel art into clean, perfectly-gridded pixel art.

Example:
    from pixel_snapper import Config, process_image_bytes

    with open("input.png", "rb") as f:
        input_bytes = f.read()

    config = Config(k_colors=16)
    output_bytes = process_image_bytes(input_bytes, config)

    with open("output.png", "wb") as f:
        f.write(output_bytes)

For grid information, use process_image_bytes_with_grid:

    from pixel_snapper import Config, process_image_bytes_with_grid

    result = process_image_bytes_with_grid(input_bytes, config)
    print(f"Grid: {len(result.col_cuts)-1}x{len(result.row_cuts)-1} cells")
"""
from .cli import (
    ProcessingResult,
    main,
    process_image,
    process_image_bytes,
    process_image_bytes_with_grid,
)
from .config import Config, PixelSnapperError

__all__ = [
    "Config",
    "PixelSnapperError",
    "ProcessingResult",
    "main",
    "process_image",
    "process_image_bytes",
    "process_image_bytes_with_grid",
]

__version__ = "1.0.0"
