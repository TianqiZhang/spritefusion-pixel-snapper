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
"""
from .cli import main, process_image, process_image_bytes
from .config import Config, PixelSnapperError

__all__ = [
    "Config",
    "PixelSnapperError",
    "main",
    "process_image",
    "process_image_bytes",
]

__version__ = "1.0.0"
