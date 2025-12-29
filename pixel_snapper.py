"""Backward-compatible entry point for pixel snapper.

This module provides backward compatibility with the original single-file
implementation. For new code, prefer importing from the pixel_snapper package:

    from pixel_snapper import Config, process_image_bytes, main
"""
from __future__ import annotations

import sys

from pixel_snapper import (
    Config,
    PixelSnapperError,
    main,
    process_image,
    process_image_bytes,
)

# Re-export for backward compatibility
__all__ = [
    "Config",
    "PixelSnapperError",
    "main",
    "process_image",
    "process_image_bytes",
]

if __name__ == "__main__":
    sys.exit(main(sys.argv))
