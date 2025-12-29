"""Pytest fixtures for pixel_snapper tests."""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from pixel_snapper import Config


@pytest.fixture
def default_config() -> Config:
    """Return a default Config instance."""
    return Config()


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a 64x64 test image with a visible grid pattern.

    The image has 8x8 pixel cells with alternating colors,
    creating a checkerboard-like pattern.
    """
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    arr = np.array(img)

    # Create 8x8 grid cells
    cell_size = 8
    colors = [
        (255, 0, 0, 255),    # Red
        (0, 255, 0, 255),    # Green
        (0, 0, 255, 255),    # Blue
        (255, 255, 0, 255),  # Yellow
    ]

    for y in range(8):
        for x in range(8):
            color_idx = (x + y) % len(colors)
            y_start, y_end = y * cell_size, (y + 1) * cell_size
            x_start, x_end = x * cell_size, (x + 1) * cell_size
            arr[y_start:y_end, x_start:x_end] = colors[color_idx]

    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Return sample image as PNG bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def transparent_image() -> Image.Image:
    """Create a 32x32 image with transparent regions."""
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    arr = np.array(img)

    # Only fill the center 16x16 with opaque pixels
    arr[8:24, 8:24] = (255, 128, 64, 255)

    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def solid_color_image() -> Image.Image:
    """Create a 16x16 solid color image."""
    return Image.new("RGBA", (16, 16), (128, 64, 32, 255))


@pytest.fixture
def gradient_image() -> Image.Image:
    """Create a 64x64 image with horizontal gradient."""
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    arr = np.array(img)

    for x in range(64):
        gray = int(x * 255 / 63)
        arr[:, x] = (gray, gray, gray, 255)

    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def small_grid_image() -> Image.Image:
    """Create a small 4x4 image for edge case testing."""
    img = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    arr = np.array(img)

    # 2x2 grid cells
    arr[0:2, 0:2] = (255, 0, 0, 255)
    arr[0:2, 2:4] = (0, 255, 0, 255)
    arr[2:4, 0:2] = (0, 0, 255, 255)
    arr[2:4, 2:4] = (255, 255, 0, 255)

    return Image.fromarray(arr, "RGBA")


def create_test_image(
    width: int, height: int, color: Tuple[int, int, int, int] = (255, 0, 0, 255)
) -> Image.Image:
    """Helper to create a test image of specified size and color."""
    return Image.new("RGBA", (width, height), color)
