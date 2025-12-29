"""Gradient profile computation for grid detection."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from .config import PixelSnapperError


def compute_profiles(img: Image.Image) -> Tuple[List[float], List[float]]:
    """Compute gradient profiles for grid detection.

    Uses Sobel-like gradients to find edge strength along each axis.
    High values indicate potential grid boundaries.

    Args:
        img: Input RGBA image.

    Returns:
        Tuple of (column_profile, row_profile) where each profile
        is a list of gradient magnitudes.

    Raises:
        PixelSnapperError: If image is too small.
    """
    width, height = img.size
    if width < 3 or height < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    arr = np.array(img, dtype=np.uint8)
    r = arr[:, :, 0].astype(np.float64)
    g = arr[:, :, 1].astype(np.float64)
    b = arr[:, :, 2].astype(np.float64)
    a = arr[:, :, 3]

    # Convert to grayscale using standard weights
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray[a == 0] = 0.0

    # Initialize profiles
    col_proj = np.zeros(width, dtype=np.float64)
    row_proj = np.zeros(height, dtype=np.float64)

    # Compute horizontal gradients (for vertical edges -> column profile)
    if width >= 3:
        grad_x = np.abs(gray[:, 2:] - gray[:, :-2])
        col_proj[1:-1] = grad_x.sum(axis=0)

    # Compute vertical gradients (for horizontal edges -> row profile)
    if height >= 3:
        grad_y = np.abs(gray[2:, :] - gray[:-2, :])
        row_proj[1:-1] = grad_y.sum(axis=1)

    return col_proj.tolist(), row_proj.tolist()
