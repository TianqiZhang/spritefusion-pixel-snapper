"""Color space conversion utilities."""
from __future__ import annotations

import numpy as np


# D65 illuminant reference white
D65_X = 0.95047
D65_Y = 1.0
D65_Z = 1.08883

# sRGB to XYZ transformation matrix
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values to CIE LAB color space.

    Args:
        rgb: Array of shape (N, 3) with RGB values in range [0, 255].

    Returns:
        Array of shape (N, 3) with LAB values.
    """
    # Normalize to [0, 1]
    rgb_normalized = rgb / 255.0

    # sRGB gamma correction (linearize)
    linear = np.where(
        rgb_normalized <= 0.04045,
        rgb_normalized / 12.92,
        ((rgb_normalized + 0.055) / 1.055) ** 2.4,
    )

    # Convert to XYZ
    r, g, b = linear[:, 0], linear[:, 1], linear[:, 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by D65 illuminant
    x = x / D65_X
    z = z / D65_Z

    # Convert to LAB
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(
            t > delta_cubed,
            np.cbrt(t),
            t / (3 * delta ** 2) + 4.0 / 29.0,
        )

    fx, fy, fz = f(x), f(y), f(z)

    l_val = 116 * fy - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return np.stack([l_val, a_val, b_val], axis=1)
