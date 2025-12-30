"""Configuration and validation for pixel snapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class PixelSnapperError(Exception):
    """Base exception for pixel snapper errors."""

    pass


@dataclass
class Config:
    """Configuration for the pixel snapping pipeline."""

    k_colors: int = 16
    k_seed: int = 42
    input_path: str = ""
    output_path: str = ""
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8
    preview: bool = False
    timing: bool = False
    palette: Optional[str] = None
    palette_space: str = "lab"

    # Enhanced grid detection options
    use_autocorrelation: bool = True
    autocorr_min_confidence: float = 0.3
    detect_grid_offset: bool = True
    offset_search_fraction: float = 0.25
    use_uniformity_scoring: bool = True
    uniformity_candidate_steps: tuple = (8, 16, 32, 64)

    # Resolution hint (upper limit on cells)
    resolution_hint: Optional[int] = None  # Max cells on long axis


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions are within acceptable bounds.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Raises:
        PixelSnapperError: If dimensions are invalid.
    """
    if width == 0 or height == 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > 10000 or height > 10000:
        raise PixelSnapperError("Image dimensions too large (max 10000x10000)")
