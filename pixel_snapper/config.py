"""Configuration and validation for pixel snapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class PixelSnapperError(Exception):
    """Base exception for pixel snapper errors."""

    pass


DEFAULT_QWEN_PROMPT = (
    "Transform main object in the image into pixel art style, keep same pose but "
    "convert to chibi style, low-resolution, 32x32 dot matrix, coarse pixels, "
    "chibi style, 8-bit retro game sprite, blocky, hard edges, aliased, isolated "
    "on a flat white background. Minimalist details, no anti-aliasing."
)
DEFAULT_QWEN_NEGATIVE_PROMPT = "high resolution, detailed, realistic, background"


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
    pattern_output: Optional[str] = None
    pattern_format: str = "pdf"

    # Enhanced grid detection options
    use_autocorrelation: bool = True
    autocorr_min_confidence: float = 0.3
    detect_grid_offset: bool = True
    offset_search_fraction: float = 0.25
    use_uniformity_scoring: bool = True
    uniformity_candidate_steps: tuple = (8, 16, 32, 64)

    # Resolution hint (upper limit on cells)
    resolution_hint: Optional[int] = None  # Max cells on long axis

    # Qwen image edit pre-processing
    qwen_enabled: bool = False
    qwen_api_key: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_endpoint: Optional[str] = None
    qwen_prompt: str = DEFAULT_QWEN_PROMPT
    qwen_negative_prompt: str = DEFAULT_QWEN_NEGATIVE_PROMPT
    qwen_prompt_extend: bool = False
    qwen_watermark: bool = False
    qwen_output_count: int = 1
    qwen_output_index: int = 0
    qwen_size: Optional[str] = "512*512"
    qwen_seed: Optional[int] = None
    qwen_timeout: int = 120


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
