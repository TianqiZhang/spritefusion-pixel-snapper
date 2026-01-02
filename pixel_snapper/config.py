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

    # Qwen image edit pre-processing
    qwen_enabled: bool = False
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen-image-edit-plus"
    qwen_endpoint: str = (
        "https://dashscope-intl.aliyuncs.com/api/v1/services/"
        "aigc/multimodal-generation/generation"
    )
    qwen_prompt: str = (
        "Transform the subject in this photo into a chibi-style cartoon character - "
        "cute, with a slightly oversized head, large eyes, and simplified features - "
        "then remove all background and render the entire image as a very-low-resolution "
        "pixel art pattern in the style of Perler or Hama fuse beads. Use a limited "
        "palette of solid, bright colors with no gradients. Each pixel should "
        "represent a single bead (circular or square), arranged on a clear grid. "
        "Keep the composition simple and recognizable, suitable for actual bead "
        "crafting. Preserve key traits (e.g., hairstyle, species, pose, or object "
        "shape) but stylize them in an adorable, minimal chibi pixel form."
    )
    qwen_negative_prompt: str = "high resolution, detailed, realistic, background"
    qwen_prompt_extend: bool = True
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
