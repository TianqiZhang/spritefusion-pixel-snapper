"""Hough transform-based grid detection.

This module implements grid detection using Canny edge detection and
Hough line transform, adapted from the proper-pixel-art approach.

The algorithm:
1. Optionally crop border pixels
2. Optionally upscale image (helps detect edges)
3. Convert to grayscale with alpha handling
4. Apply Canny edge detection
5. Apply morphological closing to fill gaps
6. Use Hough transform to detect line segments
7. Filter to keep only near-horizontal/vertical lines
8. Cluster nearby lines together
9. Estimate pixel width from median line spacing
10. Fill in missing grid lines based on pixel width
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("pixel_snapper")

# Try to import opencv, make it optional
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Alpha threshold for determining pixel opacity
ALPHA_THRESHOLD = 128


def _crop_border(img: Image.Image, num_pixels: int = 2) -> Image.Image:
    """Crop border pixels from image.

    Sometimes AI-generated images have artifacts at the edges.
    """
    width, height = img.size
    if width <= num_pixels * 2 or height <= num_pixels * 2:
        return img
    box = (num_pixels, num_pixels, width - num_pixels, height - num_pixels)
    return img.crop(box)


def _clamp_alpha_to_grayscale(img: Image.Image) -> np.ndarray:
    """Convert RGBA image to grayscale, replacing transparent pixels.

    Transparent pixels (alpha < threshold) are replaced with a color
    that contrasts with the image content to help edge detection.

    Args:
        img: Input image (will be converted to RGBA).

    Returns:
        Grayscale numpy array.
    """
    rgba = np.array(img.convert("RGBA"))
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # Create grayscale from RGB
    gray = np.array(img.convert("L"))

    # Find a background value that contrasts with the image
    # Use the inverse of the mean grayscale value
    opaque_mask = alpha >= ALPHA_THRESHOLD
    if np.any(opaque_mask):
        mean_gray = np.mean(gray[opaque_mask])
        bg_value = 255 if mean_gray < 128 else 0
    else:
        bg_value = 255

    # Replace transparent pixels with background
    gray[~opaque_mask] = bg_value

    return gray


def _upscale_nearest(img: Image.Image, factor: int) -> Image.Image:
    """Upscale image using nearest neighbor interpolation."""
    if factor <= 1:
        return img
    width, height = img.size
    return img.resize((width * factor, height * factor), Image.Resampling.NEAREST)


def detect_grid_hough(
    img: Image.Image,
    canny_low: int = 50,
    canny_high: int = 200,
    closure_kernel_size: int = 8,
    hough_threshold: int = 100,
    hough_min_line_len: int = 50,
    hough_max_line_gap: int = 10,
    angle_threshold_deg: float = 15.0,
    cluster_threshold: int = 4,
    outlier_trim_fraction: float = 0.2,
    crop_border: int = 2,
    upscale_factor: int = 2,
) -> Optional[Tuple[List[int], List[int]]]:
    """Detect grid using Canny edge detection and Hough transform.

    Args:
        img: Input RGBA image.
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        closure_kernel_size: Size of kernel for morphological closing.
        hough_threshold: Accumulator threshold for Hough transform.
        hough_min_line_len: Minimum line length to detect.
        hough_max_line_gap: Maximum gap between line segments to join.
        angle_threshold_deg: Max angle deviation from horizontal/vertical.
        cluster_threshold: Max distance to cluster lines together.
        outlier_trim_fraction: Fraction of outliers to trim when computing median.
        crop_border: Number of border pixels to crop (0 to disable).
        upscale_factor: Upscale factor before detection (1 to disable).

    Returns:
        Tuple of (col_cuts, row_cuts) or None if opencv unavailable or no grid found.
    """
    if not OPENCV_AVAILABLE:
        logger.debug("OpenCV not available, skipping Hough detection")
        return None

    original_img = img  # Keep reference to original
    original_width, original_height = img.size

    # Crop border pixels (helps with AI-generated image artifacts)
    if crop_border > 0:
        img = _crop_border(img, crop_border)

    # Upscale image (helps detect pixel edges)
    if upscale_factor > 1:
        img = _upscale_nearest(img, upscale_factor)

    # Convert to grayscale with alpha handling
    gray = _clamp_alpha_to_grayscale(img)
    height, width = gray.shape

    # Apply Canny edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Apply morphological closing to fill gaps in edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (closure_kernel_size, closure_kernel_size)
    )
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Detect lines using probabilistic Hough transform
    hough_lines = cv2.HoughLinesP(
        closed_edges,
        rho=1.0,
        theta=np.deg2rad(1),
        threshold=hough_threshold,
        minLineLength=hough_min_line_len,
        maxLineGap=hough_max_line_gap,
    )

    # Always include image boundaries (using width-1, height-1 like proper-pixel-art)
    lines_x: List[int] = [0, width - 1]
    lines_y: List[int] = [0, height - 1]

    if hough_lines is not None:
        angle_threshold_rad = np.deg2rad(angle_threshold_deg)

        for line in hough_lines[:, 0]:
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            angle = abs(np.arctan2(dy, dx))

            # Near vertical (angle close to 90 degrees)
            if angle > np.deg2rad(90) - angle_threshold_rad:
                lines_x.append(round((x1 + x2) / 2))
            # Near horizontal (angle close to 0)
            elif angle < angle_threshold_rad:
                lines_y.append(round((y1 + y2) / 2))

    logger.debug(
        f"Hough detected {len(lines_x)-2} vertical, {len(lines_y)-2} horizontal lines"
    )

    # Cluster nearby lines (scale threshold by upscale factor)
    scaled_cluster_threshold = cluster_threshold * upscale_factor
    lines_x = _cluster_lines(lines_x, scaled_cluster_threshold)
    lines_y = _cluster_lines(lines_y, scaled_cluster_threshold)

    # Check if we got a non-trivial mesh
    if len(lines_x) <= 3 and len(lines_y) <= 3:
        logger.debug("Hough detection found trivial mesh, trying without upscale")
        # Try again without upscaling if we got trivial results
        if upscale_factor > 1:
            return detect_grid_hough(
                original_img,
                canny_low=canny_low,
                canny_high=canny_high,
                closure_kernel_size=closure_kernel_size,
                hough_threshold=hough_threshold,
                hough_min_line_len=hough_min_line_len,
                hough_max_line_gap=hough_max_line_gap,
                angle_threshold_deg=angle_threshold_deg,
                cluster_threshold=cluster_threshold,
                outlier_trim_fraction=outlier_trim_fraction,
                crop_border=crop_border,
                upscale_factor=1,  # Disable upscaling on retry
            )
        return None

    # Estimate pixel width from line gaps
    pixel_width = _get_pixel_width([lines_x, lines_y], outlier_trim_fraction)

    if pixel_width is None or pixel_width < 2:
        logger.debug(f"Invalid pixel width {pixel_width}, returning raw lines")
        # Scale back to original coordinates
        scale = upscale_factor
        lines_x = [max(0, min(original_width, round(x / scale + crop_border))) for x in lines_x]
        lines_y = [max(0, min(original_height, round(y / scale + crop_border))) for y in lines_y]
        # Fix boundaries to match original image
        lines_x[0] = 0
        lines_x[-1] = original_width
        lines_y[0] = 0
        lines_y[-1] = original_height
        return lines_x, lines_y

    logger.debug(f"Hough estimated pixel width: {pixel_width:.1f}")

    # Complete the mesh by filling gaps
    col_cuts = _homogenize_lines(lines_x, pixel_width)
    row_cuts = _homogenize_lines(lines_y, pixel_width)

    # Scale back to original image coordinates
    scale = upscale_factor
    col_cuts = [max(0, min(original_width, round(x / scale + crop_border))) for x in col_cuts]
    row_cuts = [max(0, min(original_height, round(y / scale + crop_border))) for y in row_cuts]

    # Ensure boundaries are exact
    col_cuts[0] = 0
    col_cuts[-1] = original_width
    row_cuts[0] = 0
    row_cuts[-1] = original_height

    # Remove any duplicates that may have appeared from rounding
    col_cuts = sorted(set(col_cuts))
    row_cuts = sorted(set(row_cuts))

    logger.debug(
        f"Hough final grid: {len(col_cuts)-1}x{len(row_cuts)-1} cells"
    )

    return col_cuts, row_cuts


def _cluster_lines(lines: List[int], threshold: int = 4) -> List[int]:
    """Cluster nearby line positions and return median of each cluster.

    Args:
        lines: List of line positions.
        threshold: Maximum distance to consider lines as same cluster.

    Returns:
        Clustered line positions (one per cluster).
    """
    if not lines:
        return []

    lines = sorted(lines)
    clusters: List[List[int]] = [[lines[0]]]

    for pos in lines[1:]:
        if abs(pos - clusters[-1][-1]) <= threshold:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    return [int(np.median(cluster)) for cluster in clusters]


def _get_pixel_width(
    line_collections: List[List[int]], trim_fraction: float = 0.2
) -> Optional[float]:
    """Estimate pixel width from line spacing.

    Args:
        line_collections: List of line position lists (x and y).
        trim_fraction: Fraction of outliers to trim from each end.

    Returns:
        Estimated pixel width or None if cannot determine.
    """
    all_gaps: List[int] = []

    for lines in line_collections:
        if len(lines) >= 2:
            gaps = np.diff(lines)
            all_gaps.extend(gaps)

    if not all_gaps:
        return None

    gaps_array = np.array(all_gaps)

    # Filter outliers using percentile trimming
    low = np.percentile(gaps_array, 100 * trim_fraction)
    high = np.percentile(gaps_array, 100 * (1 - trim_fraction))
    middle = gaps_array[(gaps_array >= low) & (gaps_array <= high)]

    if len(middle) == 0:
        middle = gaps_array

    return float(np.median(middle))


def _homogenize_lines(lines: List[int], pixel_width: float) -> List[int]:
    """Fill in missing grid lines based on expected pixel width.

    Args:
        lines: Sorted list of line positions.
        pixel_width: Expected spacing between lines.

    Returns:
        Complete list of line positions with gaps filled.
    """
    if len(lines) < 2:
        return lines

    result: List[int] = []

    for i in range(len(lines) - 1):
        start = lines[i]
        end = lines[i + 1]
        section_width = end - start

        # Calculate how many pixels fit in this section
        num_pixels = max(1, round(section_width / pixel_width))
        section_pixel_width = section_width / num_pixels

        # Add intermediate lines
        for n in range(num_pixels):
            result.append(start + round(n * section_pixel_width))

    # Add the final boundary
    result.append(lines[-1])

    return result
