"""Image quantization using K-means clustering or palette matching."""
from __future__ import annotations

import random

import numpy as np
from PIL import Image

from .color import rgb_to_lab
from .config import Config, PixelSnapperError
from .palette import Palette, load_palette


def _find_nearest_chunked(
    data: np.ndarray,
    targets: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Find index of nearest target for each data point using chunked processing.

    Uses squared Euclidean distance. Processes in chunks to limit memory usage
    when dealing with large arrays.

    Args:
        data: Array of shape (N, D) with data points.
        targets: Array of shape (K, D) with target points.
        chunk_size: Chunk size for memory-efficient processing.

    Returns:
        Array of shape (N,) with indices of nearest targets.
    """
    nearest = np.empty(data.shape[0], dtype=np.int64)
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        chunk = data[start:end]
        diff = chunk[:, None, :] - targets[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        nearest[start:end] = np.argmin(dists, axis=1)
    return nearest


def quantize_image(img: Image.Image, config: Config) -> Image.Image:
    """Quantize an image to reduce colors.

    Args:
        img: Input RGBA image.
        config: Configuration options.

    Returns:
        Quantized RGBA image.

    Raises:
        PixelSnapperError: If configuration is invalid.
    """
    if config.k_colors <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    if config.palette:
        palette = load_palette(config.palette)
        return palette_quantize(img, palette, config.palette_space)

    return kmeans_quantize(img, config)


def kmeans_quantize(img: Image.Image, config: Config) -> Image.Image:
    """Quantize image using K-means++ clustering.

    Args:
        img: Input RGBA image.
        config: Configuration options.

    Returns:
        Quantized RGBA image.
    """
    arr = np.array(img, dtype=np.uint8)
    alpha = arr[:, :, 3]
    mask = alpha != 0
    opaque_pixels = arr[mask][:, :3].astype(np.float64)
    n_pixels = opaque_pixels.shape[0]

    if n_pixels == 0:
        return img.copy()

    rng = random.Random(config.k_seed)
    k = min(config.k_colors, n_pixels)

    # K-means++ initialization
    centroids = _kmeans_init(opaque_pixels, k, rng)

    # K-means iterations
    centroids = _kmeans_iterate(
        opaque_pixels, centroids, config.max_kmeans_iterations
    )

    # Apply quantization
    return _apply_centroids(arr, mask, centroids)


def _kmeans_init(
    pixels: np.ndarray, k: int, rng: random.Random
) -> np.ndarray:
    """Initialize centroids using K-means++ algorithm.

    Args:
        pixels: Array of shape (N, 3) with RGB values.
        k: Number of clusters.
        rng: Random number generator.

    Returns:
        Array of shape (k, 3) with initial centroids.
    """
    n_pixels = pixels.shape[0]
    centroids = np.zeros((k, 3), dtype=np.float64)

    # First centroid is random
    first_idx = rng.randrange(n_pixels)
    centroids[0] = pixels[first_idx]
    distances = np.full(n_pixels, np.inf, dtype=np.float64)

    # Remaining centroids use distance-weighted sampling
    for i in range(1, k):
        last_c = centroids[i - 1]
        d_sq = np.sum((pixels - last_c) ** 2, axis=1)
        distances = np.minimum(distances, d_sq)
        sum_sq_dist = float(distances.sum())

        if sum_sq_dist <= 0.0:
            idx = rng.randrange(n_pixels)
        else:
            r = rng.random() * sum_sq_dist
            cumulative = 0.0
            idx = 0
            for j, d in enumerate(distances):
                cumulative += float(d)
                if cumulative >= r:
                    idx = j
                    break
        centroids[i] = pixels[idx]

    return centroids


def _kmeans_iterate(
    pixels: np.ndarray,
    centroids: np.ndarray,
    max_iterations: int,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Run K-means iterations until convergence.

    Args:
        pixels: Array of shape (N, 3) with RGB values.
        centroids: Initial centroids of shape (k, 3).
        max_iterations: Maximum number of iterations.
        chunk_size: Chunk size for memory-efficient processing.

    Returns:
        Final centroids of shape (k, 3).
    """
    k = centroids.shape[0]
    n_pixels = pixels.shape[0]
    prev_centroids = centroids.copy()

    for iteration in range(max_iterations):
        sums = np.zeros((k, 3), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)

        # Process in chunks to limit memory usage
        for start in range(0, n_pixels, chunk_size):
            end = min(start + chunk_size, n_pixels)
            chunk = pixels[start:end]
            diff = chunk[:, None, :] - centroids[None, :, :]
            dists = np.sum(diff * diff, axis=2)
            labels = np.argmin(dists, axis=1)

            for idx in range(k):
                mask_idx = labels == idx
                if mask_idx.any():
                    sums[idx] += chunk[mask_idx].sum(axis=0)
                    counts[idx] += int(mask_idx.sum())

        # Update centroids
        new_centroids = centroids.copy()
        for idx in range(k):
            if counts[idx] > 0:
                new_centroids[idx] = sums[idx] / float(counts[idx])

        # Check convergence
        if iteration > 0:
            movement = np.sum((new_centroids - prev_centroids) ** 2, axis=1).max()
            if movement < 0.01:
                return new_centroids

        prev_centroids = new_centroids.copy()
        centroids = new_centroids

    return centroids


def _apply_centroids(
    arr: np.ndarray,
    mask: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 100_000,
) -> Image.Image:
    """Apply centroids to create quantized image.

    Args:
        arr: Original image array of shape (H, W, 4).
        mask: Boolean mask for non-transparent pixels.
        centroids: Centroids of shape (k, 3).
        chunk_size: Chunk size for processing.

    Returns:
        Quantized PIL Image.
    """
    flat = arr.reshape(-1, 4)
    flat_mask = flat[:, 3] != 0
    rgb = flat[flat_mask, :3].astype(np.float64)

    nearest = _find_nearest_chunked(rgb, centroids, chunk_size)

    new_rgb = np.rint(centroids[nearest]).clip(0, 255).astype(np.uint8)
    flat_out = flat.copy()
    flat_out[flat_mask, :3] = new_rgb
    out_arr = flat_out.reshape(arr.shape)
    return Image.fromarray(out_arr, "RGBA")


def palette_quantize(
    img: Image.Image, palette: Palette, space: str
) -> Image.Image:
    """Quantize image using a fixed color palette.

    Args:
        img: Input RGBA image.
        palette: Color palette to match against.
        space: Color space for matching ("rgb" or "lab").

    Returns:
        Quantized RGBA image.
    """
    arr = np.array(img, dtype=np.uint8)
    flat = arr.reshape(-1, 4)
    mask = flat[:, 3] != 0
    rgb = flat[mask, :3].astype(np.float64)

    if rgb.size == 0:
        return img.copy()

    palette_rgb = np.array(palette.rgb, dtype=np.float64)
    palette_lab = np.array(palette.lab, dtype=np.float64)

    if space == "lab":
        data = rgb_to_lab(rgb)
        target_palette = palette_lab
    else:
        data = rgb
        target_palette = palette_rgb

    nearest = _find_nearest_chunked(data, target_palette)

    mapped = np.rint(palette_rgb[nearest]).clip(0, 255).astype(np.uint8)
    flat_out = flat.copy()
    flat_out[mask, :3] = mapped
    out_arr = flat_out.reshape(arr.shape)
    return Image.fromarray(out_arr, "RGBA")
