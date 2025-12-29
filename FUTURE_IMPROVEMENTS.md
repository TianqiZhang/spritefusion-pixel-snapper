# Future Grid Detection Improvements

This document tracks potential improvements to the grid detection algorithm that haven't been implemented yet.

## Implemented (v1.1)

- **Autocorrelation-based period detection** - FFT-based step size estimation, more robust to noise
- **Grid offset detection** - Handles images with margins/borders
- **Uniformity scoring** - Multi-hypothesis grid selection based on cell color uniformity

---

## Pending Improvements

### 1. Profile Smoothing

**Impact:** Medium | **Effort:** Low

Apply Gaussian smoothing to gradient profiles before peak detection to reduce noise from anti-aliasing and compression artifacts.

**Location:** `pixel_snapper/profile.py`

```python
from scipy.ndimage import gaussian_filter1d

def compute_profiles(img: Image.Image, smooth_sigma: float = 1.5) -> Tuple[List[float], List[float]]:
    """Compute gradient profiles with optional smoothing."""
    # ... existing gradient computation ...

    # Apply Gaussian smoothing to reduce noise
    if smooth_sigma > 0:
        col_proj = gaussian_filter1d(col_proj, sigma=smooth_sigma)
        row_proj = gaussian_filter1d(row_proj, sigma=smooth_sigma)

    return col_proj.tolist(), row_proj.tolist()
```

**Config additions:**
```python
smooth_profiles: bool = True
smooth_sigma: float = 1.5
```

---

### 2. Sobel Gradient Kernels

**Impact:** Medium | **Effort:** Low

Replace the simple 2-pixel difference gradient with proper 3x3 Sobel operators. Sobel kernels provide better noise resistance and detect diagonal edges.

**Location:** `pixel_snapper/profile.py`

```python
from scipy.ndimage import sobel

def compute_profiles_sobel(img: Image.Image) -> Tuple[List[float], List[float]]:
    """Compute profiles using proper Sobel operators."""
    arr = np.array(img, dtype=np.uint8)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    a = arr[:,:,3]

    # Convert to grayscale
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)
    gray[a == 0] = 0.0

    # Full Sobel operators (3x3 kernels)
    grad_x = np.abs(sobel(gray, axis=1))  # Detects vertical edges
    grad_y = np.abs(sobel(gray, axis=0))  # Detects horizontal edges

    col_profile = grad_x.sum(axis=0)  # Sum vertically for column boundaries
    row_profile = grad_y.sum(axis=1)  # Sum horizontally for row boundaries

    return col_profile.tolist(), row_profile.tolist()
```

**Config additions:**
```python
use_sobel_gradients: bool = True
```

**Notes:**
- Requires `scipy` dependency (already used elsewhere)
- Could also try Scharr operator for even better rotation invariance

---

### 3. Adaptive Search Window

**Impact:** Low | **Effort:** Medium

Dynamically adjust the walker search window based on local profile characteristics. When edges are clear (high local variance), use a smaller window for precision. When edges are weak, use a larger window.

**Location:** `pixel_snapper/grid.py`

```python
def walk_adaptive(
    profile: Sequence[float],
    step_size: float,
    limit: int,
    config: Config
) -> List[int]:
    """Walk with adaptive search window based on profile confidence."""
    arr = np.array(profile, dtype=np.float64)

    # Compute global statistics
    global_std = np.std(arr)
    global_mean = np.mean(arr)

    # Confidence: high std relative to mean = clear edges
    confidence = min(global_std / (global_mean + 1e-6), 2.0) / 2.0

    # Adaptive ratio: confident = smaller window (more precise)
    # Range: 0.175 (high confidence) to 0.525 (low confidence)
    base_ratio = config.walker_search_window_ratio
    adaptive_ratio = base_ratio * (1.5 - confidence * 0.5)

    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * adaptive_ratio, config.walker_min_search_window)

    while current_pos < float(limit):
        target = current_pos + step_size
        if target >= float(limit):
            cuts.append(limit)
            break

        # Could also compute local confidence and adjust window per-step
        start_search = max(int(target - search_window), int(current_pos + 1.0))
        end_search = min(int(target + search_window), limit)

        # ... rest of walk logic ...

    return cuts
```

**Config additions:**
```python
adaptive_search_window: bool = False
adaptive_window_min_ratio: float = 0.175
adaptive_window_max_ratio: float = 0.525
```

---

### 4. Edge-Aware Resampling

**Impact:** Low | **Effort:** Medium

During majority-vote resampling, weight central pixels higher than edge pixels. This reduces color bleeding when grid boundaries are slightly misaligned.

**Location:** `pixel_snapper/resample.py`

```python
def resample_weighted(
    quantized: Image.Image,
    col_cuts: List[int],
    row_cuts: List[int],
    edge_weight: float = 0.5
) -> Image.Image:
    """Resample with center-weighted voting to reduce edge bleeding."""
    arr = np.array(quantized, dtype=np.uint8)
    height, width = arr.shape[:2]

    out_w = len(col_cuts) - 1
    out_h = len(row_cuts) - 1
    output = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for j in range(out_h):
        for i in range(out_w):
            x0, x1 = col_cuts[i], col_cuts[i + 1]
            y0, y1 = row_cuts[j], row_cuts[j + 1]

            cell = arr[y0:y1, x0:x1]
            cell_h, cell_w = cell.shape[:2]

            if cell_h == 0 or cell_w == 0:
                continue

            # Create weight matrix: 1.0 at center, edge_weight at edges
            wy = np.ones(cell_h)
            wx = np.ones(cell_w)
            if cell_h > 2:
                wy[0] = wy[-1] = edge_weight
            if cell_w > 2:
                wx[0] = wx[-1] = edge_weight
            weights = np.outer(wy, wx)

            # Weighted voting (pack RGBA, accumulate weights per color)
            packed = (cell[:,:,0].astype(np.uint32) << 24 |
                     cell[:,:,1].astype(np.uint32) << 16 |
                     cell[:,:,2].astype(np.uint32) << 8 |
                     cell[:,:,3].astype(np.uint32))

            unique_colors, inverse = np.unique(packed, return_inverse=True)
            color_weights = np.zeros(len(unique_colors))
            np.add.at(color_weights, inverse.ravel(), weights.ravel())

            best_color = unique_colors[np.argmax(color_weights)]
            output[j, i] = [
                (best_color >> 24) & 0xFF,
                (best_color >> 16) & 0xFF,
                (best_color >> 8) & 0xFF,
                best_color & 0xFF
            ]

    return Image.fromarray(output, mode="RGBA")
```

**Config additions:**
```python
edge_aware_resampling: bool = False
resampling_edge_weight: float = 0.5
```

---

## Additional Ideas (Not Fully Designed)

### 5. Color-Space Gradient Detection

Compute gradients in LAB color space instead of grayscale for more perceptually accurate edge detection.

### 6. Multi-Scale Analysis

Run grid detection at multiple image scales (0.5x, 1x, 2x) and combine results for more robust detection.

### 7. Machine Learning Grid Detection

Train a small CNN to predict grid cell size and offset directly from the image. Would require training data.

### 8. Grid Regularity Constraints

Add a regularization term that penalizes non-uniform cell sizes, encouraging more consistent grids.

---

## Implementation Priority

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| 1 | Profile Smoothing | Easy win, reduces noise sensitivity |
| 2 | Sobel Kernels | Better edge detection with minimal code change |
| 3 | Edge-Aware Resampling | Fixes visible artifacts at cell boundaries |
| 4 | Adaptive Search Window | Minor improvement, more complexity |
