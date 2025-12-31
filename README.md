# Pixel Snapper

A Python tool to convert messy, AI-generated pixel art into perfectly gridded pixel art.

## Why?

**Current AI image models can't generate proper grid-based pixel art:**

- Pixels are inconsistent in size and position
- Grid resolution drifts across the image
- Colors aren't tied to a strict palette

**Pixel Snapper fixes this:**

- Detects and snaps pixels to a perfect grid
- Quantizes colors using K-means clustering
- Optionally maps colors to craft bead palettes (Perler, Artkal, Hama, etc.)

## Installation

```bash
pip install numpy pillow
```

For development (includes pytest):
```bash
pip install -r requirements-dev.txt
```

## Usage

### Command Line

```bash
# Basic usage
python pixel_snapper.py input.png output.png

# Specify number of colors (default: 16)
python pixel_snapper.py input.png output.png 8

# Use a bead palette
python pixel_snapper.py input.png output.png --palette perler

# Preview with grid visualization
python pixel_snapper.py input.png output.png --preview

# Show timing information
python pixel_snapper.py input.png output.png --timing
```

### Options

| Option | Description |
|--------|-------------|
| `k_colors` | Number of colors for K-means quantization (default: 16) |
| `--palette NAME` | Use a predefined palette (perler, artkal_a, hama, etc.) |
| `--palette-space rgb\|lab` | Color matching space (default: lab) |
| `--preview` | Show side-by-side preview with grid overlay |
| `--timing` | Print timing breakdown |

### Python API

```python
from pixel_snapper import Config, process_image_bytes

# Load image
with open("input.png", "rb") as f:
    input_bytes = f.read()

# Process with default settings
output_bytes = process_image_bytes(input_bytes)

# Or customize
config = Config(k_colors=8, palette="perler")
output_bytes = process_image_bytes(input_bytes, config)

# Save result
with open("output.png", "wb") as f:
    f.write(output_bytes)
```

For access to detected grid information:

```python
from pixel_snapper import Config, process_image_bytes_with_grid

result = process_image_bytes_with_grid(input_bytes)
print(f"Detected grid: {len(result.col_cuts)-1}x{len(result.row_cuts)-1} cells")
print(f"Column boundaries: {result.col_cuts}")
print(f"Row boundaries: {result.row_cuts}")
```

## Available Palettes

| Palette | Colors | Description |
|---------|--------|-------------|
| `perler` | 62 | Perler beads |
| `perler_mini` | 55 | Perler mini beads |
| `perler_caps` | 19 | Perler caps |
| `artkal_a` | 138 | Artkal A-series (2.6mm) |
| `artkal_c` | 132 | Artkal C-series (2.6mm) |
| `artkal_r` | 99 | Artkal R-series (5mm) |
| `artkal_s` | 206 | Artkal S-series (5mm) |
| `hama` | 68 | Hama beads |
| `nabbi` | 55 | Nabbi beads |
| `diamondDotz` | 456 | Diamond Dotz |
| `yant` | 163 | Yant beads |

## How It Works

1. **Color Quantization**: Reduces colors using K-means++ clustering or maps to a fixed palette
2. **Edge Detection**: Computes gradient profiles to find grid boundaries
3. **Grid Detection**: Multiple methods compete to find the best grid (see below)
4. **Elastic Walking**: Traces grid boundaries, snapping to detected edges
5. **Stabilization**: Ensures consistent grid across both axes
6. **Resampling**: Extracts one color per cell using majority vote

## Grid Detection Methods

Pixel Snapper uses multiple grid detection methods and scores them to find the best fit. Each candidate grid is evaluated on **uniformity** (cells should have consistent colors) and **edge alignment** (grid lines should match detected edges).

### Method Comparison

Tested across 8 sample images (average combined score, higher is better):

| Method | Avg Score | Description |
|--------|-----------|-------------|
| **autocorr** | **1.162** | FFT-based autocorrelation to find periodic patterns |
| hough | 1.074 | Canny edge detection + Hough line transform |
| fixed(8) | 1.057 | Fixed 8px cell size |
| fixed(16) | 1.054 | Fixed 16px cell size |
| peak-based | 1.042 | Median spacing between gradient peaks |
| fixed(32) | 0.765 | Fixed 32px cell size |

### Key Insights

- **Autocorrelation performs best overall** — highest average score and most consistent (narrow min-max range)
- **Hough transform is #2** — works well when clear pixel edges exist
- **Fixed step sizes are competitive** — useful fallbacks when signal-based detection fails
- **Peak-based has failure modes** — can produce very wrong grids on images without clear periodicity
- **No single method wins all images** — the multi-method scoring approach is essential

## Project Structure

```
pixel_snapper/
├── __init__.py     # Public API
├── cli.py          # CLI entry point
├── config.py       # Configuration
├── color.py        # RGB/LAB conversion
├── palette.py      # Palette loading
├── quantize.py     # K-means quantization
├── profile.py      # Edge detection
├── grid.py         # Grid detection (autocorr, peak-based)
├── hough.py        # Hough transform grid detection
├── scoring.py      # Grid candidate scoring
└── resample.py     # Majority-vote resampling

tests/              # Comprehensive test suite (157 tests)
colors/             # Bead palette CSV files
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License
