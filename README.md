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
pip install numpy Pillow
```

Optional (enable Hough-based detection):
```bash
pip install opencv-python
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

# Use a bead palette and emit a printable pattern PDF
python pixel_snapper.py input.png output.png --palette perler --pattern-format pdf

# By default, pattern output is saved as output_pattern.pdf

# Override bead pattern output path or format
python pixel_snapper.py input.png output.png --palette perler --pattern-out pattern.pdf

# Preview with grid visualization
python pixel_snapper.py input.png output.png --preview

# Show timing information
python pixel_snapper.py input.png output.png --timing

# Provide a resolution hint (max cells on long axis)
python pixel_snapper.py input.png output.png --resolution-hint 128

# Enable debug logging
python pixel_snapper.py input.png output.png --debug

# Photo -> Qwen cute pixel art -> Pixel Snapper
# Requires DASHSCOPE_API_KEY in environment
python pixel_snapper.py photo.jpg output.png --qwen
```

### Options

| Option | Description |
|--------|-------------|
| `k_colors` | Number of colors for K-means quantization (default: 16) |
| `--palette NAME` | Use a predefined palette (perler, artkal_a, hama, etc.) |
| `--pattern-out PATH` | Override bead pattern output path (requires `--palette`) |
| `--pattern-format pdf\|png` | Bead pattern output format (default: pdf) |
| `--palette-space rgb\|lab` | Color matching space (default: lab) |
| `--resolution-hint N` | Hint for max cells on the long axis (filters candidates) |
| `--preview` | Show preview with grid overlay (requires GUI) |
| `--timing` | Print timing breakdown |
| `--debug` | Enable debug logging |
| `--qwen` | Enable Qwen image edit pre-step for photo-to-pixel-art |

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

# Photo -> Qwen cute pixel art -> Pixel Snapper
config = Config(qwen_enabled=True)
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
3. **Grid Detection**: Multiple methods compete to find the best grid
4. **Elastic Walking**: Traces grid boundaries, snapping to detected edges
5. **Stabilization**: Ensures consistent grid across both axes
6. **Resampling**: Extracts one color per cell using majority vote

Grid detection uses multiple candidate generators (autocorr, peak-based, Hough,
fixed step sizes) and scores them for the best fit. See
`benchmark_results.json` for the latest benchmark data.

## Project Notes

See `AGENTS.md` for a module map and contributor-focused notes.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License
