# CLAUDE.md — Pixel Snapper

## What this project does

Converts messy AI-generated pixel art into perfectly gridded pixel art.
Detects grid boundaries, quantizes colors, and resamples to a clean grid.

## Running tests

```bash
python -m pytest tests/ -v
```

## Running the CLI

```bash
python -m pixel_snapper input.png output.png [k_colors] \
  [--palette NAME] [--resample auto|majority|center|mean|palette_aware] \
  [--pattern-out PATH] [--pattern-format pdf|png] \
  [--palette-space rgb|lab] [--resolution-hint N] \
  [--preview] [--debug] [--timing] [--qwen]
```

## Module map

### Library (`pixel_snapper/`)

| Module | Purpose |
|--------|---------|
| `__init__.py` | Public API re-exports |
| `__main__.py` | `python -m pixel_snapper` entry point |
| `pipeline.py` | Core processing: `process_image_bytes`, `process_image_bytes_with_grid`, `ProcessingResult` |
| `cli.py` | Argument parsing, preview rendering, `main()` |
| `config.py` | `Config` dataclass, `PixelSnapperError`, validation |
| `grid.py` | Step size estimation (peaks, autocorrelation), offset detection, walk/stabilize |
| `scoring.py` | Candidate scoring: uniformity, edge alignment, size penalty |
| `candidates.py` | Candidate generation, deduplication, filtering |
| `profile.py` | Gradient profile computation for edge detection |
| `quantize.py` | K-means++ color quantization or palette matching |
| `resample.py` | Grid cell resampling (majority, mean, center, palette_aware) + fidelity metric |
| `reconstruction.py` | Reconstruction-based grid detection |
| `hough.py` | OpenCV Hough-based grid detection (optional, needs opencv-python) |
| `palette.py` | CSV palette loading and resolution (lru_cached) |
| `pattern.py` | Bead pattern rendering (PDF/PNG) |
| `color.py` | RGB ↔ LAB color conversion |
| `ground_truth.py` | Ground truth data model, JSON I/O, accuracy metrics, `get_test_images()` |
| `qwen.py` | DashScope/Qwen image-edit integration |

### Tools (`tools/`)

| Script | Purpose |
|--------|---------|
| `benchmark.py` | Benchmark grid detection across testdata images |
| `ground_truth_editor.py` | Tkinter GUI for creating/editing ground truth cuts |
| `gt_eval.py` | Evaluate pipeline against ground truth (`--verbose`, `--sweep`, `--palette NAME`) |

### Tests (`tests/`)

Pytest suite. Fixtures in `conftest.py` generate synthetic grid images.
Config in `pytest.ini`.

### Data

- `colors/` — Palette CSV files (Perler, Artkal, Hama, etc.)
- `testdata/` — Manual test images for benchmarks
- `testdata/ground_truth/` — Human-verified grid cut JSON files

## Pipeline overview

1. (Optional) Qwen pre-processing: photo → pixel art
2. Load image, validate dimensions
3. Quantize colors (K-means++ or palette matching)
4. Compute gradient profiles (edge detection)
5. Estimate step sizes (autocorrelation + peak-based)
6. Generate candidate grids (autocorr, peaks, Hough, reconstruction, fixed steps)
7. Score candidates (uniformity + edge alignment + size penalty)
8. Stabilize winning grid
9. Resample to final output

## Resampling

`--resample` controls the per-cell color selection strategy. Default is `auto`.

| Method | Behavior |
|--------|----------|
| `auto` | `palette_aware` when `--palette` is set, `majority` otherwise |
| `majority` | Most common color per cell |
| `mean` | Alpha-weighted mean RGB (may produce non-palette colors) |
| `center` | Center pixel, falls back to majority if transparent |
| `palette_aware` | Mean RGB → snap to nearest palette color in LAB. Requires `--palette` |

Fidelity is measured via `compute_fidelity()` in `resample.py`: Lanczos downscale as reference, Delta-E (CIE76) in LAB space. Accepts optional palette arrays to measure end-to-end quality after palette mapping.

## Key conventions

- All processing functions take/return `bytes` (PNG/JPEG)
- `Config` dataclass holds all tuning knobs
- Grid cuts are integer pixel positions, always sorted, always include 0 and dimension
- Debug logging via `logging.getLogger("pixel_snapper")`
- OpenCV is optional — Hough detection is skipped if not installed
