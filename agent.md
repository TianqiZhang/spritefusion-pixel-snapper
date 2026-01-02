# Pixel Snapper Agent Guide

## What this repo is
- Converts imperfect pixel art into a clean, uniform grid with a constrained palette.
- CLI lives in `pixel_snapper/cli.py`; `pixel_snapper.py` is a legacy entry point that forwards to the package.
- Core pipeline: quantize -> edge profiles -> grid detection + scoring -> stabilization -> resample.

## Quick start (local)
- Install deps: `pip install -r requirements-dev.txt`
- Basic run: `python pixel_snapper.py input.png output.png`
- Add palette: `python pixel_snapper.py input.png output.png --palette perler`
- Preview: `python pixel_snapper.py input.png output.png --preview` (uses `PIL.Image.show`, needs a GUI)
- Tests: `pytest tests/ -v`
- Benchmark grid scoring: `python benchmark_scoring.py`

## Data flow (high level)
1. Optional Qwen pre-processing (photo -> pixel art) in `pixel_snapper/qwen.py`.
2. Load image + validate size in `pixel_snapper/cli.py` and `pixel_snapper/config.py`.
3. Quantize colors with K-means or a fixed palette in `pixel_snapper/quantize.py`.
4. Compute gradient profiles in `pixel_snapper/profile.py`.
5. Estimate step size (autocorr + peaks), generate candidate grids, and score in
   `pixel_snapper/grid.py` and `pixel_snapper/scoring.py`.
6. Stabilize grid across axes in `pixel_snapper/grid.py`.
7. Resample by majority vote in `pixel_snapper/resample.py`.
8. Output PNG bytes to file or caller.

## Key modules and responsibilities
- `pixel_snapper/cli.py`: argument parsing, preview rendering, orchestrates pipeline.
- `pixel_snapper/config.py`: `Config` dataclass; validation and defaults.
- `pixel_snapper/grid.py`: step size estimation, offset detection, walk/stabilize cuts.
- `pixel_snapper/scoring.py`: candidate scoring (uniformity + edge alignment + size penalty).
- `pixel_snapper/profile.py`: gradient profiles for edge detection.
- `pixel_snapper/quantize.py`: K-means++ quantization or palette matching.
- `pixel_snapper/palette.py`: CSV palette resolution and loading.
- `pixel_snapper/hough.py`: OpenCV Hough-based grid detection (optional).
- `pixel_snapper/qwen.py`: DashScope/Qwen image-edit integration.

## Config and flags (core)
- `Config.k_colors`: color count for K-means (ignored when using palette).
- `Config.palette` + `Config.palette_space`: palette name/path and matching space (`rgb` or `lab`).
- `Config.use_autocorrelation`, `Config.autocorr_min_confidence`: autocorr step size detection.
- `Config.use_uniformity_scoring`: candidate grid scoring on/off.
- `Config.detect_grid_offset`: finds best grid offset for margin-heavy images.
- `Config.resolution_hint`: max cells on long axis; used to filter candidates.
- `Config.timing`/`--timing`: prints step timings in `process_image_bytes_with_grid`.
- `--debug`: enables `pixel_snapper` logger debug output.

## Qwen / DashScope integration
- Enabled via `--qwen` or any `--qwen-*` option.
- Requires `DASHSCOPE_API_KEY` env var (or `Config.qwen_api_key`).
- Default endpoint in CLI is Singapore (`dashscope-intl`); docs in `docs/aliyun_model.md`.
- Network access is required; in sandboxed environments this may need approval.

## Palettes
- Palette CSVs live in `colors/`.
- `pixel_snapper/palette.py` expects RGB in columns 3-5 and LAB in columns 9-11.
- Palettes can be referenced by name (`perler`) or by explicit path.

## Tests and fixtures
- Pytest config: `pytest.ini` (`-v --tb=short`).
- Fixtures in `tests/conftest.py` generate synthetic grid images.
- `testdata/` contains sample inputs and expected outputs for integration tests.

## Common pitfalls
- `--preview` uses `PIL.Image.show` which requires a GUI; WSL shells often fail to open it.
- OpenCV is optional; if not installed, Hough detection is skipped (`pixel_snapper/hough.py`).
- Image dimensions over 10000x10000 raise `PixelSnapperError`.

## Useful entry points for modifications
- Add new detection logic: `pixel_snapper/grid.py` + `pixel_snapper/scoring.py`.
- Tweak profile/edge detection: `pixel_snapper/profile.py`.
- Change quantization behavior: `pixel_snapper/quantize.py`.
- Extend CLI flags: `pixel_snapper/cli.py` and `pixel_snapper/config.py`.
