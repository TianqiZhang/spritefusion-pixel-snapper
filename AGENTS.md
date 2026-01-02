# Pixel Snapper Agent Guide

## Scope
- Internal notes for contributors. Usage and CLI flags live in `README.md`.

## Entry points
- CLI: `pixel_snapper/cli.py` (with `pixel_snapper.py` as a legacy wrapper).
- Public API (POC; not stable): `pixel_snapper/__init__.py` (common entry points like
  `process_image_bytes*`, `process_image`, `main`).

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

## Config notes (tuning)
- CLI flags map to `Config`; see `README.md` for the flag list.
- Low-level tuning (not exposed in CLI): `peak_threshold_multiplier`,
  `peak_distance_filter`, `walker_*`, `max_step_ratio`,
  `uniformity_candidate_steps`.
- Grid selection toggles: `use_autocorrelation`, `autocorr_min_confidence`,
  `detect_grid_offset`, `use_uniformity_scoring`, `resolution_hint`.

## Qwen / DashScope integration
- Implemented in `pixel_snapper/qwen.py` and invoked early in
  `process_image_bytes_with_grid`.
- Requires `DASHSCOPE_API_KEY` (or `Config.qwen_api_key`) and network access.
- Request/response shape is summarized in `docs/aliyun_model.md`.

## Palettes
- Palette CSVs live in `colors/`.
- Column layout (0-based indexes): `[reference_code, name, symbol, rgb_r, rgb_g, rgb_b,
  hsl_h, hsl_s, hsl_l, lab_l, lab_a, lab_b, contributor]`.
- `pixel_snapper/palette.py` reads RGB from columns 3-5 and LAB from columns 9-11.
- Palettes can be referenced by name (`perler`) or by explicit path.

## Tests and fixtures
- Pytest config: `pytest.ini` (`-v --tb=short`).
- Fixtures in `tests/conftest.py` generate synthetic grid images.
- `testdata/` contains manual test images used by `benchmark_scoring.py` (not used in tests).
- Benchmark script: `benchmark_scoring.py`.

## Common pitfalls
- `--preview` uses `PIL.Image.show` which requires a GUI; WSL shells often fail to open it.
- OpenCV is optional; if not installed, Hough detection is skipped (`pixel_snapper/hough.py`).
- Image dimensions over 10000x10000 raise `PixelSnapperError`.

## Useful entry points for modifications
- Add new detection logic: `pixel_snapper/grid.py` + `pixel_snapper/scoring.py`.
- Tweak profile/edge detection: `pixel_snapper/profile.py`.
- Change quantization behavior: `pixel_snapper/quantize.py`.
- Extend CLI flags: `pixel_snapper/cli.py` and `pixel_snapper/config.py`.
