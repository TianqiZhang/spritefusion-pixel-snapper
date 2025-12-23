# Work Log

Project: spritefusion-pixel-snapper

Summary:
- Added Python CLI in `pixel_snapper.py` alongside Rust, with `--preview` window.
- Added timing breakdown (`--timing`) and NumPy toggle (`--no-numpy`) for perf checks.
- Added palette CSV support via `--palette` (path or name in `colors/`).
- Palette matching supports `--palette-space lab|rgb` (default `lab`), using CSV Lab fields.
- Timing notes: NumPy path is much faster for quantize/profiles; resample can be slower.

Behavior notes:
- Color differences can appear due to palette whites not being pure white.
- Skin tone test showed near-equal colors (#fee5c9 vs #fee5c8); perceived shift was mostly background.

Repo state notes:
- Fork remote: `https://github.com/TianqiZhang/spritefusion-pixel-snapper.git` (remote name `fork`).
- Untracked images used for testing were not committed (zelda images).
