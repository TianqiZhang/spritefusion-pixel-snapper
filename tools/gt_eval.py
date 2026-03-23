#!/usr/bin/env python
"""Ground-truth evaluation: measure pipeline accuracy against human-verified grids.

Runs each ground-truth image through the pipeline, compares predicted cuts against
GT cuts, performs oracle analysis (does any candidate match GT?), and computes
cell-level diffs between predicted and GT resampled outputs.

Usage:
    python tools/gt_eval.py [--verbose]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixel_snapper import Config, process_image_bytes_with_grid
from pixel_snapper.ground_truth import (
    GroundTruth,
    cut_position_error,
    grid_accuracy,
    ground_truth_dir,
    load_ground_truth,
)
from pixel_snapper.resample import resample, resample_palette_aware, compute_fidelity
from pixel_snapper.scoring import ScoredCandidate, score_all_candidates
from pixel_snapper.profile import compute_profiles
from pixel_snapper.palette import load_palette


def cell_diff(
    img: Image.Image,
    pred_cols: List[int],
    pred_rows: List[int],
    gt_cols: List[int],
    gt_rows: List[int],
) -> Dict[str, Any]:
    """Resample image with both grids and count differing cells.

    Only comparable when cell counts match. Returns None-like results otherwise.
    """
    pred_cells = (len(pred_cols) - 1, len(pred_rows) - 1)
    gt_cells = (len(gt_cols) - 1, len(gt_rows) - 1)

    if pred_cells != gt_cells:
        return {
            "comparable": False,
            "pred_cells": f"{pred_cells[0]}x{pred_cells[1]}",
            "gt_cells": f"{gt_cells[0]}x{gt_cells[1]}",
        }

    pred_img = resample(img, pred_cols, pred_rows)
    gt_img = resample(img, gt_cols, gt_rows)

    pred_arr = np.array(pred_img)
    gt_arr = np.array(gt_img)

    # Compare cell by cell (exact match on RGBA)
    diff_mask = np.any(pred_arr != gt_arr, axis=2)
    n_diff = int(np.sum(diff_mask))
    n_total = pred_cells[0] * pred_cells[1]

    return {
        "comparable": True,
        "total_cells": n_total,
        "differing_cells": n_diff,
        "match_rate": round(1.0 - n_diff / n_total, 4) if n_total > 0 else 0.0,
    }


def oracle_analysis(
    scored_candidates: List[ScoredCandidate],
    gt: GroundTruth,
) -> Dict[str, Any]:
    """Find which candidate best matches GT, regardless of ranking.

    Returns info about the best-matching candidate and whether it was ranked #1.
    """
    if not scored_candidates:
        return {"no_candidates": True}

    best_match: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    all_candidates_gt: List[Dict[str, Any]] = []

    for c in scored_candidates:
        col_err = cut_position_error(list(c.col_cuts), gt.col_cuts)
        row_err = cut_position_error(list(c.row_cuts), gt.row_cuts)

        # Combined error: mean of both axes' mean errors + penalty for missing/extra
        mean_err = (col_err["mean_abs_error"] + row_err["mean_abs_error"]) / 2
        missing_penalty = (col_err["missing_cuts"] + row_err["missing_cuts"]) * 5.0
        extra_penalty = (col_err["extra_cuts"] + row_err["extra_cuts"]) * 2.0
        combined = mean_err + missing_penalty + extra_penalty

        cand_info = {
            "source": c.source,
            "rank": c.rank,
            "cells": f"{c.cells_x}x{c.cells_y}",
            "combined_score": round(c.combined_score, 4),
            "uniformity": round(c.uniformity_score, 4),
            "edge": round(c.edge_score, 4),
            "gt_col_mean_err": col_err["mean_abs_error"],
            "gt_row_mean_err": row_err["mean_abs_error"],
            "gt_combined_error": round(combined, 2),
        }
        all_candidates_gt.append(cand_info)

        if combined < best_score:
            best_score = combined
            best_match = {
                "source": c.source,
                "rank": c.rank,
                "combined_score": round(c.combined_score, 4),
                "gt_col_error": col_err,
                "gt_row_error": row_err,
                "gt_combined_error": round(combined, 2),
                "cells": f"{c.cells_x}x{c.cells_y}",
            }

    # Also get the winner's (rank=1) GT error for comparison
    winner = scored_candidates[0]
    winner_col_err = cut_position_error(list(winner.col_cuts), gt.col_cuts)
    winner_row_err = cut_position_error(list(winner.row_cuts), gt.row_cuts)

    return {
        "best_candidate": best_match,
        "best_is_winner": best_match["rank"] == 1 if best_match else False,
        "winner_source": winner.source,
        "winner_rank_1_gt_error": {
            "col": winner_col_err,
            "row": winner_row_err,
        },
        "all_candidates_gt": all_candidates_gt,
    }


def evaluate_image(
    image_path: Path, gt: GroundTruth, config: Config, verbose: bool = False,
    palette_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full evaluation on a single GT image."""
    with open(image_path, "rb") as f:
        input_bytes = f.read()

    result = process_image_bytes_with_grid(input_bytes, config)

    # 1. Winner accuracy vs GT (these are POST-stabilization cuts)
    accuracy = grid_accuracy(
        list(result.col_cuts), list(result.row_cuts),
        gt.col_cuts, gt.row_cuts,
    )

    # 2. Oracle analysis (these are PRE-stabilization candidate cuts)
    oracle = oracle_analysis(result.scored_candidates or [], gt)

    # 3. Check if stabilization helped or hurt
    winner = result.scored_candidates[0] if result.scored_candidates else None
    stabilization_info = None
    if winner:
        pre_accuracy = grid_accuracy(
            list(winner.col_cuts), list(winner.row_cuts),
            gt.col_cuts, gt.row_cuts,
        )
        pre_col_err = pre_accuracy["col_error"]["mean_abs_error"]
        pre_row_err = pre_accuracy["row_error"]["mean_abs_error"]
        post_col_err = accuracy["col_error"]["mean_abs_error"]
        post_row_err = accuracy["row_error"]["mean_abs_error"]
        stabilization_info = {
            "pre_cells": pre_accuracy["pred_cells"],
            "post_cells": accuracy["pred_cells"],
            "pre_col_err": pre_col_err,
            "pre_row_err": pre_row_err,
            "post_col_err": post_col_err,
            "post_row_err": post_row_err,
            "helped": (post_col_err + post_row_err) <= (pre_col_err + pre_row_err),
        }

    # 4. Cell-level diff (using quantized image for fair comparison)
    img = result.quantized_img
    cdiff = cell_diff(img, list(result.col_cuts), list(result.row_cuts), gt.col_cuts, gt.row_cuts)

    # 5. Fidelity measurement (how well resampled output preserves visual appearance)
    original_img = Image.open(image_path).convert("RGBA")
    resampled_img = resample(img, list(result.col_cuts), list(result.row_cuts))
    fidelity = compute_fidelity(original_img, resampled_img, list(result.col_cuts), list(result.row_cuts))

    # 6. End-to-end fidelity with palette (if provided)
    palette_fidelity = {}
    if palette_name:
        palette = load_palette(palette_name)
        palette_rgb = np.array(palette.rgb, dtype=np.float64)
        palette_lab = np.array(palette.lab, dtype=np.float64)

        # Majority vote resampled -> palette mapped (what the pipeline does without palette-aware)
        majority_fid = compute_fidelity(
            original_img, resampled_img, list(result.col_cuts), list(result.row_cuts),
            palette_rgb=palette_rgb, palette_lab=palette_lab,
        )
        palette_fidelity["majority"] = {
            "mean_delta_e": round(majority_fid.mean_delta_e, 2),
            "p90_delta_e": round(majority_fid.p90_delta_e, 2),
        }

        # Palette-aware resampling (mean -> snap to palette)
        pa_img = resample_palette_aware(
            img, list(result.col_cuts), list(result.row_cuts),
            palette_rgb, palette_lab,
        )
        pa_fid = compute_fidelity(
            original_img, pa_img, list(result.col_cuts), list(result.row_cuts),
            palette_rgb=palette_rgb, palette_lab=palette_lab,
        )
        palette_fidelity["palette_aware"] = {
            "mean_delta_e": round(pa_fid.mean_delta_e, 2),
            "p90_delta_e": round(pa_fid.p90_delta_e, 2),
        }

    return {
        "filename": image_path.name,
        "winner_source": result.scored_candidates[0].source if result.scored_candidates else "none",
        "winner_cells": f"{len(result.col_cuts)-1}x{len(result.row_cuts)-1}",
        "gt_cells": f"{gt.cells_x}x{gt.cells_y}",
        "gt_algorithm": gt.metadata.get("algorithm_source", "unknown"),
        "accuracy": accuracy,
        "oracle": oracle,
        "stabilization": stabilization_info,
        "cell_diff": cdiff,
        "fidelity": {
            "mean_delta_e": round(fidelity.mean_delta_e, 2),
            "median_delta_e": round(fidelity.median_delta_e, 2),
            "p90_delta_e": round(fidelity.p90_delta_e, 2),
            "max_delta_e": round(fidelity.max_delta_e, 2),
            "num_cells": fidelity.num_cells,
        },
        "palette_fidelity": palette_fidelity,
        "num_candidates": len(result.scored_candidates) if result.scored_candidates else 0,
    }


def print_image_result(r: Dict[str, Any], verbose: bool = False) -> None:
    """Print evaluation result for one image."""
    acc = r["accuracy"]
    oracle = r["oracle"]
    cdiff = r["cell_diff"]

    print(f"\n{'='*60}")
    print(f"  {r['filename']}")
    print(f"{'='*60}")

    # Cell count
    match_str = "MATCH" if acc["cell_count_match"] else "MISMATCH"
    print(f"  Cells: predicted {acc['pred_cells']} vs GT {acc['gt_cells']} [{match_str}]")

    # Winner info
    print(f"  Winner: {r['winner_source']} (out of {r['num_candidates']} candidates)")
    print(f"  GT was created from: {r['gt_algorithm']}")

    # Cut position errors
    col_e = acc["col_error"]
    row_e = acc["row_error"]
    print(f"  Cut errors (winner):")
    print(f"    col: mean={col_e['mean_abs_error']:.1f}px  max={col_e['max_error']}px  missing={col_e['missing_cuts']}  extra={col_e['extra_cuts']}")
    print(f"    row: mean={row_e['mean_abs_error']:.1f}px  max={row_e['max_error']}px  missing={row_e['missing_cuts']}  extra={row_e['extra_cuts']}")

    # Oracle analysis
    best = oracle.get("best_candidate")
    if best:
        if oracle["best_is_winner"]:
            print(f"  Oracle: winner IS the best match for GT")
        else:
            print(f"  Oracle: BETTER candidate exists!")
            print(f"    Best match: {best['source']} (rank #{best['rank']}, score={best['combined_score']})")
            bc_col = best["gt_col_error"]
            bc_row = best["gt_row_error"]
            print(f"    Its GT errors: col={bc_col['mean_abs_error']:.1f}px  row={bc_row['mean_abs_error']:.1f}px")

    # Cell diff
    if cdiff.get("comparable"):
        pct = cdiff["match_rate"] * 100
        print(f"  Cell match rate: {pct:.1f}% ({cdiff['differing_cells']}/{cdiff['total_cells']} differ)")
    else:
        print(f"  Cell diff: not comparable (different grid sizes)")

    # Stabilization
    stab = r.get("stabilization")
    if stab:
        if stab["pre_cells"] != stab["post_cells"]:
            label = "CHANGED grid" if not stab["helped"] else "changed grid"
            print(f"  Stabilization: {label} {stab['pre_cells']} -> {stab['post_cells']}")
            print(f"    Error before: col={stab['pre_col_err']:.1f}px  row={stab['pre_row_err']:.1f}px")
            print(f"    Error after:  col={stab['post_col_err']:.1f}px  row={stab['post_row_err']:.1f}px")
            print(f"    Verdict: {'helped' if stab['helped'] else 'HURT'}")
        else:
            print(f"  Stabilization: no cell count change")

    # Fidelity
    fid = r.get("fidelity")
    if fid:
        print(f"  Fidelity: mean_dE={fid['mean_delta_e']:.2f}  p90={fid['p90_delta_e']:.2f}  max={fid['max_delta_e']:.2f}")

    # Palette fidelity (end-to-end after palette mapping)
    pfid = r.get("palette_fidelity")
    if pfid:
        maj = pfid.get("majority", {})
        pa = pfid.get("palette_aware", {})
        if maj and pa:
            diff = maj["mean_delta_e"] - pa["mean_delta_e"]
            better = "palette_aware" if diff > 0 else "majority"
            print(f"  Palette fidelity (end-to-end):")
            print(f"    majority:      mean_dE={maj['mean_delta_e']:.2f}  p90={maj['p90_delta_e']:.2f}")
            print(f"    palette_aware: mean_dE={pa['mean_delta_e']:.2f}  p90={pa['p90_delta_e']:.2f}  [{better} wins by {abs(diff):.2f}]")

    if verbose:
        all_cands = oracle.get("all_candidates_gt", [])
        if all_cands:
            print(f"\n  All candidates vs GT (sorted by scoring rank):")
            for c in sorted(all_cands, key=lambda x: x["rank"]):
                marker = " <-- best GT match" if c["gt_combined_error"] == (best["gt_combined_error"] if best else -1) else ""
                print(f"    #{c['rank']:2d} {c['source']:30s} {c['cells']:10s} "
                      f"score={c['combined_score']:+.4f}  "
                      f"unif={c['uniformity']:.1f}  edge={c['edge']:.3f}  "
                      f"gt_err: col={c['gt_col_mean_err']:.1f}px row={c['gt_row_mean_err']:.1f}px{marker}")


def run_weight_sweep(
    testdata_dir: Path, gt_dir: Path, gt_files: List[Path], verbose: bool,
) -> None:
    """Re-score cached candidates with different weight combos to find optimal weights."""
    import io
    from pixel_snapper.candidates import (
        deduplicate_candidates,
        filter_by_resolution_hint,
        generate_candidates,
    )
    from pixel_snapper.grid import (
        estimate_step_size,
        estimate_step_size_autocorr,
        resolve_step_sizes,
    )
    from pixel_snapper.scoring import compute_expected_step
    from pixel_snapper.quantize import quantize_image

    config = Config(
        use_uniformity_scoring=True,
        use_autocorrelation=True,
    )

    # Step 1: Run pipeline once per image, cache intermediate data
    print("Caching candidates for all GT images...")
    cached = []  # (gt, quantized_img, profile_x, profile_y, candidates, expected_step, w, h)

    for gt_path in gt_files:
        gt = load_ground_truth(gt_path)
        image_path = testdata_dir / gt.image_file
        if not image_path.exists():
            continue

        with open(image_path, "rb") as f:
            input_bytes = f.read()

        result = process_image_bytes_with_grid(input_bytes, config)
        img = result.quantized_img
        w, h = img.size
        profile_x, profile_y = compute_profiles(img)

        # Extract raw candidates from scored candidates
        raw_candidates = []
        for c in (result.scored_candidates or []):
            raw_candidates.append((list(c.col_cuts), list(c.row_cuts), c.step_size, c.source))

        # Compute expected_step (same as pipeline)
        step_x_autocorr, conf_x = estimate_step_size_autocorr(profile_x, config)
        step_y_autocorr, conf_y = estimate_step_size_autocorr(profile_y, config)
        step_x_peaks = estimate_step_size(profile_x, config)
        step_y_peaks = estimate_step_size(profile_y, config)
        expected_step_x = compute_expected_step(step_x_autocorr, step_x_peaks, conf_x)
        expected_step_y = compute_expected_step(step_y_autocorr, step_y_peaks, conf_y)
        if expected_step_x is not None and expected_step_y is not None:
            expected_step, _ = resolve_step_sizes(expected_step_x, expected_step_y, w, h, config)
        elif expected_step_x is not None:
            expected_step = expected_step_x
        else:
            expected_step = expected_step_y

        cached.append((gt, img, profile_x, profile_y, raw_candidates, expected_step, w, h))
        print(f"  Cached {gt.image_file}: {len(raw_candidates)} candidates, expected_step={expected_step:.1f}" if expected_step else f"  Cached {gt.image_file}: {len(raw_candidates)} candidates")

    # Step 2: Sweep weights
    edge_weights = [0.0, 0.1, 0.2, 0.3, 0.5]
    size_penalty_weights = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    size_penalty_tolerances = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]

    print(f"\nSweeping {len(edge_weights)}x{len(size_penalty_weights)}x{len(size_penalty_tolerances)} = {len(edge_weights)*len(size_penalty_weights)*len(size_penalty_tolerances)} weight combinations...\n")

    all_combos = []

    for ew in edge_weights:
        for spw in size_penalty_weights:
            for spt in size_penalty_tolerances:
                total_gt_error = 0.0
                all_match = True

                for gt, img, px, py, candidates, exp_step, w, h in cached:
                    scored = score_all_candidates(
                        img, px, py, candidates, w, h,
                        uniformity_weight=1.0,
                        edge_weight=ew,
                        expected_step=exp_step,
                        size_penalty_tolerance=spt,
                        size_penalty_weight=spw,
                    )
                    if not scored:
                        continue

                    winner = scored[0]
                    col_err = cut_position_error(list(winner.col_cuts), gt.col_cuts)
                    row_err = cut_position_error(list(winner.row_cuts), gt.row_cuts)

                    mean_err = (col_err["mean_abs_error"] + row_err["mean_abs_error"]) / 2
                    missing = col_err["missing_cuts"] + row_err["missing_cuts"]
                    extra = col_err["extra_cuts"] + row_err["extra_cuts"]
                    total_gt_error += mean_err + missing * 10.0 + extra * 2.0

                    pred_cx = len(winner.col_cuts) - 1
                    pred_cy = len(winner.row_cuts) - 1
                    if pred_cx != gt.cells_x or pred_cy != gt.cells_y:
                        all_match = False

                all_combos.append((total_gt_error, ew, spw, spt, all_match))

    all_combos.sort(key=lambda x: x[0])

    print("Top 10 weight combinations:")
    for i, (err, ew, spw, spt, match) in enumerate(all_combos[:10]):
        print(f"  {i+1}. edge={ew}, penalty={spw}, tol={spt} -> error={err:.2f}  all_match={match}")

    best_metric, ew, spw, spt, match = all_combos[0]
    print(f"Best weights: edge={ew}, size_penalty={spw}, tolerance={spt}")
    print(f"  Total GT error: {best_metric:.2f}  Cell count all match: {match}")

    # Show top 10
    print(f"\nDetailed results with best weights (edge={ew}, penalty={spw}, tol={spt}):")
    for gt, img, px, py, candidates, exp_step, w, h in cached:
        scored = score_all_candidates(
            img, px, py, candidates, w, h,
            uniformity_weight=1.0,
            edge_weight=ew,
            expected_step=exp_step,
            size_penalty_tolerance=spt,
            size_penalty_weight=spw,
        )
        if not scored:
            continue
        winner = scored[0]
        col_err = cut_position_error(list(winner.col_cuts), gt.col_cuts)
        row_err = cut_position_error(list(winner.row_cuts), gt.row_cuts)
        cells = f"{winner.cells_x}x{winner.cells_y}"
        gt_cells = f"{gt.cells_x}x{gt.cells_y}"
        match_str = "MATCH" if cells.split("x") == gt_cells.split("x") or (winner.cells_x == gt.cells_x and winner.cells_y == gt.cells_y) else "MISMATCH"
        print(f"  {gt.image_file:16s}: winner={winner.source:30s} {cells} vs GT {gt_cells} [{match_str}]  "
              f"col_err={col_err['mean_abs_error']:.1f}px  row_err={row_err['mean_abs_error']:.1f}px")


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sweep = "--sweep" in sys.argv

    # Parse --palette NAME
    palette_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--palette" and i + 1 < len(sys.argv):
            palette_name = sys.argv[i + 1]
            break

    repo_root = Path(__file__).parent.parent
    testdata_dir = repo_root / "testdata"
    gt_dir = ground_truth_dir(testdata_dir)

    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print("No ground truth files found.")
        sys.exit(1)

    if sweep:
        run_weight_sweep(testdata_dir, gt_dir, gt_files, verbose)
        return

    print(f"Found {len(gt_files)} ground truth files")

    config = Config(
        use_uniformity_scoring=True,
        use_autocorrelation=True,
    )

    results = []
    for gt_path in gt_files:
        gt = load_ground_truth(gt_path)
        image_path = testdata_dir / gt.image_file

        if not image_path.exists():
            print(f"  SKIP {gt.image_file}: image not found")
            continue

        print(f"Processing {gt.image_file}...", end=" ", flush=True)
        r = evaluate_image(image_path, gt, config, verbose, palette_name=palette_name)
        results.append(r)
        print("done")

    # Print per-image results
    for r in results:
        print_image_result(r, verbose)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    n = len(results)
    if n == 0:
        print("  No images evaluated.")
        return

    # Aggregate metrics
    cell_matches = sum(1 for r in results if r["accuracy"]["cell_count_match"])
    oracle_wins = sum(1 for r in results if r["oracle"].get("best_is_winner", False))

    col_errors = [r["accuracy"]["col_error"]["mean_abs_error"] for r in results
                  if r["accuracy"]["col_error"]["mean_abs_error"] != float("inf")]
    row_errors = [r["accuracy"]["row_error"]["mean_abs_error"] for r in results
                  if r["accuracy"]["row_error"]["mean_abs_error"] != float("inf")]

    cell_match_rates = [r["cell_diff"]["match_rate"] for r in results
                        if r["cell_diff"].get("comparable", False)]

    fidelities = [r["fidelity"]["mean_delta_e"] for r in results
                  if r.get("fidelity") and r["fidelity"]["mean_delta_e"] < float("inf")]

    print(f"  Cell count match: {cell_matches}/{n}")
    print(f"  Scoring picks best candidate: {oracle_wins}/{n}")

    if col_errors:
        print(f"  Avg col cut error: {sum(col_errors)/len(col_errors):.2f}px")
    if row_errors:
        print(f"  Avg row cut error: {sum(row_errors)/len(row_errors):.2f}px")
    if cell_match_rates:
        print(f"  Avg cell match rate: {sum(cell_match_rates)/len(cell_match_rates)*100:.1f}%")
    if fidelities:
        print(f"  Avg fidelity (mean dE): {sum(fidelities)/len(fidelities):.2f}")

    # Palette fidelity summary
    maj_fids = [r["palette_fidelity"]["majority"]["mean_delta_e"] for r in results
                if r.get("palette_fidelity") and "majority" in r["palette_fidelity"]]
    pa_fids = [r["palette_fidelity"]["palette_aware"]["mean_delta_e"] for r in results
               if r.get("palette_fidelity") and "palette_aware" in r["palette_fidelity"]]
    if maj_fids and pa_fids:
        avg_maj = sum(maj_fids) / len(maj_fids)
        avg_pa = sum(pa_fids) / len(pa_fids)
        winner = "palette_aware" if avg_pa < avg_maj else "majority"
        print(f"  Avg palette fidelity (majority):      {avg_maj:.2f}")
        print(f"  Avg palette fidelity (palette_aware):  {avg_pa:.2f}  [{winner} wins]")


if __name__ == "__main__":
    main()
