"""Ground truth data model, JSON I/O, and accuracy metrics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GroundTruth:
    """Human-verified grid cuts for a test image."""

    image_file: str  # basename only, e.g. "ash.png"
    image_width: int
    image_height: int
    col_cuts: List[int]
    row_cuts: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cells_x(self) -> int:
        return len(self.col_cuts) - 1

    @property
    def cells_y(self) -> int:
        return len(self.row_cuts) - 1


def ground_truth_dir(testdata_dir: Path) -> Path:
    """Return the ground truth subdirectory under testdata."""
    return testdata_dir / "ground_truth"


def load_ground_truth(json_path: Path) -> GroundTruth:
    """Load a GroundTruth from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return GroundTruth(
        image_file=data["image_file"],
        image_width=data["image_width"],
        image_height=data["image_height"],
        col_cuts=data["col_cuts"],
        row_cuts=data["row_cuts"],
        metadata=data.get("metadata", {}),
    )


def save_ground_truth(gt: GroundTruth, json_path: Path) -> None:
    """Save a GroundTruth to a JSON file."""
    now = datetime.now().isoformat(timespec="seconds")
    meta = dict(gt.metadata)
    if "created" not in meta:
        meta["created"] = now
    meta["modified"] = now

    data = {
        "version": 1,
        "image_file": gt.image_file,
        "image_width": gt.image_width,
        "image_height": gt.image_height,
        "col_cuts": gt.col_cuts,
        "row_cuts": gt.row_cuts,
        "cells_x": gt.cells_x,
        "cells_y": gt.cells_y,
        "metadata": meta,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def cut_position_error(predicted: List[int], gt: List[int]) -> Dict[str, Any]:
    """Compare predicted cuts against ground truth cuts.

    Uses greedy closest-match: for each ground truth cut, find the nearest
    predicted cut.  Unmatched predictions are "extra", unmatched GT are "missing".

    Returns dict with: mean_abs_error, max_error, missing_cuts, extra_cuts.
    """
    if not gt or not predicted:
        return {
            "mean_abs_error": float("inf"),
            "max_error": float("inf"),
            "missing_cuts": len(gt),
            "extra_cuts": len(predicted),
        }

    pred_sorted = sorted(predicted)
    gt_sorted = sorted(gt)

    # Match each GT cut to nearest predicted cut
    used_pred: set[int] = set()
    errors: List[int] = []
    missing = 0

    for g in gt_sorted:
        best_idx = -1
        best_dist = float("inf")
        for i, p in enumerate(pred_sorted):
            if i in used_pred:
                continue
            d = abs(p - g)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0 and best_dist <= max(5, len(gt_sorted)):
            used_pred.add(best_idx)
            errors.append(int(best_dist))
        else:
            missing += 1

    extra = len(pred_sorted) - len(used_pred)

    mean_err = sum(errors) / len(errors) if errors else float("inf")
    max_err = max(errors) if errors else float("inf")

    return {
        "mean_abs_error": round(mean_err, 2),
        "max_error": max_err,
        "missing_cuts": missing,
        "extra_cuts": extra,
    }


def grid_accuracy(
    pred_col: List[int],
    pred_row: List[int],
    gt_col: List[int],
    gt_row: List[int],
) -> Dict[str, Any]:
    """Compute overall grid accuracy between predicted and ground truth.

    Returns dict with: cell_count_match, col_error, row_error.
    """
    col_err = cut_position_error(pred_col, gt_col)
    row_err = cut_position_error(pred_row, gt_row)

    pred_cells_x = len(pred_col) - 1 if len(pred_col) > 1 else 0
    pred_cells_y = len(pred_row) - 1 if len(pred_row) > 1 else 0
    gt_cells_x = len(gt_col) - 1 if len(gt_col) > 1 else 0
    gt_cells_y = len(gt_row) - 1 if len(gt_row) > 1 else 0

    return {
        "cell_count_match": (pred_cells_x == gt_cells_x and pred_cells_y == gt_cells_y),
        "pred_cells": f"{pred_cells_x}x{pred_cells_y}",
        "gt_cells": f"{gt_cells_x}x{gt_cells_y}",
        "col_error": col_err,
        "row_error": row_err,
    }
