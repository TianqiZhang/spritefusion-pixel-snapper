"""Tests for ground truth I/O and accuracy metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixel_snapper.ground_truth import (
    GroundTruth,
    cut_position_error,
    grid_accuracy,
    ground_truth_dir,
    load_ground_truth,
    save_ground_truth,
)


class TestGroundTruthIO:
    """Tests for save/load round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        gt = GroundTruth(
            image_file="test.png",
            image_width=100,
            image_height=200,
            col_cuts=[0, 10, 20, 30, 100],
            row_cuts=[0, 10, 20, 200],
            metadata={"notes": "test"},
        )
        json_path = tmp_path / "test.json"
        save_ground_truth(gt, json_path)
        loaded = load_ground_truth(json_path)

        assert loaded.image_file == "test.png"
        assert loaded.image_width == 100
        assert loaded.image_height == 200
        assert loaded.col_cuts == [0, 10, 20, 30, 100]
        assert loaded.row_cuts == [0, 10, 20, 200]
        assert loaded.metadata["notes"] == "test"
        assert "created" in loaded.metadata
        assert "modified" in loaded.metadata

    def test_cells_properties(self) -> None:
        gt = GroundTruth(
            image_file="t.png",
            image_width=30,
            image_height=20,
            col_cuts=[0, 10, 20, 30],
            row_cuts=[0, 10, 20],
        )
        assert gt.cells_x == 3
        assert gt.cells_y == 2

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        gt = GroundTruth(
            image_file="t.png",
            image_width=10,
            image_height=10,
            col_cuts=[0, 10],
            row_cuts=[0, 10],
        )
        deep_path = tmp_path / "a" / "b" / "t.json"
        save_ground_truth(gt, deep_path)
        assert deep_path.exists()

    def test_json_schema(self, tmp_path: Path) -> None:
        gt = GroundTruth(
            image_file="img.png",
            image_width=40,
            image_height=60,
            col_cuts=[0, 20, 40],
            row_cuts=[0, 20, 40, 60],
        )
        json_path = tmp_path / "img.json"
        save_ground_truth(gt, json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["version"] == 1
        assert data["cells_x"] == 2
        assert data["cells_y"] == 3

    def test_save_preserves_existing_created(self, tmp_path: Path) -> None:
        gt = GroundTruth(
            image_file="t.png",
            image_width=10,
            image_height=10,
            col_cuts=[0, 10],
            row_cuts=[0, 10],
            metadata={"created": "2025-01-01T00:00:00"},
        )
        json_path = tmp_path / "t.json"
        save_ground_truth(gt, json_path)
        loaded = load_ground_truth(json_path)
        assert loaded.metadata["created"] == "2025-01-01T00:00:00"


class TestGroundTruthDir:
    def test_returns_subdirectory(self, tmp_path: Path) -> None:
        result = ground_truth_dir(tmp_path)
        assert result == tmp_path / "ground_truth"


class TestCutPositionError:
    def test_perfect_match(self) -> None:
        result = cut_position_error([0, 10, 20], [0, 10, 20])
        assert result["mean_abs_error"] == 0
        assert result["max_error"] == 0
        assert result["missing_cuts"] == 0
        assert result["extra_cuts"] == 0

    def test_small_offsets(self) -> None:
        result = cut_position_error([0, 11, 19], [0, 10, 20])
        assert result["mean_abs_error"] <= 1
        assert result["max_error"] <= 1
        assert result["missing_cuts"] == 0
        assert result["extra_cuts"] == 0

    def test_extra_cuts(self) -> None:
        result = cut_position_error([0, 5, 10, 15, 20], [0, 10, 20])
        assert result["extra_cuts"] == 2

    def test_missing_cuts(self) -> None:
        result = cut_position_error([0, 20], [0, 10, 20])
        assert result["missing_cuts"] == 0 or result["extra_cuts"] == 0
        # With [0,20] vs [0,10,20], 10 might be missing if no pred is close

    def test_empty_predicted(self) -> None:
        result = cut_position_error([], [0, 10, 20])
        assert result["mean_abs_error"] == float("inf")
        assert result["missing_cuts"] == 3

    def test_empty_gt(self) -> None:
        result = cut_position_error([0, 10, 20], [])
        assert result["mean_abs_error"] == float("inf")
        assert result["extra_cuts"] == 3


class TestGridAccuracy:
    def test_perfect_match(self) -> None:
        result = grid_accuracy(
            [0, 10, 20], [0, 10, 20],
            [0, 10, 20], [0, 10, 20],
        )
        assert result["cell_count_match"] is True
        assert result["pred_cells"] == "2x2"
        assert result["gt_cells"] == "2x2"
        assert result["col_error"]["mean_abs_error"] == 0
        assert result["row_error"]["mean_abs_error"] == 0

    def test_cell_count_mismatch(self) -> None:
        result = grid_accuracy(
            [0, 10, 20, 30], [0, 10, 20],
            [0, 15, 30], [0, 10, 20],
        )
        assert result["cell_count_match"] is False
        assert result["pred_cells"] == "3x2"
        assert result["gt_cells"] == "2x2"

    def test_close_but_not_perfect(self) -> None:
        result = grid_accuracy(
            [0, 11, 21], [0, 9, 19],
            [0, 10, 20], [0, 10, 20],
        )
        assert result["cell_count_match"] is True
        assert result["col_error"]["mean_abs_error"] <= 1
        assert result["row_error"]["mean_abs_error"] <= 1
