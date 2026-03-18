#!/usr/bin/env python
"""Benchmark script to evaluate grid detection scoring across test images.

This script processes all images in the testdata folder and collects statistics
about which detection methods perform best. Results are saved to a JSON file
for comparison across code changes.

Usage:
    python tools/benchmark.py [--output results.json]
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixel_snapper import Config, process_image_bytes_with_grid
from pixel_snapper.ground_truth import get_test_images, grid_accuracy, ground_truth_dir, load_ground_truth
from pixel_snapper.scoring import ScoredCandidate


@dataclass
class ImageResult:
    """Results for a single image."""
    filename: str
    width: int
    height: int
    winner_source: str
    winner_grid_size: str
    winner_score: float
    num_candidates: int
    all_candidates: List[Dict[str, Any]]
    processing_time_ms: float
    winner_col_cuts: Optional[List[int]] = None
    winner_row_cuts: Optional[List[int]] = None
    ground_truth_accuracy: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    timestamp: str
    num_images: int
    total_time_ms: float
    wins_by_source: Dict[str, int]
    avg_score_by_source: Dict[str, float]
    avg_rank_by_source: Dict[str, float]
    source_appearances: Dict[str, int]
    image_results: List[Dict[str, Any]]


def process_image(image_path: Path, config: Config) -> ImageResult:
    """Process a single image and collect scoring data."""
    with open(image_path, 'rb') as f:
        input_bytes = f.read()

    start_time = time.perf_counter()
    result = process_image_bytes_with_grid(input_bytes, config)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Get dimensions from the already-decoded quantized image
    width, height = result.quantized_img.size

    # Extract candidate information
    candidates_data = []
    if result.scored_candidates:
        for c in result.scored_candidates:
            candidates_data.append({
                'source': c.source,
                'grid_size': c.grid_size,
                'cells_x': c.cells_x,
                'cells_y': c.cells_y,
                'step_size': round(c.step_size, 2),
                'uniformity_score': round(c.uniformity_score, 4),
                'edge_score': round(c.edge_score, 4),
                'combined_score': round(c.combined_score, 4),
                'rank': c.rank,
            })

    winner = result.scored_candidates[0] if result.scored_candidates else None

    return ImageResult(
        filename=image_path.name,
        width=width,
        height=height,
        winner_source=winner.source if winner else 'none',
        winner_grid_size=winner.grid_size if winner else 'N/A',
        winner_score=round(winner.combined_score, 4) if winner else 0.0,
        num_candidates=len(result.scored_candidates) if result.scored_candidates else 0,
        all_candidates=candidates_data,
        processing_time_ms=round(elapsed_ms, 2),
        winner_col_cuts=list(result.col_cuts),
        winner_row_cuts=list(result.row_cuts),
    )


def normalize_source(source: str) -> str:
    """Normalize source name for grouping (e.g., 'autocorr(16.0x16.0)' -> 'autocorr').

    Note: fixed(N) sources are kept as-is to analyze each step size separately.
    """
    if source.startswith('autocorr'):
        return 'autocorr'
    elif source.startswith('hint'):
        return 'hint'
    elif source.startswith('recon'):
        return 'recon'
    # Keep fixed(N) as-is to see individual step sizes
    return source


def run_benchmark(testdata_dir: Path, config: Config) -> BenchmarkResults:
    """Run benchmark on all test images."""
    images = get_test_images(testdata_dir)

    if not images:
        print(f"No test images found in {testdata_dir}")
        sys.exit(1)

    print(f"Found {len(images)} test images")
    print("-" * 60)

    image_results: List[ImageResult] = []
    total_start = time.perf_counter()

    for img_path in images:
        print(f"Processing {img_path.name}...", end=" ", flush=True)
        result = process_image(img_path, config)

        # Check for ground truth
        gt_path = ground_truth_dir(testdata_dir) / f"{img_path.stem}.json"
        if gt_path.exists() and result.winner_col_cuts and result.winner_row_cuts:
            gt = load_ground_truth(gt_path)
            result.ground_truth_accuracy = grid_accuracy(
                result.winner_col_cuts, result.winner_row_cuts,
                gt.col_cuts, gt.row_cuts,
            )

        image_results.append(result)
        print(f"Winner: {result.winner_source} ({result.winner_grid_size}) "
              f"score={result.winner_score:.3f} [{result.num_candidates} candidates]")

    total_time_ms = (time.perf_counter() - total_start) * 1000

    # Aggregate statistics
    wins_by_source: Dict[str, int] = defaultdict(int)
    scores_by_source: Dict[str, List[float]] = defaultdict(list)
    ranks_by_source: Dict[str, List[int]] = defaultdict(list)
    appearances_by_source: Dict[str, int] = defaultdict(int)

    for result in image_results:
        # Count wins (normalized)
        winner_normalized = normalize_source(result.winner_source)
        wins_by_source[winner_normalized] += 1

        # Collect scores and ranks for all candidates
        for cand in result.all_candidates:
            source_normalized = normalize_source(cand['source'])
            scores_by_source[source_normalized].append(cand['combined_score'])
            ranks_by_source[source_normalized].append(cand['rank'])
            appearances_by_source[source_normalized] += 1

    # Calculate averages
    avg_score_by_source = {
        src: round(sum(scores) / len(scores), 4)
        for src, scores in scores_by_source.items()
    }
    avg_rank_by_source = {
        src: round(sum(ranks) / len(ranks), 2)
        for src, ranks in ranks_by_source.items()
    }

    return BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        num_images=len(images),
        total_time_ms=round(total_time_ms, 2),
        wins_by_source=dict(wins_by_source),
        avg_score_by_source=avg_score_by_source,
        avg_rank_by_source=avg_rank_by_source,
        source_appearances=dict(appearances_by_source),
        image_results=[asdict(r) for r in image_results],
    )


def print_summary(results: BenchmarkResults) -> None:
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Images processed: {results.num_images}")
    print(f"Total time: {results.total_time_ms:.0f}ms")
    print()

    print("WINS BY SOURCE:")
    print("-" * 40)
    sorted_wins = sorted(results.wins_by_source.items(), key=lambda x: -x[1])
    for source, wins in sorted_wins:
        pct = wins / results.num_images * 100
        print(f"  {source:20s}: {wins:2d} ({pct:5.1f}%)")
    print()

    print("AVERAGE SCORE BY SOURCE:")
    print("-" * 40)
    sorted_scores = sorted(results.avg_score_by_source.items(), key=lambda x: -x[1])
    for source, score in sorted_scores:
        appearances = results.source_appearances.get(source, 0)
        print(f"  {source:20s}: {score:.4f} (n={appearances})")
    print()

    print("AVERAGE RANK BY SOURCE:")
    print("-" * 40)
    sorted_ranks = sorted(results.avg_rank_by_source.items(), key=lambda x: x[1])
    for source, rank in sorted_ranks:
        appearances = results.source_appearances.get(source, 0)
        print(f"  {source:20s}: {rank:.2f} (n={appearances})")

    # Ground truth comparison
    gt_rows = []
    for ir in results.image_results:
        acc = ir.get("ground_truth_accuracy")
        if acc is not None:
            gt_rows.append((ir["filename"], acc))

    if gt_rows:
        print()
        print("GROUND TRUTH COMPARISON:")
        print("-" * 60)
        for fname, acc in gt_rows:
            col_err = acc["col_error"]["mean_abs_error"]
            row_err = acc["row_error"]["mean_abs_error"]
            match_str = "(match)" if acc["cell_count_match"] else "(MISMATCH)"
            cells_str = f"{acc['pred_cells']} vs {acc['gt_cells']} {match_str}" if not acc["cell_count_match"] else f"{acc['gt_cells']} {match_str}"
            print(f"  {fname:16s}: col_err={col_err:.1f}px  row_err={row_err:.1f}px  cells: {cells_str}")


def main() -> None:
    """Main entry point."""
    # Parse arguments
    output_file = "benchmark_results.json"
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
        elif sys.argv[1] == '--output' and len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = sys.argv[1]

    # Find testdata directory
    repo_root = Path(__file__).parent.parent
    testdata_dir = repo_root / "testdata"

    if not testdata_dir.exists():
        print(f"Error: testdata directory not found at {testdata_dir}")
        sys.exit(1)

    # Use default config with uniformity scoring enabled
    config = Config(
        use_uniformity_scoring=True,
        use_autocorrelation=True,
    )

    # Run benchmark
    results = run_benchmark(testdata_dir, config)

    # Print summary
    print_summary(results)

    # Save to JSON
    output_path = repo_root / output_file
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
