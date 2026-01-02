#!/usr/bin/env python3
"""Record feedback for a Qwen prompt-tuning round."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASE_DIR = Path(__file__).resolve().parent


def _load_session(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = ROOT_DIR / path
    if candidate.exists():
        return candidate
    return BASE_DIR / path


def _load_round(log_path: Path, round_id: int) -> Dict[str, Any]:
    if not log_path.exists():
        raise SystemExit(f"Round log not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("round_id") == round_id:
                return record
    raise SystemExit(f"Round {round_id} not found in {log_path}")


def _select_prompt(record: Dict[str, Any], index: int) -> str:
    results = record.get("results", [])
    for item in results:
        if item.get("index") == index and item.get("prompt"):
            return item["prompt"]
    prompts = record.get("prompts", [])
    if 0 < index <= len(prompts):
        return prompts[index - 1]
    raise SystemExit(f"Prompt index {index} not found in round log")


def _validate_indices(indices: List[int], max_index: int, label: str) -> None:
    for idx in indices:
        if idx < 1 or idx > max_index:
            raise SystemExit(f"{label} index {idx} is out of range (1-{max_index})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record feedback for a Qwen prompt-tuning round."
    )
    parser.add_argument(
        "--session",
        default=str(BASE_DIR / "session.json"),
        help="Path to session.json",
    )
    parser.add_argument("--round", type=int, required=True, help="Round id to log")
    parser.add_argument("--good", nargs="*", type=int, default=[], help="Good indices")
    parser.add_argument("--bad", nargs="*", type=int, default=[], help="Bad indices")
    parser.add_argument("--notes", default="", help="Notes on what worked")
    parser.add_argument(
        "--best-index",
        type=int,
        help="Prompt index to write into best_prompt.txt",
    )

    args = parser.parse_args()

    session_path = Path(args.session)
    session = _load_session(session_path)
    log_path = _resolve_path(session.get("log_file", "rounds.jsonl"))
    feedback_log = _resolve_path(session.get("feedback_log_file", "feedback.jsonl"))
    best_prompt_file = _resolve_path(session.get("best_prompt_file", "best_prompt.txt"))

    record = _load_round(log_path, args.round)
    prompt_count = len(record.get("prompts", []))
    if prompt_count == 0:
        raise SystemExit("Round record missing prompts; cannot validate indices.")

    _validate_indices(args.good, prompt_count, "Good")
    _validate_indices(args.bad, prompt_count, "Bad")

    best_prompt: Optional[str] = None
    if args.best_index is not None:
        _validate_indices([args.best_index], prompt_count, "Best")
        best_prompt = _select_prompt(record, args.best_index)
        best_prompt_file.parent.mkdir(parents=True, exist_ok=True)
        best_prompt_file.write_text(best_prompt + "\n", encoding="utf-8")

    feedback_record = {
        "round_id": args.round,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "good": args.good,
        "bad": args.bad,
        "notes": args.notes,
    }
    if args.best_index is not None:
        feedback_record["best_index"] = args.best_index
        feedback_record["best_prompt"] = best_prompt

    feedback_log.parent.mkdir(parents=True, exist_ok=True)
    with feedback_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(feedback_record, ensure_ascii=True) + "\n")

    print(f"Feedback logged to {feedback_log}")
    if args.best_index is not None:
        print(f"best_prompt.txt updated at {best_prompt_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
