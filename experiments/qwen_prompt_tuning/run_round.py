#!/usr/bin/env python3
"""Run a Qwen prompt-tuning round and log outputs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pixel_snapper.config import Config, PixelSnapperError
from pixel_snapper.qwen import maybe_apply_qwen_edit

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


def _next_round_id(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    count = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count + 1


def _load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                prompt = line.strip()
                if prompt:
                    prompts.append(prompt)
    if not prompts:
        raise SystemExit("No prompts provided. Use --prompts or --prompt-file.")
    return prompts


def _build_config(session: Dict[str, Any], prompt: str) -> Config:
    return Config(
        qwen_enabled=True,
        qwen_api_key=session.get("qwen_api_key"),
        qwen_model=session.get("qwen_model"),
        qwen_endpoint=session.get("qwen_endpoint"),
        qwen_prompt=prompt,
        qwen_negative_prompt=session.get("qwen_negative_prompt"),
        qwen_prompt_extend=session.get("qwen_prompt_extend", True),
        qwen_watermark=session.get("qwen_watermark", False),
        qwen_output_count=session.get("qwen_output_count", 1),
        qwen_output_index=session.get("qwen_output_index", 0),
        qwen_size=session.get("qwen_size"),
        qwen_seed=session.get("qwen_seed"),
        qwen_timeout=session.get("qwen_timeout", 120),
    )


def _qwen_settings(session: Dict[str, Any]) -> Dict[str, Any]:
    settings = {}
    for key, value in session.items():
        if key.startswith("qwen_") and key != "qwen_api_key":
            settings[key] = value
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Qwen prompt-tuning round and log outputs."
    )
    parser.add_argument(
        "--session",
        default=str(BASE_DIR / "session.json"),
        help="Path to session.json",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Prompt variants (space separated).",
    )
    parser.add_argument(
        "--prompt-file",
        help="Text file with one prompt per line.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes for the round.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and only log prompts.",
    )

    args = parser.parse_args()
    prompts = _load_prompts(args)

    session_path = Path(args.session)
    session = _load_session(session_path)

    test_image_path = _resolve_path(session.get("test_image_path", ""))
    if not test_image_path.exists():
        raise SystemExit(
            f"Test image not found: {test_image_path}. Update session.json."
        )

    output_dir = _resolve_path(session.get("output_dir", "rounds"))
    log_path = _resolve_path(session.get("log_file", "rounds.jsonl"))

    round_id = _next_round_id(log_path)
    round_dir = output_dir / f"round_{round_id:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    input_bytes = test_image_path.read_bytes()

    results: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        entry: Dict[str, Any] = {"index": idx, "prompt": prompt}
        if args.dry_run:
            entry["skipped"] = True
            results.append(entry)
            continue
        config = _build_config(session, prompt)
        try:
            output_bytes = maybe_apply_qwen_edit(input_bytes, config)
            output_path = round_dir / f"variant_{idx:02d}.png"
            output_path.write_bytes(output_bytes)
            entry["output_path"] = str(output_path)
        except PixelSnapperError as exc:
            entry["error"] = str(exc)
        results.append(entry)

    record = {
        "round_id": round_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "test_image_path": str(test_image_path),
        "round_dir": str(round_dir),
        "prompts": prompts,
        "results": results,
        "notes": args.notes,
        "qwen_settings": _qwen_settings(session),
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Round {round_id} logged to {log_path}")
    if not args.dry_run:
        print(f"Outputs saved to {round_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
