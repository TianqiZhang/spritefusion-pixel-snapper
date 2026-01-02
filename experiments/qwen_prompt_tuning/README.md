# Qwen Prompt Tuning

This folder is a lightweight, append-only workflow for iterating on Qwen image-edit
prompts against a fixed test photo.

## Files
- `session.json`: Session settings (test photo path, Qwen params, output dir).
- `rounds.jsonl`: Append-only log of each generation round.
- `feedback.jsonl`: Append-only log of human feedback per round.
- `best_prompt.txt`: Current best prompt (update when you pick a winner).
- `rounds/`: Output folders per round with generated images.

## Quick start
1) Edit `session.json` and set `test_image_path`.
2) Run a round with prompt variants.
3) Record feedback (and optionally update `best_prompt.txt`).

## Run a round
```bash
python experiments/qwen_prompt_tuning/run_round.py \
  --prompts "prompt A" "prompt B" "prompt C" \
  --notes "baseline variants"
```

## Record feedback
```bash
python experiments/qwen_prompt_tuning/record_feedback.py \
  --round 1 \
  --good 2 \
  --bad 1 3 \
  --notes "B keeps silhouettes clean" \
  --best-index 2
```

## Requirements
- `DASHSCOPE_API_KEY` set in the environment (region-specific).
- Network access to DashScope Qwen.
