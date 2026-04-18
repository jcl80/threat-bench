"""Threshold sweep for a predictions JSONL with per-post scores.

Reuses existing scores — does not re-run the model. Produces a table of
precision/recall/F1 across thresholds vs both ground truths.

Usage:
    python -m bench.sweep_threshold --predictions bench/results/<run_dir>/predictions.jsonl
    python -m bench.sweep_threshold --predictions ... --min 0.1 --max 0.9 --step 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.scorer import score


def sweep(
    predictions_path: Path,
    data_path: Path,
    thresholds: list[float],
) -> dict:
    # Load raw predictions with scores
    rows = []
    with open(predictions_path) as f:
        for line in f:
            rows.append(json.loads(line))

    results = {"thresholds": [], "gpt5_mini": [], "gpt5": []}

    for t in thresholds:
        # Rewrite predictions to a temp-in-memory file at this threshold
        tmp_path = predictions_path.parent / f".sweep_{t:.2f}.jsonl"
        with open(tmp_path, "w") as f:
            for r in rows:
                rethresholded = {
                    "snapshot_id": r["snapshot_id"],
                    "predicted": 1 if r["score"] >= t else 0,
                    "score": r["score"],
                }
                f.write(json.dumps(rethresholded) + "\n")

        for gt in ["gpt5_mini", "gpt5"]:
            r = score(tmp_path, data_path, gt)
            results[gt].append(r["overall"])

        results["thresholds"].append(t)
        tmp_path.unlink()

    return results


def print_table(results: dict) -> None:
    thresholds = results["thresholds"]

    for gt in ["gpt5_mini", "gpt5"]:
        print(f"\n{'='*72}")
        print(f"Sweep vs {gt}")
        print(f"{'='*72}")
        print(f"{'thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Acc':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
        for t, m in zip(thresholds, results[gt]):
            print(
                f"{t:>5.2f}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
                f"{m['f1']:>6.3f}  {m['accuracy']:>6.3f}  "
                f"{m['tp']:>5d}  {m['fp']:>5d}  {m['fn']:>5d}"
            )

        # Highlight best F1
        best = max(results[gt], key=lambda m: m["f1"])
        best_t = thresholds[results[gt].index(best)]
        print(f"  best F1: {best['f1']:.3f} at threshold {best_t:.2f} "
              f"(P={best['precision']:.3f}, R={best['recall']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep over saved scores")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL path")
    parser.add_argument("--data", default="bench/data/posts.jsonl", help="Ground truth data path")
    parser.add_argument("--min", type=float, default=0.1, help="Min threshold")
    parser.add_argument("--max", type=float, default=0.9, help="Max threshold")
    parser.add_argument("--step", type=float, default=0.05, help="Step size")
    parser.add_argument("--save", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.step

    results = sweep(Path(args.predictions), Path(args.data), thresholds)
    print_table(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
