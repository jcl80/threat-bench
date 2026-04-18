"""Model-agnostic scorer for binary post classification.

Compares a predictions JSONL against ground-truth labels from prepare_data,
reporting precision, recall, F1 overall and by subreddit tier.

Predictions format (one JSON per line):
    {"snapshot_id": 12345, "predicted": 1, "score": 0.87}

Usage:
    python -m bench.scorer --predictions bench/results/nli_deberta/predictions.jsonl --ground-truth gpt5
    python -m bench.scorer --predictions bench/results/nli_deberta/predictions.jsonl --ground-truth gpt5_mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TIERS = ["threat_dense", "ambiguous", "benign"]


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(precision, recall)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "support_pos": tp + fn,
        "support_neg": fp + tn,
    }


def score(
    predictions_path: Path,
    data_path: Path,
    ground_truth: str,
) -> dict:
    label_key = f"label_{ground_truth}"

    # Load ground truth
    gt = {}
    tier_map = {}
    sub_map = {}
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            sid = row["snapshot_id"]
            gt[sid] = row[label_key]
            tier_map[sid] = row["tier"]
            sub_map[sid] = row["subreddit"]

    # Load predictions
    preds = {}
    with open(predictions_path) as f:
        for line in f:
            row = json.loads(line)
            preds[row["snapshot_id"]] = row["predicted"]

    # Check coverage
    missing = set(gt.keys()) - set(preds.keys())
    if missing:
        print(f"WARNING: {len(missing)} posts in ground truth have no prediction")

    # Compute metrics
    def compute(subset_ids):
        tp = fp = fn = tn = 0
        for sid in subset_ids:
            if sid not in preds:
                continue
            truth = gt[sid]
            pred = preds[sid]
            if truth == 1 and pred == 1:
                tp += 1
            elif truth == 0 and pred == 1:
                fp += 1
            elif truth == 1 and pred == 0:
                fn += 1
            else:
                tn += 1
        return _metrics(tp, fp, fn, tn)

    all_ids = list(gt.keys())
    results = {
        "ground_truth": ground_truth,
        "total_posts": len(all_ids),
        "predictions_count": len(preds),
        "overall": compute(all_ids),
        "by_tier": {},
        "by_subreddit": {},
    }

    for tier in TIERS:
        tier_ids = [sid for sid in all_ids if tier_map[sid] == tier]
        if tier_ids:
            results["by_tier"][tier] = compute(tier_ids)

    for sub in sorted(set(sub_map.values())):
        sub_ids = [sid for sid in all_ids if sub_map[sid] == sub]
        if sub_ids:
            results["by_subreddit"][sub] = compute(sub_ids)

    return results


def print_report(results: dict) -> None:
    gt = results["ground_truth"]
    print(f"\n{'='*60}")
    print(f"Scorer Report (ground truth: {gt})")
    print(f"{'='*60}")
    print(f"Posts: {results['total_posts']} | Predictions: {results['predictions_count']}")

    def row(label, m):
        print(f"  {label:20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}  "
              f"(+{m['support_pos']}/-{m['support_neg']})")

    print(f"\n--- Overall ---")
    row("all", results["overall"])

    print(f"\n--- By Tier ---")
    for tier in TIERS:
        if tier in results["by_tier"]:
            row(tier, results["by_tier"][tier])

    print(f"\n--- By Subreddit ---")
    for sub, m in results["by_subreddit"].items():
        row(sub, m)

    print()


def main():
    parser = argparse.ArgumentParser(description="Score binary predictions")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL path")
    parser.add_argument("--data", default="bench/data/posts.jsonl", help="Ground truth data path")
    parser.add_argument("--ground-truth", required=True, choices=["gpt5_mini", "gpt5"],
                        help="Which model's labels to score against")
    parser.add_argument("--save", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    results = score(Path(args.predictions), Path(args.data), args.ground_truth)
    print_report(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
