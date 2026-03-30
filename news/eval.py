"""Evaluate news classification results against human labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confusion_matrix(results: list[dict]) -> dict:
    """Compute TP/FP/FN/TN from results with model_answer and ground_truth."""
    tp = fp = fn = tn = parse_fail = 0
    for r in results:
        answer = r.get("model_answer")
        truth = r.get("ground_truth", "").lower()
        if not answer:
            parse_fail += 1
            continue
        if truth == "yes" and answer == "yes":
            tp += 1
        elif truth == "no" and answer == "yes":
            fp += 1
        elif truth == "yes" and answer == "no":
            fn += 1
        elif truth == "no" and answer == "no":
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "parse_fail": parse_fail}


def metrics_from_cm(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "accuracy": round(accuracy, 4),
    }


def speed_metrics(results: list[dict]) -> dict:
    times = [r["wall_clock_seconds"] for r in results if r.get("wall_clock_seconds")]
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)

    if not times:
        return {}

    total_wall = sum(times)
    return {
        "total_articles": len(results),
        "total_wall_seconds": round(total_wall, 1),
        "mean_seconds_per_article": round(total_wall / len(times), 3),
        "median_seconds_per_article": round(sorted(times)[len(times) // 2], 3),
        "articles_per_minute": round(len(times) / (total_wall / 60), 1) if total_wall > 0 else 0,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def split_by(results: list[dict], key_fn) -> dict[str, list[dict]]:
    groups = {}
    for r in results:
        k = key_fn(r)
        groups.setdefault(k, []).append(r)
    return groups


def evaluate(results_path: str) -> dict:
    results = load_results(results_path)

    # Overall
    cm = confusion_matrix(results)
    met = metrics_from_cm(cm)

    # Split: has_summary vs title-only
    by_summary = split_by(results, lambda r: "with_summary" if r.get("has_summary") else "title_only")
    summary_splits = {}
    for label, group in sorted(by_summary.items()):
        gcm = confusion_matrix(group)
        summary_splits[label] = {"n": len(group), "confusion": gcm, **metrics_from_cm(gcm)}

    # Split: by month
    by_month = split_by(results, lambda r: r.get("date", "")[:7] or "unknown")
    month_splits = {}
    for label, group in sorted(by_month.items()):
        gcm = confusion_matrix(group)
        month_splits[label] = {"n": len(group), "confusion": gcm, **metrics_from_cm(gcm)}

    speed = speed_metrics(results)

    return {
        "total": len(results),
        "confusion": cm,
        "metrics": met,
        "speed": speed,
        "by_summary": summary_splits,
        "by_month": month_splits,
    }


def print_report(eval_data: dict, run_name: str = "") -> None:
    cm = eval_data["confusion"]
    met = eval_data["metrics"]
    sp = eval_data.get("speed", {})

    print(f"\n{'='*60}")
    if run_name:
        print(f"  NEWS EVAL: {run_name}")
    print(f"  Articles: {eval_data['total']}")
    print(f"{'='*60}\n")

    # Confusion matrix
    print("  CONFUSION MATRIX")
    print(f"                    Predicted YES   Predicted NO")
    print(f"    Actual YES      {cm['tp']:>8}        {cm['fn']:>8}")
    print(f"    Actual NO       {cm['fp']:>8}        {cm['tn']:>8}")
    if cm["parse_fail"]:
        print(f"    Parse failures: {cm['parse_fail']}")
    print()

    # Metrics
    print("  METRICS")
    print(f"    Precision:  {met['precision']:.1%}")
    print(f"    Recall:     {met['recall']:.1%}")
    print(f"    F1:         {met['f1']:.1%}")
    print(f"    FPR:        {met['fpr']:.1%}")
    print(f"    Accuracy:   {met['accuracy']:.1%}")
    print()

    # Speed
    if sp:
        print("  SPEED")
        print(f"    Mean:     {sp['mean_seconds_per_article']:.2f}s/article")
        print(f"    Median:   {sp['median_seconds_per_article']:.2f}s/article")
        print(f"    Rate:     {sp['articles_per_minute']:.0f} articles/min")
        if sp.get("total_tokens"):
            print(f"    Tokens:   {sp['total_tokens']:,} total ({sp['prompt_tokens']:,} prompt, {sp['completion_tokens']:,} completion)")
        print()

    # Summary split
    print("  BY SUMMARY AVAILABILITY")
    for label, data in eval_data.get("by_summary", {}).items():
        print(f"    {label:<15} n={data['n']:<6} P={data['precision']:.1%}  R={data['recall']:.1%}  F1={data['f1']:.1%}  FPR={data['fpr']:.1%}")
    print()

    # Month split
    print("  BY MONTH")
    for label, data in eval_data.get("by_month", {}).items():
        print(f"    {label:<15} n={data['n']:<6} P={data['precision']:.1%}  R={data['recall']:.1%}  F1={data['f1']:.1%}  FPR={data['fpr']:.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate news classification results")
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--name", default=None, help="Run name for report header")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--save", default=None, help="Save eval JSON to path")

    args = parser.parse_args()
    eval_data = evaluate(args.results)

    if args.json:
        print(json.dumps(eval_data, indent=2))
    else:
        print_report(eval_data, args.name or args.results)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"Saved eval to {args.save}")


if __name__ == "__main__":
    main()
