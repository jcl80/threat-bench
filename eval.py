"""Field-level agreement scoring between model output and Go pipeline baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jaccard(a: list, b: list) -> float:
    """Set overlap between two lists."""
    sa, sb = set(s.lower() for s in a), set(s.lower() for s in b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def mae(a: float | int | None, b: float | int | None) -> float | None:
    """Mean absolute error between two values. None if both are None."""
    if a is None and b is None:
        return None
    return abs((a or 0) - (b or 0))


def evaluate(baseline_path: str, model_output_path: str) -> dict:
    """Compare model output against baseline.

    Baseline: each row is a confirmed threat analysis with post_snapshot_ids,
    threat_categories, severity_score, importance, etc.

    Model output: each row has analysis_id, flagged_posts (only flagged ones).

    Matching: by analysis_id. A baseline analysis is "detected" if the model
    flagged at least one post with matching evidence (same post_snapshot_id).
    """
    # Load baseline
    with open(baseline_path) as f:
        baselines = {b["analysis_id"]: b for line in f for b in [json.loads(line)]}

    # Load model output
    with open(model_output_path) as f:
        model_runs = {r["analysis_id"]: r for line in f for r in [json.loads(line)]}

    comparisons = []

    for analysis_id, baseline in baselines.items():
        model_run = model_runs.get(analysis_id)
        if not model_run:
            continue

        # Which posts did the baseline flag? (from evidence post_snapshot_ids)
        baseline_flagged_snapshots = set()
        for ev in baseline["stage1"].get("evidence", []):
            if ev.get("post_snapshot_id"):
                baseline_flagged_snapshots.add(ev["post_snapshot_id"])

        # Which posts did the model flag?
        model_flagged = model_run.get("flagged_posts", [])
        total_posts = model_run.get("total_posts", len(baseline["post_snapshot_ids"]))

        # Map model flagged posts by snapshot_id (using post_index to look up)
        # post_index is 1-based into the post_snapshot_ids list
        model_flagged_snapshots = set()
        model_categories = []
        model_importance = []
        model_severity = []
        model_weirdness = []
        model_confidence = []
        model_geo_country = []
        model_geo_region = []
        model_evidence_counts = []

        snapshot_ids = baseline["post_snapshot_ids"]
        for fp in model_flagged:
            idx = fp.get("post_index", 0)
            if 1 <= idx <= len(snapshot_ids):
                sid = snapshot_ids[idx - 1]
                model_flagged_snapshots.add(sid)

                # Only collect field data for posts that match baseline-flagged posts
                if sid in baseline_flagged_snapshots:
                    model_categories.extend(fp.get("categories", []))
                    model_importance.append(fp.get("importance"))
                    model_severity.append(fp.get("severity_score"))
                    model_weirdness.append(fp.get("weirdness"))
                    model_confidence.append(fp.get("confidence"))
                    model_geo_country.append(fp.get("geography_country", ""))
                    model_geo_region.append(fp.get("geography_region", ""))
                    model_evidence_counts.append(len(fp.get("evidence", [])))

        # Detection: did model find the same threat?
        detected = bool(baseline_flagged_snapshots & model_flagged_snapshots)

        comp = {
            "analysis_id": analysis_id,
            "subreddit": baseline["subreddit"],
            "total_posts": total_posts,
            "baseline_flagged": len(baseline_flagged_snapshots),
            "model_flagged": len(model_flagged),
            "detected": detected,
            # Field comparisons (baseline values are per-analysis, not per-post)
            "categories": {
                "baseline": baseline.get("threat_categories", []),
                "model": model_categories,
                "jaccard": jaccard(
                    baseline.get("threat_categories", []),
                    model_categories,
                ),
            },
            "confidence": {
                "baseline": baseline["stage1"]["confidence"],
                "model": model_confidence,
                "mae": mae(
                    baseline["stage1"]["confidence"],
                    sum(model_confidence) / len(model_confidence) if model_confidence else 0,
                ),
            },
            "severity_score": {
                "baseline": baseline.get("severity_score"),
                "model": model_severity,
                "mae": mae(
                    baseline.get("severity_score"),
                    model_severity[0] if model_severity else None,
                ) if detected else None,
            },
            "importance": {
                "baseline": baseline.get("importance"),
                "model": model_importance,
                "mae": mae(
                    baseline.get("importance"),
                    model_importance[0] if model_importance else None,
                ) if detected else None,
            },
            "weirdness": {
                "baseline": baseline.get("weirdness"),
                "model": model_weirdness,
                "mae": mae(
                    baseline.get("weirdness"),
                    model_weirdness[0] if model_weirdness else None,
                ) if detected else None,
            },
            "geography_country": {
                "baseline": baseline.get("geography_country", ""),
                "model": model_geo_country[0] if model_geo_country else "",
                "match": (
                    (baseline.get("geography_country", "") or "").upper()
                    == (model_geo_country[0] if model_geo_country else "").upper()
                ) if detected else False,
            },
            "geography_region": {
                "baseline": baseline.get("geography_region", ""),
                "model": model_geo_region[0] if model_geo_region else "",
                "match": (
                    (baseline.get("geography_region", "") or "").lower()
                    == (model_geo_region[0] if model_geo_region else "").lower()
                ) if detected else False,
            },
        }
        comparisons.append(comp)

    return {
        "overall": aggregate(comparisons),
        "per_analysis": comparisons,
    }


def aggregate(comparisons: list[dict]) -> dict:
    """Aggregate per-analysis comparisons into summary metrics."""
    n = len(comparisons)
    if n == 0:
        return {}

    detected = sum(1 for c in comparisons if c["detected"])
    detection_rate = detected / n if n else 0

    # Average MAE for detected analyses only
    def avg_field_mae(field: str) -> float | None:
        vals = [c[field]["mae"] for c in comparisons if c[field].get("mae") is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    # Match rates for detected analyses
    def match_rate(field: str) -> float:
        detected_comps = [c for c in comparisons if c["detected"]]
        if not detected_comps:
            return 0.0
        return sum(1 for c in detected_comps if c[field]["match"]) / len(detected_comps)

    # Average category jaccard
    cat_jaccards = [c["categories"]["jaccard"] for c in comparisons if c["detected"]]
    avg_cat_jaccard = round(sum(cat_jaccards) / len(cat_jaccards), 3) if cat_jaccards else 0.0

    # False positives: total model-flagged posts across all analyses minus true positives
    total_model_flagged = sum(c["model_flagged"] for c in comparisons)
    total_baseline_flagged = sum(c["baseline_flagged"] for c in comparisons)

    return {
        "total_analyses": n,
        "detection": {
            "detected": detected,
            "missed": n - detected,
            "detection_rate": round(detection_rate, 3),
        },
        "model_flagged_total": total_model_flagged,
        "baseline_flagged_total": total_baseline_flagged,
        "categories_jaccard": avg_cat_jaccard,
        "confidence_mae": avg_field_mae("confidence"),
        "severity_score_mae": avg_field_mae("severity_score"),
        "importance_mae": avg_field_mae("importance"),
        "weirdness_mae": avg_field_mae("weirdness"),
        "geography_country_match": round(match_rate("geography_country"), 3),
        "geography_region_match": round(match_rate("geography_region"), 3),
    }


def print_report(results: dict, model_name: str) -> None:
    """Print a human-readable eval report."""
    overall = results["overall"]
    if not overall:
        print("No data to report.")
        return

    det = overall["detection"]

    print(f"\n{'='*60}")
    print(f"  EVAL: {model_name} vs Go pipeline baseline")
    print(f"  Analyses compared: {overall['total_analyses']}")
    print(f"{'='*60}\n")

    print(f"  THREAT DETECTION")
    print(f"    Detection rate:     {det['detection_rate']:.1%} ({det['detected']}/{overall['total_analyses']})")
    print(f"    Missed threats:     {det['missed']}")
    print(f"    Model flagged:      {overall['model_flagged_total']} posts total")
    print(f"    Baseline flagged:   {overall['baseline_flagged_total']} posts total")
    print()

    print(f"  FIELD AGREEMENT (detected threats only)")
    print(f"    {'Field':<25} {'Metric':<10} {'Value'}")
    print(f"    {'-'*50}")
    print(f"    {'categories':<25} {'Jaccard':<10} {overall['categories_jaccard']:.3f}")
    if overall['confidence_mae'] is not None:
        print(f"    {'confidence':<25} {'MAE':<10} {overall['confidence_mae']:.3f}")
    if overall['severity_score_mae'] is not None:
        print(f"    {'severity_score':<25} {'MAE':<10} {overall['severity_score_mae']:.3f}")
    if overall['importance_mae'] is not None:
        print(f"    {'importance':<25} {'MAE':<10} {overall['importance_mae']:.3f}")
    if overall['weirdness_mae'] is not None:
        print(f"    {'weirdness':<25} {'MAE':<10} {overall['weirdness_mae']:.3f}")
    print(f"    {'geography_country':<25} {'Match':<10} {overall['geography_country_match']:.1%}")
    print(f"    {'geography_region':<25} {'Match':<10} {overall['geography_region_match']:.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model output against baseline")
    parser.add_argument("--baseline", default="data/baseline.jsonl",
                        help="Path to baseline.jsonl")
    parser.add_argument("--model-output", required=True,
                        help="Path to model output.jsonl (from a run)")
    parser.add_argument("--model-name", default=None,
                        help="Model name for report header")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of report")
    parser.add_argument("--save", default=None,
                        help="Save eval results to this path")

    args = parser.parse_args()

    results = evaluate(args.baseline, args.model_output)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        model_name = args.model_name or args.model_output
        print_report(results, model_name)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved eval to {args.save}")


if __name__ == "__main__":
    main()
