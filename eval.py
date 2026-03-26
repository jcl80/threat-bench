"""Evaluate model output against production pipeline baseline.

Computes: precision, recall, F1, field agreement, per-tier and per-category
breakdowns, throughput, and cost metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TIERS = {
    'collapse': 'threat-dense', 'ukraine': 'threat-dense',
    'worldnews': 'threat-dense', 'geopolitics': 'threat-dense',
    'economics': 'ambiguous', 'technology': 'ambiguous',
    'news': 'ambiguous', 'energy': 'ambiguous',
    'cooking': 'benign', 'askscience': 'benign',
    'woodworking': 'benign', 'gardening': 'benign',
}


def jaccard(a: list, b: list) -> float:
    sa, sb = set(s.lower() for s in a), set(s.lower() for s in b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def mae(a: float | int | None, b: float | int | None) -> float | None:
    if a is None and b is None:
        return None
    return abs((a or 0) - (b or 0))


def evaluate(baseline_path: str, model_output_path: str) -> dict:
    """Compare model output against baseline.

    Handles all baseline statuses: confirmed_threat, no_threat,
    false_positive, needs_review. Only confirmed_threat analyses
    have "true" flagged posts to detect.
    """
    with open(baseline_path) as f:
        baselines = {b["analysis_id"]: b for line in f for b in [json.loads(line)]}

    with open(model_output_path) as f:
        model_runs = {r["analysis_id"]: r for line in f for r in [json.loads(line)]}

    comparisons = []

    for analysis_id, baseline in baselines.items():
        model_run = model_runs.get(analysis_id)
        if not model_run:
            continue

        status = baseline.get("final_status", "")
        subreddit = baseline.get("subreddit", "")
        tier = TIERS.get(subreddit.lower(), "unknown")
        is_threat = status == "confirmed_threat"

        # Baseline flagged posts (from stage1 evidence)
        baseline_flagged_sids = set()
        for ev in baseline.get("stage1", {}).get("evidence", []):
            sid = ev.get("post_snapshot_id")
            if sid:
                baseline_flagged_sids.add(sid)

        # Model flagged posts
        model_flagged = model_run.get("flagged_posts", [])
        total_posts = model_run.get("total_posts", len(baseline.get("post_snapshot_ids", [])))
        snapshot_ids = baseline.get("post_snapshot_ids", [])

        model_flagged_sids = set()
        model_matched_categories = []
        model_matched_importance = []
        model_matched_weirdness = []
        model_matched_confidence = []
        model_matched_geo_country = []
        model_matched_geo_region = []

        for fp in model_flagged:
            idx = fp.get("post_index", 0)
            if 1 <= idx <= len(snapshot_ids):
                sid = snapshot_ids[idx - 1]
                model_flagged_sids.add(sid)

                if sid in baseline_flagged_sids:
                    model_matched_categories.extend(fp.get("categories", []))
                    model_matched_importance.append(fp.get("importance"))
                    model_matched_weirdness.append(fp.get("weirdness"))
                    model_matched_confidence.append(fp.get("confidence"))
                    model_matched_geo_country.append(fp.get("geography_country", ""))
                    model_matched_geo_region.append(fp.get("geography_region", ""))

        # Detection: did model flag at least one of the baseline-flagged posts?
        detected = bool(baseline_flagged_sids & model_flagged_sids) if is_threat else None

        # Post-level precision/recall for this batch
        # TP = model flagged AND baseline flagged
        # FP = model flagged AND NOT baseline flagged
        # FN = NOT model flagged AND baseline flagged
        tp = len(model_flagged_sids & baseline_flagged_sids)
        fp_count = len(model_flagged_sids - baseline_flagged_sids)
        fn = len(baseline_flagged_sids - model_flagged_sids)

        # For non-threat analyses, any flag is a false positive
        if not is_threat:
            fp_count = len(model_flagged)
            tp = 0
            fn = 0

        # Field comparisons (only for detected threats)
        field_data = {}
        if is_threat and detected:
            field_data = {
                "categories": {
                    "baseline": baseline.get("threat_categories", []),
                    "model": model_matched_categories,
                    "jaccard": jaccard(baseline.get("threat_categories", []), model_matched_categories),
                },
                "confidence": {
                    "baseline": baseline.get("stage1", {}).get("confidence"),
                    "model": model_matched_confidence,
                    "mae": mae(
                        baseline.get("stage1", {}).get("confidence"),
                        sum(model_matched_confidence) / len(model_matched_confidence) if model_matched_confidence else 0,
                    ),
                },
                "importance": {
                    "baseline": baseline.get("importance"),
                    "model": model_matched_importance,
                    "mae": mae(
                        baseline.get("importance"),
                        model_matched_importance[0] if model_matched_importance else None,
                    ),
                },
                "weirdness": {
                    "baseline": baseline.get("weirdness"),
                    "model": model_matched_weirdness,
                    "mae": mae(
                        baseline.get("weirdness"),
                        model_matched_weirdness[0] if model_matched_weirdness else None,
                    ),
                },
                "geography_country": {
                    "baseline": baseline.get("geography_country", ""),
                    "model": model_matched_geo_country[0] if model_matched_geo_country else "",
                    "match": (
                        (baseline.get("geography_country", "") or "").upper()
                        == (model_matched_geo_country[0] if model_matched_geo_country else "").upper()
                    ),
                },
                "geography_region": {
                    "baseline": baseline.get("geography_region", ""),
                    "model": model_matched_geo_region[0] if model_matched_geo_region else "",
                    "match": (
                        (baseline.get("geography_region", "") or "").lower()
                        == (model_matched_geo_region[0] if model_matched_geo_region else "").lower()
                    ),
                },
            }

        # Usage from model run
        usage = model_run.get("usage", {})

        comparisons.append({
            "analysis_id": analysis_id,
            "subreddit": subreddit,
            "tier": tier,
            "baseline_status": status,
            "baseline_categories": baseline.get("threat_categories", []),
            "total_posts": total_posts,
            "baseline_flagged": len(baseline_flagged_sids),
            "model_flagged": len(model_flagged),
            "detected": detected,
            "tp": tp,
            "fp": fp_count,
            "fn": fn,
            "fields": field_data,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "wall_clock_seconds": usage.get("wall_clock_seconds", 0),
            },
        })

    return {
        "overall": aggregate(comparisons),
        "by_tier": aggregate_by_tier(comparisons),
        "by_category": aggregate_by_category(comparisons),
        "per_analysis": comparisons,
    }


def aggregate(comparisons: list[dict]) -> dict:
    n = len(comparisons)
    if n == 0:
        return {}

    # Split by status
    threats = [c for c in comparisons if c["baseline_status"] == "confirmed_threat"]
    non_threats = [c for c in comparisons if c["baseline_status"] != "confirmed_threat"]

    # Detection (threats only)
    detected = sum(1 for c in threats if c["detected"])
    recall = detected / len(threats) if threats else 0

    # Note on precision: the baseline stores evidence for only the 1 primary
    # threat post per analysis (the post that survived stage 2). It does NOT
    # store all posts that stage 1 flagged. So post-level precision cannot be
    # computed accurately — TP is undercounted because many model-flagged posts
    # may be legitimately threat-related but aren't in the baseline evidence.
    # We report flag_rate as a relative comparison metric instead.
    total_tp = sum(c["tp"] for c in comparisons)
    total_fp = sum(c["fp"] for c in comparisons)
    total_fn = sum(c["fn"] for c in comparisons)

    # Field MAEs (detected threats only)
    detected_threats = [c for c in threats if c["detected"]]

    def avg_field_mae(field: str) -> float | None:
        vals = [c["fields"][field]["mae"] for c in detected_threats
                if c["fields"].get(field, {}).get("mae") is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def match_rate(field: str) -> float:
        if not detected_threats:
            return 0.0
        return sum(1 for c in detected_threats if c["fields"].get(field, {}).get("match", False)) / len(detected_threats)

    cat_jaccards = [c["fields"]["categories"]["jaccard"] for c in detected_threats if "categories" in c["fields"]]
    avg_cat_jaccard = round(sum(cat_jaccards) / len(cat_jaccards), 3) if cat_jaccards else 0.0

    # Flagging totals
    total_model_flagged = sum(c["model_flagged"] for c in comparisons)
    total_baseline_flagged = sum(c["baseline_flagged"] for c in comparisons)
    total_posts = sum(c["total_posts"] for c in comparisons)

    # False positives on non-threat analyses
    benign_fps = sum(c["model_flagged"] for c in non_threats)

    # Throughput
    total_tokens = sum(c["usage"]["total_tokens"] for c in comparisons)
    total_completion = sum(c["usage"]["completion_tokens"] for c in comparisons)
    total_prompt = sum(c["usage"]["prompt_tokens"] for c in comparisons)
    total_wall = sum(c["usage"]["wall_clock_seconds"] for c in comparisons)
    tokens_per_sec = round(total_completion / total_wall, 1) if total_wall > 0 else None

    return {
        "total_analyses": n,
        "threat_analyses": len(threats),
        "non_threat_analyses": len(non_threats),
        "detection": {
            "detected": detected,
            "missed": len(threats) - detected,
            "recall": round(recall, 3),
        },
        "post_level": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "note": "precision/F1 not meaningful — baseline only stores 1 evidence post per analysis, not all stage1 flags",
        },
        "flagging": {
            "model_flagged": total_model_flagged,
            "baseline_flagged": total_baseline_flagged,
            "total_posts": total_posts,
            "model_flag_rate": round(total_model_flagged / total_posts, 3) if total_posts else 0,
            "benign_false_positives": benign_fps,
        },
        "field_agreement": {
            "categories_jaccard": avg_cat_jaccard,
            "confidence_mae": avg_field_mae("confidence"),
            "importance_mae": avg_field_mae("importance"),
            "weirdness_mae": avg_field_mae("weirdness"),
            "geography_country_match": round(match_rate("geography_country"), 3),
            "geography_region_match": round(match_rate("geography_region"), 3),
        },
        "throughput": {
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "wall_clock_seconds": round(total_wall, 1),
            "completion_tokens_per_sec": tokens_per_sec,
        },
    }


def aggregate_by_tier(comparisons: list[dict]) -> dict:
    by_tier = defaultdict(list)
    for c in comparisons:
        by_tier[c["tier"]].append(c)
    return {tier: aggregate(comps) for tier, comps in sorted(by_tier.items())}


def aggregate_by_category(comparisons: list[dict]) -> dict:
    """Per-category detection rate among confirmed_threat analyses."""
    cat_stats = defaultdict(lambda: {"total": 0, "detected": 0})
    for c in comparisons:
        if c["baseline_status"] != "confirmed_threat":
            continue
        for cat in c.get("baseline_categories", []):
            cat_lower = cat.lower()
            cat_stats[cat_lower]["total"] += 1
            if c["detected"]:
                cat_stats[cat_lower]["detected"] += 1

    result = {}
    for cat, stats in sorted(cat_stats.items()):
        rate = stats["detected"] / stats["total"] if stats["total"] else 0
        result[cat] = {
            "total": stats["total"],
            "detected": stats["detected"],
            "detection_rate": round(rate, 3),
        }
    return result


def print_report(results: dict, model_name: str) -> None:
    overall = results["overall"]
    if not overall:
        print("No data to report.")
        return

    det = overall["detection"]
    pl = overall["post_level"]
    fl = overall["flagging"]
    fa = overall["field_agreement"]
    tp = overall["throughput"]

    print(f"\n{'='*65}")
    print(f"  EVAL: {model_name} vs production baseline")
    print(f"  Analyses: {overall['total_analyses']} ({overall['threat_analyses']} threats, {overall['non_threat_analyses']} non-threats)")
    print(f"{'='*65}\n")

    print(f"  THREAT DETECTION (analysis-level)")
    print(f"    Recall:             {det['recall']:.1%} ({det['detected']}/{overall['threat_analyses']} threats found)")
    print(f"    Missed:             {det['missed']}")
    print()

    print("  POST-LEVEL MATCHING")
    print(f"    Primary threat posts found: {pl['tp']}/{pl['tp']+pl['fn']}")
    print(f"    (Baseline stores 1 evidence post per analysis — use flag_rate for cross-model comparison)")
    print()

    print(f"  FLAGGING VOLUME")
    print(f"    Model flagged:      {fl['model_flagged']}/{fl['total_posts']} posts ({fl['model_flag_rate']:.0%})")
    print(f"    Baseline flagged:   {fl['baseline_flagged']}/{fl['total_posts']} posts")
    print(f"    False positives on non-threat batches: {fl['benign_false_positives']}")
    print()

    print(f"  FIELD AGREEMENT (detected threats only)")
    print(f"    {'Field':<25} {'Metric':<10} {'Value'}")
    print(f"    {'-'*50}")
    print(f"    {'categories':<25} {'Jaccard':<10} {fa['categories_jaccard']:.3f}")
    if fa['confidence_mae'] is not None:
        print(f"    {'confidence':<25} {'MAE':<10} {fa['confidence_mae']:.3f}")
    if fa['importance_mae'] is not None:
        print(f"    {'importance':<25} {'MAE':<10} {fa['importance_mae']:.3f}")
    if fa['weirdness_mae'] is not None:
        print(f"    {'weirdness':<25} {'MAE':<10} {fa['weirdness_mae']:.3f}")
    print(f"    {'geography_country':<25} {'Match':<10} {fa['geography_country_match']:.1%}")
    print(f"    {'geography_region':<25} {'Match':<10} {fa['geography_region_match']:.1%}")
    print()

    print(f"  THROUGHPUT")
    print(f"    Total tokens:       {tp['total_tokens']:,}")
    print(f"    Wall clock:         {tp['wall_clock_seconds']:.0f}s")
    if tp['completion_tokens_per_sec']:
        print(f"    Tokens/sec:         {tp['completion_tokens_per_sec']:.0f} (completion)")
    print()

    # Per-tier
    print(f"  PER-TIER DETECTION RATE")
    for tier, tier_data in results.get("by_tier", {}).items():
        td = tier_data.get("detection", {})
        ta = tier_data.get("threat_analyses", 0)
        nta = tier_data.get("non_threat_analyses", 0)
        fl_tier = tier_data.get("flagging", {})
        if ta > 0:
            print(f"    {tier:<20} recall={td.get('recall', 0):.0%} ({td.get('detected', 0)}/{ta} threats)  flag_rate={fl_tier.get('model_flag_rate', 0):.0%}")
        else:
            print(f"    {tier:<20} (no threats)  flag_rate={fl_tier.get('model_flag_rate', 0):.0%}  benign_fps={fl_tier.get('benign_false_positives', 0)}")
    print()

    # Per-category
    print(f"  PER-CATEGORY DETECTION RATE")
    for cat, cd in results.get("by_category", {}).items():
        print(f"    {cat:<25} {cd['detection_rate']:.0%} ({cd['detected']}/{cd['total']})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model output against baseline")
    parser.add_argument("--baseline", default="data/baseline.jsonl")
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", default=None)

    args = parser.parse_args()
    results = evaluate(args.baseline, args.model_output)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results, args.model_name or args.model_output)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved eval to {args.save}")


if __name__ == "__main__":
    main()
