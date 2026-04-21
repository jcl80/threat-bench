"""Operating-point tables: at each target recall, find the highest threshold
that still achieves it, report precision/F1/flag rate.

Prints four side-by-side markdown tables:
  1. Zero-shot on training test (478 posts, filtered from full 2385)
  2. Zero-shot on in-distribution holdout (1686 posts)
  3. Zero-shot on OOD holdout (1020 posts)
  4. Stacked ensemble on in-distribution holdout

Usage:
    python3 bench/operating_points.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
TARGETS = [0.80, 0.85, 0.90, 0.95, 0.98]


def reconstruct_test_split_ids(posts_path: str) -> set[int]:
    """Recreate the 478-post test split used by stacked_ensemble/validate_holdout."""
    with open(posts_path) as f:
        posts = [json.loads(l) for l in f]
    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos); rnd.shuffle(neg)
    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    test = pos[split_p:] + neg[split_n:]
    return {p["snapshot_id"] for p in test}


def load_probs_and_labels(
    predictions_path: str,
    prob_field: str,
    labels_from_predictions: bool,
    posts_path: str | None,
    filter_ids: set[int] | None,
) -> tuple[list[float], list[int]]:
    """Return (probs, labels) aligned by position.

    If labels_from_predictions: the predictions file has a `label` field.
    Else: read labels from posts_path keyed by snapshot_id.
    If filter_ids is given: keep only those snapshot_ids.
    """
    labels_by_id: dict[int, int] = {}
    if not labels_from_predictions:
        assert posts_path, "posts_path required when labels_from_predictions=False"
        with open(posts_path) as f:
            for line in f:
                r = json.loads(line)
                labels_by_id[r["snapshot_id"]] = int(r["label_gpt5"])

    probs, labels = [], []
    with open(predictions_path) as f:
        for line in f:
            r = json.loads(line)
            sid = r["snapshot_id"]
            if filter_ids is not None and sid not in filter_ids:
                continue
            if labels_from_predictions:
                y = int(r["label"])
            else:
                if sid not in labels_by_id:
                    continue
                y = labels_by_id[sid]
            probs.append(float(r[prob_field]))
            labels.append(y)
    return probs, labels


def metrics_at(probs: list[float], labels: list[int], thr: float) -> dict:
    tp = fp = fn = tn = 0
    for p, y in zip(probs, labels):
        pred = 1 if p >= thr else 0
        if y == 1 and pred == 1: tp += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def find_op_point(probs: list[float], labels: list[int],
                  target_recall: float) -> dict:
    """Highest threshold that still achieves >= target_recall.

    Sweep finely (0.001 steps). If the model can't hit target_recall at any
    threshold, return the lowest-threshold result (max recall achievable).
    """
    best = None
    t = 0.001
    # Walk from high to low; first threshold meeting recall >= target is the answer
    # (since recall is monotonically non-decreasing as threshold drops).
    candidates = []
    while t <= 0.999 + 1e-9:
        m = metrics_at(probs, labels, round(t, 4))
        candidates.append(m)
        t += 0.001

    # Filter those that meet target
    meeting = [c for c in candidates if c["recall"] >= target_recall]
    if meeting:
        # Among meeting, pick the one with the highest threshold (least aggressive)
        best = max(meeting, key=lambda c: c["threshold"])
    else:
        # Can't hit target — return the maximum-recall point
        best = max(candidates, key=lambda c: c["recall"])

    return best


def print_table(name: str, probs: list[float], labels: list[int]) -> None:
    n = len(labels)
    n_pos = sum(labels)
    print(f"\n### {name}")
    print(f"*n={n}, positives={n_pos} ({100*n_pos/n:.1f}%)*\n")
    print("| target recall | actual recall | threshold | precision | F1 | flag rate | n flagged |")
    print("|--:|--:|--:|--:|--:|--:|--:|")
    for target in TARGETS:
        m = find_op_point(probs, labels, target)
        n_flagged = m["tp"] + m["fp"]
        flag_rate = n_flagged / n
        hit = m["recall"] >= target
        achieved = f"{m['recall']:.3f}" + ("" if hit else " ⚠️")
        print(f"| {target:.2f} | {achieved} | {m['threshold']:.3f} | "
              f"{m['precision']:.3f} | {m['f1']:.3f} | "
              f"{flag_rate:.3f} | {n_flagged} |")


def main():
    reports = [
        {
            "name": "Zero-shot DeBERTa — training test (478 posts)",
            "predictions": "bench/results/2026-04-18T23-13-31_MoritzLaurer_deberta-v3-large-zeroshot-v2.0/predictions.jsonl",
            "prob_field": "score",
            "labels_from_predictions": False,
            "posts_path": "bench/data/posts.jsonl",
            "filter_to_test_split": True,
        },
        {
            "name": "Zero-shot DeBERTa — in-distribution holdout (1686 posts)",
            "predictions": "bench/results/2026-04-21T03-40-38_MoritzLaurer_deberta-v3-large-zeroshot-v2.0/predictions.jsonl",
            "prob_field": "score",
            "labels_from_predictions": False,
            "posts_path": "bench/data/holdout/posts_holdout.jsonl",
            "filter_to_test_split": False,
        },
        {
            "name": "Zero-shot DeBERTa — OOD holdout (1020 posts)",
            "predictions": "bench/results/2026-04-21T04-18-14_MoritzLaurer_deberta-v3-large-zeroshot-v2.0/predictions.jsonl",
            "prob_field": "score",
            "labels_from_predictions": False,
            "posts_path": "bench/data/holdout_ood/posts_ood.jsonl",
            "filter_to_test_split": False,
        },
        {
            "name": "Stacked ensemble — in-distribution holdout (1686 posts)",
            "predictions": "bench/data/holdout/ensemble_predictions.jsonl",
            "prob_field": "ens_prob",
            "labels_from_predictions": True,
            "posts_path": None,
            "filter_to_test_split": False,
        },
    ]

    print("# Operating points at target recall\n")
    print(f"For each target recall, the highest threshold that still meets it.")
    print(f"Precision, F1, and flag rate reported at that threshold.\n")

    for cfg in reports:
        filter_ids = None
        if cfg["filter_to_test_split"]:
            filter_ids = reconstruct_test_split_ids(cfg["posts_path"])
        probs, labels = load_probs_and_labels(
            cfg["predictions"],
            cfg["prob_field"],
            cfg["labels_from_predictions"],
            cfg["posts_path"],
            filter_ids,
        )
        print_table(cfg["name"], probs, labels)


if __name__ == "__main__":
    main()
