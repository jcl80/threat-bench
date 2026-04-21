"""Compare the frozen-backbone probe (Mode A) vs the zero-shot baseline
on the same 478 test posts, both threshold-swept.

Zero-shot scores exist for all 2385 posts; we filter to the 478 test posts
using the same deterministic split as finetune_deberta.run().

Usage:
    python -m bench.compare_probe_vs_zeroshot \
        --probe bench/results/<finetune_dir>/predictions_test.jsonl \
        --zeroshot bench/results/<zeroshot_dir>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from bench.scorer import score

DATA_PATH = "bench/data/posts.jsonl"
SEED = 42


def get_test_ids() -> set[int]:
    """Reproduce the stratified 80/20 split used in finetune_deberta."""
    with open(DATA_PATH) as f:
        posts = [json.loads(l) for l in f]
    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    test = pos[split_p:] + neg[split_n:]
    return {p["snapshot_id"] for p in test}


def filter_to_test(pred_path: Path, test_ids: set[int], out_path: Path) -> None:
    with open(pred_path) as f, open(out_path, "w") as w:
        for line in f:
            row = json.loads(line)
            if row["snapshot_id"] in test_ids:
                w.write(line)


def sweep_overall(pred_rows: list[dict], thresholds: list[float],
                  tmp_path: Path, ground_truth: str) -> list[dict]:
    results = []
    for t in thresholds:
        with open(tmp_path, "w") as f:
            for r in pred_rows:
                out = {
                    "snapshot_id": r["snapshot_id"],
                    "predicted": 1 if r["score"] >= t else 0,
                    "score": r["score"],
                }
                f.write(json.dumps(out) + "\n")
        r = score(tmp_path, Path(DATA_PATH), ground_truth)
        row = {"threshold": t, **r["overall"]}
        results.append(row)
    tmp_path.unlink(missing_ok=True)
    return results


def best_by_f1(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["f1"])


def print_curve(name: str, rows: list[dict]) -> None:
    print(f"\n{'='*72}")
    print(f"{name}")
    print(f"{'='*72}")
    print(f"{'thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    for r in rows:
        print(f"{r['threshold']:>5.2f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}  "
              f"{r['f1']:>6.3f}  {r['tp']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}")
    best = best_by_f1(rows)
    print(f"  best F1: {best['f1']:.3f} at thr={best['threshold']:.2f} "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="predictions_test.jsonl (fine-tune)")
    parser.add_argument("--zeroshot", required=True, help="predictions.jsonl (zero-shot, all 2385)")
    parser.add_argument("--ground-truth", default="gpt5",
                        choices=["gpt5", "gpt5_mini"])
    parser.add_argument("--min", type=float, default=0.01)
    parser.add_argument("--max", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    test_ids = get_test_ids()
    print(f"Test set: {len(test_ids)} posts")

    # Load probe predictions (already test-only)
    with open(args.probe) as f:
        probe_rows = [json.loads(l) for l in f]
    assert all(r["snapshot_id"] in test_ids for r in probe_rows), \
        f"Probe predictions include non-test posts"

    # Filter zero-shot predictions to test set
    with open(args.zeroshot) as f:
        zs_all = [json.loads(l) for l in f]
    zs_rows = [r for r in zs_all if r["snapshot_id"] in test_ids]
    assert len(zs_rows) == len(test_ids), \
        f"Zero-shot covers {len(zs_rows)}/{len(test_ids)} test posts"

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    tmp = Path(args.probe).parent / ".tmp_sweep.jsonl"

    probe_curve = sweep_overall(probe_rows, thresholds, tmp, args.ground_truth)
    zs_curve = sweep_overall(zs_rows, thresholds, tmp, args.ground_truth)

    print(f"\nBoth scored on the same {len(test_ids)} test posts, vs label_{args.ground_truth}")
    print_curve(f"PROBE (fine-tuned Mode A)", probe_curve)
    print_curve(f"ZERO-SHOT (Run 1, no fine-tune)", zs_curve)

    # Head-to-head at each model's best threshold
    probe_best = best_by_f1(probe_curve)
    zs_best = best_by_f1(zs_curve)
    print(f"\n{'='*72}")
    print(f"HEAD-TO-HEAD (each at its own best F1 threshold)")
    print(f"{'='*72}")
    print(f"{'':>20}  {'thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"{'Probe (Mode A)':>20}  {probe_best['threshold']:>5.2f}  "
          f"{probe_best['precision']:>6.3f}  {probe_best['recall']:>6.3f}  "
          f"{probe_best['f1']:>6.3f}")
    print(f"{'Zero-shot':>20}  {zs_best['threshold']:>5.2f}  "
          f"{zs_best['precision']:>6.3f}  {zs_best['recall']:>6.3f}  "
          f"{zs_best['f1']:>6.3f}")
    delta = probe_best["f1"] - zs_best["f1"]
    print(f"\nF1 delta: probe - zeroshot = {delta:+.4f}")


if __name__ == "__main__":
    main()
