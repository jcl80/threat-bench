"""Error-overlap analysis between TF-IDF+LR and zero-shot MoritzLaurer.

Quantifies how disjoint the two models' errors are on the shared 478-post
test set. The key number is c/(c+d): among examples zero-shot gets wrong,
what fraction does tf-idf get right?

    >15%   meaningful complementarity — ensembling probably helps
    5–15%  mild — ensemble might eke out a point
    <5%    models agree on errors — ensembling won't help

Each model is binarized at its own best-F1 threshold (default tf-idf=0.5,
zero-shot=0.17 — both configurable).

Usage:
    python -m bench.runners.error_overlap \
        --tfidf bench/results/<tfidf_dir>/predictions.jsonl \
        --zeroshot bench/results/<zs_dir>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DATA_PATH = "bench/data/posts.jsonl"


def load_posts() -> dict[int, dict]:
    posts = {}
    with open(DATA_PATH) as f:
        for line in f:
            p = json.loads(line)
            posts[p["snapshot_id"]] = p
    return posts


def load_tfidf(path: Path, test_only: bool = True) -> dict[int, float]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if test_only and r.get("split") != "test":
                continue
            out[r["snapshot_id"]] = r["tfidf_prob"]
    return out


def load_zeroshot(path: Path) -> dict[int, float]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["snapshot_id"]] = r["score"]
    return out


def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1_pos = sum(y1) / n
    p2_pos = sum(y2) / n
    chance = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if chance >= 1.0:
        return 1.0
    return (agree - chance) / (1 - chance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--zeroshot", required=True)
    parser.add_argument("--tfidf-thr", type=float, default=0.5)
    parser.add_argument("--zs-thr", type=float, default=0.17,
                        help="Best-F1 threshold for zero-shot (Run 1 on 478 test = 0.17).")
    parser.add_argument("--out-csv", default=None,
                        help="CSV path for disagreement cases (default: next to tfidf run)")
    args = parser.parse_args()

    posts = load_posts()
    tfidf = load_tfidf(Path(args.tfidf), test_only=True)
    zs_all = load_zeroshot(Path(args.zeroshot))
    # Filter zero-shot to the same test set
    zs = {sid: score for sid, score in zs_all.items() if sid in tfidf}

    # Alignment check — fail loudly
    if set(tfidf.keys()) != set(zs.keys()):
        missing_in_zs = set(tfidf.keys()) - set(zs.keys())
        missing_in_tf = set(zs.keys()) - set(tfidf.keys())
        raise SystemExit(
            f"Test set mismatch!\n"
            f"  in tfidf but not zs: {len(missing_in_zs)} ids\n"
            f"  in zs but not tfidf: {len(missing_in_tf)} ids\n"
            f"Check that both runs used the same split."
        )
    ids = sorted(tfidf.keys())
    print(f"Aligned on {len(ids)} test posts")

    y_true = [posts[sid]["label_gpt5"] for sid in ids]
    tf_pred = [1 if tfidf[sid] >= args.tfidf_thr else 0 for sid in ids]
    zs_pred = [1 if zs[sid] >= args.zs_thr else 0 for sid in ids]

    # Per-model F1 sanity check
    def f1(pred, truth):
        tp = sum(1 for p, y in zip(pred, truth) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(pred, truth) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(pred, truth) if p == 0 and y == 1)
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        return (p_, r_, 2*p_*r_/(p_+r_) if (p_+r_) else 0.0)

    tf_p, tf_r, tf_f1 = f1(tf_pred, y_true)
    zs_p, zs_r, zs_f1 = f1(zs_pred, y_true)
    print(f"TF-IDF    @thr={args.tfidf_thr}: P={tf_p:.3f} R={tf_r:.3f} F1={tf_f1:.3f}")
    print(f"Zero-shot @thr={args.zs_thr}:  P={zs_p:.3f} R={zs_r:.3f} F1={zs_f1:.3f}")

    # 2×2 correctness confusion
    # rows = zero-shot (correct/wrong), cols = tfidf (correct/wrong)
    a = b = c = d = 0
    for p_tf, p_zs, y in zip(tf_pred, zs_pred, y_true):
        tf_correct = (p_tf == y)
        zs_correct = (p_zs == y)
        if zs_correct and tf_correct:       a += 1
        elif zs_correct and not tf_correct: b += 1
        elif not zs_correct and tf_correct: c += 1
        else:                                d += 1

    print(f"\n  2×2 correctness confusion (n={len(ids)}):")
    print(f"                     tfidf correct    tfidf wrong")
    print(f"  zs correct         {a:>6d}           {b:>6d}")
    print(f"  zs wrong           {c:>6d}           {d:>6d}")

    # Key numbers
    agreement = (a + d) / len(ids)
    kappa = cohen_kappa(tf_pred, zs_pred)
    zs_wrong_total = c + d
    complement = c / zs_wrong_total if zs_wrong_total else 0.0

    print(f"\n  Overall agreement (same correctness):  {agreement:.3f}")
    print(f"  Cohen's kappa (prediction agreement):  {kappa:.3f}")
    print(f"  c/(c+d) — of zs errors, tfidf fixes:   {complement:.3f}  "
          f"({c}/{zs_wrong_total})")

    # Decision hint
    if complement > 0.15:
        verdict = "ENSEMBLE LIKELY WORTH IT"
    elif complement > 0.05:
        verdict = "MARGINAL — ensemble might eke out a point"
    else:
        verdict = "SKIP ENSEMBLE — models agree on errors"
    print(f"\n  Verdict: {verdict}")

    # Error-type breakdown
    print(f"\n  Zero-shot error type analysis:")
    zs_fp_total = sum(1 for p_zs, y in zip(zs_pred, y_true) if p_zs == 1 and y == 0)
    zs_fn_total = sum(1 for p_zs, y in zip(zs_pred, y_true) if p_zs == 0 and y == 1)
    zs_fp_fixed = sum(1 for p_tf, p_zs, y in zip(tf_pred, zs_pred, y_true)
                      if p_zs == 1 and y == 0 and p_tf == 0)
    zs_fn_fixed = sum(1 for p_tf, p_zs, y in zip(tf_pred, zs_pred, y_true)
                      if p_zs == 0 and y == 1 and p_tf == 1)
    fp_rate = zs_fp_fixed / zs_fp_total if zs_fp_total else 0.0
    fn_rate = zs_fn_fixed / zs_fn_total if zs_fn_total else 0.0
    print(f"    Of {zs_fp_total} zs false positives, tfidf correctly calls negative: "
          f"{zs_fp_fixed} ({fp_rate:.1%})")
    print(f"    Of {zs_fn_total} zs false negatives, tfidf correctly calls positive: "
          f"{zs_fn_fixed} ({fn_rate:.1%})")

    # Disagreement CSV
    csv_path = Path(args.out_csv) if args.out_csv else \
               Path(args.tfidf).parent / "disagreements.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot_id", "subreddit", "tier", "label",
                    "tfidf_pred", "zs_pred",
                    "tfidf_prob", "zs_score",
                    "title", "body_excerpt"])
        for sid, p_tf_, p_zs_ in zip(ids, tf_pred, zs_pred):
            if p_tf_ == p_zs_:
                continue
            p = posts[sid]
            body = (p.get("body") or "").strip()[:300]
            w.writerow([sid, p.get("subreddit"), p.get("tier"),
                        p["label_gpt5"], p_tf_, p_zs_,
                        round(tfidf[sid], 4), round(zs[sid], 4),
                        (p.get("title") or "").strip(), body])
    print(f"\n  Disagreement cases saved to {csv_path} "
          f"({b + c} rows)")


if __name__ == "__main__":
    main()
