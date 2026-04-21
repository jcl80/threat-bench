"""Evaluate all three models on the fresh 1686-post holdout.

Methodology:
  1. Reconstruct the exact 1907-post training split (seed=42) from the
     original posts.jsonl — same split the stacked_ensemble's meta-LR
     was calibrated on.
  2. Fit TF-IDF + LR on those 1907 posts, predict on the 1686 holdout.
  3. Load DeBERTa zero-shot scores on the holdout (already produced by
     bench/runners/nli_deberta.py on Vast).
  4. Apply saved meta-LR weights from the stacked_ensemble run to combine.
  5. Report each model's F1 at three operating points on holdout:
     - Its chosen threshold from the training test set ("shipped thr")
     - Holdout's best-F1 threshold (diagnostic for boundary stability)
     - Default 0.5
  6. Compare to training numbers — report the F1 deltas.

Usage:
    python -m bench.runners.validate_holdout \\
        --holdout bench/data/holdout/posts_holdout.jsonl \\
        --deberta bench/results/<holdout_deberta_dir>/predictions.jsonl \\
        --stacked bench/results/<training_stacked_dir>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bench.runners.tfidf_baseline import build_premise, SEED, MAX_PREMISE_CHARS

TRAINING_DATA = "bench/data/posts.jsonl"

# Operating points we "shipped" based on training-test best-F1
SHIPPED_THR = {
    "zeroshot": 0.17,
    "tfidf":    0.44,
    "ensemble": 0.30,
}


def reconstruct_training_split(training_path: str) -> list[dict]:
    """Recover the 1907-post training half used by stacked_ensemble."""
    with open(training_path) as f:
        posts = [json.loads(l) for l in f]
    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos); rnd.shuffle(neg)
    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    train = pos[:split_p] + neg[:split_n]
    rnd.shuffle(train)
    return train


def score_at(probs: list[float], labels: list[int], thr: float) -> dict:
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
    acc = (tp + tn) / max(len(labels), 1)
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}


def best_sweep(probs, labels, start=0.05, end=0.95, step=0.01):
    best = None
    t = start
    while t <= end + 1e-9:
        m = score_at(probs, labels, round(t, 2))
        if best is None or m["f1"] > best["f1"]:
            best = m
        t += step
    return best


def load_deberta_scores(pred_path: Path, test_ids: set[int]) -> dict[int, float]:
    scores = {}
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            if r["snapshot_id"] in test_ids:
                scores[r["snapshot_id"]] = r["score"]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", required=True,
                        help="posts_holdout.jsonl (with labels)")
    parser.add_argument("--deberta", required=True,
                        help="DeBERTa predictions.jsonl on the holdout set")
    parser.add_argument("--stacked", required=True,
                        help="Training stacked ensemble metadata.json (for meta-LR weights)")
    parser.add_argument("--training-data", default=TRAINING_DATA,
                        help="Original posts.jsonl (for reconstructing 1907 training split)")
    args = parser.parse_args()

    # --- Load holdout ---
    with open(args.holdout) as f:
        holdout = [json.loads(l) for l in f]
    holdout_ids = [p["snapshot_id"] for p in holdout]
    holdout_texts = [build_premise(p) for p in holdout]
    y_holdout = np.array([p["label_gpt5"] for p in holdout])
    print(f"Holdout: {len(holdout)} posts ({int(y_holdout.sum())} pos / "
          f"{int((1-y_holdout).sum())} neg)")

    # --- Reconstruct 1907-post training split ---
    train = reconstruct_training_split(args.training_data)
    train_texts = [build_premise(p) for p in train]
    y_train = np.array([p["label_gpt5"] for p in train])
    print(f"Training split: {len(train)} posts ({int(y_train.sum())} pos)")

    # --- Fit final TF-IDF + LR on training split, predict on holdout ---
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000,
        sublinear_tf=True, strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_holdout = vec.transform(holdout_texts)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    clf.fit(X_train, y_train)
    tfidf_holdout = clf.predict_proba(X_holdout)[:, 1]
    print(f"TF-IDF refit on training split. {X_train.shape[1]} features.")

    # --- Load DeBERTa zero-shot scores on holdout ---
    zs_map = {}
    with open(args.deberta) as f:
        for line in f:
            r = json.loads(line)
            zs_map[r["snapshot_id"]] = r["score"]
    missing = [sid for sid in holdout_ids if sid not in zs_map]
    if missing:
        raise SystemExit(f"DeBERTa missing scores for {len(missing)} holdout posts")
    zs_holdout = np.array([zs_map[sid] for sid in holdout_ids])
    print(f"DeBERTa scores loaded for all {len(holdout)} holdout posts.")

    # --- Load meta-LR weights from training stacked run ---
    with open(args.stacked) as f:
        stacked_meta = json.load(f)
    w_zs = stacked_meta["meta_weights"]["zs"]
    w_tf = stacked_meta["meta_weights"]["tfidf"]
    bias = stacked_meta["meta_weights"]["bias"]
    print(f"Meta-LR weights: zs={w_zs:+.3f} tfidf={w_tf:+.3f} bias={bias:+.3f}")

    # Apply meta-LR formula: sigmoid(w_zs * zs + w_tf * tf + bias)
    logit = w_zs * zs_holdout + w_tf * tfidf_holdout + bias
    ens_holdout = 1.0 / (1.0 + np.exp(-logit))

    # --- Report ---
    models = [
        ("Zero-shot",       zs_holdout,     SHIPPED_THR["zeroshot"]),
        ("TF-IDF + LR",     tfidf_holdout,  SHIPPED_THR["tfidf"]),
        ("Ensemble",        ens_holdout,    SHIPPED_THR["ensemble"]),
    ]

    # Training baselines for delta reporting
    train_best_f1 = {"Zero-shot": 0.833, "TF-IDF + LR": 0.833, "Ensemble": 0.868}
    train_at_shipped = {"Zero-shot": 0.833, "TF-IDF + LR": 0.833, "Ensemble": 0.868}

    print(f"\n{'='*88}")
    print(f"HOLDOUT METRICS (n={len(holdout)}, gpt5-positive rate "
          f"{y_holdout.mean():.1%})")
    print(f"{'='*88}")
    print(f"{'Model':<14}  {'mode':<18}  {'thr':>5}  {'P':>6}  {'R':>6}  "
          f"{'F1':>6}  {'ΔF1 vs train':>12}")
    print("-" * 88)

    results = {}
    for name, probs, shipped_thr in models:
        m_ship = score_at(probs.tolist(), y_holdout.tolist(), shipped_thr)
        m_05   = score_at(probs.tolist(), y_holdout.tolist(), 0.5)
        m_best = best_sweep(probs.tolist(), y_holdout.tolist())

        delta_ship = m_ship["f1"] - train_at_shipped[name]
        delta_best = m_best["f1"] - train_best_f1[name]

        for mode, m, delta in [
            ("shipped-thr",   m_ship, delta_ship),
            ("thr=0.5",       m_05,   None),
            ("holdout best",  m_best, delta_best),
        ]:
            delta_str = f"{delta:+.3f}" if delta is not None else "      —"
            print(f"{name:<14}  {mode:<18}  {m['threshold']:>5.2f}  "
                  f"{m['precision']:>6.3f}  {m['recall']:>6.3f}  "
                  f"{m['f1']:>6.3f}  {delta_str:>12}")
        print("-" * 88)
        results[name] = {"shipped": m_ship, "at_0.5": m_05, "best": m_best}

    # Threshold stability check
    print(f"\nThreshold stability check — does the shipped-thr still win?")
    for name, _, ship in models:
        ship_f1 = results[name]["shipped"]["f1"]
        best_f1 = results[name]["best"]["f1"]
        best_thr = results[name]["best"]["threshold"]
        gap = best_f1 - ship_f1
        marker = " (boundary drifted!)" if gap > 0.02 else ""
        print(f"  {name:<14} shipped thr {ship:.2f} -> F1 {ship_f1:.3f}"
              f"  |  best thr {best_thr:.2f} -> F1 {best_f1:.3f}"
              f"  (gap {gap:+.3f}){marker}")

    # Save
    out = {
        "holdout_size": len(holdout),
        "gpt5_pos_rate": float(y_holdout.mean()),
        "meta_weights": stacked_meta["meta_weights"],
        "models": {
            name: {
                "shipped_thr": results[name]["shipped"],
                "at_0.5":      results[name]["at_0.5"],
                "holdout_best": results[name]["best"],
            } for name in ["Zero-shot", "TF-IDF + LR", "Ensemble"]
        },
    }
    out_path = Path(args.holdout).parent / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
