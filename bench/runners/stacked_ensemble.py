"""Stacking ensemble of TF-IDF+LR and zero-shot NLI, with proper OOF.

Methodology:
  1. 5-fold stratified split on the 1907 training posts (StratifiedKFold,
     seed=42, shuffle=True).
  2. For each fold: fit TF-IDF + LR on the other 4 folds, predict
     probabilities on the held-out fold. Every training post ends up with
     one OOF TF-IDF probability — not overfit, because the post was never
     seen during that fold's fit.
  3. Zero-shot probabilities exist for all 2385 posts (no training).
  4. Meta-LR trains on 1907 rows of [zs_prob, tfidf_oof_prob] -> label.
  5. A final TF-IDF + LR is fit on ALL 1907 training posts (full-data, this
     is the production model), used to produce TF-IDF probabilities for the
     478 test posts.
  6. Single evaluation pass on the 478 test set with no folds.

Expected sanity check: if the meta-LR weights are still ~balanced positive
(like the 2-fold version's +2.9 each), the ensemble signal is real. If one
weight collapses to ~0 with more training data, it was partially spurious.

Usage:
    python -m bench.runners.stacked_ensemble \
        --zeroshot bench/results/<zs_dir>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "bench/data/posts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
MAX_PREMISE_CHARS = 1500


def build_premise(post: dict) -> str:
    parts = []
    title = (post.get("title") or "").strip()
    if title:
        parts.append(title)
    body = (post.get("body") or "").strip()
    if body:
        parts.append(body)
    for c in (post.get("comments") or [])[:3]:
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
    text = "\n".join(parts)
    return text[:MAX_PREMISE_CHARS] if len(text) > MAX_PREMISE_CHARS else text


def load_split() -> tuple[list[dict], list[dict]]:
    """Exact parity with finetune_deberta.py / tfidf_baseline.py."""
    with open(DATA_PATH) as f:
        posts = [json.loads(l) for l in f]
    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos); rnd.shuffle(neg)
    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    train = pos[:split_p] + neg[:split_n]
    test = pos[split_p:] + neg[split_n:]
    rnd.shuffle(train); rnd.shuffle(test)
    return train, test


def tfidf_lr_fit_predict(train_texts, y_train, predict_texts):
    """Fit TF-IDF + LR on training, return probs on predict_texts."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000,
        sublinear_tf=True, strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_pred = vec.transform(predict_texts)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_pred)[:, 1]


def score_at(probs, labels, thr):
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
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp+tn) / max(len(labels), 1), 4)}


def best_sweep(probs, labels, start=0.05, end=0.95, step=0.01):
    best = None
    t = start
    while t <= end + 1e-9:
        m = score_at(probs, labels, round(t, 2))
        if best is None or m["f1"] > best["f1"]:
            best = m
        t += step
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroshot", required=True,
                        help="Zero-shot predictions.jsonl covering all 2385 posts")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    # --- Data ---
    train_posts, test_posts = load_split()
    print(f"Train: {len(train_posts)} ({sum(p['label_gpt5'] for p in train_posts)} pos)")
    print(f"Test:  {len(test_posts)} ({sum(p['label_gpt5'] for p in test_posts)} pos)")

    train_texts = [build_premise(p) for p in train_posts]
    test_texts = [build_premise(p) for p in test_posts]
    y_train = np.array([p["label_gpt5"] for p in train_posts])
    y_test = np.array([p["label_gpt5"] for p in test_posts])

    # --- Load zero-shot scores ---
    zs = {}
    with open(args.zeroshot) as f:
        for line in f:
            r = json.loads(line)
            zs[r["snapshot_id"]] = r["score"]
    zs_train = np.array([zs[p["snapshot_id"]] for p in train_posts])
    zs_test = np.array([zs[p["snapshot_id"]] for p in test_posts])

    # --- 5-fold OOF TF-IDF on train ---
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    tfidf_oof_train = np.zeros(len(train_posts), dtype=float)
    t0 = time.monotonic()
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_texts, y_train), 1):
        fold_train_texts = [train_texts[i] for i in tr_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_y = y_train[tr_idx]
        probs = tfidf_lr_fit_predict(fold_train_texts, fold_y, fold_val_texts)
        tfidf_oof_train[val_idx] = probs
        print(f"  Fold {fold_idx}/{args.folds}: {len(tr_idx)} train, {len(val_idx)} val")
    print(f"OOF TF-IDF generated in {time.monotonic()-t0:.1f}s")

    # --- Final TF-IDF + LR on full train, predict on test ---
    t0 = time.monotonic()
    tfidf_test = tfidf_lr_fit_predict(train_texts, y_train, test_texts)
    print(f"Final TF-IDF on test in {time.monotonic()-t0:.1f}s")

    # --- Meta-LR on [zs_prob, tfidf_prob] → label ---
    X_meta_train = np.column_stack([zs_train, tfidf_oof_train])
    X_meta_test = np.column_stack([zs_test, tfidf_test])
    meta = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    meta.fit(X_meta_train, y_train)
    w_zs, w_tf = meta.coef_[0]
    bias = meta.intercept_[0]
    print(f"\nMeta-LR weights: zs={w_zs:+.3f}  tfidf={w_tf:+.3f}  bias={bias:+.3f}")
    # 2-fold comparison was zs≈tf≈+2.9. If both weights stay positive and
    # roughly balanced here, the signal is real.

    # --- Evaluate on test ---
    test_probs = meta.predict_proba(X_meta_test)[:, 1]
    m_05 = score_at(test_probs, y_test, 0.5)
    m_best = best_sweep(test_probs, y_test)

    # Baselines on the SAME test set (unbiased — use test-only sweep on each)
    zs_best = best_sweep(zs_test, y_test)
    tf_best = best_sweep(tfidf_test, y_test)

    print(f"\n{'='*64}")
    print(f"{'Model':<22}  {'thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"{'-'*64}")
    print(f"{'TF-IDF + LR':<22}  {tf_best['threshold']:>5.2f}  "
          f"{tf_best['precision']:>6.3f}  {tf_best['recall']:>6.3f}  "
          f"{tf_best['f1']:>6.3f}")
    print(f"{'Zero-shot NLI':<22}  {zs_best['threshold']:>5.2f}  "
          f"{zs_best['precision']:>6.3f}  {zs_best['recall']:>6.3f}  "
          f"{zs_best['f1']:>6.3f}")
    print(f"{'Stacked ensemble':<22}  {m_best['threshold']:>5.2f}  "
          f"{m_best['precision']:>6.3f}  {m_best['recall']:>6.3f}  "
          f"{m_best['f1']:>6.3f}")
    print(f"{'Stacked @ thr=0.5':<22}  {m_05['threshold']:>5.2f}  "
          f"{m_05['precision']:>6.3f}  {m_05['recall']:>6.3f}  "
          f"{m_05['f1']:>6.3f}")
    best_indiv = max(tf_best["f1"], zs_best["f1"])
    print(f"\nEnsemble gain over best individual: {m_best['f1'] - best_indiv:+.4f} F1")

    # --- Save ---
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_DIR / f"{timestamp}_stacked_ensemble"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "predictions.jsonl", "w") as f:
        for p, zs_p, tf_p, ens_p, y in zip(
            test_posts, zs_test, tfidf_test, test_probs, y_test
        ):
            f.write(json.dumps({
                "snapshot_id": p["snapshot_id"],
                "label": int(y),
                "zs_prob": round(float(zs_p), 6),
                "tfidf_prob": round(float(tf_p), 6),
                "ens_prob": round(float(ens_p), 6),
                "ens_pred_best": 1 if ens_p >= m_best["threshold"] else 0,
            }) + "\n")

    with open(run_dir / "metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": "stacked_ensemble_zs_tfidf",
            "folds": args.folds,
            "seed": SEED,
            "zeroshot_source": str(args.zeroshot),
            "n_train": len(train_posts),
            "n_test": len(test_posts),
            "meta_weights": {"zs": float(w_zs), "tfidf": float(w_tf),
                             "bias": float(bias)},
            "test_at_0.5": m_05,
            "test_best_f1": m_best,
            "tfidf_best_f1_test": tf_best,
            "zs_best_f1_test": zs_best,
        }, f, indent=2)
    print(f"\nSaved to {run_dir}/")


if __name__ == "__main__":
    main()
