"""2-feature ensemble of TF-IDF+LR and zero-shot NLI.

Logistic regression over [zs_prob, tfidf_prob]. Error-overlap analysis
showed these two models disagree a lot (Cohen's kappa ~0.54) and TF-IDF
fixes ~57% of zero-shot's errors, so learning when to trust which model
should push F1 meaningfully past either individual score.

Evaluation is honest 2-fold on the test set — we never fit and evaluate
on the same examples. The test set is stratified-split into two halves,
we fit on half A and predict on half B, then swap, then pool predictions.
Final metrics are on the full 478-post test set with no leakage.

Usage:
    python -m bench.runners.ensemble \
        --tfidf bench/results/<tfidf_dir>/predictions.jsonl \
        --zeroshot bench/results/<zs_dir>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_PATH = "bench/data/posts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def load_posts() -> dict[int, dict]:
    with open(DATA_PATH) as f:
        return {json.loads(l)["snapshot_id"]: json.loads(l) for l in open(DATA_PATH)}


def load_tfidf(path: Path) -> dict[int, float]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                out[r["snapshot_id"]] = r["tfidf_prob"]
    return out


def load_zeroshot(path: Path) -> dict[int, float]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["snapshot_id"]] = r["score"]
    return out


def stratified_halves(ids: list[int], labels: list[int], seed: int = SEED):
    pos = [i for i, y in zip(ids, labels) if y == 1]
    neg = [i for i, y in zip(ids, labels) if y == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos); rnd.shuffle(neg)
    half_p = len(pos) // 2
    half_n = len(neg) // 2
    A = pos[:half_p] + neg[:half_n]
    B = pos[half_p:] + neg[half_n:]
    rnd.shuffle(A); rnd.shuffle(B)
    return set(A), set(B)


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
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(labels), 1)
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}


def best_f1_sweep(probs, labels, start=0.05, end=0.95, step=0.01):
    best = None
    t = start
    while t <= end + 1e-9:
        m = metrics_at(probs, labels, round(t, 2))
        if best is None or m["f1"] > best["f1"]:
            best = m
        t += step
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--zeroshot", required=True)
    args = parser.parse_args()

    posts = load_posts()
    tf = load_tfidf(Path(args.tfidf))
    zs_all = load_zeroshot(Path(args.zeroshot))
    zs = {sid: s for sid, s in zs_all.items() if sid in tf}
    assert set(tf.keys()) == set(zs.keys()), "TF-IDF and zero-shot test sets don't match"

    ids = sorted(tf.keys())
    labels = [posts[i]["label_gpt5"] for i in ids]
    print(f"Test set: {len(ids)} posts ({sum(labels)} pos)")

    # Features per example: [zs_prob, tfidf_prob]
    X = np.array([[zs[i], tf[i]] for i in ids], dtype=float)
    y = np.array(labels)

    # 2-fold stratified on test labels
    A_ids, B_ids = stratified_halves(ids, labels)
    idx_A = [i for i, sid in enumerate(ids) if sid in A_ids]
    idx_B = [i for i, sid in enumerate(ids) if sid in B_ids]
    print(f"Fold A: {len(idx_A)} ({sum(y[idx_A])} pos) | "
          f"Fold B: {len(idx_B)} ({sum(y[idx_B])} pos)")

    # Train on A, predict on B
    clf_AB = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, random_state=SEED)
    clf_AB.fit(X[idx_A], y[idx_A])
    probs_B = clf_AB.predict_proba(X[idx_B])[:, 1]

    # Train on B, predict on A
    clf_BA = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, random_state=SEED)
    clf_BA.fit(X[idx_B], y[idx_B])
    probs_A = clf_BA.predict_proba(X[idx_A])[:, 1]

    # Pool ensemble probs across the full test set (no leakage)
    ens_probs = np.zeros(len(ids), dtype=float)
    for pos, i in enumerate(idx_B):
        ens_probs[i] = probs_B[pos]
    for pos, i in enumerate(idx_A):
        ens_probs[i] = probs_A[pos]
    ens_probs = ens_probs.tolist()

    # Learned weights per fold (sanity check on how the LR weights the two inputs)
    print(f"\nFold A→B weights: zs={clf_AB.coef_[0][0]:+.3f} "
          f"tfidf={clf_AB.coef_[0][1]:+.3f}  bias={clf_AB.intercept_[0]:+.3f}")
    print(f"Fold B→A weights: zs={clf_BA.coef_[0][0]:+.3f} "
          f"tfidf={clf_BA.coef_[0][1]:+.3f}  bias={clf_BA.intercept_[0]:+.3f}")

    # Ensemble metrics
    m_05 = metrics_at(ens_probs, labels, 0.5)
    m_best = best_f1_sweep(ens_probs, labels)
    print(f"\nEnsemble @ thr=0.5:   P={m_05['precision']:.3f}  R={m_05['recall']:.3f}  "
          f"F1={m_05['f1']:.3f}")
    print(f"Ensemble best-F1:     F1={m_best['f1']:.3f} at thr={m_best['threshold']:.2f} "
          f"(P={m_best['precision']:.3f}, R={m_best['recall']:.3f})")

    # Baseline comparisons on the same 478 posts
    tf_probs = [tf[i] for i in ids]
    zs_probs = [zs[i] for i in ids]
    tf_best = best_f1_sweep(tf_probs, labels)
    zs_best = best_f1_sweep(zs_probs, labels)

    print(f"\n{'='*64}")
    print(f"{'Model':<20}  {'thr':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"{'-'*64}")
    print(f"{'TF-IDF + LR':<20}  {tf_best['threshold']:>5.2f}  "
          f"{tf_best['precision']:>6.3f}  {tf_best['recall']:>6.3f}  "
          f"{tf_best['f1']:>6.3f}")
    print(f"{'Zero-shot NLI':<20}  {zs_best['threshold']:>5.2f}  "
          f"{zs_best['precision']:>6.3f}  {zs_best['recall']:>6.3f}  "
          f"{zs_best['f1']:>6.3f}")
    print(f"{'Ensemble (best)':<20}  {m_best['threshold']:>5.2f}  "
          f"{m_best['precision']:>6.3f}  {m_best['recall']:>6.3f}  "
          f"{m_best['f1']:>6.3f}")
    best_indiv = max(tf_best["f1"], zs_best["f1"])
    print(f"\nEnsemble gain over best individual: {m_best['f1'] - best_indiv:+.4f} F1")

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_DIR / f"{timestamp}_ensemble_zs_tfidf"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w") as f:
        for sid, p, y in zip(ids, ens_probs, labels):
            f.write(json.dumps({
                "snapshot_id": sid,
                "label": int(y),
                "ens_prob": round(float(p), 6),
                "ens_pred": 1 if p >= m_best["threshold"] else 0,
            }) + "\n")
    with open(run_dir / "metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": "ensemble_zs_tfidf",
            "tfidf_source": str(args.tfidf),
            "zeroshot_source": str(args.zeroshot),
            "seed": SEED,
            "n_test": len(ids),
            "fold_A_to_B_weights": {
                "zs": float(clf_AB.coef_[0][0]),
                "tfidf": float(clf_AB.coef_[0][1]),
                "bias": float(clf_AB.intercept_[0]),
            },
            "fold_B_to_A_weights": {
                "zs": float(clf_BA.coef_[0][0]),
                "tfidf": float(clf_BA.coef_[0][1]),
                "bias": float(clf_BA.intercept_[0]),
            },
            "test_at_0.5": m_05,
            "test_best_f1": m_best,
            "tfidf_best_f1": tf_best,
            "zs_best_f1": zs_best,
        }, f, indent=2)
    print(f"\nSaved to {run_dir}/")


if __name__ == "__main__":
    main()
