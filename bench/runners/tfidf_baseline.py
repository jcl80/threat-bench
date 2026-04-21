"""TF-IDF + Logistic Regression baseline on the 80/20 split.

Same split, same preprocessing as the DeBERTa runs (build_premise: title +
body + top-3 comments, truncated to 1500 chars). Purely lexical — useful as
a floor for how much signal is in surface words alone, and as an ensemble
candidate whose errors may be disjoint from a semantic model's.

Output matches the other runners so threshold sweeps and comparison scripts
work unchanged:
  bench/results/<timestamp>_tfidf_lr/
    predictions.jsonl   {snapshot_id, label, tfidf_prob, tfidf_pred, split}
    metadata.json

Usage:
    python -m bench.runners.tfidf_baseline
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "bench/data/posts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
MAX_PREMISE_CHARS = 1500


def build_premise(post: dict) -> str:
    """Exact parity with bench/runners/finetune_deberta.py and nli_deberta.py."""
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
    if len(text) > MAX_PREMISE_CHARS:
        text = text[:MAX_PREMISE_CHARS]
    return text


def load_split() -> tuple[list[dict], list[dict]]:
    """Reproduce the exact 80/20 split used by finetune_deberta.py."""
    with open(DATA_PATH) as f:
        posts = [json.loads(l) for l in f]
    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    train = pos[:split_p] + neg[:split_n]
    test = pos[split_p:] + neg[split_n:]
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def score_at_threshold(probs: list[float], labels: list[int], thr: float) -> dict:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngrams", type=str, default="1,2")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    ngram_range = tuple(int(x) for x in args.ngrams.split(","))

    train_posts, test_posts = load_split()
    print(f"Train: {len(train_posts)} ({sum(p['label_gpt5'] for p in train_posts)} pos)")
    print(f"Test:  {len(test_posts)} ({sum(p['label_gpt5'] for p in test_posts)} pos)")

    train_texts = [build_premise(p) for p in train_posts]
    test_texts = [build_premise(p) for p in test_posts]
    y_train = [p["label_gpt5"] for p in train_posts]
    y_test = [p["label_gpt5"] for p in test_posts]

    t0 = time.monotonic()
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    print(f"TF-IDF features: {X_train.shape[1]} (fit in {time.monotonic()-t0:.1f}s)")

    t0 = time.monotonic()
    clf = LogisticRegression(
        C=args.C, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    clf.fit(X_train, y_train)
    print(f"LR fit in {time.monotonic()-t0:.1f}s")

    train_probs = clf.predict_proba(X_train)[:, 1].tolist()
    test_probs = clf.predict_proba(X_test)[:, 1].tolist()

    # Metrics at 0.5
    m_test_05 = score_at_threshold(test_probs, y_test, 0.5)
    m_train_05 = score_at_threshold(train_probs, y_train, 0.5)
    print(f"\nTest  @ thr=0.5:  P={m_test_05['precision']:.3f}  R={m_test_05['recall']:.3f}  "
          f"F1={m_test_05['f1']:.3f}")
    print(f"Train @ thr=0.5:  P={m_train_05['precision']:.3f}  R={m_train_05['recall']:.3f}  "
          f"F1={m_train_05['f1']:.3f}  (gap: "
          f"{m_train_05['f1']-m_test_05['f1']:+.3f})")

    # Threshold sweep on test (best-F1)
    sweep = []
    t = 0.05
    while t <= 0.95 + 1e-9:
        sweep.append(score_at_threshold(test_probs, y_test, round(t, 2)))
        t += 0.01
    best = max(sweep, key=lambda m: m["f1"])
    print(f"\nTest best-F1: F1={best['f1']:.3f} at thr={best['threshold']:.2f} "
          f"(P={best['precision']:.3f}, R={best['recall']:.3f})")

    # Output dir
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_DIR / f"{timestamp}_tfidf_lr"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Predictions (test + train for potential ensemble use)
    with open(run_dir / "predictions.jsonl", "w") as f:
        for p, y, pr in zip(test_posts, y_test, test_probs):
            row = {
                "snapshot_id": p["snapshot_id"],
                "label": int(y),
                "tfidf_prob": round(float(pr), 6),
                "tfidf_pred": 1 if pr >= 0.5 else 0,
                "split": "test",
            }
            f.write(json.dumps(row) + "\n")
        for p, y, pr in zip(train_posts, y_train, train_probs):
            row = {
                "snapshot_id": p["snapshot_id"],
                "label": int(y),
                "tfidf_prob": round(float(pr), 6),
                "tfidf_pred": 1 if pr >= 0.5 else 0,
                "split": "train",
            }
            f.write(json.dumps(row) + "\n")

    metadata = {
        "timestamp": timestamp,
        "model": "tfidf_lr",
        "ngram_range": list(ngram_range),
        "min_df": args.min_df,
        "max_features": args.max_features,
        "C": args.C,
        "seed": SEED,
        "n_features": X_train.shape[1],
        "train_size": len(train_posts),
        "test_size": len(test_posts),
        "test_metrics_at_0.5": m_test_05,
        "train_metrics_at_0.5": m_train_05,
        "test_best_f1": best,
        "test_sweep": sweep,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved to {run_dir}/")


if __name__ == "__main__":
    main()
