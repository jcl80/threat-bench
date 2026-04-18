"""Zero-shot NLI classification using DeBERTa.

Uses a zero-shot NLI model to classify each post as threat/non-threat.
The model scores the post against a hypothesis like "This text describes
a real-world threat, crisis, or dangerous event."

Install:
    pip install torch transformers

Usage:
    python -m bench.runners.nli_deberta
    python -m bench.runners.nli_deberta --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    python -m bench.runners.nli_deberta --threshold 0.5 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_HYPOTHESIS = (
    "This text describes a real-world threat, crisis, or dangerous event "
    "such as armed conflict, disease outbreak, economic collapse, "
    "political instability, natural disaster, or AI risk."
)

# DeBERTa max input is 512 tokens; leave room for hypothesis + special tokens
MAX_PREMISE_CHARS = 1500


def build_premise(post: dict) -> str:
    """Combine title + body + top comments into a single premise string."""
    parts = []
    title = post.get("title", "").strip()
    if title:
        parts.append(title)
    body = post.get("body", "").strip()
    if body:
        parts.append(body)

    # Add top comments for context (up to 3)
    for comment in post.get("comments", [])[:3]:
        if isinstance(comment, str) and comment.strip():
            parts.append(comment.strip())

    text = "\n".join(parts)
    if len(text) > MAX_PREMISE_CHARS:
        text = text[:MAX_PREMISE_CHARS]
    return text


def run(
    model_name: str = DEFAULT_MODEL,
    hypothesis: str = DEFAULT_HYPOTHESIS,
    data_path: str = "bench/data/posts.jsonl",
    threshold: float = 0.5,
    batch_size: int = 8,
    results_dir: str | None = None,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Identify entailment label index
    id2label = model.config.id2label
    entail_idx = None
    for idx, label in id2label.items():
        if "entail" in label.lower():
            entail_idx = int(idx)
            break
    if entail_idx is None:
        raise ValueError(f"Cannot find entailment label in {id2label}")
    print(f"Entailment index: {entail_idx} ({id2label[entail_idx]})")

    # Load posts
    with open(data_path) as f:
        posts = [json.loads(line) for line in f]
    print(f"Loaded {len(posts)} posts")

    # Process in batches
    predictions = []
    t_start = time.monotonic()

    for i in range(0, len(posts), batch_size):
        batch = posts[i : i + batch_size]
        premises = [build_premise(p) for p in batch]

        inputs = tokenizer(
            premises,
            [hypothesis] * len(premises),
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entail_scores = probs[:, entail_idx].cpu().tolist()

        for j, post in enumerate(batch):
            score = entail_scores[j]
            predictions.append({
                "snapshot_id": post["snapshot_id"],
                "predicted": 1 if score >= threshold else 0,
                "score": round(score, 4),
            })

        done = min(i + batch_size, len(posts))
        if done % 100 < batch_size or done == len(posts):
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(posts)}] {rate:.1f} posts/sec")

    t_total = time.monotonic() - t_start

    # Output directory
    if results_dir is None:
        results_dir = str(Path(__file__).resolve().parent.parent / "results")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = model_name.replace("/", "_")
    run_dir = Path(results_dir) / f"{timestamp}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_path = run_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Save metadata
    n_flagged = sum(p["predicted"] for p in predictions)
    metadata = {
        "timestamp": timestamp,
        "model": model_name,
        "hypothesis": hypothesis,
        "threshold": threshold,
        "batch_size": batch_size,
        "device": device,
        "total_posts": len(posts),
        "flagged": n_flagged,
        "flag_rate": round(n_flagged / len(posts), 4),
        "wall_clock_seconds": round(t_total, 1),
        "posts_per_second": round(len(posts) / t_total, 1) if t_total > 0 else 0,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone in {t_total:.0f}s ({metadata['posts_per_second']:.1f} posts/sec)")
    print(f"Flagged: {n_flagged}/{len(posts)} ({metadata['flag_rate']:.1%})")
    print(f"Saved to {run_dir}/")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Zero-shot NLI classification")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--hypothesis", default=DEFAULT_HYPOTHESIS, help="NLI hypothesis")
    parser.add_argument("--data", default="bench/data/posts.jsonl", help="Input data path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entailment score threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--results-dir", default=None, help="Results directory")
    args = parser.parse_args()

    run_dir = run(
        model_name=args.model,
        hypothesis=args.hypothesis,
        data_path=args.data,
        threshold=args.threshold,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
    )

    # Auto-score against both ground truths
    from bench.scorer import score, print_report

    pred_path = run_dir / "predictions.jsonl"
    for gt in ["gpt5_mini", "gpt5"]:
        results = score(pred_path, Path(args.data), gt)
        print_report(results)
        with open(run_dir / f"eval_{gt}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
