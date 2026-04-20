"""Flatten batch-level data into per-post records with binary labels.

Reads bench_data.jsonl (all posts) and model output.jsonl files to produce
a single JSONL where each line is one post with ground-truth labels from
gpt-5-mini and gpt-5.

Usage:
    python -m bench.prepare_data
    python -m bench.prepare_data --out bench/data/posts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TIERS = {
    "threat_dense": ["ukraine", "worldnews", "collapse", "geopolitics"],
    "ambiguous": ["Economics", "technology", "news", "energy"],
    "benign": ["Cooking", "askscience", "woodworking", "gardening"],
}
SUB_TO_TIER = {s: t for t, subs in TIERS.items() for s in subs}

GPT5_MINI_OUTPUT = ROOT / "results/2026-03-26T21-31-32_gpt-5-mini_threat_stage1/output.jsonl"
GPT5_OUTPUT = ROOT / "results/2026-03-26T22-04-39_gpt-5_threat_stage1/output.jsonl"


def _load_flagged_ids(output_path: Path) -> set[int]:
    """Return set of snapshot_ids flagged in a model output file."""
    flagged = set()
    with open(output_path) as f:
        for line in f:
            analysis = json.loads(line)
            ids = analysis["post_snapshot_ids"]
            for post in analysis["flagged_posts"]:
                if post.get("flagged", True):
                    sid = ids[post["post_index"] - 1]
                    flagged.add(sid)
    return flagged


def prepare(out_path: Path) -> None:
    flagged_mini = _load_flagged_ids(GPT5_MINI_OUTPUT)
    flagged_gpt5 = _load_flagged_ids(GPT5_OUTPUT)

    bench_path = ROOT / "data" / "bench_data.jsonl"
    posts = []
    with open(bench_path) as f:
        for line in f:
            row = json.loads(line)
            sid = row["post"]["snapshot_id"]
            title = row["post"].get("title", "")
            body = row["post"].get("body", "")

            # Collect top-level comment texts.
            # NOTE: bench_data.jsonl has swapped fields — the comment text is
            # stored under "author" and the username is under "body".
            comments = []
            for c in row.get("comments", []):
                text = c.get("author", "").strip()
                if text:
                    comments.append(text)

            posts.append({
                "snapshot_id": sid,
                "subreddit": row["subreddit"],
                "tier": SUB_TO_TIER.get(row["subreddit"], "unknown"),
                "title": title,
                "body": body,
                "comments": comments,
                "label_gpt5_mini": 1 if sid in flagged_mini else 0,
                "label_gpt5": 1 if sid in flagged_gpt5 else 0,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in posts:
            f.write(json.dumps(p) + "\n")

    # Summary
    n = len(posts)
    n_mini = sum(p["label_gpt5_mini"] for p in posts)
    n_gpt5 = sum(p["label_gpt5"] for p in posts)
    print(f"Wrote {n} posts to {out_path}")
    print(f"  label_gpt5_mini: {n_mini} positive ({n_mini/n:.1%})")
    print(f"  label_gpt5:      {n_gpt5} positive ({n_gpt5/n:.1%})")

    for tier in ["threat_dense", "ambiguous", "benign"]:
        tp = [p for p in posts if p["tier"] == tier]
        m = sum(p["label_gpt5_mini"] for p in tp)
        g = sum(p["label_gpt5"] for p in tp)
        print(f"  {tier:15s}: {len(tp):4d} posts | mini={m:4d} ({m/len(tp):.1%}) | gpt5={g:4d} ({g/len(tp):.1%})")


def main():
    parser = argparse.ArgumentParser(description="Prepare per-post benchmark data")
    parser.add_argument("--out", default="bench/data/posts.jsonl", help="Output path")
    args = parser.parse_args()
    prepare(Path(args.out))


if __name__ == "__main__":
    main()
