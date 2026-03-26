"""Pull posts + comments from the Sentinel production DB for benchmarking.

Usage:
    python3 fetch_data.py                          # pull all tiers, append to bench_data.jsonl
    python3 fetch_data.py --tier threat-dense       # only threat-dense subs
    python3 fetch_data.py --dry-run                 # show counts, don't write
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg2

# ── Target subreddits ────────────────────────────────────────────────────

TIERS = {
    "threat-dense": {
        "subreddits": ["geopolitics"],
        "posts_per_sub": 200,
        "purpose": "Detection rate (false negatives)",
    },
    "ambiguous": {
        "subreddits": ["Economics", "technology", "news", "energy"],
        "posts_per_sub": 200,
        "purpose": "Judgment calls (precision)",
    },
    "benign": {
        "subreddits": ["Cooking", "askscience", "woodworking", "gardening"],
        "posts_per_sub": 100,
        "purpose": "False positive rate (specificity)",
    },
}

COMMENT_LIMIT = 5  # Match existing bench_data format
DATABASE_URL_ENV = "SENTINEL_DATABASE_URL"
DATABASE_URL_FALLBACK = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "reddit", ".env"
)


# ── DB helpers ───────────────────────────────────────────────────────────

def get_connection_string() -> str:
    url = os.environ.get(DATABASE_URL_ENV)
    if url:
        return url
    # Try to read from reddit repo .env
    env_path = Path(DATABASE_URL_FALLBACK).resolve()
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATABASE_URL=") and not line.startswith("#"):
                    return line.split("=", 1)[1]
    raise RuntimeError(
        f"No database URL found. Set {DATABASE_URL_ENV} or ensure reddit/.env exists."
    )


def get_existing_snapshot_ids(data_path: str) -> set[int]:
    """Read existing bench_data.jsonl and return all snapshot_ids already present."""
    ids = set()
    p = Path(data_path)
    if not p.exists():
        return ids
    with open(p) as f:
        for line in f:
            row = json.loads(line)
            ids.add(row["post"]["snapshot_id"])
    return ids


def fetch_subreddit_posts(
    conn,
    subreddit: str,
    limit: int,
    comment_limit: int,
    exclude_ids: set[int],
) -> list[dict]:
    """Fetch posts + comments for a subreddit from the DB."""
    cur = conn.cursor()

    # Get subreddit metadata
    cur.execute(
        """SELECT id, name, subscribers, public_description
           FROM subreddits WHERE LOWER(name) = LOWER(%s) LIMIT 1""",
        (subreddit,),
    )
    sub_row = cur.fetchone()
    if not sub_row:
        print(f"  WARNING: r/{subreddit} not found in DB, skipping")
        return []

    sub_id, sub_name, sub_subscribers, sub_desc = sub_row

    # Fetch recent posts with comments, excluding already-exported ones
    # Pick posts that have at least a few comments for richer data
    cur.execute(
        """SELECT id, reddit_id, title, body, author, score, num_comments
           FROM post_snapshots
           WHERE subreddit_id = %s
             AND num_comments >= 3
             AND score >= 1
           ORDER BY snapshot_created_at DESC
           LIMIT %s""",
        (sub_id, limit * 3),  # fetch extra so we can skip excluded
    )
    post_rows = cur.fetchall()

    results = []
    for post_row in post_rows:
        if len(results) >= limit:
            break

        ps_id, reddit_id, title, body, author, score, num_comments = post_row

        if ps_id in exclude_ids:
            continue

        # Fetch top comments
        cur.execute(
            """SELECT body, author, score, depth
               FROM comment_snapshots
               WHERE post_snapshot_id = %s
                 AND NOT is_deleted AND NOT is_removed
               ORDER BY score DESC
               LIMIT %s""",
            (ps_id, comment_limit),
        )
        comments = [
            {
                "author": c[1] or "[deleted]",
                "body": c[0] or "",
                "score": c[2] or 0,
                "depth": c[3] or 0,
            }
            for c in cur.fetchall()
        ]

        results.append({
            "subreddit": sub_name,
            "subreddit_subscribers": sub_subscribers or 0,
            "subreddit_description": sub_desc or "",
            "post": {
                "snapshot_id": ps_id,
                "reddit_id": reddit_id,
                "title": title or "",
                "body": body or "",
                "author": author or "[deleted]",
                "score": score or 0,
                "num_comments": num_comments or 0,
            },
            "comments": comments,
        })

    cur.close()
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch bench data from Sentinel DB")
    parser.add_argument("--tier", choices=list(TIERS.keys()), default=None,
                        help="Only fetch one tier (default: all)")
    parser.add_argument("--output", default="data/bench_data.jsonl",
                        help="Output file (appends by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without writing")
    parser.add_argument("--fresh", action="store_true",
                        help="Write a new file instead of appending")
    args = parser.parse_args()

    conn_str = get_connection_string()
    conn = psycopg2.connect(conn_str)
    print(f"Connected to database")

    # Get existing IDs to avoid duplicates
    existing_ids = set() if args.fresh else get_existing_snapshot_ids(args.output)
    print(f"Existing posts in {args.output}: {len(existing_ids)}")

    tiers_to_fetch = {args.tier: TIERS[args.tier]} if args.tier else TIERS

    all_new = []
    for tier_name, tier_cfg in tiers_to_fetch.items():
        print(f"\n{'='*60}")
        print(f"Tier: {tier_name} — {tier_cfg['purpose']}")
        print(f"{'='*60}")

        for sub in tier_cfg["subreddits"]:
            target = tier_cfg["posts_per_sub"]
            print(f"\n  r/{sub} — target: {target} posts")

            if args.dry_run:
                cur = conn.cursor()
                cur.execute(
                    """SELECT COUNT(*) FROM post_snapshots ps
                       JOIN subreddits s ON ps.subreddit_id = s.id
                       WHERE LOWER(s.name) = LOWER(%s) AND ps.num_comments >= 3 AND ps.score >= 1""",
                    (sub,),
                )
                available = cur.fetchone()[0]
                cur.close()
                print(f"    Available: {available} posts (would fetch {min(target, available)})")
                continue

            posts = fetch_subreddit_posts(
                conn, sub, target, COMMENT_LIMIT,
                exclude_ids=existing_ids,
            )
            # Track new IDs to avoid within-run duplicates
            for p in posts:
                existing_ids.add(p["post"]["snapshot_id"])
            all_new.extend(posts)
            print(f"    Fetched: {len(posts)} posts")

    conn.close()

    if args.dry_run:
        print(f"\nDry run complete. No data written.")
        return

    if not all_new:
        print("\nNo new posts to write.")
        return

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.fresh else "a"
    with open(output_path, mode) as f:
        for row in all_new:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_new)} new posts to {args.output}")
    print(f"Total posts in file: {len(existing_ids)}")

    # Print summary
    from collections import Counter
    dist = Counter(r["subreddit"] for r in all_new)
    print("\nNew posts by subreddit:")
    for sub, count in dist.most_common():
        print(f"  r/{sub}: {count}")


if __name__ == "__main__":
    main()
