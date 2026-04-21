"""Fetch a fresh holdout slice from the Sentinel production DB.

Fetches the most recent analyses per subreddit until each has ~N posts,
giving a predictable balanced holdout. With the default 8 subreddits and
--per-subreddit 250, you get ~2000 posts split 50/50 threat-dense vs
ambiguous.

Output:
  <out-dir>/baseline_holdout.jsonl      # BaselineAnalysis shape (one per batch)
  <out-dir>/bench_data_holdout.jsonl    # BenchRow shape (one per post)

The script preserves the known swapped-fields comment bug in comment_snapshots
(text stored under "author", username under "body") so prepare_data.py's
patched reader works unchanged on the output.

Usage:
    # Default: 250 posts per subreddit -> ~2000 total
    python3 bench/fetch_holdout.py --since 2026-04-05

    # Custom size:
    python3 bench/fetch_holdout.py --since 2026-04-05 --per-subreddit 300

    # Subset of subreddits:
    python3 bench/fetch_holdout.py --since 2026-04-05 \\
        --subreddits ukraine,worldnews,Economics,technology
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import psycopg2

DEFAULT_SUBREDDITS = [
    # threat-dense
    "ukraine", "worldnews", "collapse", "geopolitics",
    # ambiguous
    "Economics", "technology", "news", "energy",
]

COMMENT_LIMIT = 5
DATABASE_URL_ENV = "SENTINEL_DATABASE_URL"
DATABASE_URL_FALLBACK = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "reddit", ".env"
)


def get_connection_string() -> str:
    url = os.environ.get(DATABASE_URL_ENV)
    if url:
        return url
    env_path = Path(DATABASE_URL_FALLBACK).resolve()
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATABASE_URL=") and not line.startswith("#"):
                    return line.split("=", 1)[1]
    raise RuntimeError(
        f"No DB URL. Set {DATABASE_URL_ENV} or ensure reddit/.env exists "
        f"(tried {env_path})."
    )


def jsonb(val):
    """psycopg2 returns JSONB as str (old driver) or already-parsed (new)."""
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return val
    return json.loads(val)


def fetch_analyses_per_subreddit(
    conn, since: datetime, subreddit: str, target_posts: int
):
    """Fetch analyses for ONE subreddit, newest-first, until we accumulate
    at least target_posts unique post_snapshot_ids or run out of analyses.
    Returns list of DB rows. Stops as soon as the cumulative post count
    reaches target_posts, so the final analysis may push us slightly over.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.id, s.name, s.subscribers, s.public_description,
               a.post_snapshot_ids,
               a.final_status, a.severity_score, a.threat_categories,
               a.stage1_model, a.stage1_confidence, a.stage1_reasoning,
               a.stage1_cost_usd, a.stage1_evidence,
               a.stage2_model, a.stage2_verified, a.stage2_confidence,
               a.stage2_reasoning, a.stage2_cost_usd, a.stage2_evidence,
               a.analyzed_at
        FROM analyses a
        JOIN subreddits s ON a.subreddit_id = s.id
        WHERE a.analyzed_at >= %s
          AND LOWER(s.name) = LOWER(%s)
          AND a.post_snapshot_ids IS NOT NULL
          AND jsonb_array_length(a.post_snapshot_ids) > 0
        ORDER BY a.analyzed_at DESC
        """,
        (since, subreddit),
    )

    out = []
    seen_sids: set[int] = set()
    for row in cur:
        out.append(row)
        for sid in (jsonb(row[4]) or []):
            seen_sids.add(int(sid))
        if len(seen_sids) >= target_posts:
            break
    cur.close()
    return out, len(seen_sids)


def fetch_posts_and_comments(conn, snapshot_ids: list[int]) -> dict[int, dict]:
    """Return {snapshot_id: BenchRow-shaped-except-subreddit-metadata dict}."""
    if not snapshot_ids:
        return {}
    cur = conn.cursor()

    cur.execute(
        """SELECT id, reddit_id, title, body, author, score, num_comments
           FROM post_snapshots
           WHERE id = ANY(%s)""",
        (snapshot_ids,),
    )
    posts = {}
    for row in cur.fetchall():
        ps_id, reddit_id, title, body, author, score, num_comments = row
        posts[ps_id] = {
            "post": {
                "snapshot_id": ps_id,
                "reddit_id": reddit_id or "",
                "title": title or "",
                "body": body or "",
                "author": author or "[deleted]",
                "score": score or 0,
                "num_comments": num_comments or 0,
            },
            "comments": [],
        }

    # Fetch top COMMENT_LIMIT comments per post in a single query
    cur.execute(
        """SELECT post_snapshot_id, body, author, score, depth
           FROM (
               SELECT post_snapshot_id, body, author, score, depth,
                      ROW_NUMBER() OVER (PARTITION BY post_snapshot_id
                                         ORDER BY score DESC) AS rn
               FROM comment_snapshots
               WHERE post_snapshot_id = ANY(%s)
                 AND NOT is_deleted AND NOT is_removed
           ) t
           WHERE rn <= %s""",
        (snapshot_ids, COMMENT_LIMIT),
    )
    for row in cur.fetchall():
        post_id, c_body, c_author, c_score, c_depth = row
        if post_id not in posts:
            continue
        posts[post_id]["comments"].append({
            "author": c_author or "[deleted]",
            "body": c_body or "",
            "score": c_score or 0,
            "depth": c_depth or 0,
        })

    cur.close()
    return posts


def main():
    parser = argparse.ArgumentParser(description="Fetch fresh production holdout")
    parser.add_argument("--since", required=True,
                        help="Min analyzed_at date, YYYY-MM-DD")
    parser.add_argument("--subreddits", default=",".join(DEFAULT_SUBREDDITS),
                        help="Comma-separated list (default: threat-dense + ambiguous)")
    parser.add_argument("--per-subreddit", type=int, default=250,
                        help="Target posts per subreddit (default 250 -> ~2000 total)")
    parser.add_argument("--out-dir", default="bench/data/holdout",
                        help="Output directory")
    args = parser.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d")
    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(get_connection_string())
    print(f"Fetching ~{args.per_subreddit} posts/subreddit since {args.since} "
          f"across {len(subs)} subreddits")

    analyses = []
    per_sub_counts = {}
    for sub in subs:
        rows, n_posts = fetch_analyses_per_subreddit(
            conn, since, sub, args.per_subreddit
        )
        analyses.extend(rows)
        per_sub_counts[sub] = (len(rows), n_posts)
        under = " (UNDER target)" if n_posts < args.per_subreddit else ""
        print(f"  r/{sub}: {len(rows)} analyses, {n_posts} posts{under}")
    print(f"  → {len(analyses)} analyses total")

    # Gather all unique post snapshot IDs across analyses
    all_sids: set[int] = set()
    for row in analyses:
        sids = jsonb(row[4]) or []
        all_sids.update(int(x) for x in sids)
    print(f"  → {len(all_sids)} unique posts referenced")

    # Batch-fetch posts + comments
    post_map = fetch_posts_and_comments(conn, sorted(all_sids))
    print(f"  → {len(post_map)} posts resolved "
          f"({len(all_sids) - len(post_map)} missing from snapshots table)")
    conn.close()

    # --- Write baseline_holdout.jsonl ---
    baseline_path = out_dir / "baseline_holdout.jsonl"
    written_baseline = 0
    with open(baseline_path, "w") as f:
        for row in analyses:
            (aid, sub_name, sub_subs, sub_desc, post_sids,
             final_status, severity, threat_cats,
             s1_model, s1_conf, s1_reason, s1_cost, s1_ev,
             s2_model, s2_verified, s2_conf, s2_reason, s2_cost, s2_ev,
             analyzed_at) = row
            record = {
                "analysis_id": aid,
                "subreddit": sub_name,
                "post_snapshot_ids": [int(x) for x in (jsonb(post_sids) or [])],
                "final_status": final_status or "",
                "stage1": {
                    "model": s1_model,
                    "confidence": float(s1_conf) if s1_conf is not None else 0.0,
                    "reasoning": s1_reason or "",
                    "cost_usd": float(s1_cost) if s1_cost is not None else 0.0,
                    "evidence": jsonb(s1_ev) or [],
                    "verified": None,
                },
                "stage2": {
                    "model": s2_model,
                    "confidence": float(s2_conf) if s2_conf is not None else 0.0,
                    "reasoning": s2_reason or "",
                    "cost_usd": float(s2_cost) if s2_cost is not None else 0.0,
                    "evidence": jsonb(s2_ev) or [],
                    "verified": s2_verified,
                },
                "threat_categories": jsonb(threat_cats) or [],
                "severity_score": severity,
                "importance": None,
                "weirdness": None,
                "geography_region": None,
                "geography_country": None,
                "analyzed_at": analyzed_at.isoformat() if analyzed_at else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_baseline += 1

    # --- Write bench_data_holdout.jsonl ---
    # One row per unique post, with its subreddit metadata copied from its first
    # containing analysis (posts analyzed multiple times get the same content).
    sid_to_sub = {}
    for row in analyses:
        sub_name, sub_subs, sub_desc = row[1], row[2], row[3]
        for sid in (jsonb(row[4]) or []):
            sid = int(sid)
            if sid not in sid_to_sub:
                sid_to_sub[sid] = (sub_name, sub_subs or 0, sub_desc or "")

    data_path = out_dir / "bench_data_holdout.jsonl"
    written_data = 0
    with open(data_path, "w") as f:
        for sid in sorted(post_map.keys()):
            post_row = post_map[sid]
            sub_name, sub_subs, sub_desc = sid_to_sub.get(sid, ("", 0, ""))
            record = {
                "subreddit": sub_name,
                "subreddit_subscribers": sub_subs,
                "subreddit_description": sub_desc,
                "post": post_row["post"],
                "comments": post_row["comments"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_data += 1

    print(f"\nWrote {written_baseline} analyses  -> {baseline_path}")
    print(f"Wrote {written_data} posts       -> {data_path}")

    # Summary by subreddit
    from collections import Counter
    sub_counts = Counter(sid_to_sub[sid][0] for sid in post_map.keys())
    print("\nPosts by subreddit:")
    for sub, n in sub_counts.most_common():
        print(f"  r/{sub}: {n}")


if __name__ == "__main__":
    main()
