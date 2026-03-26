"""Generate a standalone HTML browser for any run's output.

Shows every batch: which posts went in, which got flagged,
the model's reasoning and evidence for each flag.
Same style as baseline_browser.py but reads from a run's output.jsonl.

Usage:
    python3 view_run.py --run-dir results/<run>/
    python3 view_run.py --run-dir results/<run>/ --output results/<run>/standalone.html
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
from pathlib import Path

TIERS = {
    'collapse': 'threat-dense', 'ukraine': 'threat-dense',
    'worldnews': 'threat-dense', 'geopolitics': 'threat-dense',
    'economics': 'ambiguous', 'technology': 'ambiguous',
    'news': 'ambiguous', 'energy': 'ambiguous',
    'cooking': 'benign', 'askscience': 'benign',
    'woodworking': 'benign', 'gardening': 'benign',
}

TIER_COLORS = {'threat-dense': '#e03131', 'ambiguous': '#f08c00', 'benign': '#2f9e44'}
STATUS_COLORS = {
    'confirmed_threat': '#e03131', 'false_positive': '#f08c00',
    'no_threat': '#2f9e44', 'needs_review': '#868e96', 'clear': '#339af0',
}


def esc(s) -> str:
    return html_mod.escape(str(s or ''))


def main():
    parser = argparse.ArgumentParser(description="Browse a run's output standalone")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bench-data", default="data/bench_data.jsonl")
    parser.add_argument("--baseline", default="data/baseline.jsonl", help="For subreddit metadata only")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = args.output or str(run_dir / "standalone.html")

    with open(run_dir / "output.jsonl") as f:
        runs = [json.loads(l) for l in f]
    runs.sort(key=lambda r: r["analysis_id"])

    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    with open(args.bench_data) as f:
        posts_by_id = {}
        for line in f:
            r = json.loads(line)
            posts_by_id[r['post']['snapshot_id']] = r

    # Load baseline for subreddit metadata and status
    baselines = {}
    if Path(args.baseline).exists():
        with open(args.baseline) as f:
            baselines = {b["analysis_id"]: b for line in f for b in [json.loads(line)]}

    model_name = meta.get("model", "unknown")
    prompt_name = meta.get("prompt", "")
    timestamp = meta.get("timestamp", "")

    # Summary
    total_analyses = len(runs)
    total_posts = sum(r["total_posts"] for r in runs)
    total_flagged = sum(len(r["flagged_posts"]) for r in runs)
    total_tokens = meta.get("total_usage", {}).get("total_tokens", 0)
    total_wall = sum(r.get("usage", {}).get("wall_clock_seconds", 0) for r in runs)
    total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in runs)
    tps = round(total_completion / total_wall, 0) if total_wall > 0 else 0

    # Build cards
    cards_html = []

    for run in runs:
        aid = run["analysis_id"]
        subreddit = run.get("subreddit", "")
        tier = TIERS.get(subreddit.lower(), "unknown")
        tier_color = TIER_COLORS.get(tier, "#888")
        snapshot_ids = run.get("post_snapshot_ids", [])
        flagged = run.get("flagged_posts", [])
        n_flagged = len(flagged)
        n_total = run["total_posts"]
        usage = run.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        wall = usage.get("wall_clock_seconds", 0)

        # Baseline status if available
        bl = baselines.get(aid, {})
        bl_status = bl.get("final_status", "")
        bl_status_color = STATUS_COLORS.get(bl_status, "#888")

        # Map flagged posts by post_index
        flagged_by_idx = {fp["post_index"]: fp for fp in flagged}

        # Build post rows
        post_rows = []
        for i, sid in enumerate(snapshot_ids):
            idx = i + 1
            post_data = posts_by_id.get(sid)
            fp = flagged_by_idx.get(idx)
            is_flagged = fp is not None

            title = esc(post_data['post']['title'][:100]) if post_data else f"(sid={sid})"
            score = post_data['post']['score'] if post_data else 0
            nc = post_data['post']['num_comments'] if post_data else 0

            flag_badge = '<span class="badge flagged">FLAGGED</span>' if is_flagged else ''
            row_class = 'post-row flagged-row' if is_flagged else 'post-row'

            # Detail section for flagged posts
            detail = ""
            if fp:
                cats = " ".join(f'<span class="cat-tag">{esc(c)}</span>' for c in fp.get("categories", []))
                conf = fp.get("confidence", 0)
                imp = fp.get("importance", 0)
                weird = fp.get("weirdness", 0)
                geo_c = fp.get("geography_country", "")
                geo_r = fp.get("geography_region", "")
                reasoning = esc(fp.get("reasoning", ""))

                evidence_html = ""
                for ev in fp.get("evidence", []):
                    src = esc(ev.get("source", ""))
                    reason = esc(ev.get("reason", ""))
                    evidence_html += f'<div class="ev-item"><b>{src}</b>: {reason}</div>'

                geo_str = f"{esc(geo_r)} ({esc(geo_c)})" if geo_c else esc(geo_r) if geo_r else ""

                detail = f'''
                <div class="post-detail hidden">
                    <div class="detail-fields">
                        {cats}
                        <span class="detail-field">conf: {conf:.2f}</span>
                        <span class="detail-field">imp: {imp}</span>
                        <span class="detail-field">weird: {weird}</span>
                        {f'<span class="detail-field">geo: {geo_str}</span>' if geo_str else ''}
                    </div>
                    <div class="detail-reasoning">{reasoning}</div>
                    <div class="detail-evidence">{evidence_html}</div>
                </div>'''

            # Comments preview
            comments_html = ""
            if post_data and post_data.get("comments"):
                coms = post_data["comments"][:3]
                bits = []
                for ci, c in enumerate(coms):
                    bits.append(f'<div class="comment">#{ci+1} [{c["score"]}↑] {esc(c["author"])}: {esc(c["body"][:120])}</div>')
                comments_html = f'<div class="comments hidden">{"".join(bits)}</div>'

            body_preview = ""
            if post_data and post_data['post'].get('body'):
                body_preview = f'<div class="post-body hidden">{esc(post_data["post"]["body"][:250])}</div>'

            post_rows.append(f'''
            <div class="{row_class}" onclick="toggleDetail(this)">
                <div class="post-header">
                    <span class="post-idx">#{idx}</span>
                    {flag_badge}
                    <span class="post-title">{title}</span>
                    <span class="post-meta">[{score}↑ {nc}💬]</span>
                </div>
                {detail}
                {body_preview}
                {comments_html}
            </div>''')

        posts_html = "\n".join(post_rows)

        # Card header
        status_badge = ""
        if bl_status:
            status_badge = f'<span class="status-badge" style="background:{bl_status_color}">{bl_status}</span>'

        card = f'''
        <div class="analysis-card" data-sub="{esc(subreddit.lower())}" data-tier="{tier}"
             data-has-flags="{1 if n_flagged > 0 else 0}">
            <div class="card-header" onclick="this.parentElement.querySelector('.card-body').classList.toggle('collapsed')">
                <span class="tier-dot" style="background:{tier_color}" title="{tier}"></span>
                <span class="sub-name">r/{esc(subreddit)}</span>
                {status_badge}
                <span class="flag-summary">{n_flagged}/{n_total} flagged</span>
                <span class="card-meta">{tokens:,} tok | {wall:.1f}s</span>
                <span class="card-id">#{aid}</span>
            </div>
            <div class="card-body collapsed">
                <div class="posts-section">
                    {posts_html}
                </div>
            </div>
        </div>'''
        cards_html.append(card)

    # Sidebar
    from collections import Counter
    sub_counts = Counter(r.get("subreddit", "") for r in runs)
    sub_items = ""
    for sub in sorted(sub_counts.keys(), key=lambda s: (TIERS.get(s.lower(), 'z'), s)):
        tier = TIERS.get(sub.lower(), '?')
        tc = TIER_COLORS.get(tier, '#888')
        cnt = sub_counts[sub]
        flagged_in_sub = sum(len(r["flagged_posts"]) for r in runs if r.get("subreddit") == sub)
        total_in_sub = sum(r["total_posts"] for r in runs if r.get("subreddit") == sub)
        rate = flagged_in_sub / total_in_sub * 100 if total_in_sub else 0
        sub_items += f'''<div class="filter-item" onclick="filterSub('{sub.lower()}')">
            <span class="tier-dot" style="background:{tc}"></span>
            r/{esc(sub)} <span class="count">({cnt}) {rate:.0f}%</span>
        </div>'''

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(model_name)} — Run Browser</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #f1f3f5; color: #1a1a1a; display: flex; min-height: 100vh;
}}
.sidebar {{
    width: 260px; background: #fff; border-right: 1px solid #dee2e6;
    padding: 20px; position: fixed; top: 0; left: 0; bottom: 0; overflow-y: auto;
}}
.sidebar h1 {{ font-size: 16px; margin-bottom: 4px; }}
.sidebar .model-name {{ font-size: 13px; color: #666; margin-bottom: 16px; }}
.stats {{ margin-bottom: 16px; }}
.stat {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }}
.stat .num {{ font-weight: 700; }}
.sidebar h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #868e96; margin: 14px 0 6px; }}
.filter-item {{
    padding: 4px 8px; cursor: pointer; border-radius: 4px; font-size: 12px;
    display: flex; align-items: center; gap: 6px; margin-bottom: 2px;
}}
.filter-item:hover {{ background: #f1f3f5; }}
.count {{ color: #868e96; margin-left: auto; font-size: 11px; }}
.tier-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.reset-btn {{
    display: block; width: 100%; padding: 8px; margin-top: 12px;
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    cursor: pointer; font-size: 12px; text-align: center;
}}
.reset-btn:hover {{ background: #e9ecef; }}
.filter-btn {{
    display: inline-block; padding: 4px 8px; margin: 2px;
    border: 1px solid #dee2e6; border-radius: 4px; background: #fff;
    cursor: pointer; font-size: 11px;
}}
.filter-btn:hover {{ background: #f1f3f5; }}

.main {{ margin-left: 260px; padding: 20px; flex: 1; }}

.analysis-card {{
    background: #fff; border-radius: 8px; margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06); border: 1px solid #e9ecef;
}}
.analysis-card.hidden {{ display: none; }}
.card-header {{
    padding: 10px 14px; cursor: pointer; display: flex;
    align-items: center; gap: 8px; font-size: 13px;
}}
.card-header:hover {{ background: #f8f9fa; }}
.sub-name {{ font-weight: 600; }}
.flag-summary {{ font-size: 12px; color: #495057; }}
.card-meta {{ font-size: 11px; color: #adb5bd; margin-left: auto; }}
.card-id {{ font-size: 11px; color: #ced4da; font-family: monospace; }}
.status-badge {{
    padding: 1px 6px; border-radius: 3px; color: white;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
}}

.card-body {{ padding: 0 14px 14px; }}
.card-body.collapsed {{ display: none; }}

.post-row {{
    padding: 6px 8px; border-radius: 5px; cursor: pointer;
    margin-bottom: 3px; font-size: 12px; border: 1px solid #e9ecef;
}}
.post-row:hover {{ background: #f8f9fa; }}
.post-row.flagged-row {{ background: #fff5f5; border-color: #ffa8a8; }}
.post-header {{ display: flex; align-items: center; gap: 6px; }}
.post-idx {{ font-weight: 600; color: #868e96; width: 28px; }}
.post-title {{ flex: 1; }}
.post-meta {{ color: #868e96; font-size: 11px; }}
.badge {{ padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 700; }}
.badge.flagged {{ background: #e03131; color: white; }}

.post-detail {{ padding: 8px 0 4px 34px; }}
.detail-fields {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }}
.detail-field {{ font-size: 11px; color: #495057; }}
.detail-reasoning {{ font-size: 12px; color: #495057; line-height: 1.4; margin-bottom: 6px; }}
.detail-evidence {{ }}
.ev-item {{ padding: 4px 6px; margin-bottom: 3px; background: rgba(0,0,0,0.03); border-radius: 3px; font-size: 11px; line-height: 1.3; }}
.cat-tag {{ padding: 1px 5px; border-radius: 3px; font-size: 10px; background: #e9ecef; color: #495057; }}

.post-body {{ padding: 4px 0 4px 34px; font-size: 11px; color: #666; line-height: 1.4; }}
.comments {{ padding: 2px 0 4px 34px; }}
.comment {{ padding: 2px 0; font-size: 11px; color: #666; border-top: 1px solid #f1f3f5; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<div class="sidebar">
    <h1>Run Browser</h1>
    <div class="model-name">{esc(model_name)} | {esc(prompt_name)} | {esc(timestamp)}</div>
    <div class="stats">
        <div class="stat"><span>Analyses</span><span class="num">{total_analyses}</span></div>
        <div class="stat"><span>Posts</span><span class="num">{total_posts:,}</span></div>
        <div class="stat"><span>Flagged</span><span class="num">{total_flagged} ({total_flagged/total_posts:.0%})</span></div>
        <div class="stat"><span>Tokens</span><span class="num">{total_tokens:,}</span></div>
        <div class="stat"><span>Tokens/sec</span><span class="num">{tps:.0f}</span></div>
        <div class="stat"><span>Wall clock</span><span class="num">{total_wall:.0f}s</span></div>
    </div>
    <h2>Filter</h2>
    <div>
        <button class="filter-btn" onclick="filterFlags('all')">All</button>
        <button class="filter-btn" onclick="filterFlags('flagged')">Has flags</button>
        <button class="filter-btn" onclick="filterFlags('clean')">No flags</button>
    </div>
    <h2>Subreddits</h2>
    {sub_items}
    <button class="reset-btn" onclick="resetFilters()">Show all</button>
</div>
<div class="main">
    {"".join(cards_html)}
</div>
<script>
function toggleDetail(el) {{
    el.querySelectorAll('.post-detail, .post-body, .comments').forEach(d => d.classList.toggle('hidden'));
}}
function filterSub(sub) {{
    document.querySelectorAll('.analysis-card').forEach(c => {{
        c.classList.toggle('hidden', c.dataset.sub !== sub);
    }});
}}
function filterFlags(mode) {{
    document.querySelectorAll('.analysis-card').forEach(c => {{
        if (mode === 'all') c.classList.remove('hidden');
        else if (mode === 'flagged') c.classList.toggle('hidden', c.dataset.hasFlags === '0');
        else c.classList.toggle('hidden', c.dataset.hasFlags === '1');
    }});
}}
function resetFilters() {{
    document.querySelectorAll('.analysis-card').forEach(c => c.classList.remove('hidden'));
}}
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(page)
    print(f"Saved to {output_path}")
    print(f"Open with: xdg-open {output_path}")


if __name__ == "__main__":
    main()
