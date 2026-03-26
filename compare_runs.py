"""Compare two model runs side-by-side.

For each batch, shows which posts each model flagged,
where they agree, and where they disagree.

Usage:
    python3 compare_runs.py --run-a results/<run_a>/ --run-b results/<run_b>/
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


def esc(s) -> str:
    return html_mod.escape(str(s or ''))


def load_run(run_dir: Path) -> tuple[dict, dict]:
    with open(run_dir / "output.jsonl") as f:
        output = {r["analysis_id"]: r for line in f for r in [json.loads(line)]}
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    return output, meta


def get_flagged_sids(run_data: dict, snapshot_ids: list[int]) -> dict[int, dict]:
    """Map snapshot_id -> flagged post data for a run."""
    result = {}
    for fp in run_data.get("flagged_posts", []):
        idx = fp.get("post_index", 0)
        if 1 <= idx <= len(snapshot_ids):
            sid = snapshot_ids[idx - 1]
            result[sid] = fp
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare two model runs")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--baseline", default="data/baseline.jsonl")
    parser.add_argument("--bench-data", default="data/bench_data.jsonl")
    parser.add_argument("--output", default="results/compare.html")
    args = parser.parse_args()

    out_a, meta_a = load_run(Path(args.run_a))
    out_b, meta_b = load_run(Path(args.run_b))
    name_a = meta_a.get("model", "Model A").split("/")[-1]
    name_b = meta_b.get("model", "Model B").split("/")[-1]

    with open(args.baseline) as f:
        baselines = {b["analysis_id"]: b for line in f for b in [json.loads(line)]}

    with open(args.bench_data) as f:
        posts_by_id = {}
        for line in f:
            r = json.loads(line)
            posts_by_id[r['post']['snapshot_id']] = r

    # Common analysis IDs
    common_ids = sorted(set(out_a.keys()) & set(out_b.keys()))

    # Summary counters
    both_agree = 0
    only_a_flags_more = 0
    only_b_flags_more = 0
    total_a_flagged = 0
    total_b_flagged = 0
    total_posts = 0

    cards_html = []
    for aid in common_ids:
        bl = baselines.get(aid, {})
        ra = out_a[aid]
        rb = out_b[aid]
        subreddit = ra.get("subreddit", bl.get("subreddit", ""))
        tier = TIERS.get(subreddit.lower(), "unknown")
        tier_color = TIER_COLORS.get(tier, "#888")
        status = bl.get("final_status", "")
        snapshot_ids = bl.get("post_snapshot_ids", ra.get("post_snapshot_ids", []))

        flagged_a = get_flagged_sids(ra, snapshot_ids)
        flagged_b = get_flagged_sids(rb, snapshot_ids)
        sids_a = set(flagged_a.keys())
        sids_b = set(flagged_b.keys())

        total_a_flagged += len(sids_a)
        total_b_flagged += len(sids_b)
        total_posts += len(snapshot_ids)

        agree_count = len(sids_a & sids_b)
        only_a_count = len(sids_a - sids_b)
        only_b_count = len(sids_b - sids_a)

        if only_a_count > only_b_count:
            only_a_flags_more += 1
        elif only_b_count > only_a_count:
            only_b_flags_more += 1
        else:
            both_agree += 1

        # Summary badge
        if sids_a == sids_b:
            diff_badge = '<span class="diff-badge same">SAME</span>'
            diff_type = "same"
        elif only_a_count > 0 and only_b_count > 0:
            diff_badge = '<span class="diff-badge both-diff">BOTH DIFFER</span>'
            diff_type = "differ"
        elif only_a_count > 0:
            diff_badge = f'<span class="diff-badge a-more">+{only_a_count} {esc(name_a)}</span>'
            diff_type = "differ"
        else:
            diff_badge = f'<span class="diff-badge b-more">+{only_b_count} {esc(name_b)}</span>'
            diff_type = "differ"

        # Build post rows
        post_rows = []
        for i, sid in enumerate(snapshot_ids):
            post_data = posts_by_id.get(sid)
            a_flag = sid in sids_a
            b_flag = sid in sids_b

            if a_flag and b_flag:
                row_class = "post-row agree"
                label = "BOTH"
            elif a_flag and not b_flag:
                row_class = "post-row only-a"
                label = f"ONLY {name_a}"
            elif not a_flag and b_flag:
                row_class = "post-row only-b"
                label = f"ONLY {name_b}"
            else:
                row_class = "post-row neither"
                label = ""

            title = esc(post_data['post']['title'][:90]) if post_data else f"(sid={sid})"
            score = post_data['post']['score'] if post_data else 0

            # Reasoning from both models
            detail_parts = []
            if a_flag:
                fa = flagged_a[sid]
                detail_parts.append(f'''
                    <div class="reasoning-box a-box">
                        <b>{esc(name_a)}:</b> {esc(", ".join(fa.get("categories", [])))}
                        | conf={fa.get("confidence", 0):.2f} | imp={fa.get("importance", 0)}
                        <div class="r-text">{esc(fa.get("reasoning", "")[:180])}</div>
                    </div>''')
            if b_flag:
                fb = flagged_b[sid]
                detail_parts.append(f'''
                    <div class="reasoning-box b-box">
                        <b>{esc(name_b)}:</b> {esc(", ".join(fb.get("categories", [])))}
                        | conf={fb.get("confidence", 0):.2f} | imp={fb.get("importance", 0)}
                        <div class="r-text">{esc(fb.get("reasoning", "")[:180])}</div>
                    </div>''')

            detail_html = f'<div class="post-detail hidden">{"".join(detail_parts)}</div>' if detail_parts else ''

            post_rows.append(f'''
            <div class="{row_class}" onclick="this.querySelector('.post-detail')?.classList.toggle('hidden')">
                <div class="post-main">
                    <span class="post-num">#{i+1}</span>
                    {f'<span class="flag-a">A</span>' if a_flag else ''}
                    {f'<span class="flag-b">B</span>' if b_flag else ''}
                    <span class="post-title">{title}</span>
                    <span class="post-score">[{score}↑]</span>
                    {f'<span class="diff-label">{label}</span>' if label else ''}
                </div>
                {detail_html}
            </div>''')

        posts_html = "\n".join(post_rows)

        card = f'''
        <div class="card" data-diff="{diff_type}" data-tier="{tier}">
            <div class="card-header" onclick="this.parentElement.querySelector('.card-body').classList.toggle('collapsed')">
                <span class="tier-dot" style="background:{tier_color}"></span>
                <span class="card-sub">r/{esc(subreddit)}</span>
                {diff_badge}
                <span class="card-counts">
                    {esc(name_a)}: {len(sids_a)}/{len(snapshot_ids)}
                    &nbsp;|&nbsp;
                    {esc(name_b)}: {len(sids_b)}/{len(snapshot_ids)}
                </span>
                <span class="card-id">#{aid}</span>
            </div>
            <div class="card-body collapsed">
                <div class="posts-legend">
                    <span class="flag-a">A</span> = {esc(name_a)}
                    &nbsp;
                    <span class="flag-b">B</span> = {esc(name_b)}
                    &nbsp;|&nbsp;
                    <span class="legend-box agree-box"></span> both
                    <span class="legend-box a-only-box"></span> only A
                    <span class="legend-box b-only-box"></span> only B
                </div>
                {posts_html}
            </div>
        </div>'''
        cards_html.append(card)

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Compare: {esc(name_a)} vs {esc(name_b)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #f1f3f5; color: #1a1a1a;
}}
.top-bar {{
    background: #fff; border-bottom: 1px solid #dee2e6; padding: 16px 24px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 10;
}}
.top-bar h1 {{ font-size: 18px; }}
.vs {{ font-size: 14px; color: #868e96; }}
.stat {{ text-align: center; }}
.stat .num {{ font-size: 20px; font-weight: 700; }}
.stat .label {{ font-size: 10px; color: #868e96; text-transform: uppercase; }}
.filters {{ display: flex; gap: 6px; margin-left: auto; }}
.filter-btn {{
    padding: 4px 10px; border: 1px solid #dee2e6; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 12px;
}}
.filter-btn:hover {{ background: #f1f3f5; }}

.container {{ max-width: 1200px; margin: 16px auto; padding: 0 16px; }}

.card {{
    background: #fff; border-radius: 8px; margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06); border: 1px solid #e9ecef;
}}
.card.hidden {{ display: none; }}
.card-header {{
    padding: 10px 14px; cursor: pointer; display: flex;
    align-items: center; gap: 8px; font-size: 13px;
}}
.card-header:hover {{ background: #f8f9fa; }}
.tier-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.card-sub {{ font-weight: 600; }}
.card-counts {{ font-size: 12px; color: #868e96; margin-left: auto; }}
.card-id {{ font-size: 11px; color: #ced4da; font-family: monospace; }}

.diff-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; color: white; }}
.diff-badge.same {{ background: #40c057; }}
.diff-badge.both-diff {{ background: #7950f2; }}
.diff-badge.a-more {{ background: #228be6; }}
.diff-badge.b-more {{ background: #e8590c; }}

.card-body {{ padding: 0 14px 14px; }}
.card-body.collapsed {{ display: none; }}

.posts-legend {{
    font-size: 11px; color: #868e96; margin: 8px 0 6px;
    display: flex; align-items: center; gap: 8px;
}}
.legend-box {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
.agree-box {{ background: #d3f9d8; border: 1px solid #69db7c; }}
.a-only-box {{ background: #d0ebff; border: 1px solid #74c0fc; }}
.b-only-box {{ background: #ffe8cc; border: 1px solid #ffa94d; }}

.post-row {{
    padding: 5px 8px; border-radius: 4px; cursor: pointer;
    margin-bottom: 2px; font-size: 12px; border: 1px solid #e9ecef;
}}
.post-row:hover {{ filter: brightness(0.97); }}
.post-row.agree {{ background: #d3f9d8; border-color: #69db7c; }}
.post-row.only-a {{ background: #d0ebff; border-color: #74c0fc; }}
.post-row.only-b {{ background: #ffe8cc; border-color: #ffa94d; }}
.post-row.neither {{ background: #fff; }}

.post-main {{ display: flex; align-items: center; gap: 6px; }}
.post-num {{ font-weight: 600; color: #868e96; width: 24px; font-size: 11px; }}
.post-title {{ flex: 1; }}
.post-score {{ color: #868e96; font-size: 11px; }}
.diff-label {{ font-size: 10px; font-weight: 600; color: #868e96; }}

.flag-a {{ padding: 1px 4px; border-radius: 2px; font-size: 9px; font-weight: 700; background: #74c0fc; color: #1864ab; }}
.flag-b {{ padding: 1px 4px; border-radius: 2px; font-size: 9px; font-weight: 700; background: #ffa94d; color: #7c3d00; }}

.post-detail {{ padding: 6px 0 2px 30px; }}
.reasoning-box {{ padding: 6px 8px; margin-bottom: 4px; border-radius: 4px; font-size: 11px; }}
.a-box {{ background: #e7f5ff; border: 1px solid #a5d8ff; }}
.b-box {{ background: #fff4e6; border: 1px solid #ffd8a8; }}
.r-text {{ margin-top: 3px; color: #495057; line-height: 1.3; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<div class="top-bar">
    <h1><span style="color:#228be6">{esc(name_a)}</span> <span class="vs">vs</span> <span style="color:#e8590c">{esc(name_b)}</span></h1>
    <div class="stat"><div class="num">{total_a_flagged}</div><div class="label">{esc(name_a)} flagged</div></div>
    <div class="stat"><div class="num">{total_b_flagged}</div><div class="label">{esc(name_b)} flagged</div></div>
    <div class="stat"><div class="num">{total_posts}</div><div class="label">total posts</div></div>
    <div class="stat"><div class="num">{both_agree}</div><div class="label">batches agree</div></div>
    <div class="filters">
        <button class="filter-btn" onclick="filterDiff('all')">All ({len(common_ids)})</button>
        <button class="filter-btn" onclick="filterDiff('differ')">Differ</button>
        <button class="filter-btn" onclick="filterDiff('same')">Same</button>
    </div>
</div>
<div class="container">
    {"".join(cards_html)}
</div>
<script>
function filterDiff(val) {{
    document.querySelectorAll('.card').forEach(c => {{
        c.classList.toggle('hidden', val !== 'all' && c.dataset.diff !== val);
    }});
}}
</script>
</body>
</html>'''

    with open(args.output, 'w') as f:
        f.write(page)
    print(f"Saved to {args.output}")
    print(f"Open with: xdg-open {args.output}")


if __name__ == "__main__":
    main()
