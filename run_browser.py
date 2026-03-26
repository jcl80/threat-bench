"""Generate an interactive HTML browser for a benchmark run.

Shows each analysis batch side-by-side: baseline vs model output.
For each post in the batch, shows whether baseline flagged it,
whether the model flagged it, and highlights agreement/disagreement.
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
from pathlib import Path
from collections import defaultdict

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to the run directory")
    parser.add_argument("--baseline", default="data/baseline.jsonl")
    parser.add_argument("--bench-data", default="data/bench_data.jsonl")
    parser.add_argument("--output", default=None, help="Output HTML path (default: <run-dir>/browse.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = args.output or str(run_dir / "browse.html")

    # Load data
    with open(args.baseline) as f:
        baselines = {b["analysis_id"]: b for line in f for b in [json.loads(line)]}

    with open(run_dir / "output.jsonl") as f:
        model_runs = {r["analysis_id"]: r for line in f for r in [json.loads(line)]}

    with open(args.bench_data) as f:
        posts_by_id = {}
        for line in f:
            r = json.loads(line)
            posts_by_id[r['post']['snapshot_id']] = r

    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    eval_path = run_dir / "eval.json"
    eval_data = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

    model_name = meta.get("model", "unknown")

    # Build cards
    cards_html = []
    analysis_ids = sorted(set(baselines.keys()) & set(model_runs.keys()))

    summary_stats = {"total": 0, "detected": 0, "missed": 0, "no_threat_clean": 0, "no_threat_fps": 0}

    for aid in analysis_ids:
        bl = baselines[aid]
        mr = model_runs[aid]
        status = bl.get("final_status", "")
        subreddit = bl.get("subreddit", "")
        tier = TIERS.get(subreddit.lower(), "unknown")
        tier_color = TIER_COLORS.get(tier, "#888")
        snapshot_ids = bl.get("post_snapshot_ids", [])
        is_threat = status == "confirmed_threat"

        # Baseline flagged posts
        bl_flagged_sids = set()
        for ev in bl.get("stage1", {}).get("evidence", []):
            sid = ev.get("post_snapshot_id")
            if sid:
                bl_flagged_sids.add(sid)

        # Model flagged posts (map post_index to snapshot_id)
        model_flagged_sids = set()
        model_flagged_by_sid = {}
        for fp in mr.get("flagged_posts", []):
            idx = fp.get("post_index", 0)
            if 1 <= idx <= len(snapshot_ids):
                sid = snapshot_ids[idx - 1]
                model_flagged_sids.add(sid)
                model_flagged_by_sid[sid] = fp

        # Detection result
        detected = bool(bl_flagged_sids & model_flagged_sids) if is_threat else None
        summary_stats["total"] += 1
        if is_threat:
            if detected:
                summary_stats["detected"] += 1
            else:
                summary_stats["missed"] += 1
        else:
            if len(model_flagged_sids) == 0:
                summary_stats["no_threat_clean"] += 1
            else:
                summary_stats["no_threat_fps"] += 1

        # Header badge
        if is_threat:
            result_badge = '<span class="result-badge hit">HIT</span>' if detected else '<span class="result-badge miss">MISS</span>'
        else:
            if len(model_flagged_sids) == 0:
                result_badge = '<span class="result-badge clean">CLEAN</span>'
            else:
                result_badge = f'<span class="result-badge fp">FP ({len(model_flagged_sids)})</span>'

        # Build post rows
        post_rows = []
        for i, sid in enumerate(snapshot_ids):
            post_data = posts_by_id.get(sid)
            bl_flag = sid in bl_flagged_sids
            model_flag = sid in model_flagged_sids

            # Agreement status
            if bl_flag and model_flag:
                agree_class = "agree-tp"
                agree_label = "BOTH FLAGGED"
            elif bl_flag and not model_flag:
                agree_class = "agree-fn"
                agree_label = "MISSED (baseline flagged, model didn't)"
            elif not bl_flag and model_flag:
                agree_class = "agree-fp"
                agree_label = "EXTRA (model flagged, baseline didn't)"
            else:
                agree_class = "agree-tn"
                agree_label = ""

            title = esc(post_data['post']['title'][:90]) if post_data else f"(sid={sid})"
            score = post_data['post']['score'] if post_data else 0

            # Model's reasoning for this post (if flagged)
            model_detail = ""
            if model_flag and sid in model_flagged_by_sid:
                mf = model_flagged_by_sid[sid]
                cats = ", ".join(mf.get("categories", []))
                conf = mf.get("confidence", 0)
                imp = mf.get("importance", 0)
                reasoning = esc(mf.get("reasoning", "")[:200])
                model_detail = f'''
                    <div class="model-reasoning">
                        <span class="mr-field">cats: {esc(cats)}</span>
                        <span class="mr-field">conf: {conf:.2f}</span>
                        <span class="mr-field">imp: {imp}</span>
                        <div class="mr-text">{reasoning}</div>
                    </div>'''

            post_rows.append(f'''
                <div class="post-row {agree_class}" onclick="this.querySelector('.post-expand')?.classList.toggle('hidden')">
                    <div class="post-main">
                        <span class="post-num">#{i+1}</span>
                        {f'<span class="flag-bl">BL</span>' if bl_flag else ''}
                        {f'<span class="flag-model">MODEL</span>' if model_flag else ''}
                        <span class="post-title">{title}</span>
                        <span class="post-score">[{score}↑]</span>
                        {f'<span class="agree-label">{agree_label}</span>' if agree_label else ''}
                    </div>
                    <div class="post-expand hidden">
                        {model_detail}
                    </div>
                </div>''')

        posts_html = "\n".join(post_rows)

        # Baseline info
        bl_cats = ", ".join(bl.get("threat_categories", []))
        bl_conf = bl.get("stage1", {}).get("confidence", 0)
        bl_imp = bl.get("importance", "?")
        bl_reasoning = esc(bl.get("stage1", {}).get("reasoning", "")[:200])

        # Model summary
        n_model_flagged = len(mr.get("flagged_posts", []))
        usage = mr.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        wall = usage.get("wall_clock_seconds", 0)

        card = f'''
        <div class="card" data-tier="{tier}" data-status="{status}"
             data-result="{'hit' if detected else 'miss' if is_threat else 'clean' if not model_flagged_sids else 'fp'}">
            <div class="card-header" onclick="this.parentElement.querySelector('.card-body').classList.toggle('collapsed')">
                <span class="tier-dot" style="background:{tier_color}"></span>
                <span class="card-sub">r/{esc(subreddit)}</span>
                <span class="card-status status-{status}">{status}</span>
                {result_badge}
                <span class="card-info">{len(snapshot_ids)}p → baseline:{len(bl_flagged_sids)} model:{n_model_flagged}</span>
                <span class="card-id">#{aid}</span>
            </div>
            <div class="card-body collapsed">
                <div class="comparison">
                    <div class="comp-side baseline-side">
                        <div class="comp-title">Baseline (production)</div>
                        <div class="comp-field"><b>Status:</b> {status}</div>
                        <div class="comp-field"><b>Categories:</b> {esc(bl_cats)}</div>
                        <div class="comp-field"><b>Confidence:</b> {bl_conf:.2f} | <b>Importance:</b> {bl_imp}</div>
                        <div class="comp-reasoning">{bl_reasoning}</div>
                    </div>
                    <div class="comp-side model-side">
                        <div class="comp-title">{esc(model_name)}</div>
                        <div class="comp-field"><b>Flagged:</b> {n_model_flagged}/{len(snapshot_ids)} posts</div>
                        <div class="comp-field"><b>Tokens:</b> {tokens:,} | <b>Time:</b> {wall:.1f}s</div>
                    </div>
                </div>
                <div class="posts-section">
                    <div class="posts-legend">
                        <span class="flag-bl">BL</span> = baseline evidence post
                        <span class="flag-model">MODEL</span> = model flagged
                        <span class="legend-box agree-tp-box"></span> both
                        <span class="legend-box agree-fn-box"></span> missed
                        <span class="legend-box agree-fp-box"></span> extra
                    </div>
                    {posts_html}
                </div>
            </div>
        </div>'''
        cards_html.append(card)

    # Summary header
    det_rate = summary_stats['detected'] / (summary_stats['detected'] + summary_stats['missed']) * 100 if (summary_stats['detected'] + summary_stats['missed']) else 0

    # Eval summary if available
    eval_summary = ""
    if eval_data and eval_data.get("overall"):
        o = eval_data["overall"]
        fa = o.get("field_agreement", {})
        tp_data = o.get("throughput", {})
        eval_summary = f'''
        <div class="eval-summary">
            <div class="eval-row"><span>Categories Jaccard</span><span class="eval-val">{fa.get("categories_jaccard", 0):.3f}</span></div>
            <div class="eval-row"><span>Confidence MAE</span><span class="eval-val">{fa.get("confidence_mae", 0):.3f}</span></div>
            <div class="eval-row"><span>Importance MAE</span><span class="eval-val">{fa.get("importance_mae", 0):.3f}</span></div>
            <div class="eval-row"><span>Geo Country Match</span><span class="eval-val">{fa.get("geography_country_match", 0):.0%}</span></div>
            <div class="eval-row"><span>Tokens/sec</span><span class="eval-val">{tp_data.get("completion_tokens_per_sec", 0)}</span></div>
        </div>'''

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run: {esc(model_name)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #f1f3f5; color: #1a1a1a; }}

.top-bar {{
    background: #fff; border-bottom: 1px solid #dee2e6; padding: 16px 24px;
    display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 10;
}}
.top-bar h1 {{ font-size: 18px; }}
.stat {{ text-align: center; }}
.stat .num {{ font-size: 22px; font-weight: 700; }}
.stat .label {{ font-size: 11px; color: #868e96; text-transform: uppercase; }}

.filters {{ display: flex; gap: 6px; margin-left: auto; }}
.filter-btn {{
    padding: 4px 10px; border: 1px solid #dee2e6; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 12px;
}}
.filter-btn:hover {{ background: #f1f3f5; }}
.filter-btn.active {{ background: #228be6; color: white; border-color: #228be6; }}

{eval_summary and '.eval-summary { display: flex; gap: 12px; }' or ''}
.eval-row {{ text-align: center; font-size: 12px; }}
.eval-val {{ display: block; font-size: 16px; font-weight: 700; }}

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
.card-status {{ font-size: 11px; color: #868e96; }}
.card-info {{ font-size: 12px; color: #868e96; margin-left: auto; }}
.card-id {{ font-size: 11px; color: #ced4da; font-family: monospace; }}

.result-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; color: white; }}
.result-badge.hit {{ background: #40c057; }}
.result-badge.miss {{ background: #e03131; }}
.result-badge.clean {{ background: #228be6; }}
.result-badge.fp {{ background: #f08c00; }}

.card-body {{ padding: 0 14px 14px; }}
.card-body.collapsed {{ display: none; }}

.comparison {{ display: flex; gap: 12px; margin: 10px 0; }}
.comp-side {{ flex: 1; padding: 10px; border-radius: 6px; font-size: 12px; }}
.baseline-side {{ background: #fff9db; border: 1px solid #ffd43b; }}
.model-side {{ background: #d0ebff; border: 1px solid #74c0fc; }}
.comp-title {{ font-weight: 700; margin-bottom: 6px; }}
.comp-field {{ margin-bottom: 3px; }}
.comp-reasoning {{ color: #495057; margin-top: 6px; line-height: 1.4; }}

.posts-section {{ margin-top: 8px; }}
.posts-legend {{
    font-size: 11px; color: #868e96; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}}
.legend-box {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
.agree-tp-box {{ background: #d3f9d8; border: 1px solid #69db7c; }}
.agree-fn-box {{ background: #ffe3e3; border: 1px solid #ff8787; }}
.agree-fp-box {{ background: #fff3bf; border: 1px solid #ffd43b; }}

.post-row {{
    padding: 5px 8px; border-radius: 4px; cursor: pointer;
    margin-bottom: 2px; font-size: 12px; border: 1px solid #e9ecef;
}}
.post-row:hover {{ filter: brightness(0.97); }}
.agree-tp {{ background: #d3f9d8; border-color: #69db7c; }}
.agree-fn {{ background: #ffe3e3; border-color: #ff8787; }}
.agree-fp {{ background: #fff3bf; border-color: #ffd43b; }}
.agree-tn {{ background: #fff; }}

.post-main {{ display: flex; align-items: center; gap: 6px; }}
.post-num {{ font-weight: 600; color: #868e96; width: 24px; font-size: 11px; }}
.post-title {{ flex: 1; }}
.post-score {{ color: #868e96; font-size: 11px; }}
.agree-label {{ font-size: 10px; font-weight: 600; color: #868e96; }}

.flag-bl {{
    padding: 1px 4px; border-radius: 2px; font-size: 9px; font-weight: 700;
    background: #ffd43b; color: #664d03;
}}
.flag-model {{
    padding: 1px 4px; border-radius: 2px; font-size: 9px; font-weight: 700;
    background: #74c0fc; color: #1864ab;
}}

.post-expand {{ padding: 6px 0 2px 30px; }}
.model-reasoning {{ font-size: 11px; }}
.mr-field {{ margin-right: 12px; color: #495057; }}
.mr-text {{ margin-top: 4px; color: #666; line-height: 1.4; }}
.hidden {{ display: none; }}

.status-confirmed_threat {{ color: #e03131; }}
.status-no_threat {{ color: #2f9e44; }}
.status-false_positive {{ color: #f08c00; }}
.status-needs_review {{ color: #868e96; }}
</style>
</head>
<body>
<div class="top-bar">
    <h1>{esc(model_name)}</h1>
    <div class="stat"><div class="num">{det_rate:.0f}%</div><div class="label">Recall</div></div>
    <div class="stat"><div class="num">{summary_stats['detected']}/{summary_stats['detected']+summary_stats['missed']}</div><div class="label">Threats found</div></div>
    <div class="stat"><div class="num">{summary_stats['missed']}</div><div class="label">Missed</div></div>
    <div class="stat"><div class="num">{summary_stats['no_threat_fps']}</div><div class="label">Benign FPs</div></div>
    {eval_summary}
    <div class="filters">
        <button class="filter-btn" onclick="filterResult('all')">All</button>
        <button class="filter-btn" onclick="filterResult('hit')">Hits</button>
        <button class="filter-btn" onclick="filterResult('miss')">Misses</button>
        <button class="filter-btn" onclick="filterResult('fp')">FPs</button>
        <button class="filter-btn" onclick="filterResult('clean')">Clean</button>
    </div>
</div>
<div class="container">
    {"".join(cards_html)}
</div>
<script>
function filterResult(val) {{
    document.querySelectorAll('.card').forEach(c => {{
        c.classList.toggle('hidden', val !== 'all' && c.dataset.result !== val);
    }});
    document.querySelectorAll('.filter-btn').forEach(b => {{
        b.classList.toggle('active', b.textContent.toLowerCase().startsWith(val.slice(0,3)));
    }});
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
