"""Generate an interactive HTML baseline browser.

Shows every analysis batch: which posts went in, which got flagged,
evidence chain, stage1→stage2 flow, all browsable and filterable.
"""

from __future__ import annotations

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

TIER_COLORS = {
    'threat-dense': '#e03131',
    'ambiguous': '#f08c00',
    'benign': '#2f9e44',
}

STATUS_COLORS = {
    'confirmed_threat': '#e03131',
    'false_positive': '#f08c00',
    'no_threat': '#2f9e44',
    'needs_review': '#868e96',
    'clear': '#339af0',
}


def esc(s: str) -> str:
    return html_mod.escape(str(s or ''))


def main():
    with open('data/baseline.jsonl') as f:
        analyses = [json.loads(l) for l in f]
    analyses.sort(key=lambda a: (a['subreddit'].lower(), a['analysis_id']))

    with open('data/bench_data.jsonl') as f:
        posts_by_id = {}
        for line in f:
            r = json.loads(line)
            posts_by_id[r['post']['snapshot_id']] = r

    # Build summary stats
    from collections import Counter
    status_counts = Counter(a['final_status'] for a in analyses)
    sub_counts = Counter(a['subreddit'] for a in analyses)
    total_posts = len(posts_by_id)
    total_batches = len(analyses)

    # Collect flagged snapshot_ids per analysis (from stage1 evidence)
    def get_flagged_ids(analysis):
        flagged = set()
        for ev in analysis['stage1'].get('evidence', []):
            sid = ev.get('post_snapshot_id')
            if sid:
                flagged.add(sid)
        return flagged

    # Build analysis cards HTML
    cards_html = []
    for idx, a in enumerate(analyses):
        tier = TIERS.get(a['subreddit'].lower(), 'unknown')
        tier_color = TIER_COLORS.get(tier, '#888')
        status_color = STATUS_COLORS.get(a['final_status'], '#888')
        flagged_ids = get_flagged_ids(a)
        batch_size = len(a['post_snapshot_ids'])
        date_str = a['analyzed_at'][:10] if a['analyzed_at'] else '?'

        # Build posts list
        posts_rows = []
        for i, sid in enumerate(a['post_snapshot_ids']):
            post_data = posts_by_id.get(sid)
            is_flagged = sid in flagged_ids
            if post_data:
                p = post_data['post']
                title = esc(p['title'][:100])
                score = p['score']
                nc = p['num_comments']
                body_preview = esc((p.get('body') or '')[:200])
                comments = post_data.get('comments', [])
                comments_html = ''
                if comments:
                    comments_html = '<div class="comments">'
                    for ci, c in enumerate(comments[:3]):
                        comments_html += f'<div class="comment">#{ci+1} [{c["score"]}↑] {esc(c["author"])}: {esc(c["body"][:150])}</div>'
                    comments_html += '</div>'
            else:
                title = f'(snapshot {sid} not in bench_data)'
                score = nc = 0
                body_preview = ''
                comments_html = ''

            flag_badge = '<span class="badge badge-flagged">FLAGGED</span>' if is_flagged else ''
            row_class = 'post-row flagged-post' if is_flagged else 'post-row'

            posts_rows.append(f'''
                <div class="{row_class}" onclick="this.querySelector('.post-detail')?.classList.toggle('hidden')">
                    <div class="post-header">
                        <span class="post-idx">#{i+1}</span>
                        {flag_badge}
                        <span class="post-title">{title}</span>
                        <span class="post-meta">[{score}↑ {nc}💬]</span>
                        <span class="post-sid">sid={sid}</span>
                    </div>
                    <div class="post-detail hidden">
                        <div class="post-body">{body_preview}</div>
                        {comments_html}
                    </div>
                </div>
            ''')

        posts_html = '\n'.join(posts_rows)

        # Evidence section
        s1_evidence_html = ''
        for ev in a['stage1'].get('evidence', []):
            source = esc(ev.get('source', ''))
            reason = esc(ev.get('reason', ''))
            ev_title = esc(ev.get('post_title', ''))
            s1_evidence_html += f'<div class="evidence-item"><b>{source}</b>: {reason}<br><span class="ev-post">{ev_title}</span></div>'

        s2_evidence_html = ''
        for ev in a['stage2'].get('evidence', []):
            source = esc(ev.get('source', ''))
            reason = esc(ev.get('reason', ''))
            ev_title = esc(ev.get('post_title', ''))
            s2_evidence_html += f'<div class="evidence-item"><b>{source}</b>: {reason}<br><span class="ev-post">{ev_title}</span></div>'

        cats_html = ' '.join(f'<span class="cat-tag">{esc(c)}</span>' for c in a.get('threat_categories', []))
        geo = f"{esc(a.get('geography_region', ''))} ({esc(a.get('geography_country', ''))})" if a.get('geography_country') else esc(a.get('geography_region', ''))
        s1_cost = a['stage1'].get('cost_usd', 0)
        s2_cost = a['stage2'].get('cost_usd', 0)

        card = f'''
        <div class="analysis-card"
             data-sub="{esc(a['subreddit'].lower())}"
             data-tier="{tier}"
             data-status="{a['final_status']}"
             id="analysis-{a['analysis_id']}">
            <div class="card-header" onclick="this.parentElement.querySelector('.card-body').classList.toggle('collapsed')">
                <div class="card-header-left">
                    <span class="tier-dot" style="background:{tier_color}" title="{tier}"></span>
                    <span class="sub-name">r/{esc(a['subreddit'])}</span>
                    <span class="status-badge" style="background:{status_color}">{a['final_status']}</span>
                    <span class="batch-info">{batch_size} posts → {len(flagged_ids)} flagged</span>
                    {cats_html}
                </div>
                <div class="card-header-right">
                    <span class="analysis-date">{date_str}</span>
                    <span class="analysis-id">#{a['analysis_id']}</span>
                </div>
            </div>
            <div class="card-body collapsed">
                <div class="stage-flow">
                    <div class="stage-box stage1">
                        <div class="stage-header">Stage 1: {esc(a['stage1']['model'])}</div>
                        <div class="stage-field"><b>Confidence:</b> {a['stage1']['confidence']:.2f}</div>
                        <div class="stage-field"><b>Cost:</b> ${s1_cost:.4f}</div>
                        <div class="stage-reasoning">{esc(a['stage1']['reasoning'])}</div>
                        <div class="evidence-section">
                            <div class="evidence-label">Evidence:</div>
                            {s1_evidence_html}
                        </div>
                    </div>
                    <div class="stage-arrow">→</div>
                    <div class="stage-box stage2">
                        <div class="stage-header">Stage 2: {esc(a['stage2'].get('model', 'n/a'))}</div>
                        <div class="stage-field"><b>Verified:</b> {a['stage2'].get('verified', 'n/a')}</div>
                        <div class="stage-field"><b>Confidence:</b> {a['stage2'].get('confidence', 0):.2f}</div>
                        <div class="stage-field"><b>Cost:</b> ${s2_cost:.4f}</div>
                        <div class="stage-reasoning">{esc(a['stage2'].get('reasoning', ''))}</div>
                        <div class="evidence-section">
                            <div class="evidence-label">Evidence:</div>
                            {s2_evidence_html}
                        </div>
                    </div>
                </div>
                <div class="result-fields">
                    <span><b>Severity:</b> {a.get('severity_score', 'n/a')}</span>
                    <span><b>Importance:</b> {a.get('importance', 'n/a')}</span>
                    <span><b>Weirdness:</b> {a.get('weirdness', 'n/a')}</span>
                    <span><b>Geography:</b> {geo}</span>
                    <span><b>Total cost:</b> ${s1_cost + s2_cost:.4f}</span>
                </div>
                <div class="posts-section">
                    <div class="posts-header">Posts in batch ({batch_size}):</div>
                    {posts_html}
                </div>
            </div>
        </div>
        '''
        cards_html.append(card)

    # Summary stats for sidebar
    sub_summary = ''
    for sub in sorted(sub_counts.keys(), key=lambda s: (TIERS.get(s.lower(), 'z'), s)):
        tier = TIERS.get(sub.lower(), '?')
        tc = TIER_COLORS.get(tier, '#888')
        cnt = sub_counts[sub]
        sub_summary += f'<div class="filter-item" onclick="filterSub(\'{sub.lower()}\')">'
        sub_summary += f'<span class="tier-dot" style="background:{tc}"></span> r/{esc(sub)} <span class="count">({cnt})</span></div>'

    status_summary = ''
    for status in ['confirmed_threat', 'no_threat', 'false_positive', 'needs_review']:
        cnt = status_counts.get(status, 0)
        sc = STATUS_COLORS.get(status, '#888')
        status_summary += f'<div class="filter-item" onclick="filterStatus(\'{status}\')">'
        status_summary += f'<span class="tier-dot" style="background:{sc}"></span> {status} <span class="count">({cnt})</span></div>'

    page_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>threat-bench: Baseline Browser</title>
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
.sidebar h1 {{ font-size: 16px; margin-bottom: 16px; }}
.sidebar h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #868e96; margin: 16px 0 8px; }}
.stats {{ margin-bottom: 16px; }}
.stat {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }}
.stat .num {{ font-weight: 700; }}
.filter-item {{
    padding: 5px 8px; cursor: pointer; border-radius: 4px; font-size: 13px;
    display: flex; align-items: center; gap: 6px; margin-bottom: 2px;
}}
.filter-item:hover {{ background: #f1f3f5; }}
.filter-item.active {{ background: #e7f5ff; }}
.count {{ color: #868e96; margin-left: auto; }}
.tier-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.reset-btn {{
    display: block; width: 100%; padding: 8px; margin-top: 12px;
    background: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px;
    cursor: pointer; font-size: 12px; text-align: center;
}}
.reset-btn:hover {{ background: #e9ecef; }}

.main {{ margin-left: 260px; padding: 20px; flex: 1; }}
.main-header {{ margin-bottom: 16px; }}
.main-header h1 {{ font-size: 22px; margin-bottom: 4px; }}
.main-header p {{ color: #666; font-size: 14px; }}

.analysis-card {{
    background: #fff; border-radius: 10px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e9ecef;
    overflow: hidden;
}}
.analysis-card.hidden {{ display: none; }}
.card-header {{
    padding: 12px 16px; cursor: pointer; display: flex;
    justify-content: space-between; align-items: center;
    border-bottom: 1px solid transparent;
}}
.card-header:hover {{ background: #f8f9fa; }}
.card-header-left {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.card-header-right {{ display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
.sub-name {{ font-weight: 600; font-size: 14px; }}
.status-badge {{
    padding: 2px 8px; border-radius: 4px; color: white;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
}}
.batch-info {{ font-size: 12px; color: #666; }}
.analysis-date {{ font-size: 12px; color: #868e96; }}
.analysis-id {{ font-size: 11px; color: #adb5bd; font-family: monospace; }}
.cat-tag {{
    padding: 1px 6px; border-radius: 3px; font-size: 10px;
    background: #e9ecef; color: #495057; text-transform: lowercase;
}}

.card-body {{ padding: 0 16px 16px; }}
.card-body.collapsed {{ display: none; }}

.stage-flow {{ display: flex; gap: 12px; align-items: stretch; margin: 12px 0; }}
.stage-box {{
    flex: 1; padding: 12px; border-radius: 8px; font-size: 13px;
}}
.stage1 {{ background: #fff9db; border: 1px solid #ffd43b; }}
.stage2 {{ background: #d3f9d8; border: 1px solid #69db7c; }}
.stage-header {{ font-weight: 700; margin-bottom: 6px; font-size: 13px; }}
.stage-field {{ margin-bottom: 3px; }}
.stage-reasoning {{ margin: 8px 0; color: #495057; line-height: 1.4; }}
.stage-arrow {{ display: flex; align-items: center; font-size: 24px; color: #adb5bd; }}

.evidence-section {{ margin-top: 8px; }}
.evidence-label {{ font-weight: 600; font-size: 12px; margin-bottom: 4px; color: #666; }}
.evidence-item {{
    padding: 6px 8px; margin-bottom: 4px; background: rgba(0,0,0,0.03);
    border-radius: 4px; font-size: 12px; line-height: 1.4;
}}
.ev-post {{ color: #868e96; font-size: 11px; }}

.result-fields {{
    display: flex; gap: 16px; flex-wrap: wrap; padding: 10px 0;
    border-bottom: 1px solid #e9ecef; font-size: 13px;
}}

.posts-section {{ margin-top: 12px; }}
.posts-header {{ font-weight: 600; font-size: 13px; margin-bottom: 8px; }}
.post-row {{
    padding: 6px 10px; border-radius: 6px; cursor: pointer;
    margin-bottom: 3px; font-size: 12px; border: 1px solid #e9ecef;
}}
.post-row:hover {{ background: #f8f9fa; }}
.post-row.flagged-post {{ background: #fff5f5; border-color: #ffa8a8; }}
.post-header {{ display: flex; align-items: center; gap: 6px; }}
.post-idx {{ font-weight: 600; color: #868e96; width: 28px; }}
.post-title {{ flex: 1; }}
.post-meta {{ color: #868e96; font-size: 11px; }}
.post-sid {{ color: #ced4da; font-size: 10px; font-family: monospace; }}
.badge {{ padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 700; }}
.badge-flagged {{ background: #e03131; color: white; }}

.post-detail {{ padding: 8px 0 4px 34px; font-size: 12px; }}
.post-body {{ color: #495057; margin-bottom: 6px; line-height: 1.4; }}
.comments {{ }}
.comment {{ padding: 3px 0; color: #666; border-top: 1px solid #f1f3f5; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<div class="sidebar">
    <h1>Baseline Browser</h1>
    <div class="stats">
        <div class="stat"><span>Total analyses</span><span class="num">{total_batches}</span></div>
        <div class="stat"><span>Total posts</span><span class="num">{total_posts:,}</span></div>
        <div class="stat"><span>Subreddits</span><span class="num">{len(sub_counts)}</span></div>
    </div>
    <h2>Filter by status</h2>
    {status_summary}
    <h2>Filter by subreddit</h2>
    {sub_summary}
    <button class="reset-btn" onclick="resetFilters()">Show all</button>
</div>
<div class="main">
    <div class="main-header">
        <h1>Baseline: GPT-5-mini (Stage 1) + GPT-5 (Stage 2)</h1>
        <p>Production pipeline analyses from the last 14 days. Click any row to expand and see the full batch, evidence, and stage flow.</p>
    </div>
    <div id="cards">
        {''.join(cards_html)}
    </div>
</div>
<script>
function filterSub(sub) {{
    document.querySelectorAll('.analysis-card').forEach(c => {{
        c.classList.toggle('hidden', c.dataset.sub !== sub);
    }});
    document.querySelectorAll('.filter-item').forEach(f => f.classList.remove('active'));
    event.currentTarget.classList.add('active');
}}
function filterStatus(status) {{
    document.querySelectorAll('.analysis-card').forEach(c => {{
        c.classList.toggle('hidden', c.dataset.status !== status);
    }});
    document.querySelectorAll('.filter-item').forEach(f => f.classList.remove('active'));
    event.currentTarget.classList.add('active');
}}
function resetFilters() {{
    document.querySelectorAll('.analysis-card').forEach(c => c.classList.remove('hidden'));
    document.querySelectorAll('.filter-item').forEach(f => f.classList.remove('active'));
}}
</script>
</body>
</html>'''

    out = 'results/baseline.html'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        f.write(page_html)
    print(f'Saved to {out}')
    print(f'Open with: xdg-open {out}')


if __name__ == '__main__':
    main()
