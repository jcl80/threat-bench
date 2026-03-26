"""Generate an HTML report showing bench_data coverage by pipeline analyses."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import psycopg2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ───────────────────────────────────────────────────────────────

TIERS = {
    "collapse": "threat-dense", "ukraine": "threat-dense",
    "worldnews": "threat-dense", "geopolitics": "threat-dense",
    "Economics": "ambiguous", "technology": "ambiguous",
    "news": "ambiguous", "energy": "ambiguous",
    "Cooking": "benign", "AskScience": "benign",
    "askscience": "benign", "woodworking": "benign", "gardening": "benign",
}

TIER_COLORS = {
    "threat-dense": "#e03131",
    "ambiguous": "#f08c00",
    "benign": "#2f9e44",
}

STATUS_COLORS = {
    "confirmed_threat": "#e03131",
    "false_positive": "#f08c00",
    "no_threat": "#2f9e44",
    "needs_review": "#868e96",
    "clear": "#339af0",
}


def get_conn():
    import os
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent.parent.parent / "reddit" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATABASE_URL=") and not line.startswith("#"):
                    return psycopg2.connect(line.split("=", 1)[1])
    return psycopg2.connect(os.environ["SENTINEL_DATABASE_URL"])


def main():
    conn = get_conn()
    cur = conn.cursor()

    # Load bench_data
    with open("data/bench_data.jsonl") as f:
        rows = [json.loads(line) for line in f]

    bench_ids = {r["post"]["snapshot_id"] for r in rows}
    bench_by_sub = defaultdict(list)
    for r in rows:
        bench_by_sub[r["subreddit"]].append(r["post"]["snapshot_id"])

    # Get all analyses referencing our bench posts
    cur.execute("""
        SELECT a.id, s.name, a.final_status, a.post_snapshot_ids,
               a.threat_categories, a.severity_score, a.importance,
               a.stage1_model, a.stage2_model
        FROM analyses a
        JOIN subreddits s ON a.subreddit_id = s.id
        WHERE a.post_snapshot_ids IS NOT NULL
    """)

    # Map: snapshot_id -> list of (analysis_id, status, categories, ...)
    post_to_analyses = defaultdict(list)
    analysis_details = {}
    for row in cur.fetchall():
        a_id, sub, status, sids, cats, sev, imp, s1m, s2m = row
        analysis_details[a_id] = {
            "id": a_id, "subreddit": sub, "status": status,
            "categories": cats, "severity": sev, "importance": imp,
            "stage1_model": s1m, "stage2_model": s2m,
            "post_snapshot_ids": sids or [],
        }
        for sid in (sids or []):
            if sid in bench_ids:
                post_to_analyses[sid].append(a_id)

    # Also check analysis_run_id linkage
    cur.execute("SELECT id, analysis_run_id FROM post_snapshots WHERE id = ANY(%s)", (list(bench_ids),))
    has_run_id = {row[0] for row in cur.fetchall() if row[1] is not None}

    conn.close()

    covered_via_analyses = set(post_to_analyses.keys())
    covered_either = covered_via_analyses | has_run_id
    not_covered = bench_ids - covered_either

    # ── Per-subreddit stats ──────────────────────────────────────────────
    sub_stats = {}
    for sub, sids in sorted(bench_by_sub.items()):
        sids_set = set(sids)
        covered = sids_set & covered_either
        # Get verdict distribution for covered posts (via analysis linkage)
        verdicts = Counter()
        for sid in sids_set & covered_via_analyses:
            for a_id in post_to_analyses[sid]:
                verdicts[analysis_details[a_id]["status"]] += 1
                break  # count each post once (use first matching analysis)
        # Posts covered only via run_id (no specific verdict)
        run_id_only = (sids_set & has_run_id) - covered_via_analyses
        if run_id_only:
            verdicts["(in run, no verdict)"] = len(run_id_only)

        sub_stats[sub] = {
            "total": len(sids),
            "covered": len(covered),
            "missing": len(sids_set - covered),
            "verdicts": dict(verdicts),
            "tier": TIERS.get(sub, "unknown"),
        }

    # ── Build figures ────────────────────────────────────────────────────

    # Fig 1: Coverage bar per subreddit
    subs_ordered = sorted(sub_stats.keys(), key=lambda s: (TIERS.get(s, "z"), s))

    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(
        name="Covered by pipeline",
        x=[f"r/{s}" for s in subs_ordered],
        y=[sub_stats[s]["covered"] for s in subs_ordered],
        marker_color=[TIER_COLORS.get(TIERS.get(s), "#888") for s in subs_ordered],
        text=[f"{sub_stats[s]['covered']}/{sub_stats[s]['total']}" for s in subs_ordered],
        textposition="outside",
    ))
    fig_cov.add_trace(go.Bar(
        name="Not covered",
        x=[f"r/{s}" for s in subs_ordered],
        y=[sub_stats[s]["missing"] for s in subs_ordered],
        marker_color="#dee2e6",
    ))
    fig_cov.update_layout(
        barmode="stack",
        title=dict(text=f"Pipeline Coverage: {len(covered_either)}/{len(bench_ids)} bench posts have ground truth ({len(covered_either)/len(bench_ids):.1%})", font_size=18),
        yaxis_title="Posts",
        height=450,
        legend=dict(font_size=12),
    )
    # Add tier annotations
    for sub in subs_ordered:
        tier = TIERS.get(sub, "")
        fig_cov.add_annotation(
            x=f"r/{sub}", y=-0.08, yref="paper",
            text=tier, showarrow=False, font=dict(size=9, color=TIER_COLORS.get(tier, "#888")),
        )

    # Fig 2: Verdict distribution per subreddit (stacked)
    all_statuses = sorted({s for st in sub_stats.values() for s in st["verdicts"]})
    fig_verd = go.Figure()
    for status in all_statuses:
        color = STATUS_COLORS.get(status, "#adb5bd")
        fig_verd.add_trace(go.Bar(
            name=status,
            x=[f"r/{s}" for s in subs_ordered],
            y=[sub_stats[s]["verdicts"].get(status, 0) for s in subs_ordered],
            marker_color=color,
        ))
    fig_verd.update_layout(
        barmode="stack",
        title=dict(text="Pipeline Verdict Distribution per Subreddit", font_size=18),
        yaxis_title="Posts",
        height=450,
        legend=dict(font_size=12),
    )

    # Fig 3: Summary table
    table_header = ["Subreddit", "Tier", "Total Posts", "Covered", "Missing",
                    "confirmed_threat", "no_threat", "false_positive", "needs_review"]
    table_cells = [[] for _ in table_header]
    table_colors = [[] for _ in table_header]

    for sub in subs_ordered:
        s = sub_stats[sub]
        v = s["verdicts"]
        vals = [
            f"r/{sub}", s["tier"], s["total"], s["covered"], s["missing"],
            v.get("confirmed_threat", 0), v.get("no_threat", 0),
            v.get("false_positive", 0), v.get("needs_review", 0),
        ]
        row_color = "#fff5f5" if s["missing"] > 0 else "#f8f9fa"
        for i, val in enumerate(vals):
            table_cells[i].append(val)
            table_colors[i].append(row_color)

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in table_header],
                    fill_color="#343a40", font=dict(color="white", size=11),
                    align="center", height=32),
        cells=dict(values=table_cells, fill_color=table_colors,
                   font=dict(size=11), align="center", height=28),
    )])
    fig_table.update_layout(
        title=dict(text="Coverage Detail", font_size=18),
        height=max(300, len(subs_ordered) * 30 + 120),
        margin=dict(t=50, b=10),
    )

    # Fig 4: Uncovered posts list (if any)
    if not_covered:
        uncov_rows = []
        for r in rows:
            if r["post"]["snapshot_id"] in not_covered:
                uncov_rows.append(r)
        uncov_table = go.Figure(data=[go.Table(
            header=dict(values=["<b>snapshot_id</b>", "<b>Subreddit</b>", "<b>Title</b>"],
                        fill_color="#343a40", font=dict(color="white", size=11)),
            cells=dict(
                values=[
                    [r["post"]["snapshot_id"] for r in uncov_rows],
                    [f"r/{r['subreddit']}" for r in uncov_rows],
                    [r["post"]["title"][:80] for r in uncov_rows],
                ],
                fill_color="#fff3bf", font=dict(size=10), align="left", height=26,
            ),
        )])
        uncov_table.update_layout(
            title=dict(text=f"Uncovered Posts ({len(not_covered)} posts — no pipeline analysis)", font_size=16),
            height=max(200, len(uncov_rows) * 28 + 100),
            margin=dict(t=50, b=10),
        )

    # ── Assemble HTML ────────────────────────────────────────────────────
    plotly_js = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    charts = [fig_cov, fig_verd, fig_table]
    if not_covered:
        charts.append(uncov_table)

    chart_htmls = [fig.to_html(full_html=False, include_plotlyjs=False) for fig in charts]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>threat-bench: Data Coverage Report</title>
    <script src="{plotly_js}"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #f8f9fa; color: #1a1a1a; padding: 24px;
        }}
        .header {{
            max-width: 1200px; margin: 0 auto 20px;
            padding: 24px 28px; background: white;
            border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .stat {{ display: inline-block; margin-right: 32px; }}
        .stat .num {{ font-size: 28px; font-weight: 700; }}
        .stat .label {{ font-size: 13px; color: #666; }}
        .chart-card {{
            max-width: 1200px; margin: 0 auto 14px;
            background: white; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            padding: 10px 14px;
        }}
        .legend {{
            max-width: 1200px; margin: 0 auto 14px;
            padding: 12px 18px; background: white;
            border-radius: 8px; font-size: 13px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .legend span {{ margin-right: 18px; }}
        .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bench Data Coverage Report</h1>
        <div class="stat"><div class="num">{len(bench_ids):,}</div><div class="label">Total bench posts</div></div>
        <div class="stat"><div class="num" style="color:#2f9e44">{len(covered_either):,}</div><div class="label">Covered by pipeline</div></div>
        <div class="stat"><div class="num" style="color:#e03131">{len(not_covered)}</div><div class="label">Not covered</div></div>
        <div class="stat"><div class="num">{len(covered_either)/len(bench_ids):.1%}</div><div class="label">Coverage rate</div></div>
    </div>
    <div class="legend">
        <b>Tiers:</b>
        <span><span class="dot" style="background:#e03131"></span>threat-dense</span>
        <span><span class="dot" style="background:#f08c00"></span>ambiguous</span>
        <span><span class="dot" style="background:#2f9e44"></span>benign</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Verdicts:</b>
        <span><span class="dot" style="background:#e03131"></span>confirmed_threat</span>
        <span><span class="dot" style="background:#2f9e44"></span>no_threat</span>
        <span><span class="dot" style="background:#f08c00"></span>false_positive</span>
        <span><span class="dot" style="background:#868e96"></span>needs_review</span>
    </div>
    {"".join(f'<div class="chart-card">{h}</div>' for h in chart_htmls)}
</body>
</html>"""

    out = "results/coverage.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved to {out}")
    print(f"Open with: xdg-open {out}")


if __name__ == "__main__":
    main()
