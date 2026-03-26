"""Visualize threat-bench results as an interactive HTML dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ── Colours ──────────────────────────────────────────────────────────────
PALETTE = ["#10a37f", "#e8590c", "#6c5ce7", "#d63031", "#fdcb6e", "#00b894"]
BASELINE_COLOR = "#4a4a4a"

def model_colors(models: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


# ── Data loading ─────────────────────────────────────────────────────────

def load_all_evals(results_dir: str = "results") -> dict[str, dict]:
    evals = {}
    rdir = Path(results_dir)
    for run_dir in sorted(rdir.iterdir()):
        eval_path = run_dir / "eval.json"
        meta_path = run_dir / "metadata.json"
        if not eval_path.exists() or not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        model = meta["model"]
        short = model.split("/")[-1].replace("-Instruct-AWQ", "").replace("-Instruct", "")
        evals[short] = {
            "eval": json.load(open(eval_path)),
            "meta": meta,
        }
    return evals


def load_baseline_data(path: str = "data/baseline.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def short_label(b: dict) -> str:
    cats = "/".join(c.lower()[:5] for c in b["threat_categories"])
    return f"r/{b['subreddit']} | {cats} | imp={b.get('importance', '?')}"


# ── Panel builders ───────────────────────────────────────────────────────

def make_detection_heatmap(evals: dict, baselines: list[dict], colors: dict) -> go.Figure:
    models = list(evals.keys())
    labels = [short_label(b) for b in baselines]

    # z: 1=detected, 0=missed
    z = []
    hover = []
    for b in baselines:
        row_z = []
        row_h = []
        for model in models:
            per = evals[model]["eval"]["per_analysis"]
            comp = next((c for c in per if c["analysis_id"] == b["analysis_id"]), None)
            detected = comp["detected"] if comp else False
            row_z.append(1 if detected else 0)
            status = "DETECTED" if detected else "MISSED"
            flagged = comp["model_flagged"] if comp else 0
            row_h.append(
                f"<b>{status}</b><br>"
                f"Analysis: {b['analysis_id']}<br>"
                f"r/{b['subreddit']}<br>"
                f"Categories: {', '.join(b['threat_categories'])}<br>"
                f"Baseline importance: {b.get('importance', '?')}<br>"
                f"Model flagged: {flagged} posts"
            )
        z.append(row_z)
        hover.append(row_h)

    colorscale = [[0, "#e03131"], [1, "#40c057"]]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=models, y=labels,
        colorscale=colorscale, showscale=False,
        hovertext=hover, hoverinfo="text",
        zmin=0, zmax=1,
    ))

    # Add text annotations
    for i, b in enumerate(baselines):
        for j, model in enumerate(models):
            val = z[i][j]
            fig.add_annotation(
                x=models[j], y=labels[i],
                text="HIT" if val else "MISS",
                showarrow=False, font=dict(color="white", size=11, family="monospace"),
            )

    # Detection rate at top
    for j, model in enumerate(models):
        rate = evals[model]["eval"]["overall"]["detection"]["detection_rate"]
        fig.add_annotation(
            x=models[j], y=1.02, yref="paper",
            text=f"<b>{rate:.0%}</b>", showarrow=False,
            font=dict(color=colors[model], size=16),
        )

    fig.update_layout(
        title=dict(text="Did the model detect each baseline threat?", font_size=18),
        height=max(450, len(baselines) * 28 + 100),
        margin=dict(l=300, r=30, t=80, b=30),
        yaxis=dict(autorange="reversed", tickfont=dict(family="monospace", size=10)),
        xaxis=dict(side="bottom", tickfont=dict(size=13)),
    )
    return fig


def make_summary_bars(evals: dict, colors: dict) -> go.Figure:
    models = list(evals.keys())

    metrics = {
        "Detection Rate": lambda o: o["detection"]["detection_rate"],
        "Category Jaccard": lambda o: o["categories_jaccard"],
        "Geo Country Match": lambda o: o["geography_country_match"],
        "Geo Region Match": lambda o: o["geography_region_match"],
        "Confidence Agreement": lambda o: 1 - (o["confidence_mae"] or 0),
        "Importance Agreement": lambda o: 1 - (o["importance_mae"] or 0) / 10,
    }

    fig = go.Figure()
    for model in models:
        overall = evals[model]["eval"]["overall"]
        vals = [fn(overall) for fn in metrics.values()]
        fig.add_trace(go.Bar(
            name=model, x=list(metrics.keys()), y=vals,
            marker_color=colors[model], text=[f"{v:.2f}" for v in vals],
            textposition="outside", textfont=dict(size=12),
        ))

    fig.update_layout(
        title=dict(text="Summary Metrics (all 0-1, higher = better)", font_size=18),
        barmode="group", yaxis=dict(range=[0, 1.15], title="Score"),
        height=420, margin=dict(t=70, b=60),
        legend=dict(font_size=13),
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.4)
    return fig


def make_flagging_chart(evals: dict, baselines: list[dict], colors: dict) -> go.Figure:
    models = list(evals.keys())

    x_labels = [f"r/{b['subreddit']}<br>({len(b['post_snapshot_ids'])}p)<br>id={b['analysis_id']}"
                for b in baselines]

    fig = go.Figure()

    # Total posts in batch (background)
    total_posts = [len(b["post_snapshot_ids"]) for b in baselines]
    fig.add_trace(go.Bar(
        name="Total posts in batch", x=x_labels, y=total_posts,
        marker_color="#e9ecef", marker_line=dict(color="#dee2e6", width=1),
        opacity=0.6,
    ))

    # Baseline
    baseline_counts = []
    for b in baselines:
        sids = {e.get("post_snapshot_id") for e in b["stage1"].get("evidence", [])
                if e.get("post_snapshot_id")}
        baseline_counts.append(max(len(sids), 1))
    fig.add_trace(go.Bar(
        name="Baseline (confirmed)", x=x_labels, y=baseline_counts,
        marker_color=BASELINE_COLOR,
    ))

    # Models
    for model in models:
        per_id = {c["analysis_id"]: c for c in evals[model]["eval"]["per_analysis"]}
        counts = [per_id.get(b["analysis_id"], {}).get("model_flagged", 0) for b in baselines]
        fig.add_trace(go.Bar(
            name=model, x=x_labels, y=counts, marker_color=colors[model],
        ))

    # Annotation
    ann_lines = []
    for model in models:
        total = evals[model]["eval"]["overall"]["model_flagged_total"]
        ann_lines.append(f"{model}: {total}/856 ({total/856:.0%})")
    ann_lines.append(f"Baseline: 22/856 ({22/856:.1%})")

    fig.add_annotation(
        x=1.0, y=1.0, xref="paper", yref="paper",
        text="<br>".join(ann_lines), showarrow=False,
        font=dict(family="monospace", size=11),
        bgcolor="white", bordercolor="#ccc", borderwidth=1, borderpad=6,
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        title=dict(text="How many posts did each model flag per analysis?", font_size=18),
        barmode="group", height=480,
        yaxis=dict(title="Posts flagged"),
        xaxis=dict(tickfont=dict(size=8)),
        margin=dict(t=70, b=120),
        legend=dict(font_size=12),
    )
    return fig


def make_scatter_panel(evals: dict, colors: dict) -> go.Figure:
    models = list(evals.keys())
    fields = [
        ("importance", "Importance (1-10)", 0, 10),
        ("confidence", "Confidence (0-1)", 0, 1.0),
        ("weirdness", "Weirdness (1-10)", 0, 10),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=[f[1] for f in fields],
                        horizontal_spacing=0.08)

    for col, (field, label, lo, hi) in enumerate(fields, 1):
        # Diagonal
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#adb5bd", dash="dash", width=1.5),
            showlegend=(col == 1), name="Perfect agreement",
        ), row=1, col=col)

        for model in models:
            per = evals[model]["eval"]["per_analysis"]
            bx, my, hovers = [], [], []
            for comp in per:
                if not comp["detected"]:
                    continue
                bval = comp[field].get("baseline")
                mvals = comp[field].get("model", [])
                if bval is None or not mvals:
                    continue
                mval = mvals[0] if isinstance(mvals, list) else mvals
                if mval is None:
                    continue
                bx.append(bval)
                my.append(mval)
                hovers.append(
                    f"r/{comp['subreddit']}<br>"
                    f"Analysis: {comp['analysis_id']}<br>"
                    f"Baseline: {bval}<br>"
                    f"Model: {mval}<br>"
                    f"Diff: {mval - bval:+.2f}"
                )

            fig.add_trace(go.Scatter(
                x=bx, y=my, mode="markers",
                marker=dict(color=colors[model], size=9, opacity=0.7,
                            line=dict(color="white", width=1)),
                name=model, showlegend=(col == 1),
                hovertext=hovers, hoverinfo="text",
            ), row=1, col=col)

        fig.update_xaxes(title_text=f"Baseline", range=[lo - 0.3, hi + 0.3], row=1, col=col)
        fig.update_yaxes(title_text=f"Model", range=[lo - 0.3, hi + 0.3],
                         scaleanchor=f"x{col}" if col > 1 else "x", row=1, col=col)

    fig.update_layout(
        title=dict(text="Field Agreement: Baseline vs Model (detected threats only)", font_size=18),
        height=450, margin=dict(t=80, b=50),
        legend=dict(font_size=12),
    )
    return fig


def make_geography_table(evals: dict, baselines: list[dict]) -> go.Figure:
    models = list(evals.keys())

    header_vals = ["Subreddit", "ID", "Baseline Country", "Baseline Region"] + \
                  [f"{m} Country" for m in models]

    cell_vals = [[] for _ in header_vals]
    cell_colors = [[] for _ in header_vals]

    for b in baselines:
        cell_vals[0].append(f"r/{b['subreddit']}")
        cell_vals[1].append(str(b["analysis_id"]))
        cell_vals[2].append(b.get("geography_country") or "(global)")
        cell_vals[3].append(b.get("geography_region") or "(global)")
        for c in range(4):
            cell_colors[c].append("#f8f9fa" if c < 2 else "#e9ecef")

        for j, model in enumerate(models):
            per = evals[model]["eval"]["per_analysis"]
            comp = next((c for c in per if c["analysis_id"] == b["analysis_id"]), None)
            col_idx = 4 + j
            if comp and comp["detected"]:
                mc = comp["geography_country"]["model"] or "(global)"
                match = comp["geography_country"]["match"]
                cell_vals[col_idx].append(mc)
                cell_colors[col_idx].append("#d3f9d8" if match else "#ffe3e3")
            else:
                cell_vals[col_idx].append("(not detected)")
                cell_colors[col_idx].append("#fff3bf")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#343a40", font=dict(color="white", size=11),
            align="center", height=32,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colors,
            font=dict(size=11), align="center", height=28,
        ),
    )])

    fig.update_layout(
        title=dict(text="Geography: Does the model assign the right country?", font_size=18),
        height=max(350, len(baselines) * 30 + 120),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── Assemble HTML ────────────────────────────────────────────────────────

def build_html(evals: dict, baselines: list[dict]) -> str:
    models = list(evals.keys())
    colors = model_colors(models)

    figs = [
        make_detection_heatmap(evals, baselines, colors),
        make_summary_bars(evals, colors),
        make_scatter_panel(evals, colors),
        make_flagging_chart(evals, baselines, colors),
        make_geography_table(evals, baselines),
    ]

    # Build self-contained HTML
    plotly_js = "https://cdn.plot.ly/plotly-2.35.2.min.js"

    divs = []
    scripts = []
    for i, fig in enumerate(figs):
        div_id = f"chart{i}"
        divs.append(f'<div id="{div_id}" style="margin-bottom: 12px;"></div>')
        fig_json = fig.to_json()
        scripts.append(
            f"Plotly.newPlot('{div_id}', "
            f"{fig_json}.data, {fig_json}.layout, "
            f"{{responsive: true}});"
        )

    # Actually use to_html for each figure, simpler
    chart_htmls = []
    for fig in figs:
        chart_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Summary stats for header
    summary_rows = ""
    for model in models:
        o = evals[model]["eval"]["overall"]
        d = o["detection"]
        tokens = evals[model]["meta"].get("total_usage", {}).get("total_tokens", 0)
        summary_rows += f"""
        <tr>
            <td style="font-weight:600; color:{colors[model]}">{model}</td>
            <td>{d['detection_rate']:.0%} ({d['detected']}/{o['total_analyses']})</td>
            <td>{o['model_flagged_total']}</td>
            <td>{o['categories_jaccard']:.3f}</td>
            <td>{o['confidence_mae']:.3f}</td>
            <td>{o['importance_mae']:.2f}</td>
            <td>{o['geography_country_match']:.0%}</td>
            <td>{tokens:,}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>threat-bench: Model Comparison</title>
    <script src="{plotly_js}"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #f8f9fa; color: #1a1a1a; padding: 24px;
        }}
        .header {{
            max-width: 1400px; margin: 0 auto 24px;
            padding: 28px 32px; background: white;
            border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 6px; }}
        .header p {{ color: #666; font-size: 14px; margin-bottom: 18px; }}
        .summary-table {{
            width: 100%; border-collapse: collapse; font-size: 14px;
        }}
        .summary-table th {{
            text-align: left; padding: 8px 12px; background: #f1f3f5;
            border-bottom: 2px solid #dee2e6; font-size: 12px;
            text-transform: uppercase; letter-spacing: 0.5px; color: #666;
        }}
        .summary-table td {{
            padding: 8px 12px; border-bottom: 1px solid #e9ecef;
            font-variant-numeric: tabular-nums;
        }}
        .chart-card {{
            max-width: 1400px; margin: 0 auto 16px;
            background: white; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            padding: 12px 16px; overflow: hidden;
        }}
        .note {{
            max-width: 1400px; margin: 0 auto 16px;
            padding: 14px 18px; background: #fff3cd;
            border-radius: 8px; border-left: 4px solid #ffc107;
            font-size: 13px; color: #664d03;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>threat-bench: Model Comparison Dashboard</h1>
        <p>Comparing model outputs against GPT-5 baseline ({len(baselines)} confirmed threat analyses, 856 total posts)</p>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Detection Rate</th>
                    <th>Posts Flagged</th>
                    <th>Category Jaccard</th>
                    <th>Confidence MAE</th>
                    <th>Importance MAE</th>
                    <th>Geo Country</th>
                    <th>Total Tokens</th>
                </tr>
            </thead>
            <tbody>{summary_rows}</tbody>
        </table>
    </div>

    <div class="note">
        <strong>Note:</strong> severity_score is excluded — the Stage 1 prompt does not ask for it,
        so all model values are null. The reported severity MAE in raw eval.json is meaningless.
        Baseline has 22 confirmed threats across 3 subreddits (collapse, ukraine, worldnews).
    </div>

    {"".join(f'<div class="chart-card">{h}</div>' for h in chart_htmls)}
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/comparison.html"

    evals = load_all_evals(results_dir)
    baselines = load_baseline_data()

    if not evals:
        print("No eval.json files found in results/")
        sys.exit(1)

    print(f"Found evals for: {', '.join(evals.keys())}")
    baselines.sort(key=lambda b: (b["subreddit"], b["analysis_id"]))

    html = build_html(evals, baselines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Saved to {output_path}")
    print(f"Open with: xdg-open {output_path}")


if __name__ == "__main__":
    main()
