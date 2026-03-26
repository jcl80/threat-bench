# threat-bench

Can we replace GPT-5-mini in Sentinel's threat detection pipeline with a cheaper model?

## Problem

Sentinel runs a two-stage pipeline that analyzes Reddit posts for real-world threats (conflict, health crises, economic instability, etc.):

1. **Stage 1 (triage):** GPT-5-mini scans batches of posts from a subreddit and flags potential threats
2. **Stage 2 (verification):** GPT-5 reviews flagged posts and confirms or rejects them

Stage 1 runs on every post across hundreds of subreddits — it's the expensive part. This benchmark tests whether cheaper models can replace GPT-5-mini in Stage 1 without missing real threats or flooding Stage 2 with false positives.

## How it works

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Batch of    │     │  Stage 1 Model   │     │  Comparison  │
│  15 posts    │────▸│  (model under    │────▸│  vs baseline │
│  from a sub  │     │   test)          │     │  (GPT-5-mini)│
└─────────────┘     └──────────────────┘     └──────────────┘
```

We take real batches that the production pipeline already analyzed, feed them to a cheaper model, and compare field-by-field against the production output.

## Baseline

The baseline is **98 real production analyses from the last 2 weeks**, covering 2,385 posts across 12 subreddits in 3 tiers:

| Tier | Subreddits | Purpose |
|------|-----------|---------|
| Threat-dense | collapse, ukraine, worldnews, geopolitics | Detection rate — can the model find real threats? |
| Ambiguous | Economics, technology, news, energy | Precision — does it correctly handle borderline content? |
| Benign | Cooking, askscience, woodworking, gardening | False positive rate — does it flag irrelevant content? |

Each analysis includes:
- The exact batch of posts that was sent to the model
- Which post(s) Stage 1 flagged, with evidence and reasoning
- Stage 2 verification result (confirmed/rejected)
- Categories, severity, importance, geography, confidence, cost

**Browse the baseline interactively:** open `results/baseline.html` in a browser to see every batch, its posts, which were flagged, and the full evidence chain.

## Data

```
data/
├── bench_data.jsonl     # 2,385 posts with comments (input to models)
├── baseline.jsonl       # 98 production analyses (ground truth)
└── deprecated/          # previous dataset (22 analyses, 856 posts)
```

**bench_data.jsonl** — one line per post:
```json
{
  "subreddit": "collapse",
  "subreddit_subscribers": 537715,
  "post": {
    "snapshot_id": 6373170,
    "title": "Colorado River talks collapse as crisis deepens",
    "body": "...",
    "score": 299,
    "num_comments": 19
  },
  "comments": [{"author": "...", "body": "...", "score": 10, "depth": 0}]
}
```

**baseline.jsonl** — one line per analysis (a batch of posts + pipeline result):
```json
{
  "analysis_id": 716411,
  "subreddit": "collapse",
  "post_snapshot_ids": [6373171, 6373170, ...],
  "final_status": "confirmed_threat",
  "stage1": {
    "model": "gpt-5-mini",
    "confidence": 0.90,
    "reasoning": "Interstate talks on water allocation...",
    "evidence": [{"source": "post_title", "post_snapshot_id": 6373170, "reason": "..."}]
  },
  "stage2": {
    "model": "gpt-5",
    "verified": true,
    "confidence": 0.82
  },
  "threat_categories": ["NATURAL_DISASTER", "ECONOMIC"],
  "severity_score": 4,
  "importance": 8
}
```

The `post_snapshot_ids` in each analysis map to `snapshot_id` in bench_data.jsonl — that's how you reconstruct the exact batch the model saw.

## Running a benchmark

```bash
# 1. Run a model against all baseline batches
python3 runner.py --prompt threat_stage1 --model gpt-4o --workers 10

# 2. Evaluate against baseline
python3 eval.py \
  --baseline data/baseline.jsonl \
  --model-output results/<run_dir>/output.jsonl \
  --model-name "gpt-4o" \
  --save results/<run_dir>/eval.json

# 3. Update the comparison dashboard
python3 visualize.py
# -> results/comparison.html
```

## What we measure

| Field | Metric | What it tells you |
|-------|--------|-------------------|
| Threat detection | Recall (did model find confirmed threats?) | Misses = real threats go undetected |
| False positives | Posts flagged beyond baseline | Noise = wasted Stage 2 calls |
| `categories` | Jaccard set overlap | Does it classify threats correctly? |
| `confidence` | MAE | How calibrated is the model? |
| `importance` | MAE | Does it rate severity similarly? |
| `geography_country` | Exact match | Does it identify the right country? |
| `geography_region` | Exact match | Does it identify the right region? |

## Visualizations

All generated as self-contained HTML files — open in any browser.

| File | What it shows | How to generate |
|------|--------------|-----------------|
| `results/baseline.html` | Browse every baseline analysis: posts, evidence, stage flow | `python3 baseline_browser.py` |
| `results/comparison.html` | Side-by-side model comparison dashboard | `python3 visualize.py` |
| `results/coverage.html` | Data coverage report | `python3 coverage_report.py` |

## Project structure

```
threat-bench/
├── data/
│   ├── bench_data.jsonl          # 2,385 input posts
│   ├── baseline.jsonl            # 98 production analyses (ground truth)
│   └── deprecated/               # previous dataset
│
├── prompts/
│   ├── formatter.py              # shared post formatting
│   ├── threat_stage1.py          # Stage 1: triage prompt
│   ├── threat_stage2.py          # Stage 2: verification prompt
│   ├── ai_stage1.py              # AI-specific triage
│   └── ai_stage2.py              # AI-specific verification
│
├── results/                      # benchmark runs + visualizations
│
├── schema.py                     # Pydantic models
├── runner.py                     # run prompts against models
├── eval.py                       # field-level scoring
├── visualize.py                  # comparison dashboard
├── baseline_browser.py           # baseline explorer
├── coverage_report.py            # data coverage report
├── fetch_data.py                 # pull data from Sentinel DB
└── pyproject.toml
```

## Setup

```bash
pip install -r requirements.txt  # or: uv sync

# API key for the model you're testing
echo "OPENAI_API_KEY=sk-..." > .env
```
