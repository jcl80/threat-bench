# threat-bench

Benchmarking whether cheaper LLMs can replicate GPT-5's output on a two-stage threat analysis pipeline for Reddit posts.

## Background

Sentinel runs a Go-based threat analysis pipeline that uses GPT-5 to analyze Reddit posts for threat signals (conflict, health, economic, political, AI risk, etc.). This project benchmarks whether cheaper models (GPT-4o, GPT-4o-mini, open models) can produce equivalent outputs — field by field — at lower cost.

## Project structure

```
threat-bench/
├── data/
│   ├── bench_data.jsonl          # 856 input posts (from Go pipeline)
│   └── baseline.jsonl            # 22 GPT-5 confirmed threat analyses (reference)
│
├── prompts/
│   ├── formatter.py              # shared post formatting logic
│   ├── threat_stage1.py          # triage: CONFLICT, HEALTH, ECONOMIC, etc.
│   ├── threat_stage2.py          # verification + severity_score
│   ├── ai_stage1.py              # triage: AI_CAPABILITY, AI_SAFETY, etc.
│   └── ai_stage2.py              # verification + severity_score
│
├── results/                      # generated benchmark runs
│   ├── <timestamp>_<model>_<prompt>/
│   │   ├── metadata.json         # model, git hash, timestamp, token usage
│   │   ├── output.jsonl          # flagged posts per analysis
│   │   └── eval.json             # field-level comparison vs baseline
│   └── index.json                # master list of all runs
│
├── schema.py                     # Pydantic models (input, output, baseline)
├── runner.py                     # runs prompts against any OpenAI-compatible model
├── eval.py                       # field-level agreement scoring
└── pyproject.toml
```

## Setup

```bash
# requires uv
uv sync

# add API key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Running a benchmark

```bash
# run threat stage 1 with a model
uv run python runner.py --prompt threat_stage1 --model gpt-4o-mini --workers 10

# evaluate against baseline
uv run python eval.py \
  --baseline data/baseline.jsonl \
  --model-output results/<run_dir>/output.jsonl \
  --model-name "gpt-4o-mini" \
  --save results/<run_dir>/eval.json
```

## What we're measuring

For each analysis, compared against the Go pipeline's GPT-5 baseline:

| Field | Metric |
|-------|--------|
| Threat detection | Detection rate (did model find the confirmed threat?) |
| `categories` | Set overlap (Jaccard) |
| `confidence` | MAE (mean absolute error) |
| `severity_score` | MAE |
| `importance` | MAE |
| `weirdness` | MAE |
| `geography_country` | Exact match |
| `geography_region` | Exact match |

The goal is knowing **which fields degrade** with cheaper models.

## Results so far

| Metric | gpt-4o-mini | gpt-4o |
|---|---|---|
| Detection rate | 72.7% (16/22) | 90.9% (20/22) |
| Categories (Jaccard) | 0.865 | 0.808 |
| Confidence MAE | 0.237 | 0.130 |
| Importance MAE | 1.438 | 1.500 |
| Geography country | 87.5% | 90.0% |

## Models to test

1. `gpt-4o-mini` — cheapest OpenAI baseline ✅
2. `gpt-4o` — stronger OpenAI model ✅
3. `gpt-5-mini` — same model as Go pipeline Stage 1 (needs org verification)
4. `meta-llama/Llama-3.3-70b-instruct` — open model via Together/Groq
5. `Qwen/Qwen2.5-72B-Instruct` — strong on structured output
