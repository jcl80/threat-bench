"""Run prompts against any OpenAI-compatible model and return validated output."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from schema import BenchRow, BaselineAnalysis, StageResponse

load_dotenv()

PROMPT_MODULES = {
    "threat_stage1": "prompts.threat_stage1",
    "threat_stage2": "prompts.threat_stage2",
    "ai_stage1": "prompts.ai_stage1",
    "ai_stage2": "prompts.ai_stage2",
}


def clean_json_response(text: str) -> str:
    """Strip markdown code fences from model output."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def load_prompt_module(prompt_name: str):
    """Dynamically import a prompt module."""
    import importlib
    module_path = PROMPT_MODULES[prompt_name]
    return importlib.import_module(module_path)


def git_hash() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def load_bench_data(data_path: str) -> dict[int, BenchRow]:
    """Load bench_data.jsonl into a dict keyed by snapshot_id."""
    rows = {}
    with open(data_path) as f:
        for line in f:
            row = BenchRow.model_validate(json.loads(line))
            rows[row.post.snapshot_id] = row
    return rows


def load_baseline(baseline_path: str) -> list[BaselineAnalysis]:
    """Load baseline analyses."""
    with open(baseline_path) as f:
        return [BaselineAnalysis.model_validate(json.loads(line)) for line in f]


def build_batch_from_baseline(
    analysis: BaselineAnalysis,
    all_posts: dict[int, BenchRow],
):
    """Reconstruct the batch of posts that was sent to the model for a given analysis.

    Returns a dict compatible with SubredditBatch shape for the prompt builder.
    """
    from schema import Comment as SchemaComment

    posts = []
    for sid in analysis.post_snapshot_ids:
        if sid not in all_posts:
            continue
        row = all_posts[sid]
        posts.append({
            "snapshot_id": row.post.snapshot_id,
            "reddit_id": row.post.reddit_id,
            "title": row.post.title,
            "body": row.post.body,
            "author": row.post.author,
            "score": row.post.score,
            "num_comments": row.post.num_comments,
            "comments": [c.model_dump() for c in row.comments],
        })

    # Use metadata from first available post
    first_row = all_posts[analysis.post_snapshot_ids[0]]
    return {
        "subreddit": first_row.subreddit,
        "subreddit_subscribers": first_row.subreddit_subscribers,
        "subreddit_description": first_row.subreddit_description,
        "posts": posts,
    }


def run_batch(
    client: OpenAI,
    model: str,
    prompt_name: str,
    batch: dict,
    flagged_posts: list[dict] | None = None,
    max_retries: int = 3,
) -> tuple[StageResponse, dict]:
    """Run a single batch through the model.

    Returns (validated_response, usage_info).
    """
    from schema import SubredditBatch
    batch_obj = SubredditBatch.model_validate(batch)

    mod = load_prompt_module(prompt_name)

    if prompt_name.endswith("_stage2"):
        if flagged_posts is None:
            raise ValueError("Stage 2 prompts require flagged_posts")
        prompt = mod.build_prompt(batch_obj, flagged_posts)
    else:
        prompt = mod.build_prompt(batch_obj)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.monotonic()
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            wall_clock = time.monotonic() - t0

            raw = completion.choices[0].message.content
            cleaned = clean_json_response(raw)
            data = json.loads(cleaned)
            response = StageResponse.model_validate(data)

            usage = {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
                "wall_clock_seconds": round(wall_clock, 3),
            }

            return response, usage

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                backoff = attempt * 10
                print(f"  [RETRY] Attempt {attempt}/{max_retries} failed: {e} — retrying in {backoff}s")
                time.sleep(backoff)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


CHUNK_SIZE = 15  # Match Go pipeline default


def run_chunked_batch(
    client: OpenAI,
    model: str,
    prompt_name: str,
    batch: dict,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[list, dict]:
    """Run a batch, chunking if needed. Returns (all_posts, total_usage)."""
    from schema import PostAnalysis
    all_posts_list = batch["posts"]

    if len(all_posts_list) <= chunk_size:
        response, usage = run_batch(client, model, prompt_name, batch)
        return response.posts, usage

    # Chunk and run each
    all_results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "wall_clock_seconds": 0}

    for offset in range(0, len(all_posts_list), chunk_size):
        chunk_posts = all_posts_list[offset:offset + chunk_size]
        chunk_batch = {**batch, "posts": chunk_posts}

        response, usage = run_batch(client, model, prompt_name, chunk_batch)

        # Remap post_index from chunk-local to global
        for post in response.posts:
            post.post_index += offset

        all_results.extend(response.posts)
        for k in total_usage:
            total_usage[k] += usage[k]

    return all_results, total_usage


def _run_single(
    client: OpenAI,
    model: str,
    prompt_name: str,
    analysis,
    all_posts: dict,
    index: int,
    total: int,
) -> dict:
    """Run a single analysis batch. Used by both sequential and parallel modes."""
    batch = build_batch_from_baseline(analysis, all_posts)
    n_posts = len(batch["posts"])
    print(f"[{index}/{total}] r/{batch['subreddit']} ({n_posts} posts) — analysis {analysis.analysis_id}...")

    posts_results, usage = run_chunked_batch(client, model, prompt_name, batch)
    flagged = [p for p in posts_results if p.flagged]

    print(f"  → {len(flagged)}/{n_posts} flagged | tokens: {usage['total_tokens']}")

    return {
        "analysis_id": analysis.analysis_id,
        "subreddit": batch["subreddit"],
        "post_snapshot_ids": analysis.post_snapshot_ids,
        "total_posts": n_posts,
        "flagged_posts": [p.model_dump() for p in flagged],
        "usage": usage,
    }


def run_benchmark(
    model: str,
    prompt_name: str,
    data_path: str = "data/bench_data.jsonl",
    baseline_path: str = "data/baseline.jsonl",
    results_dir: str = "results",
    max_workers: int = 5,
    base_url: str | None = None,
) -> Path:
    """Run a prompt against all baseline batches and save results.

    Returns path to the run directory.
    """
    client = OpenAI(base_url=base_url) if base_url else OpenAI()

    # Load data
    all_posts = load_bench_data(data_path)
    baseline = load_baseline(baseline_path)
    print(f"Loaded {len(all_posts)} posts and {len(baseline)} baseline analyses")
    print(f"Running with {max_workers} parallel workers\n")

    # Create run directory
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = model.replace("/", "_").replace(":", "_")
    run_dir = Path(results_dir) / f"{timestamp}_{model_slug}_{prompt_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Filter out analyses with empty post_snapshot_ids
    baseline = [a for a in baseline if a.post_snapshot_ids]
    print(f"Running {len(baseline)} analyses (skipped empty batches)\n")

    # Run batches in parallel
    all_results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "wall_clock_seconds": 0}
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single, client, model, prompt_name,
                analysis, all_posts, i + 1, len(baseline),
            ): analysis
            for i, analysis in enumerate(baseline)
        }

        for future in as_completed(futures):
            analysis = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                for k in total_usage:
                    total_usage[k] += result["usage"][k]
            except Exception as e:
                errors.append((analysis.analysis_id, str(e)))
                print(f"  [ERROR] analysis {analysis.analysis_id}: {e}")

    if errors:
        print(f"\n{len(errors)} analyses failed (saved {len(all_results)} successful)")

    # Sort by analysis_id for consistent output
    all_results.sort(key=lambda r: r["analysis_id"])

    # Save results
    with open(run_dir / "output.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model": model,
        "prompt": prompt_name,
        "git_commit": git_hash(),
        "data_path": data_path,
        "baseline_path": baseline_path,
        "total_analyses": len(baseline),
        "total_posts": len(all_posts),
        "total_usage": total_usage,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update index
    index_path = Path(results_dir) / "index.json"
    index = []
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    index.append({
        "run_dir": run_dir.name,
        "timestamp": timestamp,
        "model": model,
        "prompt": prompt_name,
        "total_tokens": total_usage["total_tokens"],
    })
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved to {run_dir}/")
    print(f"Total tokens: {total_usage['total_tokens']}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run threat-bench prompts against models")
    parser.add_argument("--prompt", required=True, choices=list(PROMPT_MODULES.keys()),
                        help="Which prompt to run")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-5, gpt-4o-mini)")
    parser.add_argument("--data", default="data/bench_data.jsonl", help="Path to bench_data.jsonl")
    parser.add_argument("--baseline", default="data/baseline.jsonl", help="Path to baseline.jsonl")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--workers", type=int, default=10, help="Parallel API calls (default: 10)")
    parser.add_argument("--base-url", default=None, help="Custom API base URL (e.g., http://localhost:8080/v1)")

    args = parser.parse_args()
    run_benchmark(
        model=args.model,
        prompt_name=args.prompt,
        data_path=args.data,
        baseline_path=args.baseline,
        results_dir=args.results_dir,
        max_workers=args.workers,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
