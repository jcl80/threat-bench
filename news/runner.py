"""Run news article classification against any OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def load_data(path: str) -> list[dict]:
    articles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def load_prompt_template(variant: str) -> str:
    prompt_dir = Path(__file__).parent / "prompts"
    if variant == "a":
        path = prompt_dir / "prompt_a_production.txt"
    elif variant == "b":
        path = prompt_dir / "prompt_b_bare.txt"
    elif variant == "c":
        path = prompt_dir / "prompt_c_with_reasoning.txt"
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return path.read_text()


def build_prompt(template: str, article: dict) -> str:
    title = article.get("title", "")
    summary = article.get("summary", "") or ""
    importance_reasoning = article.get("importance_reasoning", "") or ""
    return (template
            .replace("{title}", title)
            .replace("{summary}", summary)
            .replace("{importance_reasoning}", importance_reasoning))


def parse_yes_no(text: str) -> str | None:
    """Extract yes/no from model response, handling thinking tokens."""
    # Strip <think>...</think> blocks (reasoning models via vLLM)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        return None

    # Take last non-empty line (thinking text before answer)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1].lower().rstrip(".")

    if last in ("yes", "no"):
        return last

    first_word = last.split()[0].rstrip(".,:") if last.split() else ""
    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"

    # Last resort: unambiguous presence
    has_yes = "yes" in last
    has_no = "no" in last
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None


def classify_article(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float | None = None,
    extra_body: dict | None = None,
    max_retries: int = 3,
) -> dict:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            if extra_body:
                kwargs["extra_body"] = extra_body

            t0 = time.monotonic()
            completion = client.chat.completions.create(**kwargs)
            wall_clock = time.monotonic() - t0

            raw = completion.choices[0].message.content or ""
            answer = parse_yes_no(raw)

            usage = {}
            if completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            return {
                "raw_response": raw,
                "answer": answer,
                "wall_clock_seconds": round(wall_clock, 3),
                "usage": usage,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(attempt * 5)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


def _run_one(
    client: OpenAI,
    model: str,
    template: str,
    article: dict,
    index: int,
    total: int,
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
) -> dict:
    prompt = build_prompt(template, article)
    result = classify_article(
        client, model, prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )

    ground_truth = article.get("relevant_per_human_check", "").lower()
    has_summary = bool(article.get("summary", ""))
    date_str = article.get("date", "")

    status = "ok" if result["answer"] else "PARSE_FAIL"
    match = ""
    if result["answer"] and ground_truth in ("yes", "no"):
        match = "correct" if result["answer"] == ground_truth else "wrong"

    print(
        f"[{index}/{total}] id={article['id']} "
        f"answer={result['answer']} truth={ground_truth} "
        f"{match} ({result['wall_clock_seconds']:.1f}s) "
        f"{status}"
    )

    return {
        "id": article["id"],
        "title": article.get("title", ""),
        "date": date_str,
        "has_summary": has_summary,
        "ground_truth": ground_truth,
        "model_answer": result["answer"],
        "raw_response": result["raw_response"],
        "wall_clock_seconds": result["wall_clock_seconds"],
        "usage": result["usage"],
    }


def run_benchmark(
    model: str,
    prompt_variant: str,
    data_path: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_workers: int = 10,
    max_tokens: int = 2048,
    temperature: float | None = None,
    no_thinking: bool = False,
    results_dir: str | None = None,
) -> Path:
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)

    articles = load_data(data_path)
    template = load_prompt_template(prompt_variant)
    print(f"Loaded {len(articles)} articles")
    print(f"Model: {model} | Prompt: {prompt_variant} | Workers: {max_workers}")
    if no_thinking:
        print("Reasoning suppressed (no_thinking=True)")
    print()

    extra_body = None
    if no_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    # Output directory
    if results_dir is None:
        results_dir = str(Path(__file__).parent / "results")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = model.replace("/", "_").replace(":", "_")
    thinking_tag = "_nothink" if no_thinking else ""
    run_dir = Path(results_dir) / f"{timestamp}_{model_slug}_prompt{prompt_variant}{thinking_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run
    all_results = []
    errors = []
    total = len(articles)
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one, client, model, template, article,
                i + 1, total, max_tokens, temperature, extra_body,
            ): article
            for i, article in enumerate(articles)
        }

        for future in as_completed(futures):
            article = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                errors.append((article["id"], str(e)))
                print(f"  [ERROR] id={article['id']}: {e}")

    t_total = time.monotonic() - t_start

    # Sort by id for consistent output
    all_results.sort(key=lambda r: r["id"])

    # Save results
    with open(run_dir / "results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Quick stats
    answered = [r for r in all_results if r["model_answer"]]
    correct = [r for r in answered if r["model_answer"] == r["ground_truth"]]
    parse_fails = [r for r in all_results if not r["model_answer"]]

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model": model,
        "prompt_variant": prompt_variant,
        "no_thinking": no_thinking,
        "data_path": data_path,
        "git_commit": git_hash(),
        "max_workers": max_workers,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "total_articles": total,
        "completed": len(all_results),
        "errors": len(errors),
        "parse_failures": len(parse_fails),
        "accuracy": round(len(correct) / len(answered), 4) if answered else 0,
        "wall_clock_total_seconds": round(t_total, 1),
        "articles_per_minute": round(len(all_results) / (t_total / 60), 1) if t_total > 0 else 0,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if errors:
        with open(run_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done: {len(all_results)}/{total} articles in {t_total:.0f}s")
    print(f"Accuracy: {len(correct)}/{len(answered)} ({metadata['accuracy']:.1%})")
    print(f"Parse failures: {len(parse_fails)}")
    print(f"Speed: {metadata['articles_per_minute']:.0f} articles/min")
    print(f"Saved to {run_dir}/")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run news classification benchmark")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", required=True, choices=["a", "b", "c"],
                        help="Prompt variant: a (production), b (bare), c (with reasoning)")
    parser.add_argument("--data", default="data/eye-of-sauron/bench_simplified.jsonl",
                        help="Path to data JSONL")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature (omit for reasoning models)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Suppress reasoning for thinking models")
    parser.add_argument("--results-dir", default=None, help="Results directory")

    args = parser.parse_args()
    run_benchmark(
        model=args.model,
        prompt_variant=args.prompt,
        data_path=args.data,
        base_url=args.base_url,
        api_key=args.api_key,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        no_thinking=args.no_thinking,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
