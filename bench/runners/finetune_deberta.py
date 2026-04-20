"""Fine-tune DeBERTa for binary threat classification.

Three modes for an ablation that isolates the value of NLI pre-training:

  A  (nli_continue): MoritzLaurer/deberta-v3-large-zeroshot-v2.0, keep the
     3-class NLI head, train in (premise, hypothesis) NLI format. Label 1
     maps to entailment, label 0 to contradiction.
  B  (swap_head): Same MoritzLaurer checkpoint, but replace the 3-class head
     with a fresh 2-class classification head. Classification format (post
     as a single text input, no hypothesis).
  C  (base_plus_head): microsoft/deberta-v3-large (no NLI pre-training),
     fresh 2-class classification head. Same classification format as B.

All three use the same stratified 80/20 split on label_gpt5, same hypothesis
wording, same hyperparameters. Only the model config and input format differ.

Outputs per run (mirror existing results/ layout so sweep_threshold.py works):
  predictions_test.jsonl    — scored test set (478 posts)
  predictions_train.jsonl   — scored train set (sanity check on overfit)
  metadata.json             — config + train/test metrics + timing
  (no model weights saved by default — 1.6GB each, not worth it for an
   ablation. Add --save-model to keep them.)

Usage:
    python -m bench.runners.finetune_deberta --mode A
    python -m bench.runners.finetune_deberta --mode B --epochs 3
    python -m bench.runners.finetune_deberta --mode C --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATA_PATH = "bench/data/posts.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

HYPOTHESIS = (
    "This text describes a real-world threat, crisis, or dangerous event "
    "such as armed conflict, disease outbreak, economic collapse, "
    "political instability, natural disaster, or AI risk."
)

MAX_PREMISE_CHARS = 1500
MAX_TOKENS = 512
SEED = 42

MODE_CONFIG = {
    "A": {
        "name": "nli_continue",
        "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "num_labels": 2,  # MoritzLaurer v2 is binary NLI: entailment/not_entailment
        "slug": "nli_continue",
    },
    "B": {
        "name": "swap_head",
        "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "num_labels": 2,
        "slug": "swap_head",
    },
    "C": {
        "name": "base_plus_head",
        "model": "microsoft/deberta-v3-large",
        "num_labels": 2,
        "slug": "base_plus_head",
    },
}


def build_premise(post: dict) -> str:
    parts = []
    title = (post.get("title") or "").strip()
    if title:
        parts.append(title)
    body = (post.get("body") or "").strip()
    if body:
        parts.append(body)
    for c in (post.get("comments") or [])[:3]:
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
    text = "\n".join(parts)
    if len(text) > MAX_PREMISE_CHARS:
        text = text[:MAX_PREMISE_CHARS]
    return text


def load_split(data_path: str) -> tuple[list[dict], list[dict]]:
    """Stratified 80/20 split on label_gpt5 with fixed seed."""
    with open(data_path) as f:
        posts = [json.loads(l) for l in f]

    rnd = random.Random(SEED)
    pos = [p for p in posts if p["label_gpt5"] == 1]
    neg = [p for p in posts if p["label_gpt5"] == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)

    split_p = int(len(pos) * 0.8)
    split_n = int(len(neg) * 0.8)
    train = pos[:split_p] + neg[:split_n]
    test = pos[split_p:] + neg[split_n:]
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def _nli_entail_idx(model) -> int:
    """Find the entailment class index. Other index = not-entailment (binary head)."""
    for idx, label in model.config.id2label.items():
        if "entail" in label.lower() and "not" not in label.lower():
            return int(idx)
    raise ValueError(f"No entailment label in {model.config.id2label}")


def encode_dataset(
    posts: list[dict],
    tokenizer,
    mode: str,
    entail_idx: int | None = None,
) -> list[dict]:
    out = []
    for p in posts:
        premise = build_premise(p)
        label_bin = int(p["label_gpt5"])
        if mode == "A":
            enc = tokenizer(
                premise,
                HYPOTHESIS,
                truncation="only_first",
                max_length=MAX_TOKENS,
                padding=False,
            )
            # Binary NLI: label 1 -> entailment index, label 0 -> the other
            not_entail_idx = 1 - entail_idx
            enc["labels"] = entail_idx if label_bin == 1 else not_entail_idx
        else:
            enc = tokenizer(
                premise,
                truncation=True,
                max_length=MAX_TOKENS,
                padding=False,
            )
            enc["labels"] = label_bin
        enc["snapshot_id"] = p["snapshot_id"]
        out.append(enc)
    return out


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {k: v for k, v in r.items() if k != "snapshot_id"}


def predict(model, tokenizer, posts: list[dict], mode: str,
            entail_idx: int | None, device: str, batch_size: int) -> list[dict]:
    """Return per-post {snapshot_id, predicted, score} at threshold 0.5."""
    model.eval()
    out = []
    for i in range(0, len(posts), batch_size):
        batch = posts[i : i + batch_size]
        premises = [build_premise(p) for p in batch]
        if mode == "A":
            inputs = tokenizer(
                premises,
                [HYPOTHESIS] * len(premises),
                return_tensors="pt",
                truncation="only_first",
                max_length=MAX_TOKENS,
                padding=True,
            ).to(device)
        else:
            inputs = tokenizer(
                premises,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
                padding=True,
            ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            if mode == "A":
                scores = probs[:, entail_idx].cpu().tolist()
            else:
                scores = probs[:, 1].cpu().tolist()

        for p, s in zip(batch, scores):
            out.append({
                "snapshot_id": p["snapshot_id"],
                "predicted": 1 if s >= 0.5 else 0,
                "score": round(float(s), 6),
            })
    return out


def score_binary(preds: list[dict], posts: list[dict]) -> dict:
    gt = {p["snapshot_id"]: p["label_gpt5"] for p in posts}
    tp = fp = fn = tn = 0
    for r in preds:
        truth = gt[r["snapshot_id"]]
        pred = r["predicted"]
        if truth == 1 and pred == 1: tp += 1
        elif truth == 0 and pred == 1: fp += 1
        elif truth == 1 and pred == 0: fn += 1
        else: tn += 1
    p_ = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p_, 4), "recall": round(r_, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
    }


def run(mode: str, epochs: int, batch_size: int, lr: float,
        save_model: bool, limit: int | None, fp32: bool = False) -> Path:
    cfg = MODE_CONFIG[mode]
    set_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Mode {mode} ({cfg['name']}) | device={device} | model={cfg['model']}")

    # --- Data ---
    train_posts, test_posts = load_split(DATA_PATH)
    if limit is not None:
        train_posts = train_posts[:limit]
        test_posts = test_posts[: max(4, limit // 4)]
        print(f"[LIMIT={limit}] smoke-test mode: tiny dataset, skip the real metrics")
    print(f"Train: {len(train_posts)} ({sum(p['label_gpt5'] for p in train_posts)} pos)")
    print(f"Test:  {len(test_posts)} ({sum(p['label_gpt5'] for p in test_posts)} pos)")

    # --- Model ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=cfg["num_labels"]
    )
    if mode == "B":
        # The MoritzLaurer v2 checkpoint's 2-class head is shape-compatible,
        # so HF keeps its weights by default. For the ablation we want a
        # *fresh* head — reinitialize it explicitly.
        std = model.config.initializer_range
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=std)
        torch.nn.init.zeros_(model.classifier.bias)
        print("Mode B: reinitialized classifier head (fresh random weights)")
    model.to(device)

    entail_idx = _nli_entail_idx(model) if mode == "A" else None
    if mode == "A":
        print(f"Mode A label map: id2label={dict(model.config.id2label)} "
              f"entail_idx={entail_idx} not_entail_idx={1 - entail_idx}")
        print(f"  → label_gpt5=1 (threat) → labels={entail_idx} (entailment)")
        print(f"  → label_gpt5=0 (safe)   → labels={1 - entail_idx} (not_entail)")

    # --- Tokenize ---
    train_records = encode_dataset(train_posts, tokenizer, mode, entail_idx)
    test_records = encode_dataset(test_posts, tokenizer, mode, entail_idx)
    train_ds = ListDataset(train_records)

    # --- Sanity check: verify labels reach the model and influence loss ---
    print("\n=== PRE-TRAIN SANITY CHECK ===")
    from transformers import DataCollatorWithPadding
    _collator = DataCollatorWithPadding(tokenizer, padding="longest")
    # Pick 8 records, mix of pos/neg labels
    _pos = [r for r in train_records if r["labels"] == (entail_idx if mode == "A" else 1)][:4]
    _neg = [r for r in train_records if r["labels"] == (1 - entail_idx if mode == "A" else 0)][:4]
    _batch_records = _pos + _neg
    _batch_in = _collator([{k: v for k, v in r.items() if k != "snapshot_id"}
                           for r in _batch_records])
    _batch_in = {k: v.to(device) for k, v in _batch_in.items()}
    print(f"Batch shapes: input_ids={_batch_in['input_ids'].shape} "
          f"labels={_batch_in['labels'].tolist()}")
    model.eval()
    with torch.no_grad():
        _out = model(**_batch_in)
        print(f"Logits[0:3]: {_out.logits[:3].cpu().tolist()}")
        print(f"Per-example CE losses:")
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        _losses = loss_fn(_out.logits, _batch_in["labels"]).cpu().tolist()
        for i, (r, l) in enumerate(zip(_batch_records, _losses)):
            print(f"  snapshot_id={r['snapshot_id']} label={r['labels']} loss={l:.4f}")
        # Flip labels, recompute loss — if labels matter, loss changes
        flipped = 1 - _batch_in["labels"]
        _losses_flipped = loss_fn(_out.logits, flipped).cpu().tolist()
        print(f"Mean loss (real labels): {sum(_losses)/len(_losses):.4f}")
        print(f"Mean loss (flipped):     {sum(_losses_flipped)/len(_losses_flipped):.4f}")
        print(f"=> labels DO affect loss (values differ): "
              f"{abs(sum(_losses)-sum(_losses_flipped))>1e-3}")
    model.train()
    print("=== END SANITY CHECK ===\n")

    # --- Collator (dynamic padding) ---
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer, padding="longest")

    # --- Output dir ---
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_DIR / f"{timestamp}_finetune_{cfg['slug']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    # warmup_ratio is silently dropped in newer transformers — use warmup_steps.
    # BF16 is stable here (same dynamic range as FP32, just less precision).
    total_steps = (len(train_posts) // batch_size) * epochs
    warmup_steps = max(100, total_steps // 10)
    if fp32:
        use_bf16 = False
        use_fp16 = False
        precision = "fp32"
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
        precision = "bf16"
    elif torch.cuda.is_available():
        use_bf16 = False
        use_fp16 = True
        precision = "fp16"
    else:
        use_bf16 = False
        use_fp16 = False
        precision = "fp32 (cpu)"

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
    )
    print(f"Training config: lr={lr} warmup_steps={warmup_steps}/{total_steps} "
          f"grad_clip=1.0 precision={precision}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    t_start = time.monotonic()
    trainer.train()
    t_train = time.monotonic() - t_start
    print(f"Training done in {t_train:.1f}s")

    # --- Predict on both sets ---
    t_pred_start = time.monotonic()
    test_preds = predict(model, tokenizer, test_posts, mode, entail_idx, device, batch_size * 2)
    train_preds = predict(model, tokenizer, train_posts, mode, entail_idx, device, batch_size * 2)
    t_pred = time.monotonic() - t_pred_start

    # --- Score ---
    test_metrics = score_binary(test_preds, test_posts)
    train_metrics = score_binary(train_preds, train_posts)

    print("\nTest metrics:")
    print(f"  P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}  "
          f"F1={test_metrics['f1']:.3f}  Acc={test_metrics['accuracy']:.3f}")
    print("Train metrics (overfit sanity check):")
    print(f"  P={train_metrics['precision']:.3f}  R={train_metrics['recall']:.3f}  "
          f"F1={train_metrics['f1']:.3f}  Acc={train_metrics['accuracy']:.3f}")

    # --- Save ---
    with open(run_dir / "predictions_test.jsonl", "w") as f:
        for r in test_preds:
            f.write(json.dumps(r) + "\n")
    with open(run_dir / "predictions_train.jsonl", "w") as f:
        for r in train_preds:
            f.write(json.dumps(r) + "\n")

    metadata = {
        "timestamp": timestamp,
        "mode": mode,
        "mode_name": cfg["name"],
        "base_model": cfg["model"],
        "num_labels": cfg["num_labels"],
        "hypothesis": HYPOTHESIS if mode == "A" else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_tokens": MAX_TOKENS,
        "max_premise_chars": MAX_PREMISE_CHARS,
        "seed": SEED,
        "device": device,
        "bf16": use_bf16,
        "train_size": len(train_posts),
        "test_size": len(test_posts),
        "train_seconds": round(t_train, 1),
        "predict_seconds": round(t_pred, 1),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if save_model:
        model.save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "model")
        print(f"Saved model weights to {run_dir/'model'}")
    else:
        import shutil
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    print(f"\nSaved to {run_dir}/")
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="DeBERTa-large is unstable early; 5e-6 is the safe default. "
                             "Try 1e-5 if underfitting after stable training.")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--fp32", action="store_true",
                        help="Force FP32 training (avoids BF16 precision loss at low LR)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Truncate train set for smoke testing (e.g. --limit 30)")
    args = parser.parse_args()

    run(args.mode, args.epochs, args.batch_size, args.lr,
        args.save_model, args.limit, fp32=args.fp32)


if __name__ == "__main__":
    main()
