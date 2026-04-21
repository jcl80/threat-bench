# Research log — DeBERTa zero-shot as Stage 1 triage

Chronological record of the zero-shot DeBERTa experiments against the 2,385-post
threat-bench baseline. Each section records what was done, what the numbers
showed, and (where relevant) where earlier claims in the session were wrong.

**Ground-truth note:** `label_gpt5_mini` and `label_gpt5` in `bench/data/posts.jsonl`
are each model's independent Stage-1 flag decisions (not Stage-2 verification).
Comparisons to `gpt5` measure "does DeBERTa agree with gpt-5's Stage-1 judgement?"
not "is this truly a threat?" That ambiguity is unresolved — see open questions.

---

## 1. Setup

- Model: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
- Hardware: single RTX 4090 on Vast.ai (~14s per 2,385-post pass)
- Hypothesis: *"This text describes a real-world threat, crisis, or dangerous event
  such as armed conflict, disease outbreak, economic collapse, political
  instability, natural disaster, or AI risk."*
- Per-post premise = `title + body + top-3 comments`, joined with `\n`, truncated
  to 1500 characters.

**What production sends (for reference, from `prompts/threat_stage1.py`):**
- Subreddit name + description
- Per post: title + full body (no cap) + top-5 comments (with scores/authors/indent)
- Detailed threat taxonomy, severity rubric, geography rules
- Multiple posts per call (batch)

DeBERTa gets far less context and no cross-post batch.

---

## 2. Run 1 — default threshold 0.5 (commit `559ee81`)

Predictions dir: `bench/results/2026-04-18T23-13-31_MoritzLaurer_deberta-v3-large-zeroshot-v2.0/`

- Speed: 198 posts/sec, 12 s end-to-end
- Flagged: 1072 / 2385 (45.0%)

| Ground truth | P     | R     | F1    |
|--------------|-------|-------|-------|
| gpt5_mini    | 0.980 | 0.553 | 0.707 |
| gpt5         | 0.865 | 0.706 | 0.777 |

High precision, low recall. Default threshold is too conservative.

---

## 3. Threshold sweep on Run 1 predictions

Scores were saved per post, so sweeping is a re-threshold operation — no
re-inference needed (`bench/sweep_threshold.py`).

**Sweep vs gpt5 (relevant slice, Run 1):**

| thr  | P     | R     | F1    | flags |
|------|-------|-------|-------|-------|
| 0.01 | 0.634 | 0.975 | 0.769 | 2019  |
| 0.03 | 0.699 | 0.937 | 0.801 | 1760  |
| 0.05 | 0.732 | 0.916 | 0.814 | 1644  |
| 0.07 | 0.750 | 0.887 | 0.813 | 1555  |
| 0.09 | 0.768 | 0.874 | 0.818 | 1494  |
| 0.10 | 0.775 | 0.862 | 0.816 | 1460  |
| 0.50 | 0.865 | 0.706 | 0.777 | 1072  |

Best F1 vs gpt5: **0.818 at threshold 0.09** (P=0.768, R=0.874).

**Early claim I made that was wrong:** I first swept only 0.10–0.90 and reported
the optimal point as thr=0.10 with R=0.86, then told you DeBERTa's recall
"ceiling" was 0.86. It wasn't — the sweep just didn't go low enough.

---

## 4. Data bug — swapped fields

You flagged that gpt models might be seeing more than DeBERTa. Investigating
showed: `data/bench_data.jsonl` has **swapped** `author`/`body` fields in comments.
The actual comment **text** is stored under `"author"`, the username is stored
under `"body"`.

- `prepare_data.py` was extracting `c["body"]` → got usernames (like
  `"Fearless_Ad_7379"`), not comment text. So DeBERTa's 3 "comments" were author
  strings — pure noise.
- `prompts/formatter.py` renders both fields (`COMMENT n: [score↑] {author}: {body}`),
  so gpt5/gpt5-mini saw the real comment text — just with weird labelling
  (username shown as "body"). Production had the signal; DeBERTa did not.

**Fix:** patched `prepare_data.py` to read from `"author"` (where the text lives
in this corrupted data).

---

## 5. Run 2 — with real comment text

Predictions dir: `bench/results/2026-04-20T18-21-59_MoritzLaurer_deberta-v3-large-zeroshot-v2.0/`

- Speed: 175 posts/sec, 14 s (slower because premises are longer now)
- Flagged: 948 / 2385 (39.8%)

| Ground truth | P     | R     | F1    |
|--------------|-------|-------|-------|
| gpt5_mini    | 0.978 | 0.487 | 0.650 |
| gpt5         | 0.869 | 0.627 | 0.729 |

**Recall went down** relative to Run 1 at the same threshold.

---

## 6. Apples-to-apples threshold comparison, Run 1 vs Run 2

**Vs gpt5 — same thresholds, two data preparations:**

| thr  | Run 1 P | Run 1 R     | Run 1 flags | Run 2 P | Run 2 R | Run 2 flags |
|------|---------|-------------|-------------|---------|---------|-------------|
| 0.01 | 0.634   | **0.975**   | 2019        | 0.709   | 0.932   | 1727        |
| 0.03 | 0.699   | 0.937       | 1760        | 0.765   | 0.855   | 1468        |
| 0.05 | 0.732   | 0.916       | 1644        | 0.789   | 0.822   | 1369        |
| 0.09 | 0.768   | 0.874       | 1494        | 0.815   | 0.766   | 1236        |

**Adding real comment text lowered recall at every threshold**, while slightly
raising precision at the low-threshold end. Best F1 vs gpt5:
- Run 1: 0.818 at thr 0.09
- Run 2: 0.807 at thr 0.03

**Run 1 (no-comment-text) is the stronger DeBERTa configuration.**

---

## 7. Why comments hurt — truncation is *not* the cause

First hypothesis: the 1500-char cap was truncating body/title to fit comments.
Verified directly:

- 12% of posts exceed the 1500-char cap.
- Of those truncated, **88%** (249/284) retained full title+body and lost only
  comment text.
- Only **1.5% of all posts** (35/2385) had the body itself clipped.
- title+body median: 78 chars. Rarely approaches the cap.

So title/body survive almost always. Real explanation: adding long off-topic
comment text **dilutes the entailment signal** for a zero-shot NLI classifier —
the CLS token attends to the whole premise, and noise drags the entailment
probability down. This is a known failure mode of zero-shot NLI when irrelevant
text is appended to a relevant core.

---

## 8. Mistakes I made in this session (for the record)

1. **First sweep bottomed out at threshold 0.10** — I reported "best recall 0.86"
   as if it were a ceiling. It wasn't; I hadn't tested lower.
2. **After fixing the comment bug, I said recall "jumped from 0.86 to 0.93"** —
   wrong, because it was comparing Run 2's thr=0.01 to Run 1's thr=0.10. On
   apples-to-apples comparison, Run 1 has higher recall at every threshold.
3. **I said "comments pushed out title and body"** — wrong. Verified: title+body
   are always prepended first and almost always preserved.
4. **I framed Run 1's thr=0.01 as "DeBERTa flags more than mini, worse on both
   axes"** — true only at that threshold. At any threshold ≥ 0.03, DeBERTa flags
   *fewer* posts than mini.

---

## 9. Current best picture

Using **Run 1 predictions** (no real comments — counterintuitively stronger):

| thr  | Flags | vs mini  | Misses | R vs gpt5 | P vs gpt5 |
|------|-------|----------|--------|-----------|-----------|
| 0.01 | 2019  | **+6.2%**  | 33     | 0.975     | 0.634     |
| 0.03 | 1760  | **−7.5%**  | 83     | 0.937     | 0.699     |
| 0.05 | 1644  | −13.6%   | 110    | 0.916     | 0.732     |
| 0.09 | 1494  | −21.5%   | 166    | 0.874     | 0.768     |

Baseline for comparison:
- **mini** (current prod Stage 1): 1902 flags, 0 misses of gpt5-positive posts
  (definitionally, since `label_gpt5_mini` is mini's own output).
- **Mini's precision vs gpt5:** 1314 / 1902 = **0.691**. So at thr ≥ 0.05, DeBERTa
  is **more precise than mini** when judged against gpt5's flag set.

### Cost intuition at thr 0.03 (Run 1)
- Eliminate all 2385 mini Stage-1 calls (replace with free DeBERTa inference).
- Send 1760 posts to gpt5 Stage 2 (vs mini's 1902 → −7.5%).
- Miss 83 of 1314 gpt5-confirmed threats → **6.3% miss rate**.

Net: stage-1 LLM cost eliminated, stage-2 calls slightly reduced. Product cost:
6.3% recall loss on gpt5's opinion.

### Cost intuition at thr 0.01 (Run 1)
- Flag 2019 posts → 738 FP (gpt5 rejects) + 1281 TP → miss 33 of 1314.
- Sends **6.2% more** to stage 2 than mini — slight cost *increase* at stage 2
  but still net savings because mini Stage-1 is eliminated.
- Miss rate: 2.5%.

---

## 10. Open questions

1. **Is gpt5's Stage 1 the right ground truth?** `label_gpt5` is gpt5's own
   Stage-1 output, not a downstream human-verified truth. Some r/ukraine
   fundraiser / satellite posts appear in the "confirmed" set that look like
   noise on inspection. Suggest: sample 30 `label_gpt5=1` and 30 `label_gpt5=0`
   posts, human-adjudicate, see if gpt5 is over/under-classifying.

2. **Would a narrower hypothesis improve zero-shot?** The current 6-clause
   hypothesis maps loosely to gpt5's threat categories; missed FNs concentrated
   in r/technology (tech harms, privacy) and r/Economics (trade wars, supply
   shocks) — not classic "war/disease/disaster" vocabulary. Multi-hypothesis
   max-pool might help, but zero-shot performance is capped around R≈0.97 at
   very low precision. Diminishing returns.

3. **Does SetFit do better?** The 2385 labeled posts are more than enough for
   few-shot SetFit. Train on `label_gpt5_mini` (or `label_gpt5`), test on
   held-out subreddits or stratified folds. This is the honest next experiment
   — a classifier that adapts to the actual distribution rather than a generic
   entailment hypothesis.

---

## 11. Infra / data notes

- `bench_data.jsonl` has a swapped-fields bug; `prepare_data.py` compensates
  with a note. If upstream is fixed, revert the patch.
- Vast.ai default templates pre-load vLLM, eating ~22 GB VRAM. Kill the PID
  from `nvidia-smi` before running.
- Results are portable: `predictions.jsonl` stores per-post scores so any new
  threshold experiment is a seconds-long re-score, not a GPU run.

---

## 12. Decision point before SetFit

If you read only one row: **Run 1 DeBERTa at threshold 0.03** gets **R=0.94
vs gpt5, P=0.70, sending 7.5% fewer posts to Stage 2 than mini**. That's a
defensible candidate for replacing mini *if* the product can tolerate a ~6%
miss rate on gpt5's call.

If you can't, SetFit (next).

---

## 13. Fine-tune ablation (frozen-backbone linear probe)

After a long detour through collapsed/NaN full fine-tune runs (`adam_epsilon`
default 1e-8 was pushing DeBERTa-v3-large into early weight explosions
regardless of LR), we pivoted to frozen-backbone linear probes on top of
pre-trained feature extractors. Same stratified 80/20 split
(seed=42, on `label_gpt5`), same preprocessing, same hyperparameters across
three configs:

| Mode | Backbone | Input format | Test F1 |
|------|----------|--------------|---------|
| A | MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | (post, hypothesis) pair | **0.825** |
| B | MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | post alone | 0.699 |
| C | microsoft/deberta-v3-large | post alone (pooler + head trainable) | 0.719 |

### 13.1 Why Mode B collapsed — structural, not optimization

Mode B (post alone, no hypothesis) collapsed to the class prior (F1=0.70,
P=0.56, R=0.92) while Mode A (post + hypothesis, NLI-faithful format) trained
cleanly to F1=0.825 under identical configs. The gap is structural, not a
matter of optimization. The backbone
(`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) has been task-adapted on
large-scale NLI: its `[CLS]` token is trained as a relational summary scoring
whether a hypothesis follows from a premise, and its disentangled-attention
patterns are specialized for cross-segment aggregation. Mode A activates
that machinery with a fixed hypothesis —

> *"This text describes a real-world threat, crisis, or dangerous event such
> as armed conflict, disease outbreak, economic collapse, political
> instability, natural disaster, or AI risk."*

— placing examples on the entailment axis the backbone already discriminates
along; a linear probe then just reweights features the backbone has organized
for it. Mode B removes the second segment, leaving the cross-segment heads
with nothing to attend across and yielding a generic `[CLS]` byproduct rather
than a discriminative one. No linear boundary with adequate margin exists in
that feature space, so the probe settles at the next-best minimum — uniform
prediction, loss ≈ log(2).

### 13.2 Mode C result — isolates NLI adaptation absent the NLI format

**Mode C (microsoft/deberta-v3-large, hypothesis-free, pooler trainable):
F1=0.719.**

Within noise of Mode B (0.699). This isolates the contribution of NLI
adaptation to the backbone *independent of hypothesis format*: none.
MoritzLaurer's NLI pre-training adds value only when paired with the NLI
input format — it doesn't make the `[CLS]` representations for standalone
text more threat-discriminative.

Full attribution of the gap between zero-shot and a trained classifier:
- ~+0.12 F1 from the hypothesis input (A vs B/C)
- ~0 from backbone NLI pre-training absent that format (B vs C)
- ~0 from fine-tuning on top of the NLI head (probe ties zero-shot threshold-
  swept: 0.825 ≈ 0.830)

**Methodological note on Mode C:** initial Mode C runs had a pooler bug.
`microsoft/deberta-v3-large` ships *without* a classification pooler, so
HuggingFace random-initialized it. `--freeze-backbone` then froze a random
pooler along with the backbone, destroying the signal path. First Mode C
result was F1=0.713 on random features. After auto-detecting and unfreezing
newly-initialized tensors, F1 moved to 0.719 — marginally better. The fix
is in `bench/runners/finetune_deberta.py` via the `missing_keys` check.

### 13.3 Decision

Stop fine-tuning. The zero-shot MoritzLaurer model at threshold ~0.17
(F1=0.830 on the 478 test posts) is the strongest candidate on this backbone
at this data scale. Nothing trained beats it.

Next: either validate the zero-shot on fresh holdout (3k new posts through
the production pipeline to get gpt5 labels) or try a fundamentally different
feature space (TF-IDF + LR, SetFit contrastive tuning).

---

## 14. TF-IDF + Logistic Regression baseline

Added a lexical baseline — `TfidfVectorizer(ngram=(1,2), min_df=2,
max_features=20000, sublinear_tf=True)` + `LogisticRegression(C=1.0,
class_weight='balanced')`. Trained on the same 1907-post split, scored on
the 478 test.

**Result on 478 test:** F1=0.833 at threshold 0.44.

Same F1 as zero-shot MoritzLaurer (0.830). A bag-of-words model matches a
435M-parameter NLI transformer on this task at this scale, in ~10 seconds
of CPU training. The task has a lot of lexical signal.

## 15. Error-overlap analysis and stacked ensemble

If TF-IDF and zero-shot have disjoint errors, ensembling could push past
either one. Checked this directly:

```
                         tfidf correct    tfidf wrong
  zs correct              328               57
  zs wrong                 53               40
```

- **c/(c+d) = 0.570** — of 93 zero-shot errors, TF-IDF fixes 53.
- Cohen's kappa = 0.538 — only moderate agreement. The two models make
  genuinely different decisions, not correlated ones.
- Of 57 zero-shot false positives, TF-IDF correctly rejects 33 (57.9%).
- Of 36 zero-shot false negatives, TF-IDF correctly catches 20 (55.6%).

Well above the ~15% complementarity threshold. Ensemble was worth building.

### 15.1 Proper stacking with out-of-fold predictions

First pass was a 2-fold meta-LR on the 478 test set. Data-starved. Rebuilt
with proper stacking:

1. 5-fold StratifiedKFold on the 1907 training posts.
2. For each fold, fit TF-IDF + LR on 4 folds, predict on held-out → every
   training post has an OOF TF-IDF probability that didn't see it during
   fit.
3. Zero-shot probabilities are already unbiased (no training) on all posts.
4. Meta-LR fits on 1907 × [zs_prob, tfidf_oof_prob] → label.
5. A final TF-IDF + LR on the full 1907 training set produces probabilities
   for the 478 test posts.
6. Single eval pass on the 478 test, no leakage.

**Result on 478 test:** F1=0.868 at threshold 0.30. Gain over best individual:
**+0.034 F1**. Meta-LR weights: `zs=+2.518, tfidf=+5.532, bias=-3.793`.

Both weights clearly positive → the ensemble uses both signals. TF-IDF
weight is ~2× zero-shot — interpreted either as "TF-IDF carries more
discriminative signal on this distribution" or "OOF TF-IDF probs are
softer than test-time probs and the meta-LR compensates." Either way,
neither signal dominates and both contribute.

## 16. In-distribution holdout validation

Fresh pull from production: 134 analyses, 1686 posts across the same 8
subreddits used in training, all `analyzed_at >= 2026-04-05` (everything
is strictly newer than the training baseline). New pipeline:
`bench/fetch_holdout.py` pulls posts + comments + production analyses
directly from the Sentinel DB; `--per-subreddit` gives balanced
distribution.

Labels regenerated via `runner.py` with both gpt-5-mini and gpt-5 stage 1
on the fresh data, then `bench/prepare_data.py` (now CLI-configurable)
merges into `posts_holdout.jsonl`. TF-IDF refit on the 1907 training split
and applied to the 1686 holdout; DeBERTa zero-shot rerun on Vast; saved
meta-LR weights applied to combine.

**Notable distribution shift:** gpt-5 positive rate 55% (training) → 63%
(holdout). Threat density has drifted up.

### 16.1 Results at each model's shipped threshold

| Model | Training test F1 | Holdout F1 @ shipped thr | ΔF1 |
|-------|------------------|--------------------------|-----|
| Zero-shot (thr 0.17) | 0.833 | 0.814 | **−0.019** |
| TF-IDF + LR (thr 0.44) | 0.833 | 0.829 | **−0.004** |
| **Ensemble (thr 0.30)** | **0.868** | **0.844** | **−0.024** |

### 16.2 Threshold stability

| Model | Shipped thr | Holdout-best thr | Gap (F1) |
|-------|-------------|------------------|----------|
| Zero-shot | 0.17 | 0.05 | +0.018 |
| TF-IDF + LR | 0.44 | 0.43 | +0.006 |
| Ensemble | 0.30 | 0.27 | +0.008 |

Thresholds held. The deployment decision is legitimate.

### 16.3 Important calibration finding

**Ensemble advantage shrank from +0.034 (training test) → +0.015
(holdout).** The +0.034 was partially a specific-test-set artifact. Real
lift over TF-IDF alone is closer to +0.015 F1. Still positive, still worth
shipping on these subreddits, but smaller than the training number
suggested. Lesson for future work: size gains against held-out data, not
cross-validated folds.

## 17. Out-of-distribution holdout

Fresh pull from a deliberately different subreddit set:
- `politics` (adjacent to worldnews, different community)
- `ClaudeAI` (adjacent to technology, AI-specific)
- `depression` (far — mental health)
- `personalfinance` (far — economic-adjacent)

1020 posts, 150 per subreddit target. Positive rate: 31.9% (mini) / **11.8%
(gpt5)** — much lower than Holdout A's 63%, because most of these subs
aren't threat-dense.

### 17.1 Results at each model's shipped threshold

| Model | Training test F1 | OOD F1 @ shipped thr | ΔF1 |
|-------|------------------|----------------------|-----|
| Zero-shot (thr 0.17) | 0.833 | **0.552** | −0.281 |
| TF-IDF + LR (thr 0.44) | 0.833 | 0.452 | −0.381 |
| Ensemble (thr 0.30) | 0.868 | 0.530 | −0.338 |

**Ensemble no longer wins.** Zero-shot is both the best OOD model and the
most robust (smallest F1 drop). TF-IDF is the least robust — vocabulary
effects don't transfer. Part of the absolute F1 drop is arithmetic (lower
positive rate → lower F1 for the same underlying decision quality), but
the ranking reversal (zero-shot > ensemble > TF-IDF) is a real finding.

### 17.2 Per-subreddit flag rates (gpt-5)

| Subreddit | Flag rate |
|-----------|-----------|
| politics | 33.9% |
| depression | 6.7% |
| ClaudeAI | 1.8% |
| personalfinance | 1.0% |

`politics` behaves like in-distribution threat-dense content. The other
three have so few positives (3–17 each) that per-sub metrics are noisy.
Aggregate F1 is dominated by politics.

### 17.3 Threshold instability on OOD

| Model | Shipped thr | OOD-best thr | Gap (F1) |
|-------|-------------|--------------|----------|
| Zero-shot | 0.17 | 0.13 | +0.022 |
| TF-IDF + LR | 0.44 | 0.49 | +0.032 |
| Ensemble | 0.30 | 0.37 | +0.011 |

Zero-shot and TF-IDF boundaries both drifted. Any deployment to new
subreddits should re-sweep thresholds.

## 18. Deployment recommendation

**For the 8 baseline subreddits** (ukraine, worldnews, collapse,
geopolitics, Economics, technology, news, energy), ship the **stacked
ensemble at threshold 0.30**. Expected operating point on fresh
in-distribution data: F1 ≈ 0.84, recall ≈ 0.87, precision ≈ 0.82. Sends
~20% fewer posts to Stage 2 than gpt-5-mini does while catching ~87% of
gpt-5-confirmed threats.

**For subreddits outside this set**, the ensemble is unsafe. Two options:
1. Use zero-shot alone at threshold 0.17, re-sweep per-subreddit if
   resources permit. Expected F1 depends strongly on base rate (0.55 at
   11% positive rate, likely higher on threat-denser subs).
2. Collect 500+ labeled posts on the target subreddit and refit TF-IDF +
   re-run OOF stacking.

**Things to monitor in production:**
- Positive rate drifts over time (55% → 63% in ~1 month). Track and
  occasionally re-sweep thresholds.
- Ensemble's precision is more sensitive to base rate than zero-shot's.
- TF-IDF is vocabulary-dependent; any significant vocabulary shift
  (new named entities, new topics) degrades it.

## 19. Assets

Scripts (all in `bench/`):
- `fetch_holdout.py` — production DB → fresh baseline + bench data
- `prepare_data.py` — merge runner outputs into posts.jsonl (CLI-configurable)
- `runners/nli_deberta.py` — zero-shot NLI scoring
- `runners/tfidf_baseline.py` — TF-IDF + LR baseline
- `runners/error_overlap.py` — model disagreement analysis
- `runners/ensemble.py` — 2-fold meta-LR (superseded)
- `runners/stacked_ensemble.py` — proper 5-fold OOF stacking
- `runners/finetune_deberta.py` — frozen-backbone / linear-probe fine-tune
- `runners/validate_holdout.py` — single-shot evaluation on any held-out set
- `compare_probe_vs_zeroshot.py` — head-to-head threshold sweep

Meta-LR weights (production): `zs=+2.518, tfidf=+5.532, bias=-3.793`.
Apply as `sigmoid(w_zs*zs + w_tf*tfidf + bias)`.
