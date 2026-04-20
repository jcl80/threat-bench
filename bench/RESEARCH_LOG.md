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
