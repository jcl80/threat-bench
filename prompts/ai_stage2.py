"""AI Stage 2 — verification prompt. Ported from ai_prompts.go:buildAIStage2Prompt."""

from __future__ import annotations

from schema import SubredditBatch
from prompts.formatter import format_flagged_posts_for_stage2


def build_prompt(batch: SubredditBatch, flagged_posts: list[dict]) -> str:
    signal_summary, content_text = format_flagged_posts_for_stage2(
        batch, flagged_posts, comment_limit=10,
    )

    # Stage 2 AI uses a different summary header
    signal_summary = signal_summary.replace(
        "DETECTED THREATS FROM INITIAL SCREENING:",
        "FLAGGED AI SIGNALS FROM INITIAL SCREENING:",
    )

    return f"""You are verifying AI intelligence signals flagged by an initial screening.

For each flagged post, determine: Is this a REAL, NOTABLE AI development worth including in a daily intelligence brief?
{signal_summary}
{content_text}

Subreddit: r/{batch.subreddit}

VERIFICATION CRITERIA — VERIFY if:
1. References a SPECIFIC product, company, model, research paper, or policy action (not vague "AI is changing everything")
2. Describes something that ACTUALLY HAPPENED or was OFFICIALLY ANNOUNCED (not rumor or speculation)
3. Has SUBSTANCE — the post body or comments provide real details, numbers, names, or first-hand accounts
4. Is CURRENT — a new development, not rehashing old news without new information

REJECT if:
- Pure speculation or opinion with no news peg ("I think AI will...")
- Memes, jokes, or satire without a real development underneath
- Personal anecdotes without broader signal value ("I tried the new model and it was meh")
- Promotional or marketing content with no real news
- Vague anxiety or hype with no specific referent

SEVERITY SCORING — SCORE THE DEVELOPMENT, NOT THE POST:
- A new frontier model release should be 8-10 even if discussed in a casual one-line post
- A real safety incident should be 7-10 regardless of how sensationally it's framed on Reddit
- A regulatory action should be scored by its actual scope and impact, not Reddit's reaction
- Score based on the REAL-WORLD SIGNIFICANCE of the AI development, not the quality of discussion
- When in doubt, score HIGHER — it's better to surface a signal than miss it

Respond ONLY with valid JSON:
{{
  "posts": [
    {{
      "post_index": 7,
      "flagged": true,
      "categories": ["ai_risk", "AI_CAPABILITY"],
      "confidence": 0.9,
      "severity_score": 8,
      "geography_region": "",
      "geography_country": "",
      "importance": 8,
      "weirdness": 2,
      "reasoning": "What the AI development is and why it's significant",
      "evidence": [
        {{"source": "post_title", "reason": "Confirms a specific AI product/event"}},
        {{"source": "comment", "comment_index": 3, "reason": "Expert adds context on implications"}}
      ]
    }}
  ]
}}

CRITICAL: post_index MUST match the exact POST number shown above (e.g., if you see "POST 7", return post_index: 7, NOT 1).

CATEGORIES RULE:
- ALWAYS include "ai_risk" as the FIRST category
- Add the specific subcategory: AI_CAPABILITY, AI_SAFETY, AI_GOVERNANCE, AI_LABOR, AI_MISUSE, or AI_SENTIMENT
- Preserve or correct the subcategory from Stage 1 if it was wrong

GEOGRAPHY RULES:
- geography_region: Broad area if geographically specific (e.g., "North America", "Europe", "Global")
- geography_country: ISO 3166-1 alpha-3 code if country-specific
- Most AI developments are "Global" — only set specific geography for local regulation, country-specific deployment, etc.
- Leave empty "" if not geographically specific

EVIDENCE RULES:
- source: "post_title", "post_body", or "comment"
- comment_index: required only when source is "comment" (1-based)
- severity_score: 1-10 (only if flagged=true, else 0)
- If flagged=false, evidence can be empty

You MUST return an entry for EVERY flagged post using their ORIGINAL post numbers."""
