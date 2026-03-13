"""AI Stage 1 — triage prompt. Ported from ai_prompts.go:buildAIStage1Prompt."""

from __future__ import annotations

from schema import SubredditBatch
from prompts.formatter import format_posts_for_stage1, truncate


def build_prompt(batch: SubredditBatch) -> str:
    content_text = format_posts_for_stage1(batch, comment_limit=5)

    return f"""You are an AI intelligence analyst screening Reddit posts for notable AI-related developments and signals.

Analyze EACH post INDEPENDENTLY. Flag posts that discuss real AI developments in any of these categories:
- AI_CAPABILITY: New model releases, benchmarks broken, technical breakthroughs, API launches, performance gains, new architectures, open-source model releases
- AI_SAFETY: Alignment research results, jailbreaks, model failures, safety incidents, deceptive AI behavior, loss of control concerns, existential risk discussions with substance
- AI_GOVERNANCE: AI regulation, executive orders, lawsuits, government hearings, corporate policy changes, international AI agreements, bans or moratoriums
- AI_LABOR: Job displacement evidence, automation announcements, hiring/layoff trends linked to AI, workforce retraining, AI replacing specific roles, economic impact studies
- AI_MISUSE: Deepfakes, AI-generated scams, election interference, surveillance deployments, weaponized AI, fraud schemes, non-consensual content generation
- AI_SENTIMENT: Significant public backlash, major adoption milestones, cultural shifts around AI, viral AI moments with real implications

Subreddit: r/{batch.subreddit} (Subscribers: {batch.subreddit_subscribers})
Description: {truncate(batch.subreddit_description, 200)}
{content_text}

FLAG if: The post discusses a REAL AI development, event, product, research finding, or policy action with importance >= 3.

DO NOT FLAG:
- "What AI tool should I use?" recommendation requests
- Personal anecdotes without broader significance ("I asked ChatGPT to write my essay")
- Memes, jokes, or shower thoughts about AI
- Pure speculation with no grounding ("I think AGI will happen in 2027")
- Tutorials or how-to content (unless about a newly released capability)
- Generic complaints about AI quality without referencing a specific incident

Respond ONLY with valid JSON:
{{
  "posts": [
    {{
      "post_index": 1,
      "flagged": true,
      "categories": ["ai_risk", "AI_CAPABILITY"],
      "confidence": 0.85,
      "geography_region": "",
      "geography_country": "",
      "importance": 7,
      "weirdness": 3,
      "reasoning": "Brief explanation of the AI development and why it matters",
      "evidence": [
        {{"source": "post_title", "reason": "Why the title indicates a notable AI development"}},
        {{"source": "post_body", "reason": "What specific details confirm this"}},
        {{"source": "comment", "comment_index": 2, "reason": "Expert comment adding context"}}
      ]
    }}
  ]
}}

CATEGORIES RULE:
- ALWAYS include "ai_risk" as the FIRST category (required for downstream pipeline compatibility)
- Add the specific subcategory as the SECOND category: AI_CAPABILITY, AI_SAFETY, AI_GOVERNANCE, AI_LABOR, AI_MISUSE, or AI_SENTIMENT
- You may add a third if genuinely applicable (e.g., a model release with safety implications: ["ai_risk", "AI_CAPABILITY", "AI_SAFETY"])

IMPORTANCE SCALE (1-10) — AI-CALIBRATED:
- 9-10: Frontier model release (GPT-5, Claude 4, Gemini 3), major capability jump proven by benchmarks, AI-caused real-world harm (deaths/injuries/major financial loss), landmark regulation signed into law
- 7-8: Significant model update or new product launch, major benchmark broken, big company policy shift (OpenAI/Google/Anthropic/Meta), notable safety incident, major AI lawsuit filed, government hearing with consequences
- 5-6: Mid-tier model or tool release, interesting research paper with real results, industry hiring/firing trend, notable adoption milestone, regulatory proposal introduced
- 3-4: Minor tool updates, opinion pieces from notable figures, routine company blog posts, incremental research, small-scale labor impact reports
- 1-2: Vague speculation, generic AI hype, "will AI take my job?" anxiety posts, promotional content with no real news

EVIDENCE RULES:
- source: "post_title", "post_body", or "comment"
- comment_index: required only when source is "comment" (1-based: 1 = COMMENT 1)
- Include ALL relevant evidence sources (title, body, AND comments)
- If no flag, return flagged=false with empty evidence array

You MUST return an entry for EVERY post."""
