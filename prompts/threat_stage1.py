"""Threat Stage 1 — triage prompt. Ported from twostage.go:buildStage1Prompt."""

from __future__ import annotations

from schema import SubredditBatch
from prompts.formatter import format_posts_for_stage1, truncate


def build_prompt(batch: SubredditBatch) -> str:
    content_text = format_posts_for_stage1(batch, comment_limit=5)

    return f"""You are a threat detection system analyzing Reddit posts.

Analyze EACH post INDEPENDENTLY. Each post stands alone.

Detect signals related to:
- CONFLICT: War, violence, military activity, civil unrest
- HEALTH: Disease outbreaks, pandemics, medical crises
- ECONOMIC: Currency collapse, inflation, bank runs, shortages
- POLITICAL: Coups, government instability, oppression
- NATURAL_DISASTER: Earthquakes, floods, fires, climate events
- AI_RISK: AI-induced psychosis, unhealthy AI dependency

Subreddit: r/{batch.subreddit} (Subscribers: {batch.subreddit_subscribers})
Description: {truncate(batch.subreddit_description, 200)}
{content_text}

Focus on ACTUAL EVENTS. Do NOT flag jokes, satire, historical discussions, or hypotheticals.

Respond ONLY with valid JSON:
{{
  "posts": [
    {{
      "post_index": 1,
      "flagged": true,
      "categories": ["conflict"],
      "confidence": 0.85,
      "geography_region": "Eastern Europe",
      "geography_country": "UKR",
      "importance": 7,
      "weirdness": 3,
      "reasoning": "Brief explanation",
      "evidence": [
        {{"source": "post_title", "reason": "Why the title is concerning"}},
        {{"source": "post_body", "reason": "Why the body content is concerning"}},
        {{"source": "comment", "comment_index": 2, "reason": "Why this comment is concerning"}}
      ]
    }}
  ]
}}

GEOGRAPHY RULES (CRITICAL - DO NOT USE SUBREDDIT COUNTRY):
- IGNORE the subreddit name when assigning geography. The subreddit is just where the post was found.
- Use the country AFFECTED by the threat, NOT the subreddit's country
- geography_region: Broad area (e.g., "Eastern Europe", "Middle East", "South America", "Global")
- geography_country: ISO 3166-1 alpha-3 code (UKR, USA, CHN, GBR, SYR, etc.)
- Example: Syria conflict discussed in r/europe → "SYR" (NOT any European country)
- Example: US tariffs discussed in r/spain → "USA" (NOT "ESP")
- Example: EU-wide regulation in r/france → "" with region "Europe" (NOT "FRA")
- Example: Global AI development in r/germany → "" with region "Global" (NOT "DEU")
- Leave BOTH empty "" if threat is global or not geographically specific
- When in doubt, prefer "" (empty) over guessing a country from the subreddit name

IMPORTANCE SCALE (1-10) - BE STRICT, ERR LOW:
- 9-10: Mass casualties (>100 deaths), nuclear powers in direct conflict, pandemic with high mortality
  Example: "NATO troops engage Russian forces" = 9-10
  Example: "New pathogen with 30% mortality spreading" = 9-10
- 7-8: Major regional conflict (50-100 dead), disease outbreak affecting thousands, major terrorist attack
  Example: "Artillery strikes kill 60 in border city" = 7-8
  Example: "Hospital overwhelmed with 500 cholera cases" = 7-8
- 5-6: Localized violence with multiple casualties (5-20 dead), major civil unrest with injuries
  Example: "Bombing kills 12 at market" = 5-6
  Example: "Riots leave 8 dead, citywide curfew" = 5-6
- 3-4: Single death incidents, small protests, routine crime, local policy changes, government layoffs
  Example: "Man dies in police custody" = 3-4 (tragic but single death)
  Example: "City lays off staff after budget cuts" = 3 (routine government action)
  Example: "Man shot at gym, police respond" = 3-4 (single shooting)
  Example: "Clinic announces closure" = 3-4 (local healthcare issue)
- 1-2: Speculation, rumors, questions, complaints about services, single anecdotes
  Example: "I heard there might be protests" = 2
  Example: "Is there more gun violence lately?" = 2 (question, not event)

EVIDENCE RULES:
- source: "post_title", "post_body", or "comment"
- comment_index: required only when source is "comment" (1-based: 1 = COMMENT 1)
- Include ALL relevant evidence sources (title, body, AND comments)
- If no threat, return flagged=false with empty evidence array

You MUST return an entry for EVERY post."""
