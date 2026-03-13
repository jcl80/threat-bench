"""Threat Stage 2 — verification prompt. Ported from twostage.go:buildStage2Prompt."""

from __future__ import annotations

from schema import SubredditBatch
from prompts.formatter import format_flagged_posts_for_stage2


def build_prompt(batch: SubredditBatch, flagged_posts: list[dict]) -> str:
    threat_summary, content_text = format_flagged_posts_for_stage2(
        batch, flagged_posts, comment_limit=10,
    )

    return f"""You are a verification system fact-checking threat detections.

Verify EACH flagged post INDEPENDENTLY.
{threat_summary}
{content_text}

Subreddit: r/{batch.subreddit}

VERIFICATION CRITERIA:
1. Events are CONCRETE and CURRENT (not historical/hypothetical)
2. MULTIPLE independent mentions suggest reality
3. Content shows GENUINE concern (not jokes/memes)
4. Specific DETAILS present (locations, numbers, names)

REJECT if: satire/jokes, historical, single unverified claim

Respond ONLY with valid JSON:
{{
  "posts": [
    {{
      "post_index": 7,
      "flagged": true,
      "categories": ["conflict"],
      "confidence": 0.9,
      "severity_score": 7,
      "geography_region": "Western Europe",
      "geography_country": "GBR",
      "importance": 8,
      "weirdness": 2,
      "reasoning": "Verification explanation",
      "evidence": [
        {{"source": "post_title", "reason": "Why title confirms threat"}},
        {{"source": "comment", "comment_index": 3, "reason": "Why this comment confirms"}}
      ]
    }}
  ]
}}

CRITICAL: post_index MUST match the exact POST number shown above (e.g., if you see "POST 7", return post_index: 7, NOT 1).

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

SEVERITY/IMPORTANCE SCALE (1-10) - BE STRICT, ERR LOW:
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
- comment_index: required only when source is "comment" (1-based)
- severity_score: 1-10 (only if flagged=true, else 0)
- If flagged=false, evidence can be empty

You MUST return an entry for EVERY flagged post using their ORIGINAL post numbers."""
