"""Pydantic models matching the Go output structs from pkg/analyzer/."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Threat categories ---

class ThreatCategory(str, Enum):
    CONFLICT = "conflict"
    HEALTH = "health"
    ECONOMIC = "economic"
    POLITICAL = "political"
    NATURAL_DISASTER = "natural_disaster"
    AI_RISK = "ai_risk"


class AICategory(str, Enum):
    AI_CAPABILITY = "ai_capability"
    AI_SAFETY = "ai_safety"
    AI_GOVERNANCE = "ai_governance"
    AI_LABOR = "ai_labor"
    AI_MISUSE = "ai_misuse"
    AI_SENTIMENT = "ai_sentiment"


# --- Evidence (model output) ---

class Evidence(BaseModel):
    source: str  # "post_title", "post_body", "comment"
    comment_index: Optional[int] = None
    reason: str


# --- Post analysis (output of a single post) ---

class PostAnalysis(BaseModel):
    post_index: int
    flagged: bool
    categories: list[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    severity_score: Optional[int] = Field(default=None, ge=0, le=10)
    geography_region: Optional[str] = None
    geography_country: Optional[str] = None  # ISO 3166-1 alpha-3
    importance: int = Field(default=0, ge=0, le=10)
    weirdness: int = Field(default=0, ge=0, le=10)
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


# --- Top-level response ---

class StageResponse(BaseModel):
    """The top-level object returned by every prompt."""
    posts: list[PostAnalysis]


# --- Input data models ---

class Comment(BaseModel):
    author: str
    body: str
    score: int
    depth: int = 0
    replies: list[Comment] = Field(default_factory=list)


class InputPost(BaseModel):
    snapshot_id: int
    reddit_id: str
    title: str
    body: str
    author: str
    score: int
    num_comments: int


class BenchRow(BaseModel):
    """One line in bench_data.jsonl — a single post with its comments."""
    subreddit: str
    subreddit_subscribers: int
    subreddit_description: str
    post: InputPost
    comments: list[Comment] = Field(default_factory=list)


class Post(BaseModel):
    """A post with comments, used in SubredditBatch for prompt building."""
    snapshot_id: int
    reddit_id: str
    title: str
    body: str
    author: str
    score: int
    num_comments: int
    comments: list[Comment] = Field(default_factory=list)


class SubredditBatch(BaseModel):
    """A batch of posts from one subreddit, used as input to prompt builders."""
    subreddit: str
    subreddit_subscribers: int
    subreddit_description: str
    posts: list[Post]


# --- Baseline models (Go pipeline output) ---

class BaselineEvidence(BaseModel):
    reason: str
    source: str
    post_url: Optional[str] = None
    post_title: Optional[str] = None
    post_body: Optional[str] = None
    post_snapshot_id: Optional[int] = None
    comment_url: Optional[str] = None
    comment_text: Optional[str] = None
    comment_index: Optional[int] = None
    comment_snapshot_id: Optional[int] = None


class BaselineStage(BaseModel):
    model_config = {"extra": "ignore"}

    confidence: float = 0.0
    reasoning: str = ""
    model: Optional[str] = None
    cost_usd: float = 0.0
    evidence: list[BaselineEvidence] = Field(default_factory=list)
    verified: Optional[bool] = None  # stage2 only


class BaselineAnalysis(BaseModel):
    """One line in baseline.jsonl — a production pipeline analysis."""
    model_config = {"extra": "ignore"}

    analysis_id: int
    subreddit: str
    post_snapshot_ids: list[int]
    final_status: str
    stage1: BaselineStage
    stage2: BaselineStage
    threat_categories: list[str] = Field(default_factory=list)
    severity_score: Optional[int] = None
    importance: Optional[int] = None
    weirdness: Optional[int] = None
    geography_region: Optional[str] = None
    geography_country: Optional[str] = None
    analyzed_at: Optional[str] = None
