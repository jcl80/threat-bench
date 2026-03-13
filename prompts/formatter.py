"""Shared post formatting logic — mirrors the Go content_text builder."""

from __future__ import annotations

from schema import Comment, SubredditBatch


def truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _format_comments(comments: list[Comment], limit: int) -> str:
    """Format flat comments (with depth field) into indented text."""
    text = ""
    counter = 0

    for comment in comments:
        if counter >= limit:
            break
        counter += 1
        indent = "  " * (comment.depth + 1)
        arrow = "\u21b3 " if comment.depth > 0 else ""
        text += f"{indent}{arrow}COMMENT {counter}: [{comment.score}\u2191] {comment.author}: {comment.body}\n"

    return text


def format_posts_for_stage1(batch: SubredditBatch, comment_limit: int = 5) -> str:
    """Format all posts in a batch for Stage 1 analysis."""
    text = "\n\nPOSTS TO ANALYZE:\n"
    for i, post in enumerate(batch.posts):
        text += f"\n=== POST {i + 1} ===\n"
        text += f"Title: {post.title}\n"
        if post.body:
            text += f"Body: {post.body}\n"
        text += f"Score: {post.score} | Comments: {post.num_comments}\n"

        if post.comments:
            text += "\nComments:\n"
            text += _format_comments(post.comments, limit=comment_limit)

    return text


def format_flagged_posts_for_stage2(
    batch: SubredditBatch,
    flagged_posts: list[dict],
    comment_limit: int = 10,
) -> tuple[str, str]:
    """Format flagged posts for Stage 2 verification.

    Returns (summary_text, content_text).
    """
    summary = "\n\nDETECTED THREATS FROM INITIAL SCREENING:\n"
    for fp in flagged_posts:
        summary += f"\n--- Post {fp['post_index']}: {fp.get('post_title', '')} ---\n"
        summary += f"Categories: {fp['categories']} | Importance: {fp['importance']} | Confidence: {fp['confidence']:.2f}\n"
        if fp.get("geography_country"):
            summary += f"Geography: {fp.get('geography_region', '')} ({fp['geography_country']})\n"
        summary += f"Reasoning: {fp['reasoning']}\n"

    content = "\n\nFLAGGED POSTS - FULL CONTEXT:\n"
    for fp in flagged_posts:
        idx = fp["post_index"]
        if idx < 1 or idx > len(batch.posts):
            continue
        post = batch.posts[idx - 1]

        content += f"\n=== POST {idx} ===\n"
        content += f"Title: {post.title}\n"
        content += f"Author: {post.author} | Score: {post.score} | Comments: {post.num_comments}\n"
        if post.body:
            content += f"Body:\n{post.body}\n"

        if post.comments:
            content += "\nComments:\n"
            content += _format_comments(post.comments, limit=comment_limit)

    return summary, content
