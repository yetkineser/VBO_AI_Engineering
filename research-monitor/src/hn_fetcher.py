"""
Fetch top AI/ML stories from Hacker News using the official API.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import httpx

from .config import TOPICS, settings

logger = logging.getLogger(__name__)

HN_API = "https://hacker-news.firebaseio.com/v0"


@dataclass
class HNStory:
    title: str
    url: str
    hn_url: str
    score: int
    comments: int
    author: str
    time: datetime
    matched_topics: list[str]
    matched_keywords: list[str]


def _match_topics(text: str) -> tuple[list[str], list[str]]:
    text_lower = text.lower()
    matched_topics = []
    matched_keywords = []

    general_ai_keywords = [
        "llm", "gpt", "claude", "transformer", "neural network",
        "machine learning", "deep learning", "ai agent", "rag",
        "fine-tuning", "embedding", "vector database",
        "openai", "anthropic", "hugging face",
    ]

    for topic_key, topic_cfg in TOPICS.items():
        for kw in topic_cfg.get("rss_keywords", []):
            if kw.lower() in text_lower:
                if topic_key not in matched_topics:
                    matched_topics.append(topic_key)
                if kw not in matched_keywords:
                    matched_keywords.append(kw)

    for kw in general_ai_keywords:
        if kw in text_lower and kw not in matched_keywords:
            if "general_ai" not in matched_topics:
                matched_topics.append("general_ai")
            matched_keywords.append(kw)

    return matched_topics, matched_keywords


def fetch_hn_stories(min_score: int = 20) -> list[HNStory]:
    """Fetch top HN stories and filter for AI/ML relevance."""
    stories: list[HNStory] = []

    with httpx.Client(timeout=15.0) as client:
        try:
            resp = client.get(f"{HN_API}/topstories.json")
            resp.raise_for_status()
            top_ids = resp.json()[:100]
        except Exception:
            logger.exception("Failed to fetch HN top stories")
            return []

        for story_id in top_ids:
            if len(stories) >= settings.hn_max_stories:
                break
            try:
                resp = client.get(f"{HN_API}/item/{story_id}.json")
                resp.raise_for_status()
                item = resp.json()

                if not item or item.get("type") != "story":
                    continue

                score = item.get("score", 0)
                if score < min_score:
                    continue

                title = item.get("title", "")
                url = item.get("url", "")
                searchable = f"{title} {url}"

                matched_topics, matched_keywords = _match_topics(searchable)
                if not matched_topics:
                    continue

                story = HNStory(
                    title=title,
                    url=url or f"https://news.ycombinator.com/item?id={story_id}",
                    hn_url=f"https://news.ycombinator.com/item?id={story_id}",
                    score=score,
                    comments=item.get("descendants", 0),
                    author=item.get("by", ""),
                    time=datetime.fromtimestamp(item.get("time", 0), tz=timezone.utc),
                    matched_topics=matched_topics,
                    matched_keywords=matched_keywords,
                )
                stories.append(story)

            except Exception:
                continue

    stories.sort(key=lambda s: s.score, reverse=True)
    logger.info("Fetched %d relevant HN stories", len(stories))
    return stories
