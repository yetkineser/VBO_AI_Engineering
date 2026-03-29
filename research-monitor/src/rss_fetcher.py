"""
Fetch blog posts from RSS feeds (Medium, TDS, Dev.to, personal blogs).
Filters by relevance to configured topics.
"""

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import feedparser

from .config import RSS_FEEDS, TOPICS, settings

logger = logging.getLogger(__name__)


@dataclass
class BlogPost:
    title: str
    author: str
    url: str
    source: str
    published: datetime
    summary: str
    tags: list[str]
    matched_topics: list[str]
    matched_keywords: list[str]


def _parse_date(entry: dict) -> datetime | None:
    for field in ("published", "updated"):
        raw = entry.get(field)
        if not raw:
            continue
        try:
            return parsedate_to_datetime(raw)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags (basic)."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _match_topics(text: str) -> tuple[list[str], list[str]]:
    """Return (matched topic keys, matched keywords)."""
    text_lower = text.lower()
    matched_topics = []
    matched_keywords = []

    for topic_key, topic_cfg in TOPICS.items():
        for kw in topic_cfg.get("rss_keywords", []):
            if kw.lower() in text_lower:
                if topic_key not in matched_topics:
                    matched_topics.append(topic_key)
                if kw not in matched_keywords:
                    matched_keywords.append(kw)

    return matched_topics, matched_keywords


def fetch_rss_posts(days_back: int = 2) -> list[BlogPost]:
    """Fetch recent posts from all configured RSS feeds, filtered by relevance."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    all_posts: list[BlogPost] = []
    seen_urls: set[str] = set()

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo and not feed.entries:
                logger.warning("Feed parse error for %s: %s", source_name, feed.bozo_exception)
                continue

            count = 0
            for entry in feed.entries:
                if count >= settings.rss_max_per_feed:
                    break

                pub_date = _parse_date(entry)
                if pub_date and pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                if pub_date and pub_date < cutoff:
                    continue

                url = entry.get("link", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = entry.get("title", "Untitled")
                summary_raw = entry.get("summary", entry.get("description", ""))
                summary = _strip_html(summary_raw)[:400]

                tags = [t.get("term", "") for t in entry.get("tags", [])]
                searchable = f"{title} {summary} {' '.join(tags)}"

                matched_topics, matched_keywords = _match_topics(searchable)

                if not matched_topics:
                    continue

                author = entry.get("author", "")
                if not author and hasattr(feed.feed, "title"):
                    author = feed.feed.title

                post = BlogPost(
                    title=title,
                    author=author,
                    url=url,
                    source=source_name,
                    published=pub_date or datetime.now(timezone.utc),
                    summary=summary,
                    tags=tags,
                    matched_topics=matched_topics,
                    matched_keywords=matched_keywords,
                )
                all_posts.append(post)
                count += 1

        except Exception:
            logger.exception("Failed to fetch RSS feed: %s", source_name)

    all_posts.sort(key=lambda p: len(p.matched_keywords), reverse=True)
    logger.info("Fetched %d relevant blog posts from %d feeds", len(all_posts), len(RSS_FEEDS))
    return all_posts
