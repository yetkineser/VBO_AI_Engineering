"""
Fetch trending and recently-starred GitHub repos related to configured topics.
Uses the GitHub Search API (authenticated if GITHUB_TOKEN is set).
"""

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import httpx

from .config import TOPICS, settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


@dataclass
class Repo:
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    stars_today: int
    forks: int
    language: str
    topics: list[str]
    created_at: datetime
    updated_at: datetime
    topic_match: str


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "research-monitor/1.0",
    }
    if settings.github_token:
        h["Authorization"] = f"token {settings.github_token}"
    return h


def _parse_dt(s: str | None) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_github_repos(days_back: int = 7) -> list[Repo]:
    """
    Search GitHub for repos matching configured keywords,
    created or significantly updated in the last `days_back` days.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    date_str = cutoff.strftime("%Y-%m-%d")
    all_repos: list[Repo] = []
    seen: set[str] = set()

    with httpx.Client(headers=_headers(), timeout=30.0) as client:
        for topic_key, topic_cfg in TOPICS.items():
            for keyword in topic_cfg["github_keywords"]:
                query = f"{keyword} created:>{date_str} sort:stars"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": settings.github_max_repos,
                }

                try:
                    resp = client.get(f"{GITHUB_API}/search/repositories", params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    for item in data.get("items", []):
                        fn = item["full_name"]
                        if fn in seen:
                            continue
                        seen.add(fn)

                        repo = Repo(
                            name=item["name"],
                            full_name=fn,
                            description=(item.get("description") or "")[:300],
                            url=item["html_url"],
                            stars=item["stargazers_count"],
                            stars_today=0,
                            forks=item["forks_count"],
                            language=item.get("language") or "Unknown",
                            topics=item.get("topics", []),
                            created_at=_parse_dt(item.get("created_at")),
                            updated_at=_parse_dt(item.get("updated_at")),
                            topic_match=topic_key,
                        )
                        all_repos.append(repo)

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 403:
                        logger.warning("GitHub rate limit hit, skipping remaining queries")
                        break
                    logger.exception("GitHub search failed for: %s", keyword)
                except Exception:
                    logger.exception("GitHub search failed for: %s", keyword)

    all_repos.sort(key=lambda r: r.stars, reverse=True)
    logger.info("Fetched %d repos from GitHub", len(all_repos))
    return all_repos


def fetch_github_trending(language: str = "", since: str = "daily") -> list[dict]:
    """
    Scrape GitHub trending page (unofficial - no stable API).
    Falls back gracefully if it fails.
    """
    url = f"https://github.com/trending/{language}"
    params = {"since": since, "spoken_language_code": "en"}

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return _parse_trending_html(resp.text)
    except Exception:
        logger.warning("GitHub trending scrape failed, skipping")
        return []


def _parse_trending_html(html: str) -> list[dict]:
    """Minimal parse of GitHub trending page."""
    repos = []
    lines = html.split('<article class="Box-row">')

    for block in lines[1:]:
        try:
            name_start = block.find('<a href="/')
            if name_start == -1:
                continue
            name_start += len('<a href="/')
            name_end = block.find('"', name_start)
            full_name = block[name_start:name_end].strip()

            desc = ""
            p_start = block.find('<p class="')
            if p_start != -1:
                p_content_start = block.find(">", p_start) + 1
                p_end = block.find("</p>", p_content_start)
                desc = block[p_content_start:p_end].strip()

            repos.append({
                "full_name": full_name,
                "url": f"https://github.com/{full_name}",
                "description": desc[:300],
            })
        except Exception:
            continue

    return repos[:20]
