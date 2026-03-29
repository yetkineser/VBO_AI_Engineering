"""
Fetch recent papers from arXiv using consolidated queries to avoid rate limiting.
Uses the official arXiv API via the `arxiv` package with aggressive backoff.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import arxiv

from .config import TOPICS, ARXIV_CATEGORIES, settings

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str
    categories: list[str]
    published: datetime
    topic_match: str
    relevance_keywords: list[str] = field(default_factory=list)


CONSOLIDATED_QUERIES = [
    {
        "query": "retrieval augmented generation OR RAG OR LLM agents OR AI agent",
        "topics": ["ai_engineering"],
    },
    {
        "query": "automated machine learning OR neural architecture search OR hyperparameter optimization",
        "topics": ["autoresearch"],
    },
    {
        "query": "large language model fine-tuning OR LoRA OR model quantization OR speculative decoding OR KV cache",
        "topics": ["llm_optimization"],
    },
    {
        "query": "push notification optimization OR resource allocation optimization OR multi-armed bandit",
        "topics": ["push_allocation_optimization"],
    },
    {
        "query": "fraud detection deep learning OR anomaly detection real-time OR graph neural network fraud",
        "topics": ["fraud_anomaly_detection"],
    },
    {
        "query": "text classification transformer OR sentiment analysis LLM OR recommendation system ranking",
        "topics": ["nlp_text_classification", "recommendation_systems"],
    },
    {
        "query": "prompt engineering OR vector database OR agentic AI OR LLM deployment",
        "topics": ["ai_engineering"],
    },
]


def _all_keywords() -> dict[str, list[str]]:
    """Gather all keywords per topic for matching."""
    kw_map: dict[str, list[str]] = {}
    for topic_key, topic_cfg in TOPICS.items():
        kw_map[topic_key] = (
            topic_cfg.get("arxiv_queries", [])
            + topic_cfg.get("rss_keywords", [])
        )
    return kw_map


def _match_paper_to_topics(text: str, kw_map: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Find best topic match and matched keywords for a paper."""
    text_lower = text.lower()
    best_topic = ""
    best_keywords: list[str] = []

    for topic_key, keywords in kw_map.items():
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if len(matched) > len(best_keywords):
            best_topic = topic_key
            best_keywords = matched

    return best_topic, best_keywords


def fetch_arxiv_papers(days_back: int = 2) -> list[Paper]:
    """Fetch papers from arXiv using consolidated queries to minimize API calls."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    all_papers: list[Paper] = []
    seen_ids: set[str] = set()
    kw_map = _all_keywords()

    client = arxiv.Client(
        page_size=settings.arxiv_max_results,
        delay_seconds=5.0,
        num_retries=2,
    )

    for i, cq in enumerate(CONSOLIDATED_QUERIES):
        cat_filter = " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES)
        full_query = f'all:({cq["query"]}) AND ({cat_filter})'

        search = arxiv.Search(
            query=full_query,
            max_results=settings.arxiv_max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                if result.published < cutoff:
                    continue
                paper_id = result.entry_id
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                searchable = f"{result.title} {result.summary}"
                topic, matched_kw = _match_paper_to_topics(searchable, kw_map)
                if not topic:
                    topic = cq["topics"][0]

                paper = Paper(
                    title=result.title.replace("\n", " ").strip(),
                    authors=[a.name for a in result.authors[:5]],
                    abstract=result.summary[:500].replace("\n", " ").strip(),
                    url=result.entry_id,
                    pdf_url=result.pdf_url or "",
                    categories=list(result.categories),
                    published=result.published,
                    topic_match=topic,
                    relevance_keywords=matched_kw,
                )
                all_papers.append(paper)

        except Exception:
            logger.warning("arXiv query %d/%d failed (rate limit), continuing with collected papers",
                           i + 1, len(CONSOLIDATED_QUERIES))

        if i < len(CONSOLIDATED_QUERIES) - 1:
            time.sleep(8)

    all_papers.sort(key=lambda p: len(p.relevance_keywords), reverse=True)
    logger.info("Fetched %d papers from arXiv", len(all_papers))
    return all_papers
