"""
Side-by-side comparison of all three search systems.
Shows the same query returning different results from MongoDB, Elasticsearch, and Vector search.
"""
from __future__ import annotations

from pathlib import Path

from src.mongo_store import query_by_author, query_large_files, get_stats
from src.elastic_store import search_keyword
from src.vector_store import search_semantic
from config import OUTPUT_DIR


DEMO_QUERIES = [
    "How to build an AI application",
    "Deep learning optimization techniques",
    "Career development and productivity",
    "Natural language processing transformers",
    "Data pipeline architecture",
]


def run_comparison(query: str, k: int = 5) -> dict:
    """Run the same query against all three systems and collect results."""

    # MongoDB: metadata search (search by author name as proxy)
    # MongoDB stores metadata, not text — so we show a metadata query instead
    mongo_results = query_large_files(min_bytes=500_000)[:k]

    # Elasticsearch: keyword search (BM25)
    es_results = search_keyword(query, size=k)

    # Vector search: semantic similarity (kNN)
    vec_results = search_semantic(query, k=k)

    return {
        "query": query,
        "mongo_metadata": mongo_results,
        "es_keyword": es_results,
        "vector_semantic": vec_results,
    }


def format_comparison(result: dict) -> str:
    """Format a comparison result as readable markdown."""
    lines = [
        f"### Query: \"{result['query']}\"",
        "",
        "#### MongoDB — Metadata Filter (largest files)",
        "| # | Filename | Size | Words |",
        "|---|----------|------|-------|",
    ]
    for i, doc in enumerate(result["mongo_metadata"], 1):
        size_mb = doc.get("size_bytes", 0) / 1_048_576
        lines.append(
            f"| {i} | {doc['filename']} | {size_mb:.1f} MB | {doc.get('word_count', '?')} |"
        )

    lines += [
        "",
        "#### Elasticsearch — Keyword Search (BM25)",
        "| # | Filename | Score | Snippet |",
        "|---|----------|-------|---------|",
    ]
    for i, doc in enumerate(result["es_keyword"], 1):
        snippet = doc.get("snippet", "").replace("|", "\\|")[:80]
        lines.append(
            f"| {i} | {doc['filename']} | {doc['score']} | {snippet} |"
        )

    lines += [
        "",
        "#### Vector Search — Semantic Similarity (kNN)",
        "| # | Filename | Score |",
        "|---|----------|-------|",
    ]
    for i, doc in enumerate(result["vector_semantic"], 1):
        lines.append(f"| {i} | {doc['filename']} | {doc['score']} |")

    lines.append("")
    return "\n".join(lines)


def run_demo(queries: list[str] | None = None) -> str:
    """Run all demo queries and return a full markdown report."""
    queries = queries or DEMO_QUERIES
    stats = get_stats()

    sections = [
        "# Search Comparison Report",
        "",
        f"**Total documents**: {stats['total_documents']}",
        "",
        "**By file type**:",
    ]
    for ext in stats.get("by_extension", []):
        sections.append(f"- {ext['_id']}: {ext['count']} files, {ext['total_words']} words")

    sections += ["", "---", ""]

    for query in queries:
        print(f"  Querying: \"{query}\"")
        result = run_comparison(query)
        sections.append(format_comparison(result))
        sections.append("---\n")

    # Analysis section
    sections += [
        "## When Does Each System Win?",
        "",
        "| Scenario | Best system | Why |",
        "|----------|-------------|-----|",
        "| Filter by file type, size, date | **MongoDB** | Structured metadata queries are instant |",
        "| Find exact phrases or technical terms | **Elasticsearch** | BM25 excels at precise keyword matching |",
        "| Find conceptually similar content | **Vector search** | Embeddings capture meaning, not just words |",
        "| User asks a question (not keywords) | **Vector search** | Questions rarely share exact words with answers |",
        "| Combine all three | **Hybrid** | Filter → keyword rank → semantic re-rank |",
        "",
    ]

    report = "\n".join(sections)

    # Save to file
    output_path = OUTPUT_DIR / "search_comparison.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {output_path}")

    return report
