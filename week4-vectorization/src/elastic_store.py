"""
Elasticsearch operations — full-text indexing and keyword search (BM25).
"""
from __future__ import annotations

from elasticsearch import Elasticsearch, helpers

from config import ES_URL, ES_TEXT_INDEX


def get_client() -> Elasticsearch:
    """Return an Elasticsearch client (compatible with ES 8.x server)."""
    return Elasticsearch(ES_URL)


def create_text_index(es: Elasticsearch | None = None) -> None:
    """Create the full-text index if it does not exist."""
    es = es or get_client()
    if es.indices.exists(index=ES_TEXT_INDEX):
        return

    es.indices.create(index=ES_TEXT_INDEX, body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "filename":   {"type": "keyword"},
                "title":      {"type": "text"},
                "author":     {"type": "text"},
                "content":    {"type": "text", "analyzer": "standard"},
                "word_count": {"type": "integer"},
                "extension":  {"type": "keyword"},
            }
        }
    })
    print(f"  Created Elasticsearch index: {ES_TEXT_INDEX}")


def index_document(doc: dict, es: Elasticsearch | None = None) -> None:
    """Index a single document's text content."""
    es = es or get_client()
    es.index(
        index=ES_TEXT_INDEX,
        id=doc["filename"],
        document={
            "filename": doc["filename"],
            "title": doc.get("title", doc["filename"]),
            "author": doc.get("author", ""),
            "content": doc.get("text", ""),
            "word_count": doc.get("word_count", 0),
            "extension": doc.get("extension", ""),
        },
    )


def index_many(docs: list[dict], es: Elasticsearch | None = None) -> int:
    """Bulk-index multiple documents. Returns count indexed."""
    es = es or get_client()
    create_text_index(es)

    actions = [
        {
            "_index": ES_TEXT_INDEX,
            "_id": doc["filename"],
            "_source": {
                "filename": doc["filename"],
                "title": doc.get("title", doc["filename"]),
                "author": doc.get("author", ""),
                "content": doc.get("text", ""),
                "word_count": doc.get("word_count", 0),
                "extension": doc.get("extension", ""),
            },
        }
        for doc in docs
    ]
    success, _ = helpers.bulk(es, actions, raise_on_error=False)
    return success


def search_keyword(query: str, size: int = 5, es: Elasticsearch | None = None) -> list[dict]:
    """
    Full-text keyword search using BM25.
    Returns a list of {filename, title, score, snippet}.
    """
    es = es or get_client()
    resp = es.search(
        index=ES_TEXT_INDEX,
        body={
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title^2", "author"],
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 1,
                        "max_analyzed_offset": 500000,
                    }
                },
            },
        },
    )

    results = []
    for hit in resp["hits"]["hits"]:
        snippet = ""
        if "highlight" in hit and "content" in hit["highlight"]:
            snippet = hit["highlight"]["content"][0]
        results.append({
            "filename": hit["_source"]["filename"],
            "title": hit["_source"].get("title", ""),
            "score": round(hit["_score"], 3),
            "snippet": snippet,
        })
    return results


def count_documents(es: Elasticsearch | None = None) -> int:
    """Return the number of documents in the text index."""
    es = es or get_client()
    if not es.indices.exists(index=ES_TEXT_INDEX):
        return 0
    return es.count(index=ES_TEXT_INDEX)["count"]
