"""
Hybrid search — v2 of the pipeline.

Stores chunked documents in a single Elasticsearch index with BOTH
full text and dense vector fields. This enables:
  1. BM25 keyword search on the text
  2. kNN semantic search on the vector
  3. Reciprocal Rank Fusion (RRF) to merge both into one ranked list
  4. Score thresholding to filter out irrelevant results
"""
from __future__ import annotations

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from config import (
    ES_URL,
    ES_CHUNKS_INDEX,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    SCORE_THRESHOLD,
)

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client() -> Elasticsearch:
    return Elasticsearch(ES_URL)


# ── Index management ────���────────────────────────────────────────

def create_chunks_index(es: Elasticsearch = None) -> None:
    """Create combined text + vector index for chunks."""
    es = es or get_client()
    if es.indices.exists(index=ES_CHUNKS_INDEX):
        return

    es.indices.create(index=ES_CHUNKS_INDEX, body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id":    {"type": "keyword"},
                "filename":    {"type": "keyword"},
                "title":       {"type": "text"},
                "author":      {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "content":     {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    })
    print(f"  Created chunks index: {ES_CHUNKS_INDEX}")


def delete_chunks_index(es: Elasticsearch = None) -> None:
    """Delete the chunks index (for re-ingestion)."""
    es = es or get_client()
    if es.indices.exists(index=ES_CHUNKS_INDEX):
        es.indices.delete(index=ES_CHUNKS_INDEX)


# ── Ingestion ─────────���──────────────────────────────────────────

def index_chunks(chunks: list[dict], es: Elasticsearch = None) -> int:
    """Bulk-index chunks with text + embeddings."""
    es = es or get_client()
    create_chunks_index(es)
    model = _get_model()

    actions = []
    for chunk in chunks:
        text = chunk.get("text", "")
        embedding = model.encode(text).tolist()
        actions.append({
            "_index": ES_CHUNKS_INDEX,
            "_id": chunk["chunk_id"],
            "_source": {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "title": chunk.get("title", ""),
                "author": chunk.get("author", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "content": text,
                "embedding": embedding,
            },
        })

    if not actions:
        return 0

    success, _ = helpers.bulk(es, actions, raise_on_error=False, chunk_size=100)
    return success


# ── Search functions ─────────────────────────────────────────────

def search_keyword(query: str, size: int = 5, es: Elasticsearch = None) -> list[dict]:
    """BM25 keyword search on chunks."""
    es = es or get_client()
    resp = es.search(index=ES_CHUNKS_INDEX, body={
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content", "title^2"],
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
    })
    return _parse_hits(resp)


def search_semantic(query: str, k: int = 5, threshold: float = SCORE_THRESHOLD,
                    es: Elasticsearch = None) -> list[dict]:
    """kNN semantic search on chunks with score threshold."""
    es = es or get_client()
    model = _get_model()
    query_vector = model.encode(query).tolist()

    resp = es.search(index=ES_CHUNKS_INDEX, body={
        "size": k,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(k * 4, 20),
        },
    })

    results = _parse_hits(resp)
    # Apply score threshold
    if threshold:
        relevant = [r for r in results if r["score"] >= threshold]
        irrelevant = [r for r in results if r["score"] < threshold]
        for r in relevant:
            r["relevant"] = True
        for r in irrelevant:
            r["relevant"] = False
        return relevant + irrelevant

    return results


def search_hybrid(query: str, size: int = 5, threshold: float = SCORE_THRESHOLD,
                  es: Elasticsearch = None) -> list[dict]:
    """
    Hybrid search: BM25 + kNN combined with Reciprocal Rank Fusion.
    This is the best of both worlds — keyword precision + semantic understanding.
    """
    es = es or get_client()
    model = _get_model()
    query_vector = model.encode(query).tolist()

    # Try native RRF (ES 8.9+)
    try:
        resp = es.search(index=ES_CHUNKS_INDEX, body={
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title^2"],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": size * 2,
                "num_candidates": max(size * 4, 20),
            },
            "rank": {
                "rrf": {
                    "window_size": 100,
                    "rank_constant": 60,
                }
            },
        })
        return _parse_hits(resp)
    except Exception:
        # Fallback: manual RRF if native is not available
        return _manual_rrf(query, query_vector, size, threshold, es)


def _manual_rrf(query: str, query_vector: list, size: int,
                threshold: float, es: Elasticsearch) -> list[dict]:
    """Fallback RRF implementation when native RRF is not available."""
    bm25_results = search_keyword(query, size=size * 2, es=es)
    sem_results = search_semantic(query, k=size * 2, threshold=0, es=es)

    k = 60  # RRF constant
    scores = {}
    metadata = {}

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        metadata[cid] = r

    for rank, r in enumerate(sem_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in metadata:
            metadata[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:size]
    results = []
    for cid, rrf_score in ranked:
        entry = metadata[cid].copy()
        entry["score"] = round(rrf_score, 4)
        entry["relevant"] = True  # RRF scores are not directly comparable to threshold
        results.append(entry)

    return results


# ── Helpers ─���────────────────────────────────────────────────────

def _parse_hits(resp: dict) -> list[dict]:
    """Extract results from ES response."""
    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        snippet = ""
        if "highlight" in hit and "content" in hit["highlight"]:
            snippet = hit["highlight"]["content"][0]
        results.append({
            "chunk_id": src.get("chunk_id", ""),
            "filename": src["filename"],
            "title": src.get("title", ""),
            "author": src.get("author", ""),
            "chunk_index": src.get("chunk_index", 0),
            "total_chunks": src.get("total_chunks", 1),
            "score": round(hit["_score"], 4) if hit["_score"] else 0,
            "snippet": snippet,
        })
    return results


def count_chunks(es: Elasticsearch = None) -> int:
    """Return total chunk count."""
    es = es or get_client()
    if not es.indices.exists(index=ES_CHUNKS_INDEX):
        return 0
    return es.count(index=ES_CHUNKS_INDEX)["count"]
