"""
Vector store — generate embeddings and run kNN semantic search via Elasticsearch.
"""
from __future__ import annotations

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    MAX_TEXT_LENGTH,
    ES_URL,
    ES_VECTOR_INDEX,
)

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client() -> Elasticsearch:
    return Elasticsearch(ES_URL)


def create_vector_index(es: Elasticsearch | None = None) -> None:
    """Create the dense-vector index if it does not exist."""
    es = es or get_client()
    if es.indices.exists(index=ES_VECTOR_INDEX):
        return

    es.indices.create(index=ES_VECTOR_INDEX, body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "filename":  {"type": "keyword"},
                "title":     {"type": "text"},
                "author":    {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    })
    print(f"  Created Elasticsearch vector index: {ES_VECTOR_INDEX}")


def encode_text(text: str) -> list[float]:
    """Generate a dense embedding for the given text."""
    model = get_model()
    truncated = text[:MAX_TEXT_LENGTH]
    return model.encode(truncated).tolist()


def index_document(doc: dict, es: Elasticsearch | None = None) -> None:
    """Generate embedding and index a single document."""
    es = es or get_client()
    embedding = encode_text(doc.get("text", ""))
    es.index(
        index=ES_VECTOR_INDEX,
        id=doc["filename"],
        document={
            "filename": doc["filename"],
            "title": doc.get("title", doc["filename"]),
            "author": doc.get("author", ""),
            "embedding": embedding,
        },
    )


def index_many(docs: list[dict], es: Elasticsearch | None = None) -> int:
    """Bulk-index documents with their embeddings. Returns count indexed."""
    es = es or get_client()
    create_vector_index(es)
    model = get_model()

    actions = []
    for doc in docs:
        text = doc.get("text", "")[:MAX_TEXT_LENGTH]
        embedding = model.encode(text).tolist()
        actions.append({
            "_index": ES_VECTOR_INDEX,
            "_id": doc["filename"],
            "_source": {
                "filename": doc["filename"],
                "title": doc.get("title", doc["filename"]),
                "author": doc.get("author", ""),
                "embedding": embedding,
            },
        })

    success, _ = helpers.bulk(es, actions, raise_on_error=False)
    return success


def search_semantic(query: str, k: int = 5, es: Elasticsearch | None = None) -> list[dict]:
    """
    Semantic search using kNN on dense vectors.
    Returns a list of {filename, title, score}.
    """
    es = es or get_client()
    query_vector = encode_text(query)

    resp = es.search(
        index=ES_VECTOR_INDEX,
        body={
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": max(k * 4, 20),
            },
        },
    )

    results = []
    for hit in resp["hits"]["hits"]:
        results.append({
            "filename": hit["_source"]["filename"],
            "title": hit["_source"].get("title", ""),
            "score": round(hit["_score"], 4),
        })
    return results


def count_documents(es: Elasticsearch | None = None) -> int:
    """Return the number of documents in the vector index."""
    es = es or get_client()
    if not es.indices.exists(index=ES_VECTOR_INDEX):
        return 0
    return es.count(index=ES_VECTOR_INDEX)["count"]
