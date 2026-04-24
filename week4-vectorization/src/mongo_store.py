"""
MongoDB operations — store and query file metadata.
"""
from __future__ import annotations

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION


def get_collection():
    """Return the MongoDB collection, creating a unique index on filename."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]
    col.create_index("filename", unique=True)
    return col


def insert_metadata(doc: dict) -> str:
    """
    Insert or update a document's metadata in MongoDB.
    Skips the full text to keep MongoDB lean — text goes to Elasticsearch.

    Uses content_hash (SHA-256) to detect changes:
      - New file        → insert → returns "inserted"
      - Same hash       → skip   → returns "skipped"
      - Different hash  → update → returns "updated"
    """
    col = get_collection()
    record = {k: v for k, v in doc.items() if k != "text"}

    existing = col.find_one({"filename": doc["filename"]}, {"content_hash": 1})

    if existing is None:
        col.insert_one(record)
        return "inserted"

    if existing.get("content_hash") == doc.get("content_hash"):
        return "skipped"

    col.replace_one({"filename": doc["filename"]}, record)
    return "updated"


def insert_many(docs: list[dict]) -> dict:
    """Insert/update metadata for multiple documents. Returns counts by action."""
    counts = {"inserted": 0, "updated": 0, "skipped": 0}
    for doc in docs:
        result = insert_metadata(doc)
        counts[result] += 1
    return counts


def query_by_extension(ext: str) -> list[dict]:
    """Find all documents with a given extension (e.g., '.pdf')."""
    col = get_collection()
    return list(col.find({"extension": ext}, {"_id": 0}))


def query_large_files(min_bytes: int = 1_000_000) -> list[dict]:
    """Find files larger than min_bytes."""
    col = get_collection()
    return list(col.find(
        {"size_bytes": {"$gt": min_bytes}},
        {"_id": 0, "filename": 1, "size_bytes": 1, "word_count": 1},
    ).sort("size_bytes", -1))


def query_by_author(author: str) -> list[dict]:
    """Find all documents by a given author (case-insensitive regex)."""
    col = get_collection()
    return list(col.find(
        {"author": {"$regex": author, "$options": "i"}},
        {"_id": 0, "filename": 1, "author": 1, "title": 1},
    ))


def count_documents() -> int:
    """Return total document count in MongoDB."""
    return get_collection().count_documents({})


def get_stats() -> dict:
    """Return summary statistics about the collection."""
    col = get_collection()
    pipeline = [
        {"$group": {
            "_id": "$extension",
            "count": {"$sum": 1},
            "total_words": {"$sum": "$word_count"},
            "total_bytes": {"$sum": "$size_bytes"},
        }},
        {"$sort": {"count": -1}},
    ]
    by_ext = list(col.aggregate(pipeline))
    return {
        "total_documents": col.count_documents({}),
        "by_extension": by_ext,
    }
