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


def insert_metadata(doc: dict) -> bool:
    """
    Insert a document's metadata into MongoDB.
    Skips the full text to keep MongoDB lean — text goes to Elasticsearch.
    """
    col = get_collection()
    record = {k: v for k, v in doc.items() if k != "text"}
    try:
        col.insert_one(record)
        return True
    except DuplicateKeyError:
        return False


def insert_many(docs: list[dict]) -> int:
    """Insert metadata for multiple documents. Returns count of new inserts."""
    inserted = 0
    for doc in docs:
        if insert_metadata(doc):
            inserted += 1
    return inserted


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
