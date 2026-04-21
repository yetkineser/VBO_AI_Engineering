"""
Text chunking with overlap — splits documents into smaller pieces
so that each piece gets its own embedding vector.

Why chunk?
- Embedding models have token limits (typically 256-512 tokens)
- A 300-page book cannot be represented by a single vector
- Smaller chunks let semantic search find the *specific paragraph*
  that answers a question, not just the book that might contain it
"""
from __future__ import annotations

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: The full document text.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks. Each chunk is a string of ~chunk_size words.
        Consecutive chunks share ~overlap words at their boundaries.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """
    Take a parsed document dict and return a list of chunk dicts.
    Each chunk carries metadata from the parent document plus
    its own chunk_index and chunk text.
    """
    text = doc.get("text", "")
    chunks = chunk_text(text)

    if not chunks:
        return []

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "chunk_id": f"{doc['filename']}::chunk_{i}",
            "filename": doc["filename"],
            "title": doc.get("title", doc["filename"]),
            "author": doc.get("author", ""),
            "extension": doc.get("extension", ""),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
            "word_count": len(chunk.split()),
        })

    return results
