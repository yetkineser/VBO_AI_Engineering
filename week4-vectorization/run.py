"""
Main entry point for the Week 4 Document Ingestion Pipeline.

Usage (v1 — original):
    python run.py --ingest          Ingest all books into MongoDB + ES + vectors
    python run.py --search "query"  Search across all three systems (separate)
    python run.py --demo            Run demo queries and generate comparison report
    python run.py --stats           Show ingestion statistics

Usage (v2 — improved):
    python run.py --ingest-v2       Ingest with chunking into hybrid index
    python run.py --search-v2 "q"   Hybrid search (BM25 + kNN + RRF + threshold)
    python run.py --compare "q"     Side-by-side: v1 vs v2 search results
"""

import argparse
import sys

from config import DATA_DIR


def cmd_ingest():
    """Ingest all files from the data folder into all three systems."""
    from src.file_parser import scan_folder
    from src.mongo_store import insert_many, count_documents as mongo_count
    from src.elastic_store import index_many as es_index, count_documents as es_count
    from src.vector_store import index_many as vec_index, count_documents as vec_count

    print(f"\n{'='*60}")
    print(f"  Document Ingestion Pipeline")
    print(f"  Source: {DATA_DIR}")
    print(f"{'='*60}\n")

    # Step 1: Parse files
    print("[1/4] Parsing files...")
    docs = scan_folder(DATA_DIR)
    if not docs:
        print("No documents found. Check DATA_DIR in config.py.")
        sys.exit(1)
    print(f"  Parsed {len(docs)} documents\n")

    # Step 2: Store metadata in MongoDB
    print("[2/4] Storing metadata in MongoDB...")
    inserted = insert_many(docs)
    print(f"  Inserted {inserted} new records (total: {mongo_count()})\n")

    # Step 3: Index text in Elasticsearch
    print("[3/4] Indexing full text in Elasticsearch...")
    indexed = es_index(docs)
    print(f"  Indexed {indexed} documents (total: {es_count()})\n")

    # Step 4: Generate embeddings and index vectors
    print("[4/4] Generating embeddings and indexing vectors...")
    vec_indexed = vec_index(docs)
    print(f"  Indexed {vec_indexed} vectors (total: {vec_count()})\n")

    print(f"{'='*60}")
    print(f"  Ingestion complete!")
    print(f"  MongoDB:       {mongo_count()} documents")
    print(f"  Elasticsearch: {es_count()} documents")
    print(f"  Vector store:  {vec_count()} documents")
    print(f"{'='*60}\n")


def cmd_search(query: str):
    """Search across all three systems and print results."""
    from src.elastic_store import search_keyword
    from src.vector_store import search_semantic
    from src.mongo_store import query_by_author

    print(f"\nSearch: \"{query}\"\n")

    # Keyword search
    print("--- Elasticsearch (BM25 keyword search) ---")
    for r in search_keyword(query, size=5):
        print(f"  [{r['score']:.2f}] {r['filename']}")
        if r.get("snippet"):
            print(f"         {r['snippet'][:100]}")

    # Semantic search
    print("\n--- Vector Search (semantic similarity) ---")
    for r in search_semantic(query, k=5):
        print(f"  [{r['score']:.4f}] {r['filename']}")

    # Metadata search (show what MongoDB can do)
    print("\n--- MongoDB (metadata filter: search author) ---")
    results = query_by_author(query.split()[0] if query.split() else "")
    if results:
        for r in results[:5]:
            print(f"  {r['filename']} — {r.get('author', 'unknown')}")
    else:
        print("  (No author match — MongoDB filters metadata, not content)")

    print()


def cmd_demo():
    """Run all demo queries and generate the comparison report."""
    from src.search_demo import run_demo

    print("\nRunning search comparison demo...\n")
    run_demo()
    print("\nDone. Check outputs/search_comparison.md for the full report.\n")


def cmd_stats():
    """Show ingestion statistics."""
    from src.mongo_store import get_stats
    from src.elastic_store import count_documents as es_count
    from src.vector_store import count_documents as vec_count

    stats = get_stats()
    print(f"\n--- Ingestion Statistics ---")
    print(f"MongoDB:       {stats['total_documents']} documents")
    print(f"Elasticsearch: {es_count()} documents")
    print(f"Vector store:  {vec_count()} documents")
    print(f"\nBy file type:")
    for ext in stats.get("by_extension", []):
        print(f"  {ext['_id']}: {ext['count']} files, {ext['total_words']} words")
    print()


def cmd_ingest_v2():
    """Ingest with chunking into the hybrid index (text + vectors in one place)."""
    from src.file_parser import scan_folder
    from src.chunker import chunk_document
    from src.hybrid_store import index_chunks, count_chunks, delete_chunks_index

    print(f"\n{'='*60}")
    print(f"  Document Ingestion Pipeline v2 (with chunking)")
    print(f"  Source: {DATA_DIR}")
    print(f"{'='*60}\n")

    # Step 1: Parse files
    print("[1/3] Parsing files...")
    docs = scan_folder(DATA_DIR)
    if not docs:
        print("No documents found. Check DATA_DIR in config.py.")
        sys.exit(1)
    print(f"  Parsed {len(docs)} documents\n")

    # Step 2: Chunk documents
    print("[2/3] Chunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        if chunks:
            print(f"  {doc['filename']}: {len(chunks)} chunks")
    print(f"\n  Total: {len(all_chunks)} chunks from {len(docs)} documents\n")

    # Step 3: Index chunks with embeddings
    print("[3/3] Embedding and indexing chunks (this may take a few minutes)...")
    delete_chunks_index()
    indexed = index_chunks(all_chunks)

    from src.hybrid_store import get_client
    get_client().indices.refresh(index="document_chunks")

    print(f"\n{'='*60}")
    print(f"  v2 Ingestion complete!")
    print(f"  Chunks indexed: {count_chunks()}")
    print(f"  Avg chunks/doc: {len(all_chunks) // max(len(docs), 1)}")
    print(f"{'='*60}\n")


def cmd_search_v2(query: str):
    """Hybrid search with score threshold."""
    from src.hybrid_store import search_hybrid, search_semantic, search_keyword
    from config import SCORE_THRESHOLD

    print(f"\nHybrid Search (v2): \"{query}\"\n")

    # Hybrid (RRF)
    print("--- Hybrid (BM25 + kNN + RRF) ---")
    results = search_hybrid(query, size=5)
    if not results:
        print("  No results found.")
    for r in results:
        chunk_info = f"chunk {r['chunk_index']+1}/{r['total_chunks']}"
        print(f"  [{r['score']:.4f}] {r['filename']} ({chunk_info})")
        if r.get("snippet"):
            clean = r["snippet"].replace("<em>", "*").replace("</em>", "*")
            print(f"           {clean[:120]}")

    # Semantic with threshold
    print(f"\n--- Semantic (kNN, threshold={SCORE_THRESHOLD}) ---")
    results = search_semantic(query, k=5)
    relevant = [r for r in results if r.get("relevant", True)]
    below = [r for r in results if not r.get("relevant", True)]

    if relevant:
        for r in relevant:
            chunk_info = f"chunk {r['chunk_index']+1}/{r['total_chunks']}"
            print(f"  [{r['score']:.4f}] {r['filename']} ({chunk_info})")
    else:
        print(f"  No results above threshold ({SCORE_THRESHOLD})")

    if below:
        print(f"\n  Below threshold (would be filtered in production):")
        for r in below:
            chunk_info = f"chunk {r['chunk_index']+1}/{r['total_chunks']}"
            print(f"  [{r['score']:.4f}] {r['filename']} ({chunk_info})")

    print()


def cmd_compare(query: str):
    """Side-by-side: v1 (whole doc) vs v2 (chunked + hybrid)."""
    from src.vector_store import search_semantic as v1_semantic
    from src.elastic_store import search_keyword as v1_keyword
    from src.hybrid_store import search_hybrid, search_semantic as v2_semantic
    from config import SCORE_THRESHOLD

    print(f"\n{'='*60}")
    print(f"  v1 vs v2 Comparison: \"{query}\"")
    print(f"{'='*60}\n")

    # v1: whole-document search
    print("=== v1: Whole-Document Search ===\n")
    print("  BM25 keyword:")
    for r in v1_keyword(query, size=3):
        print(f"    [{r['score']:.2f}] {r['filename']}")

    print("\n  Semantic (whole doc, first 512 chars):")
    for r in v1_semantic(query, k=3):
        print(f"    [{r['score']:.4f}] {r['filename']}")

    # v2: chunked hybrid search
    print("\n=== v2: Chunked Hybrid Search ===\n")
    print("  Hybrid (BM25 + kNN + RRF):")
    for r in search_hybrid(query, size=3):
        chunk_info = f"chunk {r['chunk_index']+1}/{r['total_chunks']}"
        print(f"    [{r['score']:.4f}] {r['filename']} ({chunk_info})")

    print(f"\n  Semantic (chunked, threshold={SCORE_THRESHOLD}):")
    results = v2_semantic(query, k=3)
    for r in results:
        chunk_info = f"chunk {r['chunk_index']+1}/{r['total_chunks']}"
        status = "RELEVANT" if r.get("relevant", True) else "BELOW THRESHOLD"
        print(f"    [{r['score']:.4f}] {r['filename']} ({chunk_info}) — {status}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Week 4 Document Ingestion Pipeline — MongoDB + ES + Vectors"
    )
    # v1 commands
    parser.add_argument("--ingest", action="store_true", help="v1: Ingest all files")
    parser.add_argument("--search", type=str, help="v1: Search query")
    parser.add_argument("--demo", action="store_true", help="v1: Run demo queries")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    # v2 commands
    parser.add_argument("--ingest-v2", action="store_true", help="v2: Ingest with chunking")
    parser.add_argument("--search-v2", type=str, help="v2: Hybrid search")
    parser.add_argument("--compare", type=str, help="Compare v1 vs v2 results")

    args = parser.parse_args()

    if args.ingest:
        cmd_ingest()
    elif args.search:
        cmd_search(args.search)
    elif args.demo:
        cmd_demo()
    elif args.stats:
        cmd_stats()
    elif args.ingest_v2:
        cmd_ingest_v2()
    elif args.search_v2:
        cmd_search_v2(args.search_v2)
    elif args.compare:
        cmd_compare(args.compare)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
