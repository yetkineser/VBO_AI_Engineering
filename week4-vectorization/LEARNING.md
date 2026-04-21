# Week 4 Learning Guide — What to Study and Why

This page explains the concepts behind the Week 4 homework and gives you reading links for each one. Read the sections in order — each builds on the previous.

---

## 1. Document Storage for AI Systems

Before any AI system can answer questions about your data, it needs to **store** that data in a searchable way. There are three complementary approaches, and a production system uses all three.

### Reading
- [MongoDB Getting Started Tutorial](https://www.mongodb.com/docs/manual/tutorial/getting-started/) — official 30-minute walkthrough
- [What is NoSQL? (MongoDB)](https://www.mongodb.com/nosql-explained) — why flexible schemas matter

### Key takeaway
MongoDB stores **structured metadata** (author, size, date, tags) as flexible JSON documents. You do not need to define columns in advance like SQL — perfect for files where each type has different attributes.

---

## 2. Full-Text Search and Inverted Indexes

When you type a query like "transfer learning", Elasticsearch finds all documents containing those words using an **inverted index** — a map from every word to the documents that contain it. This is the same idea behind TF-IDF from Week 2, but optimised for speed.

### Reading
- [Elasticsearch: Documents and Indices](https://www.elastic.co/guide/en/elasticsearch/reference/current/documents-indices.html) — what an inverted index actually is
- [BM25: The Algorithm Behind Elasticsearch](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — BM25 is an improved TF-IDF; understand why ES uses it instead of raw TF-IDF
- [Fireship: Elasticsearch in 100 Seconds](https://www.youtube.com/watch?v=ZP0NmfyfsoM) — fastest possible overview (video)

### Key takeaway
Keyword search is **precise** — it finds exact words and phrases. But it fails when the user's words differ from the document's words ("notification system" vs "push allocation pipeline").

---

## 3. Dense Vector Embeddings and Semantic Search

You already know embeddings from Week 2 (transformer embeddings) and Week 3 (FastText/GloVe). This week, instead of computing them and throwing them away, you **store** them so you can search by meaning.

### Reading
- [What is Similarity Search? (Pinecone)](https://www.pinecone.io/learn/what-is-similarity-search/) — the best visual introduction to vector search
- [Sentence Transformers Quickstart](https://www.sbert.net/docs/quickstart.html) — the library you will use to generate embeddings
- [Elasticsearch kNN Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html) — how to store and query dense vectors in Elasticsearch

### Key takeaway
Vector search finds content by **meaning**, not by exact words. It handles synonyms, cross-language queries, and questions naturally. The trade-off: it cannot do exact phrase matching.

---

## 4. Combining All Three (Hybrid Search)

Production RAG systems do not choose one approach — they combine all three:

1. **Filter** by metadata (MongoDB): "only books published after 2020"
2. **Rank** by keyword relevance (Elasticsearch BM25): "which contain transfer learning?"
3. **Re-rank** by semantic similarity (vector search): "which are closest in meaning to the query?"

### Reading
- [Hybrid Search Explained (Qdrant)](https://qdrant.tech/articles/hybrid-search/) — how to combine keyword and vector results
- [Reciprocal Rank Fusion](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — the algorithm Elasticsearch uses to merge keyword + vector scores

### Key takeaway
Each system covers the other's weakness. Metadata filtering narrows the scope, keyword search finds exact matches, and vector search catches semantic near-misses.

---

## 5. Document Parsing

Before storing anything, you need to extract text from files. Different formats require different parsers.

### Reading
- [PyMuPDF (fitz) Tutorial](https://pymupdf.readthedocs.io/en/latest/tutorial.html) — the PDF parser we use (fast, reliable)
- [EbookLib Documentation](https://docs.sourcefabric.org/projects/ebooklib/en/latest/) — for EPUB files

### Key takeaway
PDF extraction is imperfect — scanned PDFs need OCR, some PDFs have broken character encoding. Always inspect extracted text before trusting it.

---

## 6. Where This Leads: RAG

This pipeline is the **retrieval** half of Retrieval-Augmented Generation. In Week 5–6, you will connect this storage layer to an LLM so it can generate answers grounded in your documents.

### Reading
- [RAG from Scratch (LangChain YouTube)](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) — end-to-end video series showing how retrieval feeds into generation
- [LangChain Document Loaders](https://python.langchain.com/docs/how_to/#document-loaders) — how LangChain solves the same file-loading problem with a framework

### Key takeaway
Without a retrieval layer, LLMs can only answer from their training data. With one, they can answer from **your** data — this is what makes RAG so powerful for enterprise applications.

---

## Quick Reference: Video Resources

| Topic | Link | Duration |
|-------|------|----------|
| MongoDB overview | [Fireship: MongoDB in 100 Seconds](https://www.youtube.com/watch?v=-bt_y4Loofg) | 2 min |
| Elasticsearch overview | [Fireship: Elasticsearch in 100 Seconds](https://www.youtube.com/watch?v=ZP0NmfyfsoM) | 2 min |
| Vector search | [Pinecone: What is Similarity Search?](https://www.pinecone.io/learn/what-is-similarity-search/) | 10 min read |
| RAG end-to-end | [RAG from Scratch playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) | Multi-part |

---

## How This Connects to Your Bootcamp Journey

```
Week 1: PROCESS text  →  tokenize, encode, classify (BERT on IMDb)
Week 2: ENCODE text   →  integer → one-hot → TF-IDF → transformer → zero-shot
Week 3: EMBED text    →  FastText/GloVe, cosine similarity, clustering
Week 4: STORE text    →  MongoDB + Elasticsearch + vectors     ← YOU ARE HERE
Week 5: RETRIEVE text →  LangChain retrieval chains
Week 6: GENERATE      →  RAG (retrieve + LLM = grounded answers)
Week 7: ORCHESTRATE   →  Agents and LangGraph
```

Each week builds on the previous. The embeddings you learned in Week 2–3 become the vectors you store this week. The storage layer you build this week becomes the retrieval backend for Week 5–6.

---

## 7. Beyond This Homework — How to Improve the Pipeline

Running an irrelevant query (like "best pizza recipe") against our library revealed four clear weaknesses. Each one is a step toward a production-quality retrieval system.

### 7a. Chunking — Split Documents into Smaller Pieces

We currently embed only the first 512 characters of each document. A 300-page book is represented by its first paragraph. The fix is to split documents into overlapping chunks (500–1000 tokens each) and embed each chunk separately.

**Reading**:
- [Chunking Strategies for LLM Applications (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/) — visual guide to different approaches
- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters) — practical implementations
- [Greg Kamradt: 5 Levels of Text Splitting (YouTube)](https://www.youtube.com/watch?v=8OJC21T2SL4) — from character splitting to semantic splitting in 20 minutes

### 7b. Hybrid Search — Combine Keyword + Vector Results

Instead of running three separate searches, merge BM25 and vector results into a single ranked list using Reciprocal Rank Fusion (RRF).

**Reading**:
- [Reciprocal Rank Fusion (Elasticsearch)](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — official guide
- [Hybrid Search Explained (Qdrant)](https://qdrant.tech/articles/hybrid-search/) — concept explainer with diagrams

### 7c. Score Thresholding — Let the System Say "I Don't Know"

Our pipeline always returns results, even when nothing is relevant. Setting a minimum similarity score (e.g., 0.65) prevents irrelevant documents from reaching the LLM — the first defence against hallucination.

**Reading**:
- [Reducing Hallucination in RAG (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) — production RAG guide
- [RAGAS: Evaluating RAG Pipelines](https://docs.ragas.io/en/latest/) — measuring retrieval quality

### 7d. Stronger Embedding Models

Our model (`MiniLM`, 384 dimensions) is fast but small. Larger models capture meaning better, especially for technical content. Check the MTEB leaderboard to compare options.

**Reading**:
- [MTEB Leaderboard (Hugging Face)](https://huggingface.co/spaces/mteb/leaderboard) — benchmark comparing 100+ embedding models
- [Choosing an Embedding Model for RAG (Pinecone)](https://www.pinecone.io/learn/series/rag/embedding-models-rag/) — practical selection guide

---

*Written for VBO AI & LLM Bootcamp, Week 4 (April 2026).*
