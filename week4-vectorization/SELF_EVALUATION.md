# Week 4 Self-Evaluation — Document Ingestion Pipeline

> **Assignment**: "Bir klasördeki dosyaları, hem metadata (mongo), hem metin (elastic search), hem de vektör olarak kaydedelim."
>
> **Instructor**: Erkan ŞİRİN · 100 points

---

## Requirement Checklist

### 1. Python script that reads all files from a folder

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Reads files from a folder | **Done** | `src/file_parser.py` scans `~/Desktop/week_4_researchs/` |
| Handles multiple file formats | **Done** | PDF (PyMuPDF), EPUB (EbookLib), AZW3, Markdown, plain text |
| Extracts text content | **Done** | 75 out of 78 files parsed successfully |
| Handles errors gracefully | **Done** | 3 corrupted files (bad zip) logged with `[WARN]` and skipped |

**What I learned**: PDF text extraction is not perfect. Some PDFs returned 0 words (scanned images without OCR). EPUB files occasionally have missing internal files that crash the parser. Wrapping each file in a try/except with a clear warning message is essential for real-world data.

### 2. Metadata extracted and stored in MongoDB

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Filename stored | **Done** | Used as unique key (`create_index("filename", unique=True)`) |
| File size stored | **Done** | `size_bytes` field |
| Dates stored | **Done** | `created_at` and `modified_at` from OS file stats |
| Word count stored | **Done** | Computed from extracted text (`len(text.split())`) |
| Extra metadata | **Bonus** | Also stored: `author`, `title`, `extension`, `full_path`, `page_count` (for PDFs) |
| Query capability | **Done** | `query_by_extension()`, `query_large_files()`, `query_by_author()`, `get_stats()` with aggregation pipeline |

**What I learned**: MongoDB's flexible schema is ideal for file metadata because different file types have different attributes (PDFs have page count, EPUBs do not). The `$group` aggregation pipeline is powerful for computing summary statistics (total words by extension, etc.).

### 3. Full text indexed in Elasticsearch with keyword search working

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Text indexed | **Done** | 75 documents in the `documents` index |
| Keyword search works | **Done** | `multi_match` query across `content`, `title`, `author` fields |
| Relevance ranking | **Done** | BM25 scoring with highlighted snippets |
| Bulk indexing | **Done** | Used `helpers.bulk()` for efficient batch indexing |

**What I learned**: BM25 is essentially TF-IDF with two improvements — term frequency saturation and document length normalisation. This connects directly to what I learned in Week 2 about TF-IDF encoding. The `highlight` feature is useful for showing users why a result matched, but it needs `max_analyzed_offset` for large documents.

### 4. Vector embeddings generated and stored

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Embeddings generated | **Done** | `sentence-transformers` with `paraphrase-multilingual-MiniLM-L12-v2` |
| Stored in searchable index | **Done** | Elasticsearch `dense_vector` field with `similarity: cosine` |
| kNN search works | **Done** | `search_semantic()` returns ranked results by cosine similarity |

**What I learned**: The same embedding concepts from Week 2 (transformer embeddings) and Week 3 (cosine similarity) are now being used in a production context. The key difference is that we are **storing** them instead of computing and discarding them.

### 5. Demo query showing results from all three systems

| Criterion | Status | Evidence |
|-----------|--------|----------|
| At least one demo query | **Exceeded** | 5 demo queries + 1 irrelevant stress test ("pizza tarifi") |
| Side-by-side comparison | **Done** | `src/search_demo.py` generates a full markdown report |
| Auto-generated report | **Bonus** | `outputs/search_comparison.md` — runs with `python run.py --demo` |

### 6. Brief comparison: when does keyword search win vs vector search?

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Comparison included | **Exceeded** | Detailed "When Does Each System Win?" table + per-query analysis |
| Keyword search strengths identified | **Done** | Exact phrases, technical terms, specific terminology |
| Vector search strengths identified | **Done** | Semantic similarity, question-format queries, synonym handling |
| Weaknesses of each identified | **Done** | BM25 noise with common words, vector search threshold challenges |

---

## Beyond the Requirements (Bonus Work)

### v2 Pipeline — Four Production Improvements

| Improvement | What it does | Why it matters |
|-------------|-------------|----------------|
| **Chunking** | Split 75 documents into 13,990 overlapping chunks (500 words each) | Finds the right **paragraph**, not just the right book. Semantic scores improved +0.06 to +0.10 |
| **Hybrid search (RRF)** | Merge BM25 keyword and kNN semantic results into one ranked list | Combines keyword precision with semantic understanding |
| **Score thresholding** | Filter results below 0.65 cosine similarity | First defence against feeding irrelevant context to an LLM (hallucination prevention) |
| **v1 vs v2 comparison** | `--compare` command shows old vs new results side by side | Proves each improvement measurably helped |

### Irrelevant Query Stress Test

Ran "En iyi pizza tarifi nedir?" (best pizza recipe) against a library of AI/DS books. This test revealed:

- BM25 returns confident but wrong results (matched "en" in math formulas, score 7.34)
- v1 vector search returned empty documents with identical scores (0.5929) — meaningless
- v2 vector search found paragraphs that genuinely mention pizza — more dangerous because it looks plausible
- **Lesson**: Score thresholds must be re-calibrated when changing indexing granularity (document → chunk)

### Documentation

| Document | Language | Purpose |
|----------|----------|---------|
| `README.md` | English | Project overview + Mermaid architecture diagram |
| `README_TR.md` | Turkish | Same content in Turkish |
| `GUIDE.md` | English | Step-by-step homework guide with code examples + 10 reading links |
| `LEARNING.md` | English | Learning guide: what to study and why, 20+ reading links by topic |
| `LEARNING_TR.md` | Turkish | Same content in Turkish |
| `outputs/search_comparison.md` | English | Auto-generated comparison report with analysis |
| `SELF_EVALUATION.md` | English | This file |

---

## What I Would Do Differently Next Time

1. **Start with chunking from the beginning**. Whole-document embedding was a useful baseline for comparison, but in a real project I would skip straight to chunked embeddings. The v1 → v2 comparison proved that chunking is the single highest-impact improvement.

2. **Add a cross-encoder re-ranker**. The pizza test showed that cosine similarity thresholds alone are not enough to filter irrelevant results at the chunk level. A cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores query-document pairs more carefully and would catch the false positives.

3. **Implement semantic chunking instead of fixed-size chunking**. Our current chunker splits by word count, which can cut sentences and paragraphs in the middle. A semantic chunker would split at natural topic boundaries (paragraph breaks, section headers), producing more coherent chunks.

4. **Test with a multilingual query set**. Our library contains English, Turkish, and Urdu books. The embedding model is multilingual, but we only tested with English and Turkish queries. A proper evaluation would include cross-language retrieval tests.

5. **Add proper evaluation metrics**. We compared results visually, but a production system needs automated metrics like NDCG (Normalised Discounted Cumulative Gain), MRR (Mean Reciprocal Rank), or the RAGAS framework for RAG-specific evaluation.

---

## Connection to Previous Weeks

| Week | What I built | What carried over to Week 4 |
|------|-------------|----------------------------|
| Week 1 | Text classification (BERT on IMDb) | Understanding of tokenisation and model inference |
| Week 2 | Sentiment analysis with multiple encodings | TF-IDF knowledge → directly explains how BM25 works in Elasticsearch |
| Week 3 | Turkish word embeddings (FastText/GloVe) | Cosine similarity, vector spaces → the exact same concept used in kNN search |
| **Week 4** | **Document ingestion pipeline** | **All three weeks converge: text extraction + indexing + embedding + similarity search** |

---

## Self-Assessment Score

**Requirement completion**: 6/6 (all checklist items done)

**Depth**: Went beyond the basic requirements with a v2 pipeline (chunking, hybrid search, thresholding), a stress test with analysis, and comprehensive documentation in two languages.

**Understanding demonstrated**: The comparison report shows I understand *why* each system gives different results, not just *how* to run them. The pizza test analysis connects retrieval quality to RAG hallucination — which is the real reason this pipeline matters.

**What I still need to learn**: Cross-encoder re-ranking, semantic chunking, retrieval evaluation metrics (NDCG, MRR, RAGAS). These are Week 5-6 topics.

---

*VBO AI & LLM Bootcamp, Week 4 — Yetkin Eser (April 2026)*
