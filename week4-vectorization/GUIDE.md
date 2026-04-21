# Week 4 Homework Guide — Document Ingestion Pipeline

> **Assignment**: "Bir klasördeki dosyaları, hem metadata (mongo), hem metin (elastic search), hem de vektör olarak kaydedelim."
>
> **Translation**: "Let's save files from a folder as metadata (MongoDB), as text (Elasticsearch), and as vectors."
>
> **Instructor**: Erkan ŞİRİN · Apr 16 · 100 points · Due: Tomorrow 7:59 PM

---

## What Is This Homework Really About?

This assignment is the **missing piece** between what you have already built and where the bootcamp is heading. Let me explain why.

### Your journey so far

| Week | What you built | What you learned |
|------|---------------|-----------------|
| Week 1 | Text classification (BERT/DistilBERT on IMDb) | Fine-tuning transformers, GPU training, model evaluation |
| Week 2 | Sentiment analysis with multiple encodings | Integer → One-hot → TF-IDF → Transformer embeddings → Zero-shot → Fine-tuning |
| Week 3 | Turkish word embeddings (FastText/GloVe) | Cosine similarity, K-Means clustering, word analogies, t-SNE |

### What Week 4 adds

In Weeks 1–3, you processed text and threw it at a model. But you never **stored** anything in a way that a production system could search and retrieve later. That is exactly what this homework teaches:

```
Raw files in a folder
    ├── Metadata (who wrote it, when, file type)  →  MongoDB
    ├── Full text (searchable by keywords)         →  Elasticsearch
    └── Dense vectors (searchable by meaning)      →  Vector DB / Elasticsearch
```

**This is the foundation of every RAG system.** When you build a RAG chatbot in Week 6, it will need exactly this pipeline to find relevant documents before sending them to the LLM.

---

## The Three Storage Layers Explained

### Layer 1: Metadata in MongoDB

**What is metadata?** Information *about* the file, not the file content itself:
- File name, path, size, extension
- Creation date, last modified date
- Author (if extractable)
- Number of pages, word count
- Any tags or categories you assign

**Why MongoDB?** It is a document database — each record is a flexible JSON object. Unlike SQL tables, you don't need to define columns in advance. Perfect for metadata that varies across file types (a PDF has page count, a `.txt` does not).

**What you will learn:**
- How to connect Python to MongoDB
- How to design a document schema for file metadata
- How to insert, query, and update documents
- Why NoSQL is sometimes better than SQL for semi-structured data

### Layer 2: Full Text in Elasticsearch

**What is Elasticsearch?** A search engine built for text. When you type a query like "push notification allocation", Elasticsearch finds all documents containing those words, ranks them by relevance (using BM25 — a smarter version of TF-IDF), and returns them in milliseconds.

**Why not just MongoDB for text?** MongoDB can store text, but it cannot do efficient full-text search. Elasticsearch builds an **inverted index** — for every word, it knows which documents contain it. This is the same structure Google uses.

**What you will learn:**
- How inverted indexes work (directly related to your Week 2 TF-IDF knowledge)
- How to index documents and run full-text queries
- Filtering, highlighting, and relevance scoring
- Why keyword search and semantic search are complementary, not competing

### Layer 3: Vectors (Embeddings)

**What are vectors in this context?** The dense numerical representations you already worked with in Week 2 (transformer embeddings) and Week 3 (FastText/GloVe). Now, instead of computing them and throwing them away, you **store** them so you can search by meaning later.

**Why store vectors separately?** Keyword search fails when:
- The user says "notification system" but the document says "push allocation pipeline"
- The query is in English but the document is in Turkish
- The user asks a question, not a keyword ("How does the tier system work?")

Vector search solves this by finding documents whose **meaning** is close to the query's meaning (cosine similarity — you know this from Week 3).

**Where to store vectors?** You have several options:
- **Elasticsearch** itself (since v8, it supports `dense_vector` fields with kNN search)
- **Qdrant** or **Chroma** (purpose-built vector databases — you will use these in Week 5–6)
- **MongoDB Atlas** (also supports vector search now)

**What you will learn:**
- How to generate embeddings for documents (you already know how)
- How to store and index those embeddings for fast retrieval
- How to run similarity search queries
- The trade-off between keyword search (precise) and vector search (semantic)

---

## Step-by-Step Implementation Plan

### Step 0: Prepare your folder of files

Create a `data/` folder with 10–20 mixed files:
- A few `.txt` files (plain text notes, README excerpts)
- A few `.pdf` files (papers, articles)
- A few `.md` files (your own Markdown notes)

You can use your own `CEM_Prisma/` markdown files — they are perfect test data.

### Step 1: Extract text and metadata from each file

```python
# Libraries you will need:
# pip install pymongo elasticsearch sentence-transformers
# pip install PyPDF2 python-docx  (for PDF/DOCX parsing)
# pip install pymupdf  (alternative PDF parser, often better)

import os, datetime
from pathlib import Path

def extract_file_info(file_path: str) -> dict:
    """Extract metadata + text content from a file."""
    path = Path(file_path)
    stat = path.stat()

    metadata = {
        "filename": path.name,
        "extension": path.suffix,
        "size_bytes": stat.st_size,
        "created_at": datetime.datetime.fromtimestamp(stat.st_ctime),
        "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime),
        "full_path": str(path.absolute()),
    }

    # Read text content based on file type
    if path.suffix == ".md" or path.suffix == ".txt":
        text = path.read_text(encoding="utf-8")
    elif path.suffix == ".pdf":
        text = extract_pdf_text(file_path)  # implement with PyPDF2 or pymupdf
    else:
        text = ""

    metadata["word_count"] = len(text.split())
    return metadata, text
```

### Step 2: Store metadata in MongoDB

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["week4_homework"]
collection = db["file_metadata"]

# Insert metadata for each file
for file_path in all_files:
    metadata, text = extract_file_info(file_path)
    metadata["file_id"] = metadata["filename"]  # simple ID
    collection.insert_one(metadata)

# Query example: find all PDF files larger than 10KB
results = collection.find({"extension": ".pdf", "size_bytes": {"$gt": 10000}})
```

### Step 3: Index full text in Elasticsearch

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Create an index with text analysis settings
es.indices.create(index="documents", body={
    "mappings": {
        "properties": {
            "filename": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "standard"},
            "embedding": {"type": "dense_vector", "dims": 384}  # for Step 4
        }
    }
})

# Index each document
for file_path in all_files:
    metadata, text = extract_file_info(file_path)
    es.index(index="documents", id=metadata["filename"], body={
        "filename": metadata["filename"],
        "content": text,
    })

# Full-text search example
results = es.search(index="documents", body={
    "query": {"match": {"content": "push notification allocation"}}
})
```

### Step 4: Generate and store vectors

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

for file_path in all_files:
    metadata, text = extract_file_info(file_path)

    # Generate embedding (you know this from Week 2!)
    embedding = model.encode(text[:512]).tolist()  # truncate long texts

    # Option A: Store in Elasticsearch (alongside the text)
    es.update(index="documents", id=metadata["filename"], body={
        "doc": {"embedding": embedding}
    })

    # Option B: Also store in MongoDB for reference
    collection.update_one(
        {"file_id": metadata["filename"]},
        {"$set": {"embedding": embedding}}
    )

# Semantic search with kNN
query_vector = model.encode("How does the tier system work?").tolist()
results = es.search(index="documents", body={
    "knn": {
        "field": "embedding",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": 20
    }
})
```

### Step 5: Compare search results

Build a small demo that shows the **same query** returning different results from each system:

```python
query = "push notification"

# MongoDB: metadata filter only
mongo_results = collection.find({"filename": {"$regex": "push", "$options": "i"}})

# Elasticsearch: full-text keyword search
es_text_results = es.search(index="documents", body={
    "query": {"match": {"content": query}}
})

# Elasticsearch: vector similarity search
es_vector_results = es.search(index="documents", body={
    "knn": {"field": "embedding", "query_vector": model.encode(query).tolist(), "k": 5}
})

# Print side by side — this is the "aha moment"
```

---

## What Tools Should You Use?

### Recommended stack (simplest path):

| Component | Tool | Why |
|-----------|------|-----|
| MongoDB | **Docker** (`mongo:latest`) | One command to start, no install hassle |
| Elasticsearch | **Docker** (`elasticsearch:8.x`) | Same — one command |
| Embeddings | **sentence-transformers** | You already used this in Week 2 |
| File parsing | **pymupdf** (PDF), **pathlib** (txt/md) | Lightweight |
| Orchestration | **Python script** | Keep it simple — no frameworks needed |

### Docker commands to start everything:

```bash
# Start MongoDB
docker run -d --name mongo -p 27017:27017 mongo:latest

# Start Elasticsearch (single-node, no security for local dev)
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.17.0
```

### Alternative: No Docker? Use managed free tiers

- **MongoDB Atlas** (free 512MB cluster): https://www.mongodb.com/cloud/atlas
- **Elastic Cloud** (14-day trial): https://cloud.elastic.co
- **Qdrant Cloud** (1GB free): https://cloud.qdrant.io

---

## Reading List — What to Study to Truly Understand This

### Must-Read (before you start coding)

1. **MongoDB Basics** — official "Getting Started" tutorial (30 min)
   https://www.mongodb.com/docs/manual/tutorial/getting-started/

2. **Elasticsearch: What is an Inverted Index?** — short conceptual explainer
   https://www.elastic.co/guide/en/elasticsearch/reference/current/documents-indices.html

3. **Vector Search Explained** — Pinecone's visual guide (the best introduction)
   https://www.pinecone.io/learn/what-is-similarity-search/

### Deepen Your Understanding

4. **BM25 vs TF-IDF** — why Elasticsearch uses BM25 instead of raw TF-IDF
   https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables

5. **Sentence Transformers Documentation** — the model you will use for embeddings
   https://www.sbert.net/docs/quickstart.html

6. **Elasticsearch Dense Vector Search** — how to do kNN with Elasticsearch
   https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html

### Connecting to What Comes Next (Week 5–6)

7. **LangChain Document Loaders** — how LangChain loads files (same problem, framework approach)
   https://python.langchain.com/docs/how_to/#document-loaders

8. **RAG from Scratch** — LangChain's YouTube series showing how retrieval feeds into generation
   https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x

### Video Resources

9. **Fireship: MongoDB in 100 Seconds** — fastest possible overview
   https://www.youtube.com/watch?v=-bt_y4Loofg

10. **Fireship: Elasticsearch in 100 Seconds**
    https://www.youtube.com/watch?v=ZP0NmfyfsoM

---

## How This Connects to Your Bigger Picture

```
Week 1–3: You learned to PROCESS text (tokenize, encode, embed, classify)
Week 4:   You learn to STORE text in searchable systems   ← YOU ARE HERE
Week 5:   You learn to RETRIEVE text with LangChain
Week 6:   You learn to GENERATE answers with RAG (retrieve + LLM)
Week 7:   You learn to ORCHESTRATE with agents (LangGraph)
```

This homework is the bridge. Without it, your Week 6 RAG chatbot would have nowhere to search.

---

## Submission Checklist

- [ ] Python script that reads all files from a folder
- [ ] Metadata extracted and stored in MongoDB (filename, size, dates, word count)
- [ ] Full text indexed in Elasticsearch with keyword search working
- [ ] Vector embeddings generated and stored (in ES or separate vector DB)
- [ ] At least one demo query showing results from all three systems
- [ ] Brief comparison: when does keyword search win? When does vector search win?

---

*This guide was prepared based on your Week 1–3 assignments and the VBO AI/LLM Bootcamp Week 4 curriculum (Vectorization & Semantic Search). Written in upper-intermediate English to support your language improvement goal.*
