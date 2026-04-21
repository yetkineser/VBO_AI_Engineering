"""
Configuration for the Week 4 Document Ingestion Pipeline.
Paths, model name, connection settings — all in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / "Desktop" / "week_4_researchs"
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".azw3", ".md", ".txt"}

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384          # output dimension of the model above
MAX_TEXT_LENGTH = 512         # characters to encode (used in v1 whole-doc embedding)

# ---------------------------------------------------------------------------
# Chunking (v2 improvement)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500             # tokens per chunk
CHUNK_OVERLAP = 50           # overlapping tokens between consecutive chunks

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "week4_vectorization"
MONGO_COLLECTION = "file_metadata"

# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------
ES_URL = "http://localhost:9200"
ES_TEXT_INDEX = "documents"
ES_VECTOR_INDEX = "document_vectors"
ES_CHUNKS_INDEX = "document_chunks"     # v2: combined text + vector per chunk

# ---------------------------------------------------------------------------
# Search quality (v2 improvement)
# ---------------------------------------------------------------------------
SCORE_THRESHOLD = 0.65       # minimum cosine similarity to consider relevant

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
