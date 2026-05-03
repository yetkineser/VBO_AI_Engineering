"""
Chroma vector store helpers.

Two embedding profiles:
  - "openrouter" (spec default) — text-embedding-3-small via OpenRouter
  - "ollama" — nomic-embed-text running locally via Ollama

Different embedding dims → different Chroma collections so we never mix
1536-dim OpenRouter vectors with 768-dim Ollama vectors.

The pysqlite3 shim at the top is a workaround for environments (some Linux
distros, Streamlit Cloud) where the system sqlite3 is older than Chroma's
floor. Safe no-op when the package is not installed.
"""

try:  # noqa: SIM105
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
load_dotenv()

PERSIST_DIRECTORY = str(Path(__file__).parent / "chroma_db")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small"
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_K = 4

COLLECTIONS = {
    "openrouter": "vbo-aillm-bc-rag",          # spec collection name
    "ollama": "vbo-aillm-bc-rag-ollama",       # separate, different vector dim
}


def _embeddings(profile: str) -> Embeddings:
    if profile == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing — set it in .env.")
        return OpenAIEmbeddings(
            model=OPENROUTER_EMBED_MODEL,
            openai_api_base=OPENROUTER_BASE,
            openai_api_key=api_key,
        )
    if profile == "ollama":
        from langchain_ollama import OllamaEmbeddings  # lazy import
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE)
    raise ValueError(f"Unknown embedding profile: {profile}")


def create_vector_store(documents: list[Document], profile: str = "openrouter") -> Chroma:
    """Embed documents and persist them to the profile's collection."""
    if not documents:
        raise ValueError("No documents to ingest — did the loader return an empty list?")
    collection = COLLECTIONS[profile]
    logger.info(
        "Ingesting %d chunks into collection '%s' at %s (profile=%s)",
        len(documents), collection, PERSIST_DIRECTORY, profile,
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(profile),
        collection_name=collection,
        persist_directory=PERSIST_DIRECTORY,
    )
    logger.info("Ingestion complete. Collection size: %d", vectorstore._collection.count())
    return vectorstore


def load_vector_store(profile: str = "openrouter") -> Chroma:
    """Reopen the persisted collection for the given profile."""
    if not Path(PERSIST_DIRECTORY).exists():
        raise FileNotFoundError(
            f"No persisted index at {PERSIST_DIRECTORY}. Run `python main.py ingest` first."
        )
    return Chroma(
        collection_name=COLLECTIONS[profile],
        embedding_function=_embeddings(profile),
        persist_directory=PERSIST_DIRECTORY,
    )


def get_retriever(profile: str = "openrouter", k: int = DEFAULT_K):
    """Convenience: load the profile's store and return a top-k retriever."""
    return load_vector_store(profile).as_retriever(search_kwargs={"k": k})
