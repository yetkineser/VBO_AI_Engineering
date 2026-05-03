"""
Document loader for the HR RAG chatbot.

Uses LangChain's `DirectoryLoader` (one instance per file format) to walk
hr_documents_pack/initial_docs/, then splits each document into ~500-char
chunks with 100-char overlap and attaches the 13-field metadata dict
required by the homework spec to every chunk.

Why three DirectoryLoaders instead of one?
  Each `DirectoryLoader` takes a single `loader_cls`. We need different
  loaders per extension (Docx2txtLoader / PyPDFLoader / a fallback-encoding
  TextLoader), so we run one DirectoryLoader per glob and merge the results.

Why the FallbackTextLoader subclass?
  langchain_community's TextLoader with `autodetect_encoding=True` is broken
  on chardet>=5.2 (FileEncoding rejects chardet's new `mime_type` key).
  Subclassing TextLoader to walk a small encoding list keeps DirectoryLoader
  ergonomics while sidestepping the chardet path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


class FallbackTextLoader(TextLoader):
    """TextLoader that tries common encodings in order before giving up."""

    def __init__(self, file_path: str):
        super().__init__(file_path, encoding=TEXT_ENCODINGS[0])
        self._fallback_encodings = TEXT_ENCODINGS

    def lazy_load(self):
        last_err: Exception | None = None
        for enc in self._fallback_encodings:
            try:
                self.encoding = enc
                yield from super().lazy_load()
                return
            except (UnicodeDecodeError, RuntimeError) as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Could not decode {self.file_path} with {self._fallback_encodings}"
        ) from last_err


# Per-extension loader config: (loader_cls, glob, document_type label).
LOADER_CONFIG = [
    (Docx2txtLoader,     "**/*.docx", "document"),
    (PyPDFLoader,        "**/*.pdf",  "pdf"),
    (FallbackTextLoader, "**/*.txt",  "text"),
]


def _file_metadata(path: Path, doc_type: str) -> dict:
    """Filesystem-derived metadata that does not change per chunk."""
    stat = path.stat()
    return {
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "document_type": doc_type,
        "creation_date": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def load_and_chunk(docs_dir: str | Path) -> list[Document]:
    """Use DirectoryLoader (per format) to load files, then split + stamp metadata.

    Returns a flat list of chunked Documents ready for Chroma ingestion.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks: list[Document] = []
    file_count = 0

    for loader_cls, glob, doc_type in LOADER_CONFIG:
        loader = DirectoryLoader(
            str(docs_dir),
            glob=glob,
            loader_cls=loader_cls,
            show_progress=False,
            use_multithreading=False,
        )
        raw_docs = loader.load()
        if not raw_docs:
            continue

        # DirectoryLoader returns one Document per file (multi-page PDFs return
        # one Document per page). Group by source path to compute character_count
        # and per-file metadata once.
        by_source: dict[str, list[Document]] = {}
        for d in raw_docs:
            by_source.setdefault(d.metadata.get("source", ""), []).append(d)

        for source, docs in by_source.items():
            path = Path(source)
            file_count += 1
            full_text = "\n".join(d.page_content for d in docs)
            character_count = len(full_text)
            file_meta = _file_metadata(path, doc_type)

            for raw in docs:
                page_number = raw.metadata.get("page")          # PDF page
                section_title = raw.metadata.get("title") or raw.metadata.get("section")
                for chunk_text in splitter.split_text(raw.page_content):
                    metadata = {
                        **file_meta,
                        "character_count": character_count,
                        "chunk_index": len(all_chunks),
                        "chunk_size": len(chunk_text),
                        "chunk_overlap": CHUNK_OVERLAP,
                        "ingestion_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "page_number": page_number,
                        "section_title": section_title,
                    }
                    # Chroma rejects None values — coerce to "" so every chunk has all 13 keys.
                    metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
                    all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    logger.info("Produced %d chunks across %d files", len(all_chunks), file_count)
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    here = Path(__file__).parent
    chunks = load_and_chunk(here / "hr_documents_pack" / "initial_docs")
    print(f"Loaded {len(chunks)} chunks. Sample metadata keys:")
    if chunks:
        for k in chunks[0].metadata:
            print(f"  - {k}")
