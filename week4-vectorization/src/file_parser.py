"""
Extract text and metadata from PDF, EPUB, AZW3, and Markdown files.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from config import SUPPORTED_EXTENSIONS


def extract_metadata(path: Path) -> dict:
    """Return file-level metadata (no content reading)."""
    stat = path.stat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_at": datetime.datetime.fromtimestamp(stat.st_ctime),
        "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime),
        "full_path": str(path.absolute()),
    }


def extract_pdf_text(path: Path) -> tuple[str, dict]:
    """Extract text + extra metadata from a PDF."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    text = "\n".join(pages)

    info = doc.metadata or {}
    extra = {
        "page_count": len(doc),
        "author": info.get("author", ""),
        "title": info.get("title", "") or path.stem,
    }
    doc.close()
    return text, extra


def extract_epub_text(path: Path) -> tuple[str, dict]:
    """Extract text + extra metadata from an EPUB."""
    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    # Metadata
    title = ""
    author = ""
    for meta in book.get_metadata("DC", "title"):
        title = meta[0]
        break
    for meta in book.get_metadata("DC", "creator"):
        author = meta[0]
        break

    # Content
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        parts.append(soup.get_text(separator="\n", strip=True))

    text = "\n".join(parts)
    extra = {
        "author": author,
        "title": title or path.stem,
    }
    return text, extra


def extract_markdown_text(path: Path) -> tuple[str, dict]:
    """Read a markdown or plain text file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    extra = {"title": path.stem}
    return text, extra


def parse_file(file_path: str | Path) -> dict | None:
    """
    Parse a single file and return a dict with metadata + text.
    Returns None if the file type is not supported.
    """
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None

    metadata = extract_metadata(path)

    try:
        if path.suffix.lower() == ".pdf":
            text, extra = extract_pdf_text(path)
        elif path.suffix.lower() in (".epub", ".azw3"):
            text, extra = extract_epub_text(path)
        elif path.suffix.lower() in (".md", ".txt"):
            text, extra = extract_markdown_text(path)
        else:
            return None
    except Exception as e:
        print(f"  [WARN] Could not parse {path.name}: {e}")
        return None

    metadata.update(extra)
    metadata["word_count"] = len(text.split())
    metadata["text"] = text
    return metadata


def scan_folder(folder: str | Path) -> list[dict]:
    """Scan a folder and parse all supported files."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Found {len(files)} supported files in {folder}")

    results = []
    for f in files:
        doc = parse_file(f)
        if doc:
            results.append(doc)
            print(f"  ✓ {doc['filename']} ({doc['word_count']} words)")
    return results
