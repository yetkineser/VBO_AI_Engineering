#!/usr/bin/env python3
"""Week 3 — Turkish Word Embeddings CLI.

Loads a pre-trained word embedding model, demonstrates similarity computation,
runs K-Means clustering on a curated word list, and saves all results to
`outputs/`.

Usage
-----
  python src/main.py                              # default: data/cc.tr.300.vec
  python src/main.py --model data/cc.tr.300.vec   # explicit FastText path
  python src/main.py --model data/glove.tr.300.txt --model-type glove
  python src/main.py --limit 100000               # load fewer words (faster)
  python src/main.py --k 4                        # change number of clusters
  python src/main.py --analogy                    # run analogy demo
  python src/main.py --visualise                  # generate t-SNE scatter plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure `src/` is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_utils import (
    cluster_words,
    get_word_vector,
    load_fasttext_model,
    load_glove_model,
    normalise_word,
    word_similarity,
)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_PATH = "data/cc.tr.300.vec"
DEFAULT_LIMIT = 200_000
DEFAULT_K = 3

# Words for the clustering demo — three intuitive categories
CLUSTER_WORDS = [
    # Animals
    "kedi", "köpek", "kuş", "balık", "at", "tavuk", "inek", "aslan",
    # Vehicles
    "araba", "otobüs", "uçak", "tren", "bisiklet", "gemi", "kamyon", "motosiklet",
    # Fruits
    "elma", "muz", "portakal", "çilek", "üzüm", "karpuz", "kiraz", "armut",
]

# Word pairs for similarity measurement
SIMILARITY_PAIRS = [
    ("kedi", "köpek"),
    ("araba", "otobüs"),
    ("elma", "muz"),
    ("kedi", "araba"),
    ("güzel", "çirkin"),
    ("iyi", "kötü"),
    ("büyük", "küçük"),
    ("hızlı", "yavaş"),
    ("okul", "üniversite"),
    ("doktor", "hastane"),
    ("bilgisayar", "telefon"),
    ("kitap", "dergi"),
]

# Analogy quads: (A, B, C) → we expect D ≈ C + (B - A)
ANALOGY_QUADS = [
    # Gender
    ("erkek", "kadın", "kral", "kraliçe"),
    ("erkek", "kadın", "baba", "anne"),
    ("erkek", "kadın", "oğul", "kız"),
    ("erkek", "kadın", "amca", "teyze"),
    ("erkek", "kadın", "dede", "nine"),
    # Country → Capital
    ("türkiye", "ankara", "almanya", "berlin"),
    ("türkiye", "ankara", "fransa", "paris"),
    ("türkiye", "ankara", "japonya", "tokyo"),
    ("türkiye", "ankara", "rusya", "moskova"),
    ("türkiye", "ankara", "ingiltere", "londra"),
    # Antonyms / Adjectives
    ("iyi", "kötü", "güzel", "çirkin"),
    ("sıcak", "soğuk", "büyük", "küçük"),
    ("hızlı", "yavaş", "uzun", "kısa"),
    ("zengin", "fakir", "güçlü", "zayıf"),
    # Verb tense
    ("gitmek", "geldi", "yazmak", "yazdı"),
    ("okumak", "okudu", "yazmak", "yazdı"),
    # Profession → Workplace
    ("doktor", "hastane", "öğretmen", "okul"),
    ("doktor", "hastane", "asker", "kışla"),
    # Country → Language
    ("fransa", "fransızca", "almanya", "almanca"),
    ("fransa", "fransızca", "japonya", "japonca"),
    ("fransa", "fransızca", "türkiye", "türkçe"),
    # Country → Currency (harder)
    ("türkiye", "lira", "amerika", "dolar"),
    ("türkiye", "lira", "japonya", "yen"),
    # Animal → Sound
    ("köpek", "havlamak", "kedi", "miyavlamak"),
    # Comparative
    ("iyi", "daha", "çok", "fazla"),
]

OUTPUTS_DIR = Path("outputs")

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def print_header(title: str) -> None:
    width = max(60, len(title) + 4)
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def safe_format(value: float, fmt: str = ".4f") -> str:
    """Format a float; return 'N/A (OOV)' for NaN."""
    if math.isnan(value):
        return "N/A (OOV)"
    return f"{value:{fmt}}"


# --------------------------------------------------------------------------- #
# Step 1: Model loading                                                       #
# --------------------------------------------------------------------------- #


def load_model(args: argparse.Namespace):
    """Load the embedding model based on CLI args."""
    path = args.model
    limit = args.limit

    if args.model_type == "glove":
        return load_glove_model(path, limit=limit)
    return load_fasttext_model(path, limit=limit)


# --------------------------------------------------------------------------- #
# Step 2: Word vector demo                                                    #
# --------------------------------------------------------------------------- #


def demo_word_vectors(model, words: list[str], n_dims: int = 10) -> list[dict]:
    """Print the first `n_dims` dimensions of selected word vectors."""
    print_header("Word Vectors (first {} dimensions)".format(n_dims))
    rows = []
    for w in words:
        vec = get_word_vector(model, w)
        if vec is None:
            print(f"  {w:20s}  →  OOV")
            rows.append({"word": w, "status": "OOV", "vector_preview": ""})
        else:
            preview = ", ".join(f"{v:.4f}" for v in vec[:n_dims])
            print(f"  {w:20s}  →  [{preview}, ...]")
            rows.append({"word": w, "status": "found", "vector_preview": preview})
    return rows


# --------------------------------------------------------------------------- #
# Step 3: Similarity                                                          #
# --------------------------------------------------------------------------- #


def demo_similarity(model, pairs: list[tuple[str, str]]) -> list[dict]:
    """Compute and print cosine similarity for word pairs."""
    print_header("Cosine Similarity")
    rows = []
    for w1, w2 in pairs:
        sim = word_similarity(model, w1, w2)
        print(f"  {w1:15s} ↔ {w2:15s} : {safe_format(sim)}")
        rows.append({"word1": w1, "word2": w2, "similarity": sim})
    return rows


# --------------------------------------------------------------------------- #
# Step 4: Clustering                                                          #
# --------------------------------------------------------------------------- #


def demo_clustering(model, words: list[str], k: int) -> dict[str, int]:
    """Cluster words and pretty-print the results."""
    print_header(f"K-Means Clustering (k={k})")
    assignments = cluster_words(model, words, k=k)

    if not assignments:
        print("  No valid words to cluster (all OOV).")
        return {}

    # Group by cluster label
    groups: dict[int, list[str]] = defaultdict(list)
    for w, lbl in sorted(assignments.items(), key=lambda x: x[1]):
        groups[lbl].append(w)

    for lbl in sorted(groups):
        members = ", ".join(groups[lbl])
        print(f"  Cluster {lbl}: {members}")

    return assignments


# --------------------------------------------------------------------------- #
# Step 5: Analogy                                                             #
# --------------------------------------------------------------------------- #


def demo_analogy(model, quads: list[tuple[str, str, str, str]], topn: int = 5) -> list[dict]:
    """Compute A - B + C and see if the expected D is in the top results."""
    print_header("Analogy: A is to B as C is to ?")
    rows = []

    for a, b, c, expected in quads:
        va = get_word_vector(model, a)
        vb = get_word_vector(model, b)
        vc = get_word_vector(model, c)

        if va is None or vb is None or vc is None:
            oov = [w for w, v in [(a, va), (b, vb), (c, vc)] if v is None]
            print(f"  {a} → {b} :: {c} → ?   (skipped — OOV: {', '.join(oov)})")
            rows.append({"a": a, "b": b, "c": c, "expected": expected, "result": "OOV"})
            continue

        # result_vec ≈ b - a + c
        result_vec = vb - va + vc

        # Find most similar words, excluding the input words
        try:
            most_similar = model.similar_by_vector(result_vec, topn=topn + 3)
            filtered = [(w, s) for w, s in most_similar
                        if normalise_word(w) not in {normalise_word(a), normalise_word(b), normalise_word(c)}][:topn]
        except Exception:
            filtered = []

        if filtered:
            top_word, top_sim = filtered[0]
            rest = ", ".join(f"{w} ({s:.3f})" for w, s in filtered[1:])
            hit = "✓" if normalise_word(top_word) == normalise_word(expected) else " "
            print(f"  {a} → {b} :: {c} → {top_word} ({top_sim:.3f})  [expected: {expected}] {hit}")
            if rest:
                print(f"    Other candidates: {rest}")
            rows.append({"a": a, "b": b, "c": c, "expected": expected, "result": top_word, "score": top_sim})
        else:
            print(f"  {a} → {b} :: {c} → ?  (no results)")
            rows.append({"a": a, "b": b, "c": c, "expected": expected, "result": "N/A"})

    return rows


# --------------------------------------------------------------------------- #
# Step 6: Visualisation (optional)                                            #
# --------------------------------------------------------------------------- #


def demo_visualise(model, words: list[str], assignments: dict[str, int]) -> None:
    """Generate a 2D t-SNE scatter plot of clustered words."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib or sklearn not installed — skipping visualisation")
        return

    print_header("Visualisation (t-SNE)")

    valid_words = [w for w in words if get_word_vector(model, w) is not None]
    if len(valid_words) < 3:
        print("  Not enough valid words for t-SNE.")
        return

    vectors = np.array([get_word_vector(model, w) for w in valid_words])
    labels = [assignments.get(w, -1) for w in valid_words]

    perplexity = min(5, len(valid_words) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vectors)

    # Colour palette
    unique_labels = sorted(set(labels))
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, lbl in enumerate(unique_labels):
        mask = [j for j, l in enumerate(labels) if l == lbl]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=colours[i],
            label=f"Cluster {lbl}",
            s=100,
            edgecolors="white",
            linewidth=0.5,
        )
        for j in mask:
            ax.annotate(
                valid_words[j],
                (coords[j, 0], coords[j, 1]),
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_title("Word Embeddings — t-SNE Visualisation with K-Means Clusters")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = OUTPUTS_DIR / "tsne_clusters.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# --------------------------------------------------------------------------- #
# Output saving                                                               #
# --------------------------------------------------------------------------- #


def save_similarity_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word1", "word2", "similarity"])
        writer.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["similarity"] = safe_format(r2["similarity"])
            writer.writerow(r2)
    print(f"  Saved → {path}")


def save_cluster_csv(assignments: dict[str, int], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "cluster"])
        writer.writeheader()
        for w, lbl in sorted(assignments.items(), key=lambda x: (x[1], x[0])):
            writer.writerow({"word": w, "cluster": lbl})
    print(f"  Saved → {path}")


def save_results_md(
    sim_rows: list[dict],
    assignments: dict[str, int],
    analogy_rows: list[dict] | None,
    model_info: str,
    path: Path,
) -> None:
    """Generate a human-readable Markdown report of all results."""
    lines = [
        "# Week 3 — Word Embedding Results",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Model: {model_info}*",
        "",
        "---",
        "",
        "## Cosine Similarity",
        "",
        "| Word 1 | Word 2 | Similarity |",
        "|--------|--------|------------|",
    ]
    for r in sim_rows:
        lines.append(f"| {r['word1']} | {r['word2']} | {safe_format(r['similarity'])} |")

    lines += ["", "---", "", "## K-Means Clustering", ""]
    groups: dict[int, list[str]] = defaultdict(list)
    for w, lbl in sorted(assignments.items(), key=lambda x: x[1]):
        groups[lbl].append(w)
    for lbl in sorted(groups):
        members = ", ".join(groups[lbl])
        lines.append(f"- **Cluster {lbl}:** {members}")

    if analogy_rows:
        lines += ["", "---", "", "## Analogies", ""]
        lines.append("| A | B | C | Expected | Result | Score |")
        lines.append("|---|---|---|----------|--------|-------|")
        for r in analogy_rows:
            score = safe_format(r.get("score", float("nan")))
            lines.append(f"| {r['a']} | {r['b']} | {r['c']} | {r['expected']} | {r['result']} | {score} |")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Week 3 — Turkish Word Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH),
        help=f"Path to the embedding file (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--model-type",
        choices=["fasttext", "glove"],
        default="fasttext",
        help="Type of embedding file (default: fasttext)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Load only the top-N most frequent words (default: {DEFAULT_LIMIT})",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of clusters for K-Means (default: {DEFAULT_K})",
    )
    p.add_argument(
        "--analogy",
        action="store_true",
        help="Run the analogy demo (king - man + woman = ?)",
    )
    p.add_argument(
        "--visualise",
        "--visualize",
        action="store_true",
        help="Generate a t-SNE scatter plot of clustered words",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run everything: similarity, clustering, analogy, visualisation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        args.analogy = True
        args.visualise = True

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ---- Load model ----
    model = load_model(args)
    model_info = f"{args.model_type} — {args.model} (limit={args.limit}, vocab={len(model)})"
    print(f"\nModel loaded: {model_info}")

    # ---- Word vector preview ----
    demo_words = ["kedi", "köpek", "araba", "elma", "türkiye", "bilgisayar"]
    demo_word_vectors(model, demo_words)

    # ---- Similarity ----
    sim_rows = demo_similarity(model, SIMILARITY_PAIRS)
    save_similarity_csv(sim_rows, OUTPUTS_DIR / "similarity.csv")

    # ---- Clustering ----
    assignments = demo_clustering(model, CLUSTER_WORDS, k=args.k)
    if assignments:
        save_cluster_csv(assignments, OUTPUTS_DIR / "clusters.csv")

    # ---- Analogy (optional) ----
    analogy_rows = None
    if args.analogy:
        analogy_rows = demo_analogy(model, ANALOGY_QUADS)

    # ---- Visualisation (optional) ----
    if args.visualise and assignments:
        demo_visualise(model, CLUSTER_WORDS, assignments)

    # ---- Markdown report ----
    save_results_md(sim_rows, assignments, analogy_rows, model_info, OUTPUTS_DIR / "results.md")

    print_header("Done")
    print(f"  All outputs saved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
