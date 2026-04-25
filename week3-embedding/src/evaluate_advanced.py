#!/usr/bin/env python3
"""Compare static vs contextual embeddings on Turkish benchmarks.

Models tested:
  1. FastText cc.tr.300       — static, subword-aware (baseline)
  2. BERTurk                  — contextual, Turkish-specific
  3. XLM-RoBERTa              — contextual, multilingual (100+ langs)
  4. Turkish BERT-NLI-STS     — sentence-transformer, Turkish fine-tuned
  5. Multilingual MiniLM      — sentence-transformer, multilingual

Benchmarks:
  - AnlamVer word similarity (500 pairs, Spearman ρ)
  - Turkish semantic analogies (7742 questions, Top-1/5/MRR)
  - Turkish syntactic analogies (206 questions, Top-1/5/MRR)
  - Clustering (90 words, 5 categories, ARI/NMI)

Usage:
  python src/evaluate_advanced.py                  # run all models
  python src/evaluate_advanced.py --models fasttext berturk  # specific models
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_utils import (
    load_fasttext_model,
    normalise_word,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")

# --------------------------------------------------------------------------- #
# Benchmark data                                                              #
# --------------------------------------------------------------------------- #

SIMILARITY_FILE = Path("data/anlamver_similarity.txt")
SEMANTIC_ANALOGY_FILE = Path("data/turkish-analogy-semantic.txt")
SYNTACTIC_ANALOGY_FILE = Path("data/SynAnalogyTr.txt")

CLUSTER_BENCHMARK = {
    "hayvanlar": [
        "kedi", "köpek", "kuş", "balık", "at", "tavuk", "inek", "aslan",
        "kaplan", "ayı", "kurt", "tilki", "tavşan", "fare", "yılan",
        "kartal", "kaplumbağa", "koyun", "keçi", "maymun",
    ],
    "araçlar": [
        "araba", "otobüs", "uçak", "tren", "bisiklet", "gemi", "kamyon",
        "motosiklet", "helikopter", "tramvay", "tekne", "vapur",
        "taksi", "minibüs", "metro", "kayık", "jet", "traktör",
    ],
    "meyveler": [
        "elma", "muz", "portakal", "çilek", "üzüm", "karpuz", "kiraz",
        "armut", "şeftali", "kayısı", "erik", "nar", "incir", "limon",
        "mandalina", "kavun", "ananas", "böğürtlen",
    ],
    "meslekler": [
        "doktor", "mühendis", "öğretmen", "avukat", "hemşire", "polis",
        "asker", "pilot", "aşçı", "garson", "berber", "terzi",
        "çiftçi", "şoför", "gazeteci", "mimar", "eczacı", "hakim",
    ],
    "spor": [
        "futbol", "basketbol", "voleybol", "tenis", "yüzme", "boks",
        "güreş", "atletizm", "kayak", "bisiklet", "golf", "kriket",
        "hokey", "badminton", "masa", "eskrim",
    ],
}


def print_header(title: str) -> None:
    width = max(70, len(title) + 4)
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


# --------------------------------------------------------------------------- #
# Model wrapper — unified interface for all model types                       #
# --------------------------------------------------------------------------- #


class EmbeddingModel:
    """Unified interface: get_vector(word) -> np.ndarray | None"""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, np.ndarray | None] = {}

    def get_vector(self, word: str) -> np.ndarray | None:
        key = normalise_word(word)
        if key in self._cache:
            return self._cache[key]
        vec = self._compute_vector(key)
        self._cache[key] = vec
        return vec

    def _compute_vector(self, word: str) -> np.ndarray | None:
        raise NotImplementedError

    def similar_by_vector(self, vec: np.ndarray, words: list[str], topn: int = 10) -> list[tuple[str, float]]:
        """Find most similar words to vec from a given word list."""
        results = []
        for w in words:
            wv = self.get_vector(w)
            if wv is None:
                continue
            sim = cosine_similarity(vec.reshape(1, -1), wv.reshape(1, -1))[0, 0]
            results.append((w, float(sim)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:topn]


class FastTextModel(EmbeddingModel):
    def __init__(self, path: str, limit: int = 200_000):
        super().__init__("FastText cc.tr.300", "static")
        self._kv = load_fasttext_model(path, limit=limit)
        self.dim = self._kv.vector_size

    def _compute_vector(self, word: str) -> np.ndarray | None:
        if word in self._kv:
            return self._kv[word]
        return None

    def similar_by_vector(self, vec, words=None, topn=10):
        """Use gensim's fast similar_by_vector when no word list is given."""
        if words is None:
            return self._kv.similar_by_vector(vec, topn=topn)
        return super().similar_by_vector(vec, words, topn)


class TransformerWordModel(EmbeddingModel):
    """Extract word-level embeddings from a transformer (BERT-like) model.

    Strategy: tokenize the word, run through the model, mean-pool the
    subword token embeddings from the last hidden state.
    """

    def __init__(self, model_id: str, name: str):
        super().__init__(name, "contextual")
        from transformers import AutoTokenizer, AutoModel
        import torch
        self._torch = torch

        logger.info("Loading transformer: %s", model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        self.dim = self._model.config.hidden_size

    def _compute_vector(self, word: str) -> np.ndarray | None:
        if not word:
            return None
        try:
            inputs = self._tokenizer(word, return_tensors="pt",
                                     truncation=True, max_length=32)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            # Mean pool over all tokens (excluding [CLS] and [SEP])
            hidden = outputs.last_hidden_state[0]  # (seq_len, dim)
            # Skip [CLS] (0) and [SEP] (-1)
            if hidden.shape[0] > 2:
                vec = hidden[1:-1].mean(dim=0).numpy()
            else:
                vec = hidden.mean(dim=0).numpy()
            return vec
        except Exception as e:
            logger.debug("Error encoding '%s': %s", word, e)
            return None


class SentenceTransformerWordModel(EmbeddingModel):
    """Use sentence-transformers to encode single words."""

    def __init__(self, model_id: str, name: str):
        super().__init__(name, "sentence-transformer")
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer: %s", model_id)
        self._st = SentenceTransformer(model_id, trust_remote_code=True)
        self.dim = self._st.get_sentence_embedding_dimension()

    def _compute_vector(self, word: str) -> np.ndarray | None:
        if not word:
            return None
        try:
            vec = self._st.encode(word, show_progress_bar=False)
            return vec
        except Exception as e:
            logger.debug("Error encoding '%s': %s", word, e)
            return None


# --------------------------------------------------------------------------- #
# Evaluation functions (model-agnostic)                                       #
# --------------------------------------------------------------------------- #


def eval_similarity(model: EmbeddingModel, filepath: Path) -> dict:
    """Word similarity on AnlamVer."""
    if not filepath.exists():
        return {}

    human_scores, model_scores = [], []
    oov = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            w1, w2, score = parts[0], parts[1], float(parts[2])
            total += 1

            v1 = model.get_vector(w1)
            v2 = model.get_vector(w2)
            if v1 is None or v2 is None:
                oov += 1
                continue

            sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
            human_scores.append(score)
            model_scores.append(float(sim))

    if len(human_scores) < 10:
        return {}

    rho, _ = spearmanr(human_scores, model_scores)
    r, _ = pearsonr(human_scores, model_scores)
    coverage = (total - oov) / total * 100

    return {
        "total": total, "valid": total - oov, "oov": oov,
        "coverage": coverage,
        "spearman": rho, "pearson": r,
    }


def eval_analogies(model: EmbeddingModel, filepath: Path) -> dict:
    """Word analogy evaluation."""
    if not filepath.exists():
        return {}

    # Parse
    categories: dict[str, list[tuple]] = defaultdict(list)
    current_cat = "unknown"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_cat = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) == 4:
                categories[current_cat].append(tuple(parts))

    # Build vocab for nearest-neighbour search
    all_words = set()
    for quads in categories.values():
        for a, b, c, d in quads:
            all_words.update([normalise_word(a), normalise_word(b),
                              normalise_word(c), normalise_word(d)])
    # Pre-cache all word vectors
    for w in all_words:
        model.get_vector(w)

    total = 0
    correct_1 = 0
    correct_5 = 0
    rr_sum = 0.0
    oov = 0
    cat_results = {}

    for cat, quads in sorted(categories.items()):
        c1 = c5 = cat_oov = 0
        cat_rr = 0.0
        cat_n = len(quads)

        for a, b, c, expected in quads:
            total += 1
            va = model.get_vector(a)
            vb = model.get_vector(b)
            vc = model.get_vector(c)

            if va is None or vb is None or vc is None:
                oov += 1
                cat_oov += 1
                continue

            result_vec = vb - va + vc
            exclude = {normalise_word(a), normalise_word(b), normalise_word(c)}

            # Search in all benchmark words
            candidates = model.similar_by_vector(result_vec, list(all_words - exclude), topn=5)
            predicted = [normalise_word(w) for w, _ in candidates]
            exp_norm = normalise_word(expected)

            if predicted and predicted[0] == exp_norm:
                c1 += 1
                correct_1 += 1

            if exp_norm in predicted:
                c5 += 1
                correct_5 += 1
                rank = predicted.index(exp_norm) + 1
                cat_rr += 1.0 / rank
                rr_sum += 1.0 / rank

        valid = cat_n - cat_oov
        cat_results[cat] = {
            "total": cat_n, "valid": valid, "oov": cat_oov,
            "top1": c1 / valid * 100 if valid else 0,
            "top5": c5 / valid * 100 if valid else 0,
            "mrr": cat_rr / valid if valid else 0,
        }

    valid_total = total - oov
    return {
        "total": total, "valid": valid_total, "oov": oov,
        "coverage": valid_total / total * 100 if total else 0,
        "top1": correct_1 / valid_total * 100 if valid_total else 0,
        "top5": correct_5 / valid_total * 100 if valid_total else 0,
        "mrr": rr_sum / valid_total if valid_total else 0,
        "categories": cat_results,
    }


def eval_clustering(model: EmbeddingModel, benchmark: dict[str, list[str]]) -> dict:
    """Clustering evaluation."""
    all_words = []
    true_labels = []
    label_map = {cat: i for i, cat in enumerate(sorted(benchmark))}

    for cat, words in sorted(benchmark.items()):
        for w in words:
            if model.get_vector(w) is not None:
                all_words.append(w)
                true_labels.append(label_map[cat])

    if len(all_words) < 10:
        return {}

    vectors = np.array([model.get_vector(w) for w in all_words])
    # L2 normalise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    k = len(benchmark)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    pred_labels = km.fit_predict(vectors)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # Purity
    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        confusion[p][t] += 1
    purity = sum(max(c.values()) for c in confusion.values()) / len(pred_labels)

    return {
        "total": sum(len(ws) for ws in benchmark.values()),
        "valid": len(all_words), "k": k,
        "ari": ari, "nmi": nmi, "purity": purity,
    }


# --------------------------------------------------------------------------- #
# FastText analogy (uses gensim's fast search, not brute-force)               #
# --------------------------------------------------------------------------- #


def eval_analogies_fasttext(model: FastTextModel, filepath: Path) -> dict:
    """Optimised analogy evaluation for FastText using gensim's index."""
    if not filepath.exists():
        return {}

    categories: dict[str, list[tuple]] = defaultdict(list)
    current_cat = "unknown"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_cat = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) == 4:
                categories[current_cat].append(tuple(parts))

    total = correct_1 = correct_5 = oov = 0
    rr_sum = 0.0
    cat_results = {}

    for cat, quads in sorted(categories.items()):
        c1 = c5 = cat_oov = 0
        cat_rr = 0.0

        for a, b, c, expected in quads:
            total += 1
            va = model.get_vector(a)
            vb = model.get_vector(b)
            vc = model.get_vector(c)

            if va is None or vb is None or vc is None:
                oov += 1
                cat_oov += 1
                continue

            result_vec = vb - va + vc
            exclude = {normalise_word(a), normalise_word(b), normalise_word(c)}

            try:
                most_sim = model._kv.similar_by_vector(result_vec, topn=15)
                filtered = [(w, s) for w, s in most_sim
                            if normalise_word(w) not in exclude][:5]
            except Exception:
                continue

            predicted = [normalise_word(w) for w, _ in filtered]
            exp_norm = normalise_word(expected)

            if predicted and predicted[0] == exp_norm:
                c1 += 1
                correct_1 += 1

            if exp_norm in predicted:
                c5 += 1
                correct_5 += 1
                rank = predicted.index(exp_norm) + 1
                cat_rr += 1.0 / rank
                rr_sum += 1.0 / rank

        valid = len(quads) - cat_oov
        cat_results[cat] = {
            "total": len(quads), "valid": valid, "oov": cat_oov,
            "top1": c1 / valid * 100 if valid else 0,
            "top5": c5 / valid * 100 if valid else 0,
            "mrr": cat_rr / valid if valid else 0,
        }

    valid_total = total - oov
    return {
        "total": total, "valid": valid_total, "oov": oov,
        "coverage": valid_total / total * 100 if total else 0,
        "top1": correct_1 / valid_total * 100 if valid_total else 0,
        "top5": correct_5 / valid_total * 100 if valid_total else 0,
        "mrr": rr_sum / valid_total if valid_total else 0,
        "categories": cat_results,
    }


# --------------------------------------------------------------------------- #
# Run all benchmarks for one model                                            #
# --------------------------------------------------------------------------- #


def run_benchmarks(model: EmbeddingModel) -> dict:
    """Run all benchmarks and return results dict."""
    print_header(f"Evaluating: {model.name} ({model.model_type})")
    t0 = time.time()

    # Similarity
    print("  Running AnlamVer similarity...")
    sim = eval_similarity(model, SIMILARITY_FILE)
    if sim:
        print(f"    Spearman ρ = {sim['spearman']:.4f}  "
              f"(coverage: {sim['coverage']:.1f}%)")

    # Analogies — use fast path for FastText, brute-force for transformers
    print("  Running semantic analogies...")
    if isinstance(model, FastTextModel):
        sem = eval_analogies_fasttext(model, SEMANTIC_ANALOGY_FILE)
    else:
        sem = eval_analogies(model, SEMANTIC_ANALOGY_FILE)
    if sem:
        print(f"    Top-1={sem['top1']:.1f}%  Top-5={sem['top5']:.1f}%  "
              f"MRR={sem['mrr']:.3f}  (coverage: {sem['coverage']:.1f}%)")

    print("  Running syntactic analogies...")
    if isinstance(model, FastTextModel):
        syn = eval_analogies_fasttext(model, SYNTACTIC_ANALOGY_FILE)
    else:
        syn = eval_analogies(model, SYNTACTIC_ANALOGY_FILE)
    if syn:
        print(f"    Top-1={syn['top1']:.1f}%  Top-5={syn['top5']:.1f}%  "
              f"MRR={syn['mrr']:.3f}  (coverage: {syn['coverage']:.1f}%)")

    # Clustering
    print("  Running clustering...")
    clu = eval_clustering(model, CLUSTER_BENCHMARK)
    if clu:
        print(f"    ARI={clu['ari']:.4f}  NMI={clu['nmi']:.4f}  Purity={clu['purity']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    return {
        "name": model.name,
        "type": model.model_type,
        "similarity": sim,
        "semantic_analogy": sem,
        "syntactic_analogy": syn,
        "clustering": clu,
        "time": elapsed,
    }


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def save_comparison_report(all_results: list[dict], path: Path) -> None:
    """Save a comprehensive comparison report."""
    lines = [
        "# Word Embedding Model Comparison — Turkish Benchmarks",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Model | Type | AnlamVer<br>Spearman ρ | Semantic<br>Top-5 / MRR | "
        "Syntactic<br>Top-5 / MRR | Cluster<br>ARI | Time |",
        "|-------|------|----------------------|------------------------|"
        "------------------------|------------|------|",
    ]

    for r in all_results:
        sim = r.get("similarity", {})
        sem = r.get("semantic_analogy", {})
        syn = r.get("syntactic_analogy", {})
        clu = r.get("clustering", {})

        sim_str = f"{sim.get('spearman', 0):.4f}" if sim else "—"
        sem_str = (f"{sem.get('top5', 0):.1f}% / {sem.get('mrr', 0):.3f}"
                   if sem else "—")
        syn_str = (f"{syn.get('top5', 0):.1f}% / {syn.get('mrr', 0):.3f}"
                   if syn else "—")
        clu_str = f"{clu.get('ari', 0):.4f}" if clu else "—"
        time_str = f"{r.get('time', 0):.0f}s"

        lines.append(
            f"| **{r['name']}** | {r['type']} | {sim_str} | {sem_str} | "
            f"{syn_str} | {clu_str} | {time_str} |"
        )

    # Detailed per-model results
    for r in all_results:
        lines += ["", "---", "",
                   f"## {r['name']} ({r['type']})", ""]

        sim = r.get("similarity", {})
        if sim:
            lines += [
                "### Word Similarity (AnlamVer)",
                f"- Spearman ρ = **{sim['spearman']:.4f}**, Pearson r = {sim['pearson']:.4f}",
                f"- Coverage: {sim['valid']}/{sim['total']} ({sim['coverage']:.1f}%)",
                "",
            ]

        for label, key in [("Semantic Analogies", "semantic_analogy"),
                           ("Syntactic Analogies", "syntactic_analogy")]:
            ana = r.get(key, {})
            if not ana:
                continue
            lines += [
                f"### {label}",
                f"- Top-1 = **{ana['top1']:.1f}%**, Top-5 = **{ana['top5']:.1f}%**, MRR = **{ana['mrr']:.3f}**",
                f"- Coverage: {ana['valid']}/{ana['total']} ({ana['coverage']:.1f}%)",
                "",
            ]
            if "categories" in ana:
                lines += [
                    "| Category | n | Top-1 | Top-5 | MRR |",
                    "|----------|---|-------|-------|-----|",
                ]
                for cat, cr in sorted(ana["categories"].items()):
                    lines.append(
                        f"| {cat} | {cr['valid']} | {cr['top1']:.1f}% | "
                        f"{cr['top5']:.1f}% | {cr['mrr']:.3f} |"
                    )
                lines.append("")

        clu = r.get("clustering", {})
        if clu:
            lines += [
                "### Clustering",
                f"- ARI = **{clu['ari']:.4f}**, NMI = {clu['nmi']:.4f}, Purity = {clu['purity']:.4f}",
                f"- {clu['valid']}/{clu['total']} words, k={clu['k']}",
                "",
            ]

    # Interpretation
    lines += [
        "---", "",
        "## Interpretation", "",
        "- **Spearman ρ** on AnlamVer: measures correlation with human similarity judgements. "
        "Higher is better. Static embeddings typically score 0.4–0.6, contextual 0.5–0.7.",
        "- **Analogy Top-5 / MRR**: contextual models may score lower on analogies because "
        "they were not designed for word-level vector arithmetic.",
        "- **Clustering ARI**: most models achieve near-perfect clustering on well-separated "
        "categories. The real test is on ambiguous or overlapping categories.",
        "- **Coverage**: FastText has the largest vocabulary but contextual models handle any "
        "word via subword tokenisation (100% coverage by construction).",
        "",
        "---", "",
        "*Benchmarks: AnlamVer (Ercan & Yıldız, 2018), "
        "Turkish Semantic/Syntactic Analogies (Kıyak et al.)*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {path}")


# --------------------------------------------------------------------------- #
# Model definitions                                                           #
# --------------------------------------------------------------------------- #

AVAILABLE_MODELS = {
    "fasttext": {
        "name": "FastText cc.tr.300",
        "description": "Static, subword-aware, Turkish (baseline)",
    },
    "berturk": {
        "name": "BERTurk",
        "id": "dbmdz/bert-base-turkish-cased",
        "description": "Contextual, Turkish-specific BERT",
    },
    "xlm-roberta": {
        "name": "XLM-RoBERTa-base",
        "id": "xlm-roberta-base",
        "description": "Contextual, multilingual (100+ languages)",
    },
    "turkish-nli": {
        "name": "Turkish BERT-NLI-STS",
        "id": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        "description": "Sentence-transformer, Turkish fine-tuned for similarity",
    },
    "minilm": {
        "name": "Multilingual MiniLM",
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Sentence-transformer, multilingual (50+ languages)",
    },
}


def load_model(key: str, fasttext_path: str, fasttext_limit: int) -> EmbeddingModel:
    if key == "fasttext":
        return FastTextModel(fasttext_path, limit=fasttext_limit)
    elif key in ("berturk", "xlm-roberta"):
        return TransformerWordModel(AVAILABLE_MODELS[key]["id"], AVAILABLE_MODELS[key]["name"])
    elif key in ("turkish-nli", "minilm"):
        return SentenceTransformerWordModel(AVAILABLE_MODELS[key]["id"], AVAILABLE_MODELS[key]["name"])
    else:
        raise ValueError(f"Unknown model: {key}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Compare embedding models on Turkish benchmarks")
    p.add_argument("--models", nargs="+", default=list(AVAILABLE_MODELS.keys()),
                   choices=list(AVAILABLE_MODELS.keys()),
                   help="Models to evaluate")
    p.add_argument("--fasttext-path", default="data/cc.tr.300.vec")
    p.add_argument("--fasttext-limit", type=int, default=200_000)
    args = p.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("\nModels to evaluate:")
    for key in args.models:
        m = AVAILABLE_MODELS[key]
        print(f"  - {m['name']}: {m['description']}")

    all_results = []

    for key in args.models:
        try:
            model = load_model(key, args.fasttext_path, args.fasttext_limit)
            results = run_benchmarks(model)
            all_results.append(results)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", key, e)
            import traceback
            traceback.print_exc()

    if all_results:
        # Print summary table
        print_header("FINAL COMPARISON")
        print(f"  {'Model':35s} {'Spearman':>10s} {'Sem Top-5':>10s} "
              f"{'Syn Top-5':>10s} {'ARI':>8s} {'Time':>8s}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

        for r in all_results:
            sim = r.get("similarity", {})
            sem = r.get("semantic_analogy", {})
            syn = r.get("syntactic_analogy", {})
            clu = r.get("clustering", {})

            print(f"  {r['name']:35s} "
                  f"{sim.get('spearman', 0):10.4f} "
                  f"{sem.get('top5', 0):9.1f}% "
                  f"{syn.get('top5', 0):9.1f}% "
                  f"{clu.get('ari', 0):8.4f} "
                  f"{r.get('time', 0):7.0f}s")

        save_comparison_report(all_results, OUTPUTS_DIR / "model_comparison.md")

    print_header("Done")


if __name__ == "__main__":
    main()
