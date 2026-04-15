#!/usr/bin/env python3
"""Evaluate word embeddings on Turkish benchmarks.

Benchmarks:
  1. AnlamVer — 500 Turkish word pairs with human similarity scores (Spearman ρ)
  2. Turkish Semantic Analogies — 7,742 questions across 7 categories (Top-1/5/MRR)
  3. Turkish Syntactic Analogies — 206 questions (Top-1/5/MRR)
  4. Clustering — K-Means on categorised words (ARI / NMI / Purity)

Usage:
  python src/evaluate.py
  python src/evaluate.py --model data/cc.tr.300.vec --limit 200000
  python src/evaluate.py --limit 500000   # more vocab = better coverage
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_utils import (
    cluster_words,
    get_word_vector,
    load_fasttext_model,
    load_glove_model,
    normalise_word,
    word_similarity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")

# --------------------------------------------------------------------------- #
# Benchmark paths                                                             #
# --------------------------------------------------------------------------- #

SIMILARITY_FILE = Path("data/anlamver_similarity.txt")
SEMANTIC_ANALOGY_FILE = Path("data/turkish-analogy-semantic.txt")
SYNTACTIC_ANALOGY_FILE = Path("data/SynAnalogyTr.txt")

# --------------------------------------------------------------------------- #
# Extended clustering benchmark                                               #
# --------------------------------------------------------------------------- #

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
# 1. Word Similarity (AnlamVer)                                               #
# --------------------------------------------------------------------------- #


def evaluate_similarity(model, filepath: Path) -> dict:
    """Evaluate word similarity against AnlamVer benchmark."""
    print_header("1. Word Similarity — AnlamVer (500 pairs)")

    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return {}

    human_scores = []
    model_scores = []
    oov_count = 0
    total = 0
    oov_examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            w1, w2, score = parts[0], parts[1], float(parts[2])
            total += 1

            sim = word_similarity(model, w1, w2)
            if sim is None:
                oov_count += 1
                if len(oov_examples) < 5:
                    oov_examples.append((w1, w2))
                continue

            human_scores.append(score)
            model_scores.append(sim)

    if len(human_scores) < 10:
        print(f"  Too few valid pairs ({len(human_scores)}) — skipping.")
        return {}

    spearman_rho, spearman_p = spearmanr(human_scores, model_scores)
    pearson_r, pearson_p = pearsonr(human_scores, model_scores)

    coverage = (total - oov_count) / total * 100

    print(f"  Total pairs:     {total}")
    print(f"  Valid pairs:     {total - oov_count} ({coverage:.1f}%)")
    print(f"  OOV pairs:       {oov_count} ({100 - coverage:.1f}%)")
    print()
    print(f"  Spearman ρ:      {spearman_rho:.4f}  (p={spearman_p:.2e})")
    print(f"  Pearson r:       {pearson_r:.4f}  (p={pearson_p:.2e})")
    print()

    if oov_examples:
        print(f"  OOV examples:    {', '.join(f'{w1}-{w2}' for w1, w2 in oov_examples)}")

    # Interpretation
    if spearman_rho >= 0.6:
        grade = "Good"
    elif spearman_rho >= 0.4:
        grade = "Moderate"
    elif spearman_rho >= 0.2:
        grade = "Weak"
    else:
        grade = "Poor"
    print(f"  Interpretation:  {grade} correlation with human judgements")

    return {
        "total": total,
        "valid": total - oov_count,
        "oov": oov_count,
        "coverage_pct": coverage,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "grade": grade,
    }


# --------------------------------------------------------------------------- #
# 2. Analogy evaluation                                                       #
# --------------------------------------------------------------------------- #


def evaluate_analogies(model, filepath: Path, name: str) -> dict:
    """Evaluate word analogies from a benchmark file."""
    print_header(f"2. Word Analogies — {name}")

    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return {}

    # Parse file: lines starting with ':' are category headers
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

    total = 0
    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []
    oov_count = 0
    cat_results = {}

    for cat, quads in sorted(categories.items()):
        cat_total = 0
        cat_correct_1 = 0
        cat_correct_5 = 0
        cat_oov = 0
        cat_rr = []

        for a, b, c, expected in quads:
            cat_total += 1
            total += 1

            va = get_word_vector(model, a)
            vb = get_word_vector(model, b)
            vc = get_word_vector(model, c)
            ve = get_word_vector(model, expected)

            if va is None or vb is None or vc is None:
                oov_count += 1
                cat_oov += 1
                continue

            result_vec = vb - va + vc
            exclude = {normalise_word(a), normalise_word(b), normalise_word(c)}

            try:
                most_similar = model.similar_by_vector(result_vec, topn=15)
                filtered = [(w, s) for w, s in most_similar
                            if normalise_word(w) not in exclude][:5]
            except Exception:
                continue

            if not filtered:
                continue

            predicted_words = [normalise_word(w) for w, _ in filtered]
            expected_norm = normalise_word(expected)

            if expected_norm == predicted_words[0]:
                cat_correct_1 += 1
                correct_at_1 += 1

            if expected_norm in predicted_words:
                cat_correct_5 += 1
                correct_at_5 += 1
                rank = predicted_words.index(expected_norm) + 1
                cat_rr.append(1.0 / rank)
                reciprocal_ranks.append(1.0 / rank)
            else:
                cat_rr.append(0.0)
                reciprocal_ranks.append(0.0)

        valid = cat_total - cat_oov
        acc1 = cat_correct_1 / valid * 100 if valid else 0
        acc5 = cat_correct_5 / valid * 100 if valid else 0
        mrr = np.mean(cat_rr) if cat_rr else 0

        cat_results[cat] = {
            "total": cat_total, "valid": valid, "oov": cat_oov,
            "top1": acc1, "top5": acc5, "mrr": mrr,
        }

        print(f"  {cat:30s}  n={cat_total:4d}  valid={valid:4d}  "
              f"Top-1={acc1:5.1f}%  Top-5={acc5:5.1f}%  MRR={mrr:.3f}")

    valid_total = total - oov_count
    overall_acc1 = correct_at_1 / valid_total * 100 if valid_total else 0
    overall_acc5 = correct_at_5 / valid_total * 100 if valid_total else 0
    overall_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    coverage = valid_total / total * 100 if total else 0

    print()
    print(f"  {'OVERALL':30s}  n={total:4d}  valid={valid_total:4d}  "
          f"Top-1={overall_acc1:5.1f}%  Top-5={overall_acc5:5.1f}%  MRR={overall_mrr:.3f}")
    print(f"  Coverage: {coverage:.1f}% ({oov_count} OOV skipped)")

    return {
        "name": name,
        "total": total,
        "valid": valid_total,
        "oov": oov_count,
        "coverage_pct": coverage,
        "top1_accuracy": overall_acc1,
        "top5_accuracy": overall_acc5,
        "mrr": overall_mrr,
        "categories": cat_results,
    }


# --------------------------------------------------------------------------- #
# 3. Clustering evaluation                                                    #
# --------------------------------------------------------------------------- #


def evaluate_clustering(model, benchmark: dict[str, list[str]], k: int | None = None) -> dict:
    """Evaluate clustering quality against known categories."""
    print_header("3. Clustering — Extended Benchmark (5 categories)")

    all_words = []
    true_labels = []
    label_map = {cat: i for i, cat in enumerate(sorted(benchmark))}

    for cat, words in sorted(benchmark.items()):
        for w in words:
            if get_word_vector(model, w) is not None:
                all_words.append(w)
                true_labels.append(label_map[cat])

    total_words = sum(len(words) for words in benchmark.values())
    valid_words = len(all_words)
    oov_count = total_words - valid_words

    print(f"  Total words:    {total_words}")
    print(f"  Valid (in vocab): {valid_words} ({valid_words/total_words*100:.1f}%)")
    print(f"  OOV:            {oov_count}")
    print()

    if valid_words < 10:
        print("  Too few valid words — skipping.")
        return {}

    if k is None:
        k = len(benchmark)

    assignments = cluster_words(model, all_words, k=k)
    pred_labels = [assignments[w] for w in all_words]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # Purity
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(true_labels, pred_labels):
        confusion[pred][true] += 1
    purity = sum(max(classes.values()) for classes in confusion.values()) / len(pred_labels)

    print(f"  K-Means k={k}")
    print(f"  ARI:            {ari:.4f}  (1.0 = perfect, 0.0 = random)")
    print(f"  NMI:            {nmi:.4f}  (1.0 = perfect, 0.0 = random)")
    print(f"  Purity:         {purity:.4f}  (1.0 = perfect)")
    print()

    # Show cluster composition
    inv_label_map = {v: k2 for k2, v in label_map.items()}
    cat_names = list(sorted(benchmark.keys()))

    for cluster_id in sorted(set(pred_labels)):
        members = [w for w, lbl in zip(all_words, pred_labels) if lbl == cluster_id]
        member_cats = [inv_label_map[true_labels[all_words.index(w)]] for w in members]
        cat_counts = defaultdict(int)
        for c in member_cats:
            cat_counts[c] += 1
        dominant = max(cat_counts, key=cat_counts.get)
        composition = ", ".join(f"{c}: {n}" for c, n in sorted(cat_counts.items()))
        print(f"  Cluster {cluster_id} ({len(members)} words, dominant: {dominant})")
        print(f"    {composition}")
        # Show a few example words
        print(f"    Examples: {', '.join(members[:8])}{'...' if len(members) > 8 else ''}")
        print()

    # Interpretation
    if ari >= 0.8:
        grade = "Excellent"
    elif ari >= 0.6:
        grade = "Good"
    elif ari >= 0.4:
        grade = "Moderate"
    else:
        grade = "Weak"
    print(f"  Interpretation: {grade} clustering quality")

    return {
        "total_words": total_words,
        "valid_words": valid_words,
        "oov": oov_count,
        "k": k,
        "ari": ari,
        "nmi": nmi,
        "purity": purity,
        "grade": grade,
    }


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def save_evaluation_report(
    sim_results: dict,
    sem_analogy_results: dict,
    syn_analogy_results: dict,
    cluster_results: dict,
    model_info: str,
    path: Path,
) -> None:
    """Save a comprehensive Markdown evaluation report."""
    lines = [
        "# Word Embedding Evaluation Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Model: {model_info}*",
        "",
        "---",
        "",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| Benchmark | Metric | Score | Coverage | Grade |",
        "|-----------|--------|-------|----------|-------|",
    ]

    if sim_results:
        lines.append(
            f"| AnlamVer (500 pairs) | Spearman ρ | "
            f"**{sim_results['spearman_rho']:.4f}** | "
            f"{sim_results['coverage_pct']:.1f}% | {sim_results['grade']} |"
        )

    if sem_analogy_results:
        lines.append(
            f"| Semantic Analogies ({sem_analogy_results['total']}) | Top-5 Acc / MRR | "
            f"**{sem_analogy_results['top5_accuracy']:.1f}%** / {sem_analogy_results['mrr']:.3f} | "
            f"{sem_analogy_results['coverage_pct']:.1f}% | - |"
        )

    if syn_analogy_results:
        lines.append(
            f"| Syntactic Analogies ({syn_analogy_results['total']}) | Top-5 Acc / MRR | "
            f"**{syn_analogy_results['top5_accuracy']:.1f}%** / {syn_analogy_results['mrr']:.3f} | "
            f"{syn_analogy_results['coverage_pct']:.1f}% | - |"
        )

    if cluster_results:
        lines.append(
            f"| Clustering ({cluster_results['valid_words']} words, k={cluster_results['k']}) | ARI / NMI | "
            f"**{cluster_results['ari']:.4f}** / {cluster_results['nmi']:.4f} | "
            f"{cluster_results['valid_words']/cluster_results['total_words']*100:.1f}% | {cluster_results['grade']} |"
        )

    # Detailed similarity
    if sim_results:
        lines += [
            "", "---", "",
            "## 1. Word Similarity — AnlamVer",
            "",
            f"- **Spearman ρ = {sim_results['spearman_rho']:.4f}** (p = {sim_results['spearman_p']:.2e})",
            f"- Pearson r = {sim_results['pearson_r']:.4f} (p = {sim_results['pearson_p']:.2e})",
            f"- Coverage: {sim_results['valid']}/{sim_results['total']} pairs ({sim_results['coverage_pct']:.1f}%)",
            "",
            "Spearman ρ measures rank correlation between model cosine similarities and human judgements.",
            "A score above 0.5 is considered good for static word embeddings on Turkish.",
        ]

    # Detailed analogies
    for res, title in [(sem_analogy_results, "Semantic"), (syn_analogy_results, "Syntactic")]:
        if not res:
            continue
        lines += [
            "", "---", "",
            f"## 2. Word Analogies — {title}",
            "",
            f"- Total questions: {res['total']}",
            f"- Valid (no OOV): {res['valid']} ({res['coverage_pct']:.1f}%)",
            f"- **Top-1 Accuracy: {res['top1_accuracy']:.1f}%**",
            f"- **Top-5 Accuracy: {res['top5_accuracy']:.1f}%**",
            f"- **MRR: {res['mrr']:.3f}**",
            "",
        ]

        if "categories" in res and res["categories"]:
            lines += [
                "### By Category",
                "",
                "| Category | n | Valid | Top-1 | Top-5 | MRR |",
                "|----------|---|-------|-------|-------|-----|",
            ]
            for cat, cr in sorted(res["categories"].items()):
                lines.append(
                    f"| {cat} | {cr['total']} | {cr['valid']} | "
                    f"{cr['top1']:.1f}% | {cr['top5']:.1f}% | {cr['mrr']:.3f} |"
                )

    # Detailed clustering
    if cluster_results:
        lines += [
            "", "---", "",
            "## 3. Clustering",
            "",
            f"- Words: {cluster_results['valid_words']}/{cluster_results['total_words']}",
            f"- k = {cluster_results['k']}",
            f"- **ARI = {cluster_results['ari']:.4f}** (1.0 = perfect, 0.0 = random)",
            f"- **NMI = {cluster_results['nmi']:.4f}**",
            f"- **Purity = {cluster_results['purity']:.4f}**",
        ]

    lines += ["", "---", "",
              "*Evaluation benchmarks: AnlamVer (Ercan & Yıldız, 2018), "
              "Turkish Semantic/Syntactic Analogies (Kıyak et al.)*", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved → {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Evaluate Turkish word embeddings on benchmarks")
    p.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL_PATH", "data/cc.tr.300.vec"))
    p.add_argument("--model-type", choices=["fasttext", "glove"], default="fasttext")
    p.add_argument("--limit", type=int, default=200_000)
    args = p.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load model
    t0 = time.time()
    if args.model_type == "glove":
        model = load_glove_model(args.model, limit=args.limit)
    else:
        model = load_fasttext_model(args.model, limit=args.limit)
    load_time = time.time() - t0

    model_info = f"{args.model_type} — {args.model} (limit={args.limit}, vocab={len(model)})"
    print(f"\nModel loaded in {load_time:.1f}s: {model_info}")

    # Run evaluations
    sim_results = evaluate_similarity(model, SIMILARITY_FILE)
    sem_results = evaluate_analogies(model, SEMANTIC_ANALOGY_FILE, "Turkish Semantic (7742)")
    syn_results = evaluate_analogies(model, SYNTACTIC_ANALOGY_FILE, "Turkish Syntactic (206)")
    cluster_results = evaluate_clustering(model, CLUSTER_BENCHMARK)

    # Save report
    save_evaluation_report(
        sim_results, sem_results, syn_results, cluster_results,
        model_info, OUTPUTS_DIR / "evaluation_report.md",
    )

    print_header("Evaluation Complete")


if __name__ == "__main__":
    main()
