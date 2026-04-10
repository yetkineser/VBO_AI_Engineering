# Word Embedding Evaluation Report

*Generated: 2026-04-10 13:38*
*Model: fasttext — data/cc.tr.300.vec (limit=500000, vocab=500000)*

---

## Summary

| Benchmark | Metric | Score | Coverage | Grade |
|-----------|--------|-------|----------|-------|
| AnlamVer (500 pairs) | Spearman ρ | **0.5792** | 76.2% | Moderate |
| Semantic Analogies (7742) | Top-5 Acc / MRR | **56.7%** / 0.390 | 42.5% | - |
| Syntactic Analogies (206) | Top-5 Acc / MRR | **84.7%** / 0.785 | 95.1% | - |
| Clustering (90 words, k=5) | ARI / NMI | **0.9487** / 0.9566 | 100.0% | Excellent |

---

## 1. Word Similarity — AnlamVer

- **Spearman ρ = 0.5792** (p = 1.65e-35)
- Pearson r = 0.6195 (p = 9.26e-42)
- Coverage: 381/500 pairs (76.2%)

Spearman ρ measures rank correlation between model cosine similarities and human judgements.
A score above 0.5 is considered good for static word embeddings on Turkish.

---

## 2. Word Analogies — Semantic

- Total questions: 7742
- Valid (no OOV): 3294 (42.5%)
- **Top-1 Accuracy: 28.0%**
- **Top-5 Accuracy: 56.7%**
- **MRR: 0.390**

### By Category

| Category | n | Valid | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-------|-----|
| aile | 90 | 90 | 37.8% | 66.7% | 0.486 |
| capital-world | 4446 | 332 | 22.3% | 48.8% | 0.325 |
| es-anlamlilar | 600 | 600 | 39.3% | 57.8% | 0.461 |
| para-birimi | 156 | 88 | 2.3% | 8.0% | 0.039 |
| sehir-bolge | 1344 | 1344 | 23.8% | 60.9% | 0.382 |
| zit-anlamlilar | 600 | 600 | 32.3% | 55.7% | 0.410 |
| ülke-başkent | 506 | 240 | 26.2% | 58.8% | 0.385 |

---

## 2. Word Analogies — Syntactic

- Total questions: 206
- Valid (no OOV): 196 (95.1%)
- **Top-1 Accuracy: 74.0%**
- **Top-5 Accuracy: 84.7%**
- **MRR: 0.785**

### By Category

| Category | n | Valid | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-------|-----|
| syntactic analogy | 206 | 196 | 74.0% | 84.7% | 0.785 |

---

## 3. Clustering

- Words: 90/90
- k = 5
- **ARI = 0.9487** (1.0 = perfect, 0.0 = random)
- **NMI = 0.9566**
- **Purity = 0.9778**

---

*Evaluation benchmarks: AnlamVer (Ercan & Yıldız, 2018), Turkish Semantic/Syntactic Analogies (Kıyak et al.)*
