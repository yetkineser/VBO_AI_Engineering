# Word Embedding Model Comparison — Turkish Benchmarks

*Generated: 2026-04-10 15:29*

---

## Summary

| Model | Type | AnlamVer<br>Spearman ρ | Semantic<br>Top-5 / MRR | Syntactic<br>Top-5 / MRR | Cluster<br>ARI | Time |
|-------|------|----------------------|------------------------|------------------------|------------|------|
| **FastText cc.tr.300** | static | 0.5712 | 65.1% / 0.471 | 69.8% / 0.632 | 0.9487 | 18s |
| **BERTurk** | contextual | 0.3563 | 18.1% / 0.128 | 75.2% / 0.719 | 0.4186 | 564s |
| **XLM-RoBERTa-base** | contextual | 0.0141 | 7.8% / 0.057 | 50.5% / 0.392 | 0.0198 | 568s |
| **Turkish BERT-NLI-STS** | sentence-transformer | 0.5139 | 22.4% / 0.181 | 98.5% / 0.960 | 0.6973 | 585s |
| **Multilingual MiniLM** | sentence-transformer | 0.2650 | 14.6% / 0.102 | 84.0% / 0.767 | 0.2710 | 562s |

---

## FastText cc.tr.300 (static)

### Word Similarity (AnlamVer)
- Spearman ρ = **0.5712**, Pearson r = 0.6170
- Coverage: 338/500 (67.6%)

### Semantic Analogies
- Top-1 = **35.8%**, Top-5 = **65.1%**, MRR = **0.471**
- Coverage: 2480/7742 (32.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| aile | 90 | 42.2% | 70.0% | 0.525 |
| capital-world | 55 | 21.8% | 40.0% | 0.290 |
| es-anlamlilar | 600 | 43.5% | 65.8% | 0.511 |
| para-birimi | 36 | 5.6% | 27.8% | 0.131 |
| sehir-bolge | 1050 | 32.8% | 70.5% | 0.478 |
| zit-anlamlilar | 600 | 34.2% | 58.3% | 0.435 |
| ülke-başkent | 49 | 55.1% | 71.4% | 0.615 |

### Syntactic Analogies
- Top-1 = **58.8%**, Top-5 = **69.8%**, MRR = **0.632**
- Coverage: 182/206 (88.3%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| syntactic analogy | 182 | 58.8% | 69.8% | 0.632 |

### Clustering
- ARI = **0.9487**, NMI = 0.9566, Purity = 0.9778
- 90/90 words, k=5


---

## BERTurk (contextual)

### Word Similarity (AnlamVer)
- Spearman ρ = **0.3563**, Pearson r = 0.3046
- Coverage: 500/500 (100.0%)

### Semantic Analogies
- Top-1 = **9.9%**, Top-5 = **18.1%**, MRR = **0.128**
- Coverage: 7742/7742 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| aile | 90 | 42.2% | 64.4% | 0.512 |
| capital-world | 4446 | 2.2% | 6.8% | 0.037 |
| es-anlamlilar | 600 | 40.8% | 57.2% | 0.469 |
| para-birimi | 156 | 0.0% | 0.6% | 0.002 |
| sehir-bolge | 1344 | 8.8% | 19.0% | 0.125 |
| zit-anlamlilar | 600 | 41.5% | 64.3% | 0.500 |
| ülke-başkent | 506 | 4.0% | 10.9% | 0.063 |

### Syntactic Analogies
- Top-1 = **69.9%**, Top-5 = **75.2%**, MRR = **0.719**
- Coverage: 206/206 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| syntactic analogy | 206 | 69.9% | 75.2% | 0.719 |

### Clustering
- ARI = **0.4186**, NMI = 0.5411, Purity = 0.6667
- 90/90 words, k=5


---

## XLM-RoBERTa-base (contextual)

### Word Similarity (AnlamVer)
- Spearman ρ = **0.0141**, Pearson r = 0.0756
- Coverage: 500/500 (100.0%)

### Semantic Analogies
- Top-1 = **4.5%**, Top-5 = **7.8%**, MRR = **0.057**
- Coverage: 7742/7742 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| aile | 90 | 10.0% | 20.0% | 0.136 |
| capital-world | 4446 | 0.9% | 3.3% | 0.017 |
| es-anlamlilar | 600 | 4.7% | 9.8% | 0.066 |
| para-birimi | 156 | 1.9% | 4.5% | 0.026 |
| sehir-bolge | 1344 | 15.2% | 17.5% | 0.162 |
| zit-anlamlilar | 600 | 9.2% | 17.5% | 0.122 |
| ülke-başkent | 506 | 2.0% | 6.5% | 0.033 |

### Syntactic Analogies
- Top-1 = **32.5%**, Top-5 = **50.5%**, MRR = **0.392**
- Coverage: 206/206 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| syntactic analogy | 206 | 32.5% | 50.5% | 0.392 |

### Clustering
- ARI = **0.0198**, NMI = 0.1133, Purity = 0.3333
- 90/90 words, k=5


---

## Turkish BERT-NLI-STS (sentence-transformer)

### Word Similarity (AnlamVer)
- Spearman ρ = **0.5139**, Pearson r = 0.6023
- Coverage: 500/500 (100.0%)

### Semantic Analogies
- Top-1 = **15.7%**, Top-5 = **22.4%**, MRR = **0.181**
- Coverage: 7742/7742 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| aile | 90 | 58.9% | 74.4% | 0.649 |
| capital-world | 4446 | 1.9% | 6.0% | 0.033 |
| es-anlamlilar | 600 | 90.2% | 95.7% | 0.925 |
| para-birimi | 156 | 1.9% | 6.4% | 0.037 |
| sehir-bolge | 1344 | 14.1% | 24.1% | 0.176 |
| zit-anlamlilar | 600 | 53.3% | 72.3% | 0.604 |
| ülke-başkent | 506 | 4.5% | 11.5% | 0.072 |

### Syntactic Analogies
- Top-1 = **94.2%**, Top-5 = **98.5%**, MRR = **0.960**
- Coverage: 206/206 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| syntactic analogy | 206 | 94.2% | 98.5% | 0.960 |

### Clustering
- ARI = **0.6973**, NMI = 0.7309, Purity = 0.8667
- 90/90 words, k=5


---

## Multilingual MiniLM (sentence-transformer)

### Word Similarity (AnlamVer)
- Spearman ρ = **0.2650**, Pearson r = 0.3395
- Coverage: 500/500 (100.0%)

### Semantic Analogies
- Top-1 = **7.8%**, Top-5 = **14.6%**, MRR = **0.102**
- Coverage: 7742/7742 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| aile | 90 | 30.0% | 43.3% | 0.346 |
| capital-world | 4446 | 5.2% | 11.7% | 0.075 |
| es-anlamlilar | 600 | 34.0% | 39.2% | 0.361 |
| para-birimi | 156 | 2.6% | 10.9% | 0.065 |
| sehir-bolge | 1344 | 2.5% | 7.1% | 0.041 |
| zit-anlamlilar | 600 | 4.3% | 14.3% | 0.079 |
| ülke-başkent | 506 | 15.6% | 27.5% | 0.193 |

### Syntactic Analogies
- Top-1 = **72.3%**, Top-5 = **84.0%**, MRR = **0.767**
- Coverage: 206/206 (100.0%)

| Category | n | Top-1 | Top-5 | MRR |
|----------|---|-------|-------|-----|
| syntactic analogy | 206 | 72.3% | 84.0% | 0.767 |

### Clustering
- ARI = **0.2710**, NMI = 0.4068, Purity = 0.6222
- 90/90 words, k=5

---

## Interpretation

- **Spearman ρ** on AnlamVer: measures correlation with human similarity judgements. Higher is better. Static embeddings typically score 0.4–0.6, contextual 0.5–0.7.
- **Analogy Top-5 / MRR**: contextual models may score lower on analogies because they were not designed for word-level vector arithmetic.
- **Clustering ARI**: most models achieve near-perfect clustering on well-separated categories. The real test is on ambiguous or overlapping categories.
- **Coverage**: FastText has the largest vocabulary but contextual models handle any word via subword tokenisation (100% coverage by construction).

---

*Benchmarks: AnlamVer (Ercan & Yıldız, 2018), Turkish Semantic/Syntactic Analogies (Kıyak et al.)*