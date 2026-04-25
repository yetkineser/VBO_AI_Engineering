# Evaluating Word Embeddings — Metrics, Methods & Benchmarks

Word embeddings are not a model that produces a single accuracy number. They are a **representation** — a way of encoding words as vectors. Evaluating them requires asking: *"How well does this representation capture linguistic knowledge?"*

There are two broad families of evaluation: **intrinsic** (test the vectors directly) and **extrinsic** (plug them into a downstream task and measure task performance).

---

## 1. Intrinsic Evaluation

Intrinsic methods test the embedding space itself, without training a downstream model. They are fast, cheap, and give you a quick diagnostic.

### 1.1 Word Similarity (Correlation with Human Judgements)

**What it measures:** Do cosine similarity scores agree with how humans rate word relatedness?

**How it works:**
1. Take a benchmark dataset of word pairs with human-assigned similarity scores (e.g. 1–10 scale).
2. Compute cosine similarity for each pair using your embeddings.
3. Measure the **Spearman rank correlation** (ρ) between the two rankings.

**Why Spearman, not Pearson?** We care about *ranking* ("is A-B more similar than C-D?"), not the raw numbers. Spearman captures monotonic relationships without assuming linearity.

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Spearman ρ | rank correlation between human scores and cosine similarities | [-1, +1] | +1 = perfect agreement, 0 = no correlation |
| Pearson r | linear correlation | [-1, +1] | Assumes linear relationship (less appropriate) |

**Benchmark datasets:**

| Dataset | Language | # Pairs | What it measures |
|---------|----------|---------|-----------------|
| [WordSim-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) | English | 353 | Relatedness + similarity |
| [SimLex-999](https://fh295.github.io/simlex.html) | English | 999 | Pure similarity (not relatedness) |
| [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN) | English | 3000 | Relatedness |
| [RG-65](https://aclanthology.org/J91-1003/) | English | 65 | Classic small benchmark |
| [AnlamVer](https://github.com/Wikipedia2Vec/AnlamVer) | Turkish | 500 | Turkish word similarity |

**Important distinction:** *Similarity* vs *relatedness*. "Coffee" and "tea" are **similar** (both are drinks). "Coffee" and "cup" are **related** (often appear together) but not similar. SimLex-999 measures similarity; WordSim-353 mixes both. Know which one your benchmark tests.

**Links:**
- [Faruqui & Dyer, 2014 — Community Evaluation of Word Vectors](https://www.aclweb.org/anthology/W14-1508/) — compares multiple benchmarks
- [wordvectors.org](https://wordvectors.org/) — online tool to evaluate your vectors against multiple benchmarks
- [Bakarov, 2018 — A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536) — comprehensive survey

---

### 1.2 Word Analogy (Accuracy)

**What it measures:** Can the embedding space capture relational patterns like "A is to B as C is to D"?

**How it works:**
1. Take a set of analogy quads: (A, B, C, D).
2. Compute `vec(B) - vec(A) + vec(C)`.
3. Find the nearest word to that result vector (excluding A, B, C).
4. Check if the nearest word is D.

**Metrics:**

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | % of analogies where the #1 result is the expected word |
| **Top-5 Accuracy** | % of analogies where the expected word appears in the top 5 |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank for each expected word. Rewards models that put the right answer high even if not #1 |

```
Top-1 Accuracy = correct_at_1 / total_analogies
MRR = (1/N) * Σ (1 / rank_of_correct_answer)
```

**Benchmark datasets:**

| Dataset | # Analogies | Categories |
|---------|-------------|------------|
| [Google Analogy](https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)) | 19,544 | Semantic (capitals, currency, gender) + Syntactic (tense, plural) |
| [BATS](https://vecto.space/projects/BATS/) | 99,200 | 40 relation types, more balanced than Google |
| [SemEval-2012 Task 2](https://aclanthology.org/S12-1047/) | 79 relation types | Finer-grained relational similarity |

**Caveats:**
- Analogy accuracy is **brittle**. Small vocabulary gaps (OOV) or morphological variation can cause failures that don't reflect actual embedding quality.
- Turkish is especially tricky because agglutinative morphology means the "expected" answer may appear as a different surface form (e.g. `yazdı` vs `yazdım`).
- Consider **Top-5 accuracy** or **MRR** instead of strict Top-1 for morphologically rich languages.

**Links:**
- [Mikolov et al., 2013 — Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090/) — introduced the analogy task
- [Levy & Goldberg, 2014 — Linguistic Regularities in Sparse and Explicit Word Representations](https://aclanthology.org/W14-1618/) — shows that analogy is not magic, it's a property of PMI-based methods too
- [Rogers et al., 2017 — Too Many Problems of Analogical Reasoning with Word Vectors](https://aclanthology.org/S17-1017/) — critical analysis of analogy evaluation

---

### 1.3 Word Categorisation / Clustering (Purity, NMI, ARI)

**What it measures:** If you cluster word vectors, do the resulting clusters align with known semantic categories?

**How it works:**
1. Take a set of words with known category labels (e.g. animals, vehicles, fruits).
2. Cluster the word vectors with K-Means (or another algorithm).
3. Compare predicted clusters to true labels using clustering metrics.

**Metrics:**

| Metric | Range | What it measures |
|--------|-------|-----------------|
| **Purity** | [0, 1] | Each cluster is dominated by a single class. Simple but biased toward more clusters. |
| **NMI (Normalised Mutual Information)** | [0, 1] | Mutual information between predicted and true labels, normalised. Handles different cluster counts. |
| **ARI (Adjusted Rand Index)** | [-1, 1] | Measures agreement between two clusterings, adjusted for chance. 0 = random, 1 = perfect. |
| **V-Measure** | [0, 1] | Harmonic mean of homogeneity (each cluster has one class) and completeness (each class is in one cluster). |

```python
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score(true_labels, predicted_labels)
ari = adjusted_rand_score(true_labels, predicted_labels)
```

**Why ARI over Purity?** Purity always increases with more clusters (trivially 1.0 if each word is its own cluster). ARI corrects for chance, making it comparable across different values of `k`.

**Links:**
- [sklearn — Clustering Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — excellent overview of all metrics
- [Rosenberg & Hirschberg, 2007 — V-Measure](https://aclanthology.org/D07-1043/) — homogeneity + completeness
- [Vinh et al., 2010 — Information Theoretic Measures for Clusterings Comparison](https://jmlr.org/papers/v11/vinh10a.html) — deep dive into NMI variants

---

### 1.4 Outlier Detection (OPP Score)

**What it measures:** Given a group of related words plus one outlier, can the model identify the outlier?

**How it works:**
1. Present a set like `{kedi, köpek, kuş, araba}`.
2. Compute the average cosine similarity of each word to all others.
3. The word with the lowest average similarity is the predicted outlier.
4. Check if it matches the known outlier.

**Metric:** **OPP (Outlier Position Percentage)** — accuracy of correctly identifying the outlier across all test sets.

This is a simpler, more intuitive test than analogies and less sensitive to morphological variation.

**Links:**
- [Camacho-Collados & Navigli, 2016 — Find the Word Intruder](https://aclanthology.org/D16-1153/) — formalises the outlier detection task
- [8-8-8 Dataset](https://github.com/Wikipedia2Vec/outlier-detection) — standard benchmark

---

## 2. Extrinsic Evaluation

Extrinsic methods plug embeddings into a real NLP task and measure task performance. They answer the practical question: *"Do these embeddings make my system better?"*

### 2.1 Text Classification (Accuracy, F1)

Use word embeddings as features (e.g. average word vectors per document) and train a classifier. Compare against a TF-IDF baseline.

| Metric | What it tells you |
|--------|------------------|
| **Accuracy** | Overall correctness |
| **F1 (macro)** | Balance between precision and recall, averaged across classes |
| **AUC-ROC** | Ranking quality, threshold-independent |

This is exactly what Week 2 did — and the most practical measure of embedding quality.

### 2.2 Named Entity Recognition (F1)

Use embeddings as input features to a sequence labeller (BiLSTM-CRF or similar). Measure entity-level F1.

### 2.3 Semantic Textual Similarity (Pearson/Spearman)

Average word vectors to create sentence embeddings, then correlate with human sentence similarity judgements (STS Benchmark).

**Links:**
- [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) — standard sentence similarity benchmark
- [Conneau & Kiela, 2018 — SentEval](https://arxiv.org/abs/1803.05449) — toolkit for evaluating sentence representations

---

## 3. Summary: Which Metric for Which Task?

```
┌──────────────────────────┬────────────────────────┬─────────────────────────┐
│ What you want to measure │ Method                 │ Key metric              │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│ Semantic similarity      │ Word similarity bench. │ Spearman ρ              │
│ Relational patterns      │ Word analogy           │ Top-5 Accuracy / MRR    │
│ Clustering quality       │ K-Means + gold labels  │ ARI / NMI               │
│ Outlier detection        │ Intruder identification│ OPP                     │
│ Downstream usefulness    │ Text classification    │ F1 / AUC-ROC            │
│ Sentence understanding   │ STS benchmark          │ Spearman ρ              │
└──────────────────────────┴────────────────────────┴─────────────────────────┘
```

### Recommendation for this project

For a homework-scale project using pre-trained Turkish embeddings, the most practical evaluation approach is:

1. **Word similarity** — Compute Spearman ρ against the [AnlamVer](https://github.com/Wikipedia2Vec/AnlamVer) Turkish dataset (if available) or manually create 20-30 Turkish word pairs with human scores.
2. **Analogy accuracy (Top-5 + MRR)** — More forgiving than Top-1 for Turkish morphology.
3. **Clustering ARI/NMI** — You already have ground-truth labels (animals/vehicles/fruits), so this is free.
4. **Qualitative inspection** — Look at the t-SNE plot. Are the clusters visually separated? Do the nearest neighbours make sense?

Don't chase a single number. Use multiple metrics and look for consistency across them.

---

## 4. Common Pitfalls

| Pitfall | Why it matters | What to do |
|---------|---------------|------------|
| Evaluating only with analogies | Analogy accuracy is noisy and over-emphasised in the literature | Use similarity + clustering + downstream task |
| Ignoring OOV rate | High OOV means the model never gets a chance to answer | Report OOV rate alongside accuracy |
| Comparing across vocabularies | A model with 2M words has an unfair advantage over one with 200K | Use the same `limit` or report vocabulary size |
| Using Top-1 accuracy for Turkish | Agglutinative morphology means `yazdı` ≠ `yazdım` | Use Top-5 or MRR |
| Not normalising text | `"Kedi"` ≠ `"kedi"` in the lookup | Always lowercase + strip punctuation before evaluation |
| Over-interpreting t-SNE | t-SNE is non-deterministic and distances between clusters are not meaningful | Use for qualitative insight only, not as a metric |

---

## 5. Further Reading

### Surveys & Overviews
- [Bakarov, 2018 — A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536) — the most comprehensive survey (covers 19 intrinsic and 9 extrinsic methods)
- [Schnabel et al., 2015 — Evaluation Methods for Unsupervised Word Embeddings](https://aclanthology.org/D15-1036/) — critical comparison of evaluation approaches
- [Wang et al., 2019 — Evaluating Word Embedding Models](https://aclanthology.org/P19-1070/) — recent meta-analysis

### Specific Methods
- [Finkelstein et al., 2002 — Placing Search in Context: The Concept Revisited](https://dl.acm.org/doi/10.1145/503104.503110) — WordSim-353 paper
- [Hill et al., 2015 — SimLex-999](https://aclanthology.org/J15-4004/) — why similarity ≠ relatedness
- [Levy et al., 2015 — Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) — hyper-parameters matter more than algorithms
- [Gladkova et al., 2016 — Analogy-based Detection of Morphological and Semantic Relations with Word Embeddings](https://aclanthology.org/N16-2002/) — BATS dataset

### Tools
- [wordvectors.org](https://wordvectors.org/) — evaluate your vectors online against standard benchmarks
- [VecEval](https://github.com/AKGostar/VecEval) — Python framework for embedding evaluation
- [Gensim evaluate_word_pairs()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs) — built-in evaluation in Gensim
- [Gensim evaluate_word_analogies()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies) — built-in analogy evaluation

### Turkish-Specific
- [AnlamVer — Turkish Word Similarity Dataset](https://github.com/Wikipedia2Vec/AnlamVer)
- [Ercan & Yıldız, 2018 — AnlamVer: Intrinsic Evaluation of Word Embeddings for Turkish](https://dergipark.org.tr/en/pub/tbbmd/issue/40485/484526)
