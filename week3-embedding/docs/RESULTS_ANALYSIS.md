# Results Analysis — What Our Experiments Tell Us

This document interprets the results of our five-model comparison on Turkish word embedding benchmarks. It explains **what each metric means with simple examples**, **why each model scored the way it did**, and **what we can learn from the results**.

---

## The Experiment at a Glance

We tested five embedding models on four Turkish benchmarks:

| Model | Type | How it works |
|-------|------|-------------|
| FastText cc.tr.300 | Static | Each word has one fixed vector, learned from word co-occurrence |
| BERTurk | Contextual | Each word gets a different vector depending on its context |
| XLM-RoBERTa | Contextual | Same as BERT but trained on 100+ languages |
| Turkish BERT-NLI-STS | Sentence-transformer | BERT fine-tuned to make similar sentences have similar vectors |
| Multilingual MiniLM | Sentence-transformer | Compact multilingual sentence encoder |

---

## Understanding the Metrics (with Simple Examples)

### Spearman ρ (Spearman's Rank Correlation)

**What it measures:** Do the model's similarity rankings match human rankings?

**Simple example:** Imagine three word pairs rated by humans (1–10 scale):

| Pair | Human score | Model cosine | Human rank | Model rank |
|------|-------------|-------------|------------|------------|
| kedi–köpek | 8.0 | 0.79 | 1st | 1st |
| elma–muz | 6.0 | 0.63 | 2nd | 2nd |
| kedi–araba | 2.0 | 0.37 | 3rd | 3rd |

Here **the rankings match perfectly**, so Spearman ρ = 1.0. The raw numbers don't matter — only the ordering does. If the model had ranked `elma–muz` above `kedi–köpek`, the ranks would disagree, and ρ would drop.

**Our results:**

| Model | Spearman ρ | Interpretation |
|-------|-----------|----------------|
| FastText | **0.571** | Agrees with humans 57% of the time on ranking |
| Turkish NLI-STS | **0.514** | Close second — fine-tuned for similarity |
| BERTurk | 0.356 | Weak — not designed for single-word similarity |
| MiniLM | 0.265 | Poor — multilingual dilution |
| XLM-RoBERTa | 0.014 | Random — essentially no correlation |

**Why FastText wins here:** Static embeddings give each word a stable vector trained on millions of word co-occurrences. When you ask "how similar are 'kedi' and 'köpek'?", the model has a clear answer. BERT-type models produce a context-dependent vector — when you feed just the word "kedi" with no surrounding sentence, the vector is noisy and unreliable.

**Links:**
- [Wikipedia — Spearman's Rank Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
- [Simply Psychology — Spearman's Rank (visual guide)](https://www.simplypsychology.org/spearmans-rank.html)
- [Khan Academy — Correlation (video)](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/scatterplots-and-correlation/v/correlation-coefficient-intuition-examples)

---

### Top-1 and Top-5 Accuracy (Analogy)

**What it measures:** When we compute `vec(B) - vec(A) + vec(C)`, is the expected answer D among the nearest words?

**Simple example:** Given the analogy `erkek → kadın :: baba → ?`:

1. Compute `vec(kadın) - vec(erkek) + vec(baba)` → result vector
2. Find the 5 nearest words to that vector
3. **Top-1:** Is the #1 nearest word "anne"? → if yes, Top-1 correct
4. **Top-5:** Is "anne" anywhere in the top 5? → if yes, Top-5 correct

If the top 5 results are `[babanın, anne, anneciğim, annesi, dede]`, then:
- Top-1 = ✗ (the #1 word is "babanın", not "anne")
- Top-5 = ✓ ("anne" is at position #2)

**Why Top-5 matters more than Top-1 for Turkish:** Turkish is agglutinative — `yazdı`, `yazdım`, `yazdılar` are all valid forms of "wrote". If the model returns `yazdım` but we expected `yazdı`, it's not really wrong. Top-5 gives credit for finding the right concept even if the exact surface form differs.

**Our results (semantic analogies, 7742 questions):**

| Model | Top-1 | Top-5 | What it means |
|-------|-------|-------|---------------|
| FastText | **35.8%** | **65.1%** | Gets the concept right 2/3 of the time |
| Turkish NLI-STS | 15.7% | 22.4% | Trained for sentences, not word arithmetic |
| BERTurk | 9.9% | 18.1% | Context-free vectors are unreliable |
| MiniLM | 7.8% | 14.6% | Multilingual = diluted |
| XLM-RoBERTa | 4.5% | 7.8% | Essentially random |

**Surprise: Turkish NLI-STS dominates syntactic analogies (98.5% Top-5).** Why? Syntactic analogies test morphological patterns like `verdi→verdiniz :: geldi→geldiniz`. This model was fine-tuned on Turkish NLI data, which requires understanding morphological variation — so it learned Turkish grammar deeply.

**Links:**
- [Mikolov et al., 2013 — Linguistic Regularities (the original analogy paper)](https://aclanthology.org/N13-1090/)
- [The Illustrated Word2Vec — Analogy section](https://jalammar.github.io/illustrated-word2vec/#analogy)
- [Rogers et al., 2017 — Problems of Analogical Reasoning with Word Vectors](https://aclanthology.org/S17-1017/) — why analogy tests can be misleading

---

### MRR (Mean Reciprocal Rank)

**What it measures:** On average, how high does the correct answer rank?

**Simple example:** Three analogy questions:

| Question | Correct answer position | Reciprocal Rank |
|----------|------------------------|-----------------|
| erkek→kadın :: baba→? | "anne" is #1 | 1/1 = 1.000 |
| iyi→kötü :: güzel→? | "çirkin" is #3 | 1/3 = 0.333 |
| türkiye→ankara :: fransa→? | "paris" is not in top 5 | 0.000 |

**MRR = (1.000 + 0.333 + 0.000) / 3 = 0.444**

MRR rewards models that put the right answer near the top, even if it's not #1. A model with MRR=0.5 puts the correct answer at position ~2 on average.

| MRR | Interpretation |
|-----|----------------|
| 1.0 | Always correct at position #1 |
| 0.5 | Correct answer at ~position 2 on average |
| 0.33 | Correct answer at ~position 3 on average |
| 0.0 | Never finds the correct answer |

**Links:**
- [Wikipedia — Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- [Croft et al. — Search Engines: Information Retrieval in Practice](https://ciir.cs.umass.edu/irbook/) — MRR in context of information retrieval

---

### ARI (Adjusted Rand Index) — Clustering Quality

**What it measures:** Do the clusters match the true categories, adjusted for luck?

**Simple example:** Suppose we have 6 words and 2 true categories:

```
True labels:    [animal, animal, animal, fruit, fruit, fruit]
                 kedi    köpek   kuş     elma   muz    portakal
```

**Perfect clustering (ARI = 1.0):**
```
Predicted:      [0, 0, 0, 1, 1, 1]   → Every cluster = one category
```

**Random clustering (ARI ≈ 0.0):**
```
Predicted:      [0, 1, 0, 1, 0, 1]   → Animals and fruits mixed randomly
```

**Partially correct (ARI ≈ 0.5):**
```
Predicted:      [0, 0, 1, 1, 1, 1]   → "kuş" is misplaced
```

Why ARI instead of just counting correct assignments? Because a trivial solution — putting each word in its own cluster — gets 100% on simple metrics. ARI adjusts for chance, so only meaningful clustering gets a high score.

**Our results (90 words, 5 categories):**

| Model | ARI | NMI | Purity | Grade |
|-------|-----|-----|--------|-------|
| FastText | **0.949** | 0.957 | 0.978 | Excellent — 88/90 words correct |
| Turkish NLI-STS | 0.697 | 0.731 | 0.867 | Good — some cross-category confusion |
| BERTurk | 0.419 | 0.541 | 0.667 | Moderate — categories partially mixed |
| MiniLM | 0.271 | 0.407 | 0.622 | Weak |
| XLM-RoBERTa | 0.020 | 0.113 | 0.333 | Random — no clustering structure |

**Links:**
- [Wikipedia — Rand Index](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index)
- [scikit-learn — Clustering Evaluation](https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index) — includes visual examples
- [Vinh et al., 2010 — NMI comparison](https://jmlr.org/papers/v11/vinh10a.html)

---

## The Big Picture: Why Static Embeddings Won

This result seems counterintuitive — aren't BERT models "better" than FastText? The answer is: **it depends on the task**.

```
                    Word-level tasks          Sentence-level tasks
                    (similarity, analogy,     (NLI, sentiment,
                     clustering)               paraphrase detection)
                    ─────────────────         ────────────────────
FastText            ████████████████  ★       ████
BERTurk (raw)       ██████                    ████████████
Turkish NLI-STS     ███████████               ████████████████  ★
```

**Static embeddings (FastText, GloVe, Word2Vec)** assign each word a fixed vector trained on word co-occurrence patterns across billions of tokens. This vector is stable, general-purpose, and captures word-level semantics well.

**Contextual embeddings (BERT, RoBERTa)** produce a different vector for the same word depending on the surrounding sentence. "bank" in "river bank" and "bank account" get different vectors. This is powerful for sentence understanding but **useless when you feed a single word with no context**. The model sees `[CLS] kedi [SEP]` and has nothing to work with.

**Sentence-transformers (NLI-STS, MiniLM)** add a pooling layer on top of BERT and fine-tune on similarity tasks. They produce reasonable vectors even for single words because the fine-tuning process teaches them to encode meaning independently of context length.

### The lesson

**There is no universally "best" embedding model.** The right choice depends on your task:

| Task | Best model type | Why |
|------|----------------|-----|
| Word similarity | Static (FastText) | Stable, word-level vectors |
| Word analogy | Static (FastText) | Vector arithmetic works on fixed vectors |
| Word clustering | Static (FastText) | Clean, separable word representations |
| Sentence similarity | Sentence-transformer | Fine-tuned for sentence-level comparison |
| Text classification | Sentence-transformer or fine-tuned BERT | Captures sentence-level meaning |
| Named entity recognition | Contextual (BERT) | Context-dependent word meaning |

---

## Category-Level Insights

### Where FastText excels

| Category | Top-5 | Why |
|----------|-------|-----|
| şehir–bölge | 70.5% | Geographic names are frequent in training data |
| ülke–başkent | 71.4% | Classic analogy success case |
| aile (kinship) | 70.0% | Gender/family relationships encode well |
| eş anlamlılar (synonyms) | 65.8% | Similar words → similar contexts |

### Where Turkish NLI-STS excels

| Category | Top-5 | Why |
|----------|-------|-----|
| Syntactic analogies | **98.5%** | Fine-tuned on Turkish NLI → learned morphology deeply |
| eş anlamlılar | **95.7%** | Synonym detection is core to NLI |
| zıt anlamlılar | 72.3% | NLI requires understanding contradiction |

### Where everyone struggles

| Category | Best Top-5 | Why |
|----------|-----------|-----|
| para-birimi (currency) | 27.8% (FastText) | Rare words, constantly changing relationships |
| capital-world | 41.9% (FastText) | Too many similar city/country names competing |

---

## Coverage: The Hidden Metric

Coverage is how many test items the model can actually evaluate (i.e., all words are in the vocabulary).

| Model | AnlamVer | Semantic Analogy | Why |
|-------|----------|-----------------|-----|
| FastText (200K) | 67.6% | 32.0% | Fixed vocabulary — agglutinative forms cause OOV |
| BERTurk | **100%** | **100%** | Subword tokenisation handles any string |
| XLM-RoBERTa | **100%** | **100%** | Same |
| Turkish NLI-STS | **100%** | **100%** | Same |
| MiniLM | **100%** | **100%** | Same |

FastText's 32% coverage on semantic analogies means **it couldn't even attempt 68% of the questions**. If we could increase coverage (by loading more words or using the full FastText model with subword fallback), its scores would likely change significantly.

This is an important caveat: **FastText's high accuracy is partly because it only answered the "easy" questions** — high-frequency word pairs where it has vectors. The transformer models answered everything, including rare and complex word forms.

---

## What Would Improve These Results?

| Approach | Expected improvement | Effort |
|----------|---------------------|--------|
| Load FastText with `limit=500000` | +10% coverage, mixed accuracy effect | Low — just change a parameter |
| Use full FastText model (not KeyedVectors) | OOV → 0% via subword fallback | Medium — needs `fasttext` library |
| Fine-tune BERTurk for word similarity | Spearman ρ → 0.6+ | High — needs training data |
| Use sentence-level benchmarks (STS-TR) | Transformer models will dominate | Medium — different evaluation |
| Ensemble: FastText for words + BERT for sentences | Best of both worlds | Medium |

---

## Further Reading

### Static vs Contextual Embeddings
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — why contextual embeddings are different
- [Peters et al., 2018 — ELMo: Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365) — the bridge between static and contextual
- [Ethayarajh, 2019 — How Contextual are Contextualized Word Representations?](https://aclanthology.org/D19-1006/) — measures how much BERT vectors change with context

### Embedding Evaluation
- [Bakarov, 2018 — Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536) — 19 intrinsic + 9 extrinsic methods
- [Schnabel et al., 2015 — Evaluation Methods for Unsupervised Word Embeddings](https://aclanthology.org/D15-1036/) — critical comparison
- [wordvectors.org](https://wordvectors.org/) — online evaluation tool

### Turkish NLP
- [AnlamVer Paper (Ercan & Yıldız, 2018)](https://aclanthology.org/C18-1323/) — the Turkish similarity benchmark we used
- [BERTurk (dbmdz)](https://huggingface.co/dbmdz/bert-base-turkish-cased) — model card
- [Turkish BERT-NLI-STS](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) — model card
- [Comprehensive Analysis of Static Word Embeddings for Turkish](https://arxiv.org/abs/2405.07778) — 2024 survey

### Metrics Deep Dives
- [Google ML Crash Course — Classification Metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy) — accuracy, precision, recall
- [scikit-learn — Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — ARI, NMI, V-measure with examples
- [Towards Data Science — Understanding AUC-ROC](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) — visual explanation
- [StatQuest — Machine Learning Fundamentals (playlist)](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — intuitive video series on all key metrics
