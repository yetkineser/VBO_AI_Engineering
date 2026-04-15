# Week 3 — Self-Evaluation: What We Did, What We Learned

This document walks through every homework requirement, shows what we implemented, reflects on what we learned, and points to resources for deeper understanding. It then covers the extras we built beyond the assignment.

---

## Homework Requirements

### 1. Load a pre-trained model

**Requirement:** Write `load_fasttext_model()` and `load_glove_model()` that return a `KeyedVectors` object.

**What we did:**
- Implemented both functions in `src/embedding_utils.py`
- Downloaded the full FastText Turkish model (`cc.tr.300.vec`, 4.2 GB, ~2M words)
- Added a `limit` parameter (default 200K) to control memory usage and load time
- `load_glove_model()` auto-detects whether the file has a word2vec header or not (GloVe files often omit the header)
- Added proper error messages when the file is not found

**What we learned:**
- Pre-trained word vectors follow the **word2vec text format**: first line is `<vocab_size> <dim>`, then each line is `<word> <v1> ... <v300>`
- Gensim's `KeyedVectors.load_word2vec_format()` is the standard loader — it handles both `.vec` and `.vec.gz` files
- Loading 200K words takes ~20 seconds and ~600 MB RAM. Loading the full 2M takes minutes and several GB. The `limit` parameter is essential for practical development
- GloVe and FastText files look identical in text format but GloVe sometimes omits the header line

**Deepen your understanding:**
- [Gensim — KeyedVectors docs](https://radimrehurek.com/gensim/models/keyedvectors.html) — `load_word2vec_format()` parameters
- [FastText — Pre-trained vectors](https://fasttext.cc/docs/en/crawl-vectors.html) — where to download models for 157 languages
- [Bojanowski et al., 2017 — FastText paper](https://arxiv.org/abs/1607.04606) — how FastText creates subword vectors

---

### 2. Retrieve a word vector

**Requirement:** Write `get_word_vector(model, word)` that returns the 300-dim vector or `None` for OOV.

**What we did:**
- Implemented in `src/embedding_utils.py`
- Two-step lookup: first try the raw word, then try the normalised version
- Built a Turkish-aware normalisation function (`normalise_word()`) that handles:
  - `İ` → `i` (dotted capital I → dotted lowercase i)
  - `I` → `ı` (undotted capital I → undotted lowercase ı)
  - Strip punctuation, whitespace
- Returns `None` for OOV instead of raising an exception

**What we learned:**
- Turkish casing is tricky: Python's `casefold()` converts `İ` to `i\u0307` (i + combining dot above), which doesn't match vocabulary entries. We had to use explicit character replacement
- OOV is very common in Turkish due to agglutinative morphology — `kitaplarımızda` (in our books) may be absent even when `kitap` (book) is present
- The `normalise_word()` function is critical: without it, "Kedi" and "kedi" would be treated as different words

**Deepen your understanding:**
- [Unicode — Turkish case mapping](https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf#G33992) — why Turkish I/İ is special
- [Python docs — str.casefold()](https://docs.python.org/3/library/stdtypes.html#str.casefold) — full Unicode case folding
- [Zeyrek — Turkish morphological analyser](https://github.com/obulat/zeyrek) — for lemmatising Turkish words before lookup

---

### 3. Word similarity (cosine)

**Requirement:** Write `word_similarity(model, word1, word2)` returning cosine similarity as a float.

**What we did:**
- Implemented in `src/embedding_utils.py`
- Uses `sklearn.metrics.pairwise.cosine_similarity` for the computation
- Returns `None` for OOV words, as specified by the homework
- Tested on 12 Turkish word pairs with intuitively correct results

**Results highlights:**
- `kedi ↔ köpek: 0.79` (both animals — high similarity)
- `kedi ↔ araba: 0.37` (unrelated — low similarity)
- `iyi ↔ kötü: 0.72` (antonyms score high because they appear in similar contexts)

**What we learned:**
- Cosine similarity measures direction, not magnitude — this is important because word frequency affects vector norms
- Antonyms (iyi/kötü, güzel/çirkin) score surprisingly high because they share context ("the movie was good" vs "the movie was bad")
- Cosine similarity is not the same as "synonymy" — it captures co-occurrence patterns, which includes both synonyms and antonyms

**Deepen your understanding:**
- [Machine Learning Mastery — Cosine Similarity](https://machinelearningmastery.com/cosine-similarity-for-nlp/) — worked examples
- [Hill et al., 2015 — SimLex-999](https://aclanthology.org/J15-4004/) — why similarity ≠ relatedness (explains the antonym problem)
- [Gensim — similarity()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.similarity) — built-in alternative

---

### 4. K-Means clustering

**Requirement:** Write `cluster_words(model, words, k=3)` with `random_state=42` and `n_init=10`.

**What we did:**
- Implemented in `src/embedding_utils.py`
- **L2-normalises vectors before clustering** — this is not in the homework spec but is the correct thing to do, because K-Means uses Euclidean distance and we want cosine-like behaviour
- Silently skips OOV words with a log message
- Raises `ValueError` when no valid words remain or when `k > len(valid_words)`, as specified by the homework
- Tested with 24 words (8 animals + 8 vehicles + 8 fruits) → **all 24 correctly clustered**

**What we learned:**
- `random_state=42` fixes the random seed for reproducibility — without it, you get different clusters every run
- `n_init=10` runs K-Means 10 times from different starting positions and picks the best result. K-Means can get stuck in local optima, so multiple initialisations help
- L2 normalisation makes Euclidean distance equivalent to cosine distance: `||u - v||² = 2(1 - cos(u,v))` when `||u|| = ||v|| = 1`
- Word embeddings create remarkably clean clusters — 24/24 correct with just 3 categories

**Deepen your understanding:**
- [StatQuest — K-Means Clustering (video)](https://www.youtube.com/watch?v=4b5d3muPQmA) — 8 min visual explanation
- [scikit-learn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — API reference
- [sklearn — Clustering Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — ARI, NMI, Purity

---

### 5. Command-line interface

**Requirement:** Write `main.py` that loads the model, prints vectors, computes similarities, clusters words, and saves results.

**What we did:**
- Implemented `src/main.py` with full `argparse` CLI
- Flags: `--model`, `--model-type`, `--limit`, `--k`, `--analogy`, `--visualise`, `--all`
- Environment variable support: `EMBEDDING_MODEL_PATH`
- Outputs: `similarity.csv`, `clusters.csv`, `results.md`
- Pretty-printed console output with formatted tables

**What we learned:**
- `argparse` is the standard Python CLI library — it auto-generates `--help`, validates types, and handles defaults
- Environment variables (`os.environ.get("EMBEDDING_MODEL_PATH", default)`) are a clean way to configure paths without hardcoding
- Saving results in both CSV (machine-readable) and Markdown (human-readable) makes the output useful for both analysis and documentation

**Deepen your understanding:**
- [Python docs — argparse tutorial](https://docs.python.org/3/howto/argparse.html) — step-by-step
- [The Twelve-Factor App — III. Config](https://12factor.net/config) — why config belongs in environment variables

---

## What We Did Beyond the Homework

### Extra 1: Word Analogies (`--analogy` flag)

**What it is:** Compute `vec(B) - vec(A) + vec(C)` and find the nearest word. Classic test of embedding quality.

**What we implemented:**
- 25 analogy quads across 7 categories (gender, country-capital, antonyms, verb tense, profession-workplace, country-language, country-currency)
- Filters out input words from results
- Shows top-5 candidates with scores
- Hit rate: 8/24 Top-1 correct, many more in Top-5

**What we learned:**
- Gender analogies work well in Turkish (`baba→anne ✓`, `amca→teyze` in Top-2)
- Country-capital analogies are weak (`türkiye→ankara :: fransa→?` returns "belçika" not "paris") because Ankara dominates
- Antonym and synonym analogies work well (`iyi→kötü :: güzel→çirkin ✓`)
- Verb tense: works when same paradigm (`okumak→okudu :: yazmak→yazdı ✓`) but fails across paradigms

**Links:**
- [Mikolov et al., 2013 — Linguistic Regularities](https://aclanthology.org/N13-1090/)
- [The Illustrated Word2Vec — Analogy](https://jalammar.github.io/illustrated-word2vec/#analogy)

---

### Extra 2: t-SNE Visualisation (`--visualise` flag)

**What it is:** Reduce 300-dimensional word vectors to 2D with t-SNE and plot a colour-coded scatter chart.

**What we implemented:**
- `demo_visualise()` in `main.py` using `sklearn.manifold.TSNE` and `matplotlib`
- Colour-coded by cluster assignment
- Word labels on each point
- Saved to `outputs/tsne_clusters.png`

**What we learned:**
- t-SNE preserves local neighbourhood structure — words that are similar in 300D stay close in 2D
- The three clusters (animals, vehicles, fruits) are visually well-separated
- t-SNE is non-deterministic (`random_state` helps but doesn't guarantee identical plots) and distances between clusters are not meaningful — only within-cluster structure matters

**Links:**
- [Distill.pub — How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — essential reading
- [Google Embedding Projector](https://projector.tensorflow.org/) — interactive browser tool

---

### Extra 3: Full Benchmark Evaluation (`evaluate.py`)

**What it is:** Evaluate our embeddings on real Turkish NLP benchmarks with proper metrics.

**What we implemented:**
- **AnlamVer** (500 Turkish word pairs with human scores) → Spearman ρ
- **Turkish Semantic Analogies** (7742 questions, 7 categories) → Top-1/5/MRR
- **Turkish Syntactic Analogies** (206 questions) → Top-1/5/MRR
- **Extended clustering** (90 words, 5 categories) → ARI/NMI/Purity
- Tested at both `limit=200K` and `limit=500K`

**Key results:**
- Spearman ρ = 0.571 (moderate — expected for static embeddings on Turkish)
- Semantic analogy Top-5 = 65.1%, MRR = 0.471
- Syntactic analogy Top-5 = 69.8%
- Clustering ARI = 0.949 (excellent)

**What we learned:**
- Larger vocabulary (`limit=500K`) improves coverage but can hurt analogy accuracy due to noise words competing with the correct answer
- The AnlamVer benchmark includes many rare/morphologically complex words — 32% of pairs are OOV at 200K limit
- Our FastText model scores in the expected range for Turkish static embeddings (literature reports 0.45–0.60 for Spearman ρ)

**Links:**
- [AnlamVer Paper (Ercan & Yıldız, 2018)](https://aclanthology.org/C18-1323/)
- [Comprehensive Analysis of Static Word Embeddings for Turkish (2024)](https://arxiv.org/abs/2405.07778)
- [Gensim — evaluate_word_pairs()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs)

---

### Extra 4: Five-Model Comparison (`evaluate_advanced.py`)

**What it is:** Compare static vs contextual vs sentence-transformer embeddings on the same benchmarks.

**Models tested:**
1. FastText cc.tr.300 (static)
2. BERTurk (contextual)
3. XLM-RoBERTa-base (contextual, multilingual)
4. Turkish BERT-NLI-STS (sentence-transformer)
5. Multilingual MiniLM (sentence-transformer)

**Key findings:**

| Finding | What it teaches |
|---------|----------------|
| FastText wins at word similarity (ρ=0.571) | Static embeddings give stable word-level vectors |
| Raw BERT is terrible for single words (ρ=0.356) | Contextual models need context — `[CLS] kedi [SEP]` is not enough |
| XLM-RoBERTa is worst (ρ=0.014) | Multilingual models dilute language-specific knowledge |
| Turkish NLI-STS wins syntactic analogy (98.5%) | Fine-tuning on Turkish NLI teaches morphology deeply |
| Turkish NLI-STS gets 95.7% on synonyms | NLI training = understanding when two things mean the same |
| Coverage: 100% for all transformers, 32-68% for FastText | Subword tokenisation eliminates OOV |

**The central lesson:** There is no universally "best" embedding. FastText dominates word-level tasks; sentence-transformers dominate sentence-level tasks. Choose based on your downstream need.

**Links:**
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [Ethayarajh, 2019 — How Contextual are Contextualized Representations?](https://aclanthology.org/D19-1006/)
- [Sentence-Transformers documentation](https://www.sbert.net/)
- [Bakarov, 2018 — Survey of Word Embedding Evaluation Methods](https://arxiv.org/abs/1801.09536)

---

### Extra 5: Comprehensive Documentation

**What we created:**
- `docs/HOMEWORK.md` + `_TR` — original assignment saved and translated
- `docs/LEARNING_OBJECTIVES.md` + `_TR` — 8-section study guide with curated links
- `docs/EXTRA_SUGGESTIONS.md` + `_TR` — 8 extension ideas with difficulty ratings
- `docs/EVALUATION_METRICS.md` + `_TR` — complete guide to embedding evaluation metrics
- `docs/RESULTS_ANALYSIS.md` + `_TR` — interpretation of our 5-model comparison with simple metric examples
- `docs/SELF_EVALUATION.md` + `_TR` — this document
- `README.md` + `_TR` — project overview with mermaid diagrams
- `outputs/model_comparison.md` — auto-generated detailed comparison report
- `outputs/evaluation_report.md` — auto-generated benchmark results

---

## Summary Scorecard

| Homework Requirement | Status | Quality |
|---------------------|--------|---------|
| `load_fasttext_model()` | Done | Handles .vec and .gz, limit parameter, error messages |
| `load_glove_model()` | Done | Auto-detects header, no_header fallback |
| `get_word_vector()` | Done | Turkish-aware normalisation, graceful OOV |
| `word_similarity()` | Done | Cosine via sklearn, None for OOV |
| `cluster_words()` | Done | L2 normalisation, random_state=42, n_init=10, ValueError for edge cases |
| `main.py` CLI | Done | argparse, env vars, CSV + MD output |
| `requirements.txt` | Done | Minimal dependencies |
| `.gitignore` | Done | Excludes data/*.vec*, outputs/ |
| `README.md` | Done | Mermaid diagrams, API reference, resources |

| Extra | Status |
|-------|--------|
| Word analogies (25 quads, 7 categories) | Done |
| t-SNE visualisation | Done |
| AnlamVer benchmark (500 pairs, Spearman ρ) | Done |
| Turkish analogy benchmark (7742 + 206 questions) | Done |
| Extended clustering (90 words, 5 categories, ARI/NMI) | Done |
| 5-model comparison (FastText vs BERT vs sentence-transformers) | Done |
| Evaluation at multiple vocabulary sizes (200K vs 500K) | Done |
| 14 documentation files (EN + TR) | Done |
