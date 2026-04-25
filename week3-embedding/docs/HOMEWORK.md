# Week 3 Homework — Turkish Word Embeddings

## Objective

Implement a small Python project that loads a **pre-trained Turkish word embedding model** (FastText or GloVe), retrieves word vectors, measures semantic similarity between words, and groups related words with **K-Means clustering**.

The aim of this assignment is to turn the theoretical concepts behind word embeddings (dense vectors, cosine similarity, clustering in a semantic space) into a working, reproducible pipeline.

---

## Background

Word embeddings map each word to a fixed-length dense vector such that words with similar meanings end up close to each other in the vector space. Unlike one-hot or count-based vectors, embeddings capture **semantic relationships** — e.g. `vector("kral") - vector("erkek") + vector("kadın")` is close to `vector("kraliçe")`.

For this assignment we use **pre-trained** embeddings, meaning somebody else has already trained the model on a large Turkish corpus (Wikipedia + Common Crawl). We only **consume** the vectors.

Two popular sources of pre-trained Turkish word vectors:

| Source | File | Size | Vocabulary |
|---|---|---|---|
| Facebook FastText (`cc.tr.300`) | `cc.tr.300.vec.gz` | ~4.5 GB | ~2M words |
| Stanford GloVe (community Turkish builds) | `glove.tr.300.txt` | ~1 GB | ~400K words |

Both files follow the **word2vec text format**: the first line contains `<vocab_size> <dim>`, and each subsequent line is `<word> <v1> <v2> ... <vN>`. Gensim's `KeyedVectors.load_word2vec_format()` reads this directly.

---

## Requirements

### 1. Load a pre-trained model

Write a function that loads either FastText or GloVe vectors and returns a `gensim.models.KeyedVectors` object.

```python
def load_fasttext_model(path: str) -> KeyedVectors: ...
def load_glove_model(path: str) -> KeyedVectors: ...
```

### 2. Retrieve a word vector

Given a loaded model and a word, return its 300-dimensional vector. Handle the out-of-vocabulary (OOV) case gracefully — return `None` or raise a clear error instead of crashing.

```python
def get_word_vector(model, word: str) -> np.ndarray | None: ...
```

### 3. Word similarity (cosine)

Compute the cosine similarity between two words. Return a float in `[-1, 1]`. Handle OOV words.

```python
def word_similarity(model, word1: str, word2: str) -> float: ...
```

### 4. K-Means clustering

Group a list of words into `k` clusters using K-Means (`random_state=42`, `n_init=10`). Return a dict or DataFrame mapping each word to its cluster label.

```python
def cluster_words(model, words: list[str], k: int = 3) -> dict[str, int]: ...
```

### 5. Command-line interface

Write a `main.py` script that:

1. Loads the model from a configurable path.
2. Prints a few word vectors (head only, not the full 300 dims).
3. Computes similarity for several word pairs.
4. Clusters an example word list with `k=3` and prints the grouping.
5. Saves the results to `outputs/` as CSV or Markdown.

### Example output

```
kedi ↔ köpek     : 0.78
araba ↔ otobüs  : 0.71
elma ↔ muz      : 0.64
kedi ↔ araba    : 0.12

Cluster 0: kedi, köpek, kuş, balık        (animals)
Cluster 1: araba, otobüs, uçak, tren     (vehicles)
Cluster 2: elma, muz, portakal, çilek    (fruits)
```

---

## Deliverables

- `src/embedding_utils.py` — the four functions above.
- `src/main.py` — the CLI entry point.
- `requirements.txt` — dependencies.
- `README.md` — how to download the model, install, and run.
- `outputs/` — similarity table and cluster assignments.

---

## Constraints & Tips

- **Do not commit the embedding file** to git — it is huge. Add `data/*.vec*` and `data/*.gz` to `.gitignore`.
- **Download once, reuse.** Loading the full FastText file takes minutes. Consider loading only the top N most frequent words with `limit=200000`.
- **Use `KeyedVectors`**, not the full FastText model object — it is much faster and uses far less memory.
- **Cosine similarity** = dot product of L2-normalised vectors. sklearn's `cosine_similarity` or `sklearn.metrics.pairwise.cosine_similarity` works; gensim's `model.similarity(w1, w2)` also works and is usually faster.
- **OOV handling matters.** Turkish is agglutinative, so words like "gidiyorum" may be absent even when "git" is present. FastText's subword support helps with this (if you load the full model, not just KeyedVectors), but for this assignment we stay at the KeyedVectors level.
- Normalise text (lowercase, strip punctuation) before lookup.

---

## Evaluation

Your submission will be judged on:

1. **Correctness** — do the four functions work as specified?
2. **Robustness** — does it handle OOV words and empty input?
3. **Reproducibility** — can I clone, install, and run without surprises?
4. **Code quality** — types, docstrings, a sensible CLI.
5. **Documentation** — a README that explains *what*, *how*, and *why*.

Good luck!
