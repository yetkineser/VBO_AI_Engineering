# Week 3 — Learning Objectives & Study Guide

This document unpacks the **why** behind the Week 3 homework. Each section lists a concept you are expected to understand, a short explanation, and curated links (articles, papers, videos) for deeper study.

The writing level is **upper-intermediate English** — approachable for non-native speakers but not dumbed down.

---

## 1. Why word embeddings exist

Classical text vectorisation (one-hot, Bag-of-Words, TF-IDF) represents each word as an isolated symbol. Two problems follow from that choice:

1. **No notion of similarity.** "car" and "automobile" are as different from each other as "car" and "banana".
2. **High dimensionality.** With a 100K-word vocabulary you get 100K-dimensional sparse vectors that are hard to use as features for modern neural networks.

Word embeddings fix both problems by learning a **dense, low-dimensional** vector for each word such that *similar words have similar vectors*. The similarity is learned from context: words that appear in the same kinds of sentences end up near each other.

**Study links**
- [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — the single clearest visual introduction.
- [Chris McCormick — Word2Vec Tutorial Part 1](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) — skip-gram with concrete numbers.
- [StatQuest — Word Embedding and Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0) — 15-minute intuitive video.
- [Stanford CS224N Lecture 1 — Introduction and Word Vectors](https://www.youtube.com/watch?v=rmVRLeJRkl4) — the gold-standard lecture.

---

## 2. Word2Vec, GloVe, and FastText — three flavours of the same idea

| Model | Year | Key idea | Strength |
|---|---|---|---|
| **Word2Vec** | 2013 | Predict a word from its context (CBOW) or the context from a word (skip-gram) | Fast, iconic baseline |
| **GloVe** | 2014 | Factorise a global word-cooccurrence matrix | Captures global statistics, not just local windows |
| **FastText** | 2016 | Represent each word as the sum of its **character n-grams** | Handles rare words and morphology — perfect for Turkish |

FastText is the right choice for Turkish because Turkish is **agglutinative** — a single root can produce dozens of surface forms (`kitap`, `kitabım`, `kitaplarımızda`). FastText can compose a vector for `kitaplarımızda` from subword pieces even if that exact form was never seen during training.

**Study links**
- [Mikolov et al., 2013 — Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) — original Word2Vec paper.
- [Pennington et al., 2014 — GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — original GloVe paper.
- [Bojanowski et al., 2017 — Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — the FastText paper.
- [Facebook FastText — Pre-trained vectors for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html) — where to download `cc.tr.300`.
- [Gensim docs — KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html) — the API you will use.

---

## 3. Cosine similarity — the distance metric for embeddings

Once each word is a vector, the natural question is: *how similar are two words?* For embeddings the answer is almost always **cosine similarity**:

```
cos(u, v) = (u · v) / (||u|| * ||v||)
```

Cosine similarity ignores vector **magnitude** and only cares about **direction**. This matters because embedding magnitudes are often affected by word frequency (more frequent words tend to have larger norms), and you don't want "the" to look "stronger" than "philosophy".

- `+1` — identical direction (maximally similar)
- `0`  — orthogonal (unrelated)
- `-1` — opposite direction (antonyms, in theory — in practice rarely seen)

**Study links**
- [Wikipedia — Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) — quick reference.
- [Machine Learning Mastery — How to Calculate Cosine Similarity](https://machinelearningmastery.com/cosine-similarity-for-nlp/) — worked examples in Python.
- [sklearn — cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — reference implementation.
- [Gensim — KeyedVectors.similarity](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.similarity) — the fast path.

---

## 4. K-Means clustering in embedding space

With a way to measure similarity between words, it is natural to ask: *can we automatically discover groups of related words?* K-Means is the simplest algorithm that does this.

**How it works in one paragraph:** Pick `k` initial cluster centres at random. Assign every point to its nearest centre. Recompute each centre as the mean of the points assigned to it. Repeat until nothing moves. The result is `k` clusters where each point is closer to its own centre than to any other.

Key parameters you need to know:
- `n_clusters=k` — how many groups.
- `random_state=42` — fixes the random initialisation for reproducibility.
- `n_init=10` — runs K-Means 10 times from different random seeds and keeps the best. The homework asks for this because a single run can get stuck in a bad local optimum.

One caveat: K-Means uses **Euclidean distance**, not cosine. For embedding clustering you should either (a) L2-normalise your vectors first so Euclidean distance becomes equivalent to cosine distance, or (b) use `sklearn.cluster.KMeans` on raw vectors and accept that results will be close but not identical to cosine-based clustering.

**Study links**
- [StatQuest — K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) — 8-minute visual explanation.
- [sklearn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — the API you will use.
- [sklearn — Clustering User Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means) — discussion of when K-Means works and when it doesn't.
- [Google Developers — K-Means Advantages and Disadvantages](https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages) — know the trade-offs.

---

## 5. Practical engineering lessons

The homework is small but it exercises several engineering reflexes that matter in real-world NLP work:

1. **Large files don't belong in git.** The FastText Turkish model is ~4.5 GB. Add it to `.gitignore` and document the download step in the README instead.
2. **Lazy loading matters.** Loading 2 million word vectors takes minutes and a lot of RAM. Use the `limit=` parameter of `load_word2vec_format()` to load only the top N most frequent words during development.
3. **Reproducibility.** Always set `random_state` for any algorithm that uses randomness. The homework explicitly requires `random_state=42`.
4. **OOV is not an exception, it's the default.** Assume that any word you look up might not be in the vocabulary, and return `None` or raise a clear error rather than letting the program crash.
5. **Normalise before you look up.** `"Kedi"`, `"KEDİ"`, and `"kedi."` should all map to the same vector. Lowercasing and stripping punctuation is the minimum.

**Study links**
- [Gensim — memory-efficient loading](https://radimrehurek.com/gensim/models/keyedvectors.html#how-to-obtain-word-vectors) — `limit` and `binary` parameters.
- [The Twelve-Factor App — III. Config](https://12factor.net/config) — why model paths belong in environment variables, not code.
- [Hugging Face NLP Course — Tokenization](https://huggingface.co/learn/nlp-course/chapter6/1) — deeper background on text normalisation.

---

## 6. Turkish-specific NLP notes

Turkish has a few properties that make it an interesting test case for word embeddings:

- **Agglutinative morphology.** Suffixes stack to encode tense, person, case, possession, etc. This explodes the surface vocabulary.
- **Vowel harmony.** Vowels in suffixes change to match the root. This matters for subword models like FastText.
- **Relatively free word order.** Context windows still work, but the "neighbours" of a word can be more variable than in English.
- **Special characters.** `ı`, `İ`, `ş`, `ğ`, `ö`, `ü`, `ç` — make sure your tokenisation and normalisation preserve these correctly.

**Study links**
- [Zemberek — Turkish NLP toolkit](https://github.com/ahmetaa/zemberek-nlp) — lemmatisation, morphology, etc.
- [Zeyrek — Python port of Zemberek's morphology](https://github.com/obulat/zeyrek) — lighter-weight alternative.
- [Turkish NLP Resources (awesome list)](https://github.com/topics/turkish-nlp) — GitHub aggregator.
- [Stemming and Lemmatization for Turkish (blog)](https://towardsdatascience.com/text-preprocessing-for-turkish-4f1abb72a9d8) — practical walkthrough.

---

## 7. What "understanding" this homework looks like

After finishing the homework you should be able to answer, without looking things up:

- What does a word embedding vector represent? Where do the numbers come from?
- Why is cosine similarity preferred over Euclidean distance for embeddings?
- What is the difference between Word2Vec, GloVe, and FastText in one sentence each?
- Why does FastText handle Turkish better than Word2Vec?
- Why do we set `random_state` and `n_init` on K-Means?
- How would you debug the case where two "obviously similar" words have low cosine similarity in your model?

If any of those feel shaky, revisit the corresponding section above.

---

## 8. Going beyond the homework

If you finish early and want to push further, here are three directions that teach a lot:

1. **Analogy tasks** — compute `king - man + woman` and find the nearest word. Classic smoke test.
2. **Visualisation** — run PCA or t-SNE on your clustered words and draw a 2D scatter plot. Seeing the clusters makes the concept much more concrete.
3. **Compare models** — load both FastText and GloVe, run the same similarity and clustering tasks on both, and report which one agrees more with your intuition.

These are covered in detail in the companion `EXTRA_SUGGESTIONS.md` file in this folder.
