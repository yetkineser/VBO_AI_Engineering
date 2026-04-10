# Week 3 — Extra Suggestions: What Else Could Be Done?

This document covers **extensions beyond the homework** that are both educational and practical. Each suggestion includes a difficulty rating, what it teaches, and a link to get started.

---

## 1. Word Analogies (king - man + woman = queen)

**Difficulty:** Easy | **What it teaches:** Vector arithmetic captures semantic relationships

Word embeddings encode relationships as directions in vector space. The classic test is:

```
vec("kral") - vec("erkek") + vec("kadın") ≈ vec("kraliçe")
```

This works because the "gender direction" is consistent across the vocabulary. You can test country-capital, verb tense, and adjective degree analogies the same way.

**Already implemented** in this project with the `--analogy` flag.

**Links:**
- [The Illustrated Word2Vec — Analogy Section](https://jalammar.github.io/illustrated-word2vec/#analogy)
- [Gensim — most_similar with positive/negative](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar)

---

## 2. t-SNE / PCA Visualisation

**Difficulty:** Easy | **What it teaches:** High-dimensional structure made visible

Reduce 300-dimensional word vectors to 2D with PCA or t-SNE and plot them on a scatter chart. Words from the same semantic category should form visible clusters.

PCA is deterministic and fast but captures only linear variance. t-SNE preserves local neighbourhood structure better but is non-deterministic and slower.

**Already implemented** in this project with the `--visualise` flag.

**Links:**
- [scikit-learn — t-SNE User Guide](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
- [Distill.pub — How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — essential reading on interpreting t-SNE
- [Google's Embedding Projector](https://projector.tensorflow.org/) — interactive browser-based exploration

---

## 3. Bias Detection in Turkish Word Embeddings

**Difficulty:** Medium | **What it teaches:** Fairness, societal bias in ML, responsible AI

Word embeddings learn from internet text, which contains stereotypes. You can measure gender bias by checking which profession words (doktor, hemşire, mühendis, öğretmen) are closer to "erkek" vs "kadın".

This is ethically important and surprisingly easy to implement — it is just cosine similarity applied to sensitive word pairs.

**Links:**
- [Bolukbasi et al., 2016 — Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520) — the foundational bias paper
- [Caliskan et al., 2017 — Semantics derived automatically from language corpora contain human-like biases](https://arxiv.org/abs/1608.07187)
- [Google AI Blog — Reducing Gender Bias in Word Embeddings](https://ai.googleblog.com/2016/07/reducing-gender-bias-in-word-embeddings.html)

---

## 4. Compare FastText vs GloVe vs Word2Vec

**Difficulty:** Medium | **What it teaches:** Model selection, Turkish morphology impact

Load two or three different embedding models, run the same similarity and clustering tasks, and create a comparison table. Key questions:

- Which model agrees more with your intuition on Turkish word pairs?
- How do OOV rates compare? (FastText should win due to subword support)
- Do cluster assignments differ?

**Links:**
- [FastText — Pre-trained Vectors for 157 Languages](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Gensim Word2Vec Turkish (if available)](https://radimrehurek.com/gensim/models/word2vec.html)

---

## 5. Sentence Embeddings with Mean Pooling

**Difficulty:** Medium | **What it teaches:** From word vectors to sentence vectors

Average all word vectors in a sentence to create a sentence-level embedding. Then compute cosine similarity between sentences. This is crude but surprisingly effective for many tasks.

```python
def sentence_embedding(model, sentence: str) -> np.ndarray:
    words = sentence.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

Compare this with proper sentence transformers (like `sentence-transformers` from Week 2) to see the gap.

**Links:**
- [Arora et al., 2017 — A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx) — SIF weighting improves mean pooling
- [Sentence-Transformers](https://www.sbert.net/) — the modern alternative

---

## 6. Train Your Own Embeddings from Scratch

**Difficulty:** Hard | **What it teaches:** How embeddings are actually created

Instead of loading pre-trained vectors, train Word2Vec or FastText on a Turkish corpus (e.g. Turkish Wikipedia dump or a news dataset). This teaches you:

- How corpus size affects quality
- How window size and vector dimensionality change the results
- Why pre-trained models are usually better — you rarely have enough data

**Links:**
- [Gensim — Word2Vec Training Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [FastText — Training on Your Own Data](https://fasttext.cc/docs/en/unsupervised-tutorial.html)
- [Turkish Wikipedia Dumps](https://dumps.wikimedia.org/trwiki/) — raw text for training

---

## 7. Embedding-Based Text Classification

**Difficulty:** Hard | **What it teaches:** Bridging embeddings with downstream tasks

Use word embeddings as features for a text classifier:

1. Encode each document as the mean (or TF-IDF-weighted mean) of its word vectors.
2. Train a Logistic Regression or SVM on top.
3. Compare with TF-IDF baseline from Week 2.

This connects Week 2 (sentiment analysis) with Week 3 (embeddings) and shows why contextual embeddings (BERT, etc.) outperform static word vectors.

**Links:**
- [scikit-learn — Text Classification Pipeline](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- Week 2 project in this repository

---

## 8. Nearest Neighbours & Semantic Search

**Difficulty:** Medium | **What it teaches:** Practical application of embeddings

Build a simple semantic search engine: given a query word, find the N most similar words. Then extend it to multi-word queries by averaging vectors. This is the core idea behind vector databases like Pinecone, Weaviate, and FAISS.

**Links:**
- [Gensim — most_similar](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Pinecone — What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)

---

## Summary

| Suggestion | Difficulty | Implemented? |
|---|---|---|
| Word Analogies | Easy | Yes (`--analogy`) |
| t-SNE Visualisation | Easy | Yes (`--visualise`) |
| Bias Detection | Medium | No |
| FastText vs GloVe | Medium | No (infrastructure ready) |
| Sentence Embeddings | Medium | No |
| Train from Scratch | Hard | No |
| Embedding Classification | Hard | No |
| Semantic Search | Medium | No |
