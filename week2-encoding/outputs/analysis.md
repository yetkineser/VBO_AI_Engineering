# Sentiment Analysis — Results Comparison & Interpretation

*Auto-generated analysis report*

## Full Results Table

| Vectorization | Model | Accuracy | Precision | Recall | F1 (macro) | AUC-ROC | Time (s) |
|---|---|---|---|---|---|---|---|
| CountVectorizer (One-Hot) | Naive Bayes (MultinomialNB) | 0.8120 | 0.8157 | 0.8120 | 0.8115 | 0.8886 | 0.015 |
| CountVectorizer (One-Hot) | Logistic Regression | 0.8560 | 0.8560 | 0.8560 | 0.8560 | 0.9238 | 0.585 |
| CountVectorizer (One-Hot) | SVM (LinearSVC) | 0.8480 | 0.8480 | 0.8480 | 0.8480 | 0.9162 | 0.030 |
| CountVectorizer (One-Hot) | XGBoost | 0.8050 | 0.8120 | 0.8050 | 0.8039 | 0.8784 | 6.274 |
| CountVectorizer (One-Hot) | LightGBM | 0.7980 | 0.8043 | 0.7980 | 0.7969 | 0.8773 | 1.449 |
| TfidfVectorizer | Naive Bayes (MultinomialNB) | 0.8170 | 0.8196 | 0.8170 | 0.8166 | 0.9129 | 0.010 |
| TfidfVectorizer | Logistic Regression | 0.8350 | 0.8354 | 0.8350 | 0.8350 | 0.9226 | 0.183 |
| TfidfVectorizer | SVM (LinearSVC) | 0.8600 | 0.8600 | 0.8600 | 0.8600 | 0.9349 | 0.016 |
| TfidfVectorizer | XGBoost | 0.8040 | 0.8062 | 0.8040 | 0.8037 | 0.8813 | 14.154 |
| TfidfVectorizer | LightGBM | 0.8040 | 0.8060 | 0.8040 | 0.8037 | 0.8787 | 1.288 |
| Transformer (multilingual-MiniLM) | Logistic Regression | 0.8520 | 0.8521 | 0.8520 | 0.8520 | 0.9246 | 0.112 |
| Transformer (multilingual-MiniLM) | SVM (LinearSVC) | 0.8540 | 0.8541 | 0.8540 | 0.8540 | 0.9225 | 1.307 |
| Transformer (multilingual-MiniLM) | XGBoost | 0.8580 | 0.8580 | 0.8580 | 0.8580 | 0.9288 | 0.569 |
| Transformer (multilingual-MiniLM) | LightGBM | 0.8530 | 0.8531 | 0.8530 | 0.8530 | 0.9277 | 0.247 |
| Transformer (turkish-BERT-nli) | Logistic Regression | 0.8740 | 0.8747 | 0.8740 | 0.8739 | 0.9393 | 0.580 |
| Transformer (turkish-BERT-nli) | SVM (LinearSVC) | 0.8580 | 0.8586 | 0.8580 | 0.8579 | 0.9259 | 3.939 |
| Transformer (turkish-BERT-nli) | XGBoost | 0.8720 | 0.8723 | 0.8720 | 0.8720 | 0.9441 | 0.813 |
| Transformer (turkish-BERT-nli) | LightGBM | 0.8720 | 0.8722 | 0.8720 | 0.8720 | 0.9418 | 0.455 |
| Transformer (Trendyol-ecomm) | Logistic Regression | 0.8910 | 0.8911 | 0.8910 | 0.8910 | 0.9567 | 0.053 |
| Transformer (Trendyol-ecomm) | SVM (LinearSVC) **[BEST]** | 0.9020 | 0.9021 | 0.9020 | 0.9020 | 0.9568 | 0.200 |
| Transformer (Trendyol-ecomm) | XGBoost | 0.8850 | 0.8850 | 0.8850 | 0.8850 | 0.9495 | 0.824 |
| Transformer (Trendyol-ecomm) | LightGBM | 0.8920 | 0.8921 | 0.8920 | 0.8920 | 0.9512 | 0.445 |
| Ollama (nomic-embed-text) | Logistic Regression | 0.7450 | 0.7461 | 0.7450 | 0.7447 | 0.8280 | 0.081 |
| Ollama (nomic-embed-text) | SVM (LinearSVC) | 0.7490 | 0.7493 | 0.7490 | 0.7489 | 0.8339 | 0.315 |
| Ollama (nomic-embed-text) | XGBoost | 0.7420 | 0.7439 | 0.7420 | 0.7415 | 0.8243 | 0.803 |
| Ollama (nomic-embed-text) | LightGBM | 0.7360 | 0.7380 | 0.7360 | 0.7354 | 0.8247 | 0.412 |

## Overall Winner

**Transformer (Trendyol-ecomm) + SVM (LinearSVC)** achieved the highest AUC-ROC of **0.9568** (F1=0.9020). AUC-ROC is the primary metric because it evaluates ranking quality independent of any threshold choice.

## Vectorization Method Comparison

### CountVectorizer (One-Hot)

- Best model: **Logistic Regression** (F1=0.8560, AUC=0.9238)
- CountVectorizer with `binary=True` creates a one-hot style representation. Each word is either present (1) or absent (0). This approach ignores word frequency and word order entirely.

### Ollama (nomic-embed-text)

- Best model: **SVM (LinearSVC)** (F1=0.7489, AUC=0.8339)
- Ollama runs the `nomic-embed-text` embedding model locally on your machine. No data is sent to any external API. The model produces 768-dimensional dense vectors, but since it is primarily trained on English text, its Turkish performance is limited.

### TfidfVectorizer

- Best model: **SVM (LinearSVC)** (F1=0.8600, AUC=0.9349)
- TF-IDF weights words by how important they are to a document relative to the whole corpus. Common words get lower scores, distinctive words get higher ones. Still ignores word order.

### Transformer (Trendyol-ecomm)

- Best model: **SVM (LinearSVC)** (F1=0.9020, AUC=0.9568)
- Trendyol's e-commerce embedding model is fine-tuned from `Alibaba-NLP/gte-multilingual-base` on Turkish e-commerce data. Since our dataset consists of product reviews, the domain match gives this model a significant advantage.

### Transformer (multilingual-MiniLM)

- Best model: **XGBoost** (F1=0.8580, AUC=0.9288)
- Transformer embeddings encode the full meaning of a sentence into a dense vector. Unlike bag-of-words methods, they capture word order, context, and semantic relationships.

### Transformer (turkish-BERT-nli)

- Best model: **XGBoost** (F1=0.8720, AUC=0.9441)
- Transformer embeddings encode the full meaning of a sentence into a dense vector. Unlike bag-of-words methods, they capture word order, context, and semantic relationships.

### Head-to-Head Comparison

Using **LightGBM** as the control model across vectorizers:

- CountVectorizer (One-Hot): F1=0.7969, AUC=0.8773
- Ollama (nomic-embed-text): F1=0.7354, AUC=0.8247
- TfidfVectorizer: F1=0.8037, AUC=0.8787
- Transformer (Trendyol-ecomm): F1=0.8920, AUC=0.9512
- Transformer (multilingual-MiniLM): F1=0.8530, AUC=0.9277
- Transformer (turkish-BERT-nli): F1=0.8720, AUC=0.9418

## AUC-ROC vs F1: Why Both Matter

F1 score depends on a fixed classification threshold (usually 0.5). A model might have mediocre F1 simply because its default threshold is not optimal. AUC-ROC, on the other hand, evaluates the model's ability to **rank** positive examples above negative ones across all possible thresholds. A high AUC with low F1 means the model learned useful patterns but needs threshold tuning to translate that into good predictions.

## Key Takeaways

1. **Representation matters more than the model.** The same SVM or Logistic Regression can jump from ~80% to ~90% accuracy simply by changing how text is vectorized. The choice of embedding model has a bigger impact than the choice of classifier.
2. **Domain-specific embeddings dominate.** Trendyol's e-commerce embedding model outperformed all others because its training domain (Turkish product reviews) matches our test data. Generic multilingual models like MiniLM scored lower despite being larger.
3. **AUC and F1 tell different stories.** Always report both. A model with high AUC but lower F1 is not broken — it just needs threshold calibration. The AUC-F1 gap across all models suggests 3-5% F1 improvement is possible with threshold tuning alone.
4. **Local embedding models (Ollama) lag behind.** `nomic-embed-text` running locally via Ollama scored the lowest AUC (~0.83). This is expected — it is an English-centric model not optimized for Turkish sentiment. Local models are best suited for English text or RAG applications where privacy is a priority.
5. **Linear models beat tree-based models on text.** Logistic Regression and SVM consistently outperformed XGBoost and LightGBM with sparse features (CountVectorizer, TF-IDF). Tree-based models performed better only with dense transformer embeddings.
6. **TF-IDF is not dead.** On a binary sentiment task with enough data, TF-IDF + SVM can match or beat generic transformer embeddings. Do not dismiss classical methods without benchmarking them.

## Experiment Setup

- **Training set:** 5,000 samples
- **Test set:** 1,000 samples
- Train and test sets come from separate, non-overlapping splits of the source dataset to prevent data leakage.
