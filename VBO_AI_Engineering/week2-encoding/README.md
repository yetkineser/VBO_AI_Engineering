# Week 2: Text Encoding & Sentiment Analysis

This project tackles the challenge of turning raw text into numerical representations that machine learning models can understand, and then building a sentiment classifier on top of those representations. It starts with the basics (integer encoding, one-hot, TF-IDF) and progressively introduces modern approaches (transformer embeddings, zero-shot classification, and fine-tuning a pre-trained Turkish BERT model).

---

## What You Will Learn

| Topic | What is expected |
|-------|-----------------|
| **Text to numbers** | ML models require vector/matrix inputs. You will convert words and sentences into numerical form through various encoding methods. |
| **Integer encoding** | Assigning each word a unique index from a vocabulary: e.g. `"good" -> 42`. Simple, but the ordering is meaningless. |
| **One-hot encoding** | Creating a binary dimension for each word in the vocabulary. If the word is present, its dimension is 1; otherwise 0. |
| **TF-IDF** | Weighting words by how distinctive they are in a document relative to the entire corpus. Common words get lower scores. |
| **Sentiment analysis** | A classification task that predicts the emotional tone of a text (e.g. positive / negative). |
| **Classical ML models** | Naive Bayes, Logistic Regression, SVM, XGBoost, LightGBM — no deep learning required for the core assignment. |
| **Transformer embedding** | Using a pre-trained language model to encode sentences into dense, meaning-rich vectors. |
| **Zero-shot classification** | Predicting sentiment without any training data, using a pre-trained NLI model. |
| **Fine-tuning** | Adapting a pre-trained Turkish BERT model to our specific sentiment task. |

---

## Core Concepts Explained

### 1. Text Vectorization

Computers cannot process raw text directly. Before feeding text to any model, you need to convert it into a sequence of numbers or a vector. This step is commonly known as **feature extraction** or **text vectorization**.

### 2. Integer Encoding

You build a **vocabulary** from all training texts and assign each unique word an integer index (0 to |V|-1). A sentence then becomes a sequence of these indices.

**Important caveat:** The numerical ordering is arbitrary — index 42 is not "greater" than index 7 in any meaningful sense. That is why integer encoding alone is rarely used as input to classical ML models. It serves mainly as a stepping stone toward one-hot or embedding-based representations.

### 3. One-Hot & Bag of Words

- **For labels:** With 3 classes, each sample becomes `[1,0,0]`, `[0,1,0]`, or `[0,0,1]`.
- **For text (Bag of Words):** You create a vector as wide as the vocabulary. For each word in a sentence, its corresponding position is set to 1 (or to its frequency count). This produces **high-dimensional, sparse** vectors. In scikit-learn, `CountVectorizer(binary=True)` implements this approach.

### 4. TF-IDF

TF-IDF goes one step beyond raw word counts. It down-weights words that appear in many documents (like "the" or "is") and up-weights words that are distinctive to a particular document. scikit-learn's `TfidfVectorizer` handles this automatically.

### 5. Building a Sentiment Classifier

The typical workflow is:

1. Prepare labeled text data (e.g. positive / negative reviews).
2. Convert text to numerical features using one of the methods above.
3. Split into training and test sets.
4. Train a classifier and evaluate with metrics (accuracy, F1, AUC-ROC).

---

## Modern Approaches (Beyond the Assignment)

This project goes beyond the basic homework requirements by implementing five additional approaches. Each one shows a different trade-off between data requirements, compute cost, and performance.

### 1. Transformer Embedding (`--transformer`)

CountVectorizer and TF-IDF treat words independently — they cannot distinguish "the movie was great" from "great was the movie". **Transformer embeddings** solve this by encoding the entire sentence into a dense vector that captures word order, context, and semantic relationships.

This project compares three transformer models to demonstrate the impact of domain and language match:

| Model | Dimensions | Language | Domain | AUC-ROC |
|-------|-----------|----------|--------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual (50+) | General | 0.9288 |
| `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 768 | Turkish | NLI / STS | 0.9441 |
| `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0` | 768 | Turkish + Multilingual | E-commerce | **0.9568** |

The Trendyol model achieved the best results because our dataset consists of product reviews — the same domain it was fine-tuned on. This demonstrates that **domain match matters more than model size**.

**When to use:** You have labeled data and want significantly better features without training a deep learning model from scratch.

### 2. Local Embedding with Ollama (`--ollama`)

Ollama runs the `nomic-embed-text` embedding model entirely on your local machine. No data is sent to any external API — this is the only fully private option.

The trade-off is performance: `nomic-embed-text` is an English-centric model with limited Turkish understanding (AUC = 0.83 on our Turkish dataset). For English text or RAG applications where privacy is a priority, it is a solid choice.

**When to use:** Privacy is critical, you are working with English text, or you need a local embedding solution without internet access.

### 3. Zero-Shot Classification (`--zero-shot`)

Zero-shot classification requires **no training data at all**. It uses a pre-trained Natural Language Inference (NLI) model to evaluate whether a sentence matches a given label hypothesis (e.g. "This text expresses a positive sentiment").

This project uses `xlm-roberta-large-xnli`, a cross-lingual model that works across 100+ languages including Turkish.

**When to use:** You have no labeled data, you are exploring a new domain, or you need a quick baseline before investing in data collection.

### 4. Fine-Tuning BERTurk (`--finetune`)

Fine-tuning takes a pre-trained Turkish BERT model (`dbmdz/bert-base-turkish-cased`) and adapts it to our specific sentiment task. The model already understands Turkish grammar, word meanings, and context from its pre-training on a large corpus. We add a classification head and train it on our labeled data for a few epochs.

This is the most powerful approach but requires more compute (GPU recommended) and labeled data.

**When to use:** You have labeled data and need the highest possible accuracy. This is the industry standard for text classification tasks.

### Approach Comparison

| Approach | Training Required? | Data Required? | Relative Strength |
|---|---|---|---|
| CountVectorizer + ML | Yes | Yes | Low |
| TF-IDF + ML | Yes | Yes | Medium |
| Transformer Embedding + ML (generic) | No (encode only) + Yes (ML) | Yes | Medium-High |
| Transformer Embedding + ML (domain-specific) | No (encode only) + Yes (ML) | Yes | **High** |
| Ollama Local Embedding + ML | No (encode only) + Yes (ML) | Yes | Low (for Turkish) |
| Zero-Shot | No | No | Medium-High |
| Fine-Tuning BERTurk | Yes (GPU recommended) | Yes | Highest |

---

## Usage

```bash
# Core assignment — classical ML only
python src/sentiment_analysis.py

# With Zeyrek morphological analysis for Turkish
python src/sentiment_analysis.py --zeyrek

# Add transformer embeddings (MiniLM + turkish-BERT-nli + Trendyol-ecomm)
python src/sentiment_analysis.py --transformer

# Add local Ollama embedding (requires: ollama pull nomic-embed-text)
python src/sentiment_analysis.py --ollama

# Add zero-shot classification (no training needed)
python src/sentiment_analysis.py --zero-shot

# Add BERTurk fine-tuning
python src/sentiment_analysis.py --finetune

# Run everything
python src/sentiment_analysis.py --all

# Use a custom CSV dataset with cross-dataset validation
python src/sentiment_analysis.py --data data/train.csv --test-data data/test.csv --transformer

# Full comparison with all embedding methods
python src/sentiment_analysis.py --data data/turkish_sentiment_binary_5k.csv \
    --test-data data/external_test_1k.csv --transformer --ollama
```

### Output Files

All outputs are saved to `outputs/`:

| File | Description |
|------|-------------|
| `run_log.txt` | Full timestamped log of every step |
| `comparison_results.csv` | All model results in tabular format |
| `analysis.md` | Auto-generated interpretation of the results with commentary |

---

## Recommended Implementation Steps

1. Choose or create a small sentiment dataset (at least two classes: positive/negative). The script includes 40 built-in Turkish sentences as a fallback.
2. **Integer encoding:** Demonstrate the concept with `CountVectorizer` or manual vocabulary mapping.
3. **One-hot representation:** Use `CountVectorizer(binary=True)` for text features.
4. **TF-IDF representation:** Use `TfidfVectorizer` for weighted features.
5. **Model training:** Train `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `XGBClassifier`, `LGBMClassifier`.
6. **Comparison:** Evaluate all vectorizer-model combinations using F1 and AUC-ROC.
7. *(Optional)* Add transformer embeddings, zero-shot, or fine-tuning for a deeper comparison.

---

## Evaluation Metrics

This project reports both **F1 (macro)** and **AUC-ROC** for every model:

- **F1** depends on a fixed classification threshold (usually 0.5). It tells you how well the model performs at that specific threshold.
- **AUC-ROC** is threshold-independent. It measures the model's ability to rank positive examples above negative ones across all possible thresholds. A model with high AUC but low F1 has learned useful patterns but needs threshold calibration.

Always report both. They tell different stories about model quality.

---

## Resources

### Official Documentation & Tutorials

- [scikit-learn: Working with text data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) — complete text classification pipeline
- [scikit-learn: CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [scikit-learn: LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

### Sentiment Analysis & NLP Fundamentals

- [NLTK Book — Chapter 6: Text Classification](https://www.nltk.org/book/ch06.html) — classical NLP perspective
- [Kaggle Learn: NLP Course](https://www.kaggle.com/learn/natural-language-processing) — short, practical modules
- Jurafsky & Martin, *Speech and Language Processing* — [online draft](https://web.stanford.edu/~jurafsky/slp3/) — especially the chapters on text classification and logistic regression

### Transformer Embedding

- [Sentence-Transformers Documentation](https://www.sbert.net/) — how to use sentence embeddings with pre-trained models
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — a visual guide to the transformer architecture
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — from word vectors to sentence vectors
- [Hugging Face: paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — generic multilingual sentence transformer
- [Hugging Face: bert-base-turkish-cased-mean-nli-stsb-tr](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) — Turkish-specific sentence transformer (NLI + STS)
- [Hugging Face: Trendyol E-Commerce Embedding](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) — Trendyol's domain-specific e-commerce embedding model (best performer in our experiments)

### Local Embedding with Ollama

- [Ollama](https://ollama.com/) — run LLMs and embedding models locally
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) — 768-dim embedding model for local use
- [Ollama Embedding API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings) — `/api/embed` endpoint documentation

### Zero-Shot Classification

- [Hugging Face: Zero-Shot Classification Guide](https://huggingface.co/tasks/zero-shot-classification) — concept and usage examples
- [Yin et al., 2019 — Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — foundational paper on zero-shot text classification
- [Hugging Face: xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) — the multilingual zero-shot model used in this project

### Fine-Tuning & Transfer Learning

- [Hugging Face: Text Classification Guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) — step-by-step fine-tuning walkthrough
- [BERTurk (dbmdz)](https://huggingface.co/dbmdz/bert-base-turkish-cased) — Turkish BERT model card
- [Jay Alammar: The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — visual explanation of BERT
- [Devlin et al., 2019 — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — original BERT paper
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) — NLP course covering BERT and transfer learning

### General NLP & Deep Learning

- [Hugging Face NLP Course (free)](https://huggingface.co/learn/nlp-course) — from beginner to advanced NLP
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — intuitive explanation of Naive Bayes
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=viZrOnJclY0) — visual introduction to embeddings
- [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — comprehensive blog post on attention mechanisms
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — building intuition for how neural networks learn

### Turkish NLP Tools

- [Zeyrek](https://github.com/obulat/zeyrek) — Turkish morphological analysis library
- [Turkish Sentiment Dataset (Kaggle)](https://www.kaggle.com/datasets/winvoker/turkish-sentiment-analysis-dataset) — labeled Turkish movie reviews
- [Trendyol Open Source Models](https://huggingface.co/Trendyol) — Trendyol's HuggingFace organization with 19 open-source models (LLMs, embeddings, vision)

### Text Extraction (Optional)

- [textract (PyPI)](https://pypi.org/project/textract/) — extract text from PDF, DOCX, and other file formats
- [textract GitHub](https://github.com/deanmalmgren/textract) — supported formats and setup instructions

---

## Notes

- If you are working with **Turkish text**, basic preprocessing (lowercasing, punctuation removal) can improve results. For advanced stemming/lemmatization, the optional `--zeyrek` flag uses the Zeyrek library.
- One-hot vectors become very **sparse** with large vocabularies. scikit-learn handles sparse matrices efficiently.
- The built-in sample dataset has only 40 sentences. Results are illustrative, not reliable. For meaningful comparisons, use a dataset with 500+ samples.

---

*This project was created as part of a course assignment on text encoding and sentiment analysis.*
