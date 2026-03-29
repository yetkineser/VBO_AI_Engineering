# Week 1: GPU-Accelerated Text Classification with PyTorch

Sentiment analysis on the IMDb dataset using a fine-tuned DistilBERT model with PyTorch and Hugging Face Transformers.

## Architecture

```
Raw Text → AutoTokenizer (DistilBERT) → Token IDs + Attention Mask
    → DistilBertForSequenceClassification → Logits → Softmax → Prediction
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train the model (3 epochs, ~5 min on GPU)
python src/train.py

# Run inference on custom text
python src/inference.py "This movie was absolutely fantastic!"
```

## Project Structure

```
├── src/
│   ├── config.py        # Device setup, hyperparameters
│   ├── train.py         # Data loading, training loop, evaluation
│   └── inference.py     # Single-text prediction function
├── tests/
│   └── test_inference.py
├── outputs/             # Training logs, saved model
├── requirements.txt
└── README.md
```

## Tech Stack

- PyTorch (training loop, GPU management)
- Hugging Face Transformers (DistilBERT model + tokenizer)
- Hugging Face Datasets (IMDb dataset)
- scikit-learn (classification report)
