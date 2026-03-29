"""
Inference module: predict sentiment for custom text inputs.

Usage:
    python src/inference.py "This movie was absolutely fantastic!"
    python src/inference.py "Terrible acting and boring plot."

WHY A SEPARATE FILE:
In notebook-land, inference is a cell at the bottom of the notebook.
In production, inference is a separate module (or API endpoint) that
loads a model once and serves predictions. This is the first step
toward that pattern.
"""

import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DEVICE, MODEL_NAME, MAX_LENGTH, NUM_LABELS, OUTPUT_DIR


LABELS = {0: "Negative", 1: "Positive"}


def predict(text, model, tokenizer):
    """Predict sentiment for a single text input.

    Args:
        text: The review text to classify.
        model: The trained model (already on DEVICE).
        tokenizer: The tokenizer matching the model.

    Returns:
        dict with 'label', 'confidence', and 'text'.
    """
    model.eval()

    # Tokenize the input — same settings as training
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",  # Return PyTorch tensors (not lists)
    )

    # Move input tensors to GPU
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Inference with no gradient tracking
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_class = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    return {
        "text": text,
        "label": LABELS[predicted_class],
        "confidence": round(confidence, 4),
    }


def load_model():
    """Load trained model and tokenizer. Falls back to pre-trained if no saved model."""
    import os
    trained_path = os.path.join(OUTPUT_DIR, "model")
    if os.path.exists(trained_path):
        print(f"Loading TRAINED model from {trained_path}")
        tokenizer = AutoTokenizer.from_pretrained(trained_path)
        model = AutoModelForSequenceClassification.from_pretrained(trained_path)
    else:
        print(f"WARNING: No trained model found at {trained_path}, using pre-trained {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py \"Your text here\"")
        sys.exit(1)

    text_input = sys.argv[1]

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    model, tokenizer = load_model()

    result = predict(text_input, model, tokenizer)

    print(f"\nInput   : {result['text']}")
    print(f"Label   : {result['label']}")
    print(f"Confidence: {result['confidence']}")
