"""
Basic tests for the inference module.

WHY WRITE TESTS:
  "I ran the cell and it worked" is not testing.
  Tests catch regressions — if you change the model or tokenizer config
  and something breaks, tests tell you immediately.

Run:
    cd week1-text-classification
    python -m pytest tests/ -v
"""

import sys
import os

# Add src/ to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import predict, load_model, LABELS


def test_predict_returns_valid_label():
    """Prediction label should be one of the known labels."""
    model, tokenizer = load_model()
    result = predict("This movie was great!", model, tokenizer)
    assert result["label"] in LABELS.values()


def test_predict_returns_confidence_between_0_and_1():
    """Confidence score should be a valid probability."""
    model, tokenizer = load_model()
    result = predict("Terrible movie, worst ever.", model, tokenizer)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_returns_all_keys():
    """Result dict should have text, label, and confidence."""
    model, tokenizer = load_model()
    result = predict("It was okay.", model, tokenizer)
    assert "text" in result
    assert "label" in result
    assert "confidence" in result
