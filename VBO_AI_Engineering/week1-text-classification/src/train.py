"""
Week 1 Homework: GPU-Accelerated Text Classification with PyTorch

This script trains a Transformer model on IMDb sentiment data.
It covers all 4 homework tasks: GPU setup, tokenization, training loop, evaluation.

Run:
    python src/train.py --experiment 1   (DistilBERT, 5K samples)
    python src/train.py --experiment 2   (BERT-base, 25K samples)
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report

from config import (
    DEVICE, MODEL_NAME, NUM_LABELS,
    BATCH_SIZE, MAX_LENGTH, LEARNING_RATE, NUM_EPOCHS, SEED,
    OUTPUT_DIR, LOG_FILE, TRAIN_SAMPLES, TEST_SAMPLES, EXPERIMENT_NAME,
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Environment & GPU Setup (15 points)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT'S HAPPENING:
#   We verify that PyTorch can see the GPU, print diagnostic info,
#   and set random seeds for reproducibility.
#
# WHY THIS MATTERS:
#   In production, you ALWAYS log your environment at startup.
#   When a training run gives weird results 3 days later, the first
#   question is "what GPU was it on? what CUDA version?"
#
# YOUR DS BACKGROUND:
#   You've set random seeds before (np.random.seed). Same concept,
#   but PyTorch has its own seed system. torch.manual_seed controls
#   the CPU RNG, torch.cuda.manual_seed_all controls all GPU RNGs.
# ══════════════════════════════════════════════════════════════════════

def setup_environment():
    """Configure device, seeds, and print diagnostics."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Model           : {MODEL_NAME}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU device      : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print(f"Using device    : {DEVICE}")
    print(f"Train samples   : {TRAIN_SAMPLES or 'ALL (25K)'}")
    print(f"Test samples    : {TEST_SAMPLES or 'ALL (25K)'}")
    print("=" * 60)
    print()


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Data Preprocessing & Tokenization (25 points)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT'S HAPPENING:
#   1. Load IMDb dataset from Hugging Face (25K train, 25K test)
#   2. Tokenize text into token IDs + attention masks
#   3. Wrap in PyTorch DataLoaders for batched iteration
#
# KEY CONCEPTS YOU NEED TO UNDERSTAND:
#
#   TOKEN IDS:
#     Text → ["this", "movie", "is", "great"] → [2023, 3185, 2003, 2307]
#     Each word (or sub-word) maps to an integer in the model's vocabulary.
#     You've seen this with TF-IDF vectorization — same idea, different encoding.
#
#   ATTENTION MASK:
#     When you batch sentences of different lengths, shorter ones get padded
#     with zeros to match the longest. The attention mask is a binary tensor:
#       1 = real token (pay attention to this)
#       0 = padding (ignore this)
#     Without it, the model would treat padding as meaningful text.
#
#   MAX_LENGTH & TRUNCATION:
#     DistilBERT has a hard limit of 512 tokens. We use 256 for speed.
#     Longer reviews get cut off (truncation=True).
#     Shorter reviews get padded (padding="max_length").
#
#   WHY DataLoader AND NOT JUST A LIST:
#     DataLoader handles batching, shuffling, and (optionally) parallel
#     data loading. In production, this is the standard interface.
#     Think of it as a generator that yields (batch_x, batch_y) tuples.
# ══════════════════════════════════════════════════════════════════════

def load_and_tokenize_data():
    """Load IMDb dataset, tokenize, and create DataLoaders."""
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    train_dataset = dataset["train"].shuffle(seed=SEED)
    test_dataset = dataset["test"].shuffle(seed=SEED)

    if TRAIN_SAMPLES:
        train_dataset = train_dataset.select(range(TRAIN_SAMPLES))
    if TEST_SAMPLES:
        test_dataset = test_dataset.select(range(TEST_SAMPLES))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    # Initialize the tokenizer
    # AutoTokenizer automatically picks the right tokenizer for the model name.
    # For "distilbert-base-uncased", it loads DistilBertTokenizerFast.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        """Tokenize a batch of examples.

        This function is applied to the entire dataset at once via .map(),
        which is much faster than tokenizing one example at a time.

        Returns dict with 'input_ids', 'attention_mask' (and 'label' passthrough).
        """
        return tokenizer(
            examples["text"],
            padding="max_length",   # Pad short texts to MAX_LENGTH
            truncation=True,        # Cut long texts at MAX_LENGTH
            max_length=MAX_LENGTH,
        )

    # .map() applies the function to all examples in batches
    # batched=True means the function receives lists, not single items → much faster
    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Tell the dataset to return PyTorch tensors instead of Python lists.
    # We only keep the columns the model needs.
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create DataLoaders
    # shuffle=True for training (so the model sees data in different order each epoch)
    # shuffle=False for test (order doesn't matter for evaluation, and reproducibility)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}")
    print()

    return train_loader, test_loader, tokenizer


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Model Architecture & Training (40 points)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT'S HAPPENING:
#   1. Load a pre-trained DistilBERT with a classification head on top
#   2. Move the model to GPU
#   3. Run the training loop: forward → loss → backward → optimizer step
#
# THE TRAINING LOOP — LINE BY LINE:
#
#   model.train()
#     Puts the model in training mode. This enables dropout and
#     batch normalization (if any). Forgetting this is a classic bug.
#
#   for batch in train_loader:
#     Each batch is a dict with 'input_ids', 'attention_mask', 'label'.
#     Each is a tensor of shape (BATCH_SIZE, MAX_LENGTH) or (BATCH_SIZE,).
#
#   .to(DEVICE)
#     THIS IS THE GPU PART. Every tensor must be moved to the same device
#     as the model. If the model is on cuda and your data is on cpu,
#     you get: RuntimeError: Expected all tensors to be on the same device.
#     This is the #1 error every beginner hits.
#
#   outputs = model(input_ids=..., attention_mask=..., labels=...)
#     Forward pass. When you pass `labels`, HuggingFace automatically:
#       - Computes logits (raw predictions)
#       - Computes CrossEntropyLoss internally
#       - Returns both in `outputs.loss` and `outputs.logits`
#
#   loss.backward()
#     Backpropagation. Computes gradients for every parameter.
#     This is where PyTorch's autograd does the calculus for you.
#
#   optimizer.step()
#     Updates the weights using the gradients. AdamW = Adam + weight decay.
#     This is the standard optimizer for fine-tuning transformers.
#
#   optimizer.zero_grad()
#     Clears the gradients. PyTorch ACCUMULATES gradients by default.
#     If you forget this, gradients pile up and training explodes.
#
# YOUR DS BACKGROUND:
#   You know what gradient descent does mathematically. This is just
#   the explicit code version of what sklearn.fit() does internally.
# ══════════════════════════════════════════════════════════════════════

def train_model(train_loader, test_loader):
    """Initialize model, train for NUM_EPOCHS, validate each epoch."""

    # Load pre-trained model with a classification head
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    # Move model to GPU — the model's millions of parameters now live in GPU memory
    model.to(DEVICE)
    print(f"Model moved to: {DEVICE}")
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / 1e6
        print(f"GPU memory used by model: {mem_mb:.0f} MB")

    # Optimizer: AdamW with a small learning rate for fine-tuning
    # WHY AdamW and not SGD?
    # Transformers are sensitive to optimizer choice. AdamW (Adam with
    # decoupled weight decay) is the standard for fine-tuning BERT-family models.
    # The learning rate 2e-5 comes from the original BERT paper.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Prepare log file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        # ── Training phase ──
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move ALL tensors to the same device as the model
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Forward pass: model computes logits AND loss (because we pass labels)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass: compute gradients
            loss.backward()

            # Update weights using gradients
            optimizer.step()

            # Clear gradients for next iteration
            optimizer.zero_grad()

            total_train_loss += loss.item()
            num_train_batches += 1

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_train_loss / num_train_batches
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        epoch_time = time.time() - epoch_start

        # ── Validation phase ──
        val_accuracy = evaluate_accuracy(model, test_loader)

        log_line = (
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_line)
        log_lines.append(log_line)

    # Save training log (homework deliverable)
    with open(LOG_FILE, "w") as f:
        f.write(f"{EXPERIMENT_NAME}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Train Samples: {TRAIN_SAMPLES or 'ALL (25K)'}\n")
        f.write(f"Test Samples: {TEST_SAMPLES or 'ALL (25K)'}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write("=" * 60 + "\n\n")
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nTraining log saved to {LOG_FILE}")
    print()

    return model


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluation & Inference (20 points)
# ══════════════════════════════════════════════════════════════════════
#
# WHAT'S HAPPENING:
#   1. Run the model on the entire test set, collect predictions
#   2. Generate a sklearn classification report (precision, recall, F1)
#
# KEY CONCEPTS:
#
#   model.eval()
#     Puts the model in evaluation mode. Disables dropout.
#     Training with dropout ON → evaluation with dropout OFF.
#     Forgetting model.eval() is a classic bug that gives lower accuracy.
#
#   torch.no_grad()
#     Tells PyTorch NOT to track gradients. During evaluation, we don't
#     need gradients (no backward pass). This saves memory and speeds up
#     inference. In production, you ALWAYS wrap inference in no_grad().
#
#   logits.argmax(dim=-1)
#     Logits are raw scores, e.g. [2.1, -0.5] for [positive, negative].
#     argmax picks the index of the highest score → the predicted class.
#     This is the same as np.argmax on your sklearn predict_proba output.
# ══════════════════════════════════════════════════════════════════════

def evaluate_accuracy(model, data_loader):
    """Calculate accuracy on a dataset. Used during training for validation."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total


def full_evaluation(model, data_loader):
    """Generate classification report on the test set and save detailed results."""
    from sklearn.metrics import confusion_matrix, accuracy_score
    import json

    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
            probs = torch.softmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy().tolist())

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Positive"],
    )
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    print("CONFUSION MATRIX")
    print(f"                 Predicted Neg  Predicted Pos")
    print(f"  Actual Neg     {cm[0][0]:<14} {cm[0][1]}")
    print(f"  Actual Pos     {cm[1][0]:<14} {cm[1][1]}")
    print()

    # Save detailed test results to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_report_path = os.path.join(OUTPUT_DIR, "test_results.txt")
    with open(test_report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model          : {MODEL_NAME}\n")
        f.write(f"Device         : {DEVICE}\n")
        f.write(f"Test samples   : {len(all_labels)}\n")
        f.write(f"Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Classification Report\n")
        f.write("-" * 60 + "\n")
        f.write(report + "\n")
        f.write("Confusion Matrix\n")
        f.write("-" * 60 + "\n")
        f.write(f"                 Predicted Neg  Predicted Pos\n")
        f.write(f"  Actual Neg     {cm[0][0]:<14} {cm[0][1]}\n")
        f.write(f"  Actual Pos     {cm[1][0]:<14} {cm[1][1]}\n\n")
        f.write(f"True Negatives : {cm[0][0]}\n")
        f.write(f"False Positives: {cm[0][1]}\n")
        f.write(f"False Negatives: {cm[1][0]}\n")
        f.write(f"True Positives : {cm[1][1]}\n")

    # Save as JSON for programmatic access
    test_json_path = os.path.join(OUTPUT_DIR, "test_results.json")
    results_dict = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "test_samples": len(all_labels),
        "accuracy": round(accuracy, 4),
        "classification_report": classification_report(
            all_labels, all_predictions,
            target_names=["Negative", "Positive"],
            output_dict=True,
        ),
        "confusion_matrix": cm.tolist(),
    }
    with open(test_json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Append to training log too
    with open(LOG_FILE, "a") as f:
        f.write("\nClassification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report + "\n")
        f.write("Confusion Matrix\n")
        f.write(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}\n")

    print(f"Test results saved to {test_report_path}")
    print(f"Test results (JSON) saved to {test_json_path}")

    return report


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
#
# WHY if __name__ == "__main__":?
#   This guard ensures the code below only runs when you execute
#   `python train.py` directly. If another file imports from train.py
#   (e.g., to reuse evaluate_accuracy), the training doesn't auto-start.
#   This is Python 101 but many DS people skip it in notebooks.
# ══════════════════════════════════════════════════════════════════════

def save_model(model, tokenizer):
    """Save the trained model and tokenizer for later inference."""
    save_path = os.path.join(OUTPUT_DIR, "model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}/")


if __name__ == "__main__":
    setup_environment()
    train_loader, test_loader, tokenizer = load_and_tokenize_data()
    model = train_model(train_loader, test_loader)
    full_evaluation(model, test_loader)
    save_model(model, tokenizer)

    print(f"Done! Check {OUTPUT_DIR}/ for all results.")
