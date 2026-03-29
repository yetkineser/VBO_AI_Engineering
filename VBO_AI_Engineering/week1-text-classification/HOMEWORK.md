# Week 1 Homework: GPU-Accelerated Text Classification with PyTorch

## Objective
The goal of this assignment is to build a high-performance text classification pipeline using **PyTorch** and **Hugging Face Transformers**. You will leverage **GPU acceleration (CUDA)** to train and evaluate a model that categorizes text data efficiently.

---

## Tasks

### Task 1: Environment & GPU Setup (15 points)
1. Set up a Python environment with `torch`, `transformers`, and `datasets`.
2. Verify GPU availability using `torch.cuda.is_available()`.
3. Configure the training loop to move both the **model** and **tensors** to the `cuda` device.

### Task 2: Data Preprocessing & Tokenization (25 points)
1. Load a standard text classification dataset (e.g., *IMDb* for sentiment analysis or *AG News* for topic classification).
2. Implement a tokenizer (e.g., `BertTokenizer` or `AutoTokenizer`) to convert raw text into input IDs and attention masks.
3. Create PyTorch `DataLoader` objects with appropriate batch sizes for training and validation.

### Task 3: Model Architecture & Training (40 points)
1. Initialize a pre-trained Transformer model (e.g., `distilbert-base-uncased`) for sequence classification.
2. Define a training loop that includes:
    * Forward pass
    * Loss calculation (CrossEntropy)
    * Backpropagation
    * Optimizer step (AdamW)
3. Log the training loss and validation accuracy for at least 3 epochs.

### Task 4: Evaluation & Inference (20 points)
1. Evaluate the model on the test set and generate a **Classification Report** (Precision, Recall, F1-score).
2. Write a function that takes a custom string input, moves it to the GPU, and returns the predicted category.

---

## Deliverables
- `requirements.txt`: List of necessary libraries.
- `train.py` or `notebook.ipynb`: The complete code containing setup, training, and evaluation.
- `training_log.txt`: Output showing the loss decreasing over epochs and final accuracy.
- `inference_examples.pdf`: 3-5 examples of the model predicting categories for new text inputs.

---

## Grading
- **GPU Verification & Device Handling**: 15 points
- **Data Pipeline & Tokenization**: 25 points
- **Training Loop Implementation**: 40 points
- **Evaluation & Inference Logic**: 20 points
- **Total: 100 points**
