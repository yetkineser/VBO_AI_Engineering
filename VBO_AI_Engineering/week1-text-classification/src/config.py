"""
Centralized configuration for the training pipeline.

Supports multiple experiments via --experiment flag:
    python src/train.py --experiment 1   (DistilBERT, 5K samples)
    python src/train.py --experiment 2   (BERT-base, 25K samples)
"""

import argparse
import torch


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()
SEED = 42

EXPERIMENTS = {
    1: {
        "name": "Experiment 1: DistilBERT + 5K samples",
        "model_name": "distilbert-base-uncased",
        "num_labels": 2,
        "batch_size": 16,
        "max_length": 256,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "train_samples": 5000,
        "test_samples": 1000,
        "output_dir": "outputs/experiment_1",
    },
    2: {
        "name": "Experiment 2: BERT-base + 25K samples",
        "model_name": "bert-base-uncased",
        "num_labels": 2,
        "batch_size": 16,
        "max_length": 256,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "train_samples": None,  # None = full dataset
        "test_samples": None,
        "output_dir": "outputs/experiment_2",
    },
}


def get_experiment_config(exp_id=None):
    """Get config for a specific experiment. Defaults to experiment 1."""
    if exp_id is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", type=int, default=1, choices=EXPERIMENTS.keys())
        args, _ = parser.parse_known_args()
        exp_id = args.experiment

    cfg = EXPERIMENTS[exp_id]
    return cfg


# For backward compatibility — default experiment
_cfg = get_experiment_config()
MODEL_NAME = _cfg["model_name"]
NUM_LABELS = _cfg["num_labels"]
BATCH_SIZE = _cfg["batch_size"]
MAX_LENGTH = _cfg["max_length"]
LEARNING_RATE = _cfg["learning_rate"]
NUM_EPOCHS = _cfg["num_epochs"]
TRAIN_SAMPLES = _cfg["train_samples"]
TEST_SAMPLES = _cfg["test_samples"]
OUTPUT_DIR = _cfg["output_dir"]
LOG_FILE = f"{OUTPUT_DIR}/training_log.txt"
EXPERIMENT_NAME = _cfg["name"]
