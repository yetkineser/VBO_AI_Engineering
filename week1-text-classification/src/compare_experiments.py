"""
Compare results from multiple experiments side by side.

Usage:
    python src/compare_experiments.py
"""

import json
import os

EXPERIMENTS_DIR = "outputs"
EXPERIMENT_DIRS = ["experiment_1", "experiment_2"]


def load_results(exp_dir):
    """Load test_results.json from an experiment directory."""
    json_path = os.path.join(EXPERIMENTS_DIR, exp_dir, "test_results.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def main():
    results = {}
    for exp_dir in EXPERIMENT_DIRS:
        data = load_results(exp_dir)
        if data:
            results[exp_dir] = data

    if not results:
        print("No experiment results found. Run experiments first.")
        return

    print("=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print()

    # Header
    print(f"{'Metric':<25}", end="")
    for exp in results:
        print(f"{exp:<25}", end="")
    print()
    print("-" * (25 + 25 * len(results)))

    # Model
    print(f"{'Model':<25}", end="")
    for exp in results:
        print(f"{results[exp]['model']:<25}", end="")
    print()

    # Device
    print(f"{'Device':<25}", end="")
    for exp in results:
        print(f"{results[exp]['device']:<25}", end="")
    print()

    # Test samples
    print(f"{'Test Samples':<25}", end="")
    for exp in results:
        print(f"{results[exp]['test_samples']:<25}", end="")
    print()

    # Accuracy
    print(f"{'Accuracy':<25}", end="")
    for exp in results:
        acc = results[exp]["accuracy"]
        print(f"{acc:.4f} ({acc*100:.2f}%){'':<10}", end="")
    print()

    # Per-class metrics
    for label in ["Negative", "Positive"]:
        print()
        print(f"--- {label} ---")
        for metric in ["precision", "recall", "f1-score"]:
            print(f"  {metric:<23}", end="")
            for exp in results:
                val = results[exp]["classification_report"][label][metric]
                print(f"{val:.4f}{'':<20}", end="")
            print()

    print()
    print("=" * 70)

    # Determine winner
    if len(results) >= 2:
        accs = {exp: results[exp]["accuracy"] for exp in results}
        best = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        diff = (accs[best] - accs[worst]) * 100
        print(f"Winner: {best} (+{diff:.2f}% accuracy)")

    # Save comparison
    comparison_path = os.path.join(EXPERIMENTS_DIR, "comparison.md")
    with open(comparison_path, "w") as f:
        f.write("# Experiment Comparison — Week 1 Homework\n\n")
        f.write("| Metric | " + " | ".join(results.keys()) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(results)) + "|\n")
        f.write("| Model | " + " | ".join(r["model"] for r in results.values()) + " |\n")
        f.write("| Test Samples | " + " | ".join(str(r["test_samples"]) for r in results.values()) + " |\n")
        f.write("| **Accuracy** | " + " | ".join(f"**{r['accuracy']*100:.2f}%**" for r in results.values()) + " |\n")

        for label in ["Negative", "Positive"]:
            f.write(f"| {label} Precision | " + " | ".join(
                f"{r['classification_report'][label]['precision']:.4f}" for r in results.values()) + " |\n")
            f.write(f"| {label} Recall | " + " | ".join(
                f"{r['classification_report'][label]['recall']:.4f}" for r in results.values()) + " |\n")
            f.write(f"| {label} F1 | " + " | ".join(
                f"{r['classification_report'][label]['f1-score']:.4f}" for r in results.values()) + " |\n")

    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
