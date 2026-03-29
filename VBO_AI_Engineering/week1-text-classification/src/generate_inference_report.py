"""
Generate inference_examples — homework deliverable.

Loads the TRAINED model, runs 6 example predictions, and saves results.

Usage:
    python src/generate_inference_report.py
"""

import os
import sys
from datetime import datetime

from inference import predict, load_model
from config import DEVICE, MODEL_NAME, OUTPUT_DIR, EXPERIMENT_NAME

EXAMPLES = [
    ("Positive", "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat the entire time."),
    ("Negative", "Terrible film. The plot made no sense, the dialogue was awful, and I wanted to leave after 20 minutes."),
    ("Ambiguous", "An average movie. Some parts were enjoyable but overall it felt a bit too long and predictable."),
    ("Positive", "One of the best films I have ever seen. A masterpiece of storytelling with brilliant performances from the entire cast."),
    ("Negative", "I was really disappointed. The trailer looked promising but the movie itself was boring and poorly directed."),
    ("Tricky (Positive)", "I expected this movie to be absolutely awful, but I was dead wrong. It completely blew me away and I could not stop watching."),
]


def generate_report(results):
    """Generate inference report as both .txt and .md (convertible to PDF)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Markdown version (can be converted to PDF via browser print) ---
    md_path = os.path.join(OUTPUT_DIR, "inference_examples.md")
    with open(md_path, "w") as f:
        f.write(f"# Inference Examples — {EXPERIMENT_NAME}\n\n")
        f.write(f"**Model:** {MODEL_NAME} | **Device:** {DEVICE} | **Date:** {timestamp}\n\n")
        f.write("---\n\n")

        for i, r in enumerate(results, 1):
            expected = EXAMPLES[i-1][0]
            correct = ""
            if "Positive" in expected and r["label"] == "Positive":
                correct = "  ✅"
            elif "Negative" in expected and r["label"] == "Negative":
                correct = "  ✅"
            elif "Ambiguous" in expected or "Tricky" in expected:
                correct = "  🔍"

            f.write(f"## Example {i} (Expected: {expected})\n\n")
            f.write(f"> \"{r['text']}\"\n\n")
            f.write(f"- **Prediction:** {r['label']}{correct}\n")
            f.write(f"- **Confidence:** {r['confidence']:.2%}\n\n")

        f.write("---\n\n")
        f.write("## Summary Table\n\n")
        f.write("| # | Expected | Prediction | Confidence | Correct? |\n")
        f.write("|---|----------|-----------|------------|----------|\n")
        for i, r in enumerate(results, 1):
            expected = EXAMPLES[i-1][0]
            if "Positive" in expected and r["label"] == "Positive":
                mark = "✅"
            elif "Negative" in expected and r["label"] == "Negative":
                mark = "✅"
            elif "Ambiguous" in expected:
                mark = "🔍 (subjective)"
            elif "Tricky" in expected:
                mark = "✅" if r["label"] == "Positive" else "❌ (tricky)"
            else:
                mark = "❌"
            text_short = r["text"][:50] + "..."
            f.write(f"| {i} | {expected} | {r['label']} | {r['confidence']:.2%} | {mark} |\n")

    print(f"Markdown report saved to {md_path}")

    # --- Plain text version ---
    txt_path = os.path.join(OUTPUT_DIR, "inference_examples.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"INFERENCE EXAMPLES — {EXPERIMENT_NAME}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model  : {MODEL_NAME}\n")
        f.write(f"Device : {DEVICE}\n")
        f.write(f"Date   : {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        for i, r in enumerate(results, 1):
            expected = EXAMPLES[i-1][0]
            f.write(f"Example {i} (Expected: {expected})\n")
            f.write("-" * 50 + "\n")
            f.write(f"Input      : {r['text']}\n")
            f.write(f"Prediction : {r['label']}\n")
            f.write(f"Confidence : {r['confidence']:.2%}\n\n")

    print(f"Text report saved to {txt_path}")

    # --- PDF version ---
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Inference Examples", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, EXPERIMENT_NAME, ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, "Model: {}  |  Device: {}  |  Date: {}".format(MODEL_NAME, DEVICE, timestamp), ln=True, align="C")
        pdf.ln(8)

        for i, r in enumerate(results, 1):
            expected = EXAMPLES[i-1][0]
            if "Positive" in expected and r["label"] == "Positive":
                mark = "CORRECT"
            elif "Negative" in expected and r["label"] == "Negative":
                mark = "CORRECT"
            elif "Ambiguous" in expected:
                mark = "SUBJECTIVE"
            elif "Tricky" in expected:
                mark = "CORRECT" if r["label"] == "Positive" else "WRONG (tricky)"
            else:
                mark = "WRONG"

            # Example header
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Example {} (Expected: {})".format(i, expected), ln=True)

            # Input text
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 6, '"{}"'.format(r["text"]))
            pdf.set_text_color(0, 0, 0)

            # Prediction
            if r["label"] == "Positive":
                pdf.set_text_color(0, 128, 0)
            else:
                pdf.set_text_color(200, 0, 0)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Prediction: {}   Confidence: {:.2%}   [{}]".format(
                r["label"], r["confidence"], mark), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(5)

        # Summary table
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 10)
        col_w = [10, 35, 25, 25, 75]
        headers = ["#", "Expected", "Prediction", "Confidence", "Input (truncated)"]
        for j, h in enumerate(headers):
            pdf.cell(col_w[j], 8, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for i, r in enumerate(results, 1):
            expected = EXAMPLES[i-1][0]
            text_short = r["text"][:65] + "..."
            pdf.cell(col_w[0], 7, str(i), border=1)
            pdf.cell(col_w[1], 7, expected, border=1)
            pdf.cell(col_w[2], 7, r["label"], border=1)
            pdf.cell(col_w[3], 7, "{:.2%}".format(r["confidence"]), border=1)
            pdf.cell(col_w[4], 7, text_short, border=1)
            pdf.ln()

        pdf_path = os.path.join(OUTPUT_DIR, "inference_examples.pdf")
        pdf.output(pdf_path)
        print(f"PDF report saved to {pdf_path}")

    except ImportError:
        print("fpdf2 not installed. Run: pip install fpdf2")
        print("Then re-run this script to generate the PDF.")

    return md_path


if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    model, tokenizer = load_model()

    print(f"\nRunning {len(EXAMPLES)} inference examples...\n")

    results = []
    for i, (expected, text) in enumerate(EXAMPLES, 1):
        result = predict(text, model, tokenizer)
        results.append(result)
        status = "✓" if (
            ("Positive" in expected and result["label"] == "Positive") or
            ("Negative" in expected and result["label"] == "Negative")
        ) else "?"
        print(f"  [{status}] Example {i}: {result['label']} ({result['confidence']:.2%})")
        trunc = text[:80] + "..." if len(text) > 80 else text
        print(f"      \"{trunc}\"")
        print()

    generate_report(results)
    print("\nTo convert to PDF: open outputs/inference_examples.md in browser, then Print → Save as PDF")
