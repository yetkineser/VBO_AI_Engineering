# Inference Examples — Experiment 1: DistilBERT + 5K samples

**Model:** distilbert-base-uncased | **Device:** mps | **Date:** 2026-03-29 21:37

---

## Example 1 (Expected: Positive)

> "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat the entire time."

- **Prediction:** Negative
- **Confidence:** 51.59%

## Example 2 (Expected: Negative)

> "Terrible film. The plot made no sense, the dialogue was awful, and I wanted to leave after 20 minutes."

- **Prediction:** Negative  ✅
- **Confidence:** 50.32%

## Example 3 (Expected: Ambiguous)

> "An average movie. Some parts were enjoyable but overall it felt a bit too long and predictable."

- **Prediction:** Negative  🔍
- **Confidence:** 51.82%

## Example 4 (Expected: Positive)

> "One of the best films I have ever seen. A masterpiece of storytelling with brilliant performances from the entire cast."

- **Prediction:** Negative
- **Confidence:** 51.95%

## Example 5 (Expected: Negative)

> "I was really disappointed. The trailer looked promising but the movie itself was boring and poorly directed."

- **Prediction:** Negative  ✅
- **Confidence:** 50.70%

## Example 6 (Expected: Tricky (Positive))

> "I expected this movie to be absolutely awful, but I was dead wrong. It completely blew me away and I could not stop watching."

- **Prediction:** Positive  ✅
- **Confidence:** 50.65%

---

## Summary Table

| # | Expected | Prediction | Confidence | Correct? |
|---|----------|-----------|------------|----------|
| 1 | Positive | Negative | 51.59% | ❌ |
| 2 | Negative | Negative | 50.32% | ✅ |
| 3 | Ambiguous | Negative | 51.82% | 🔍 (subjective) |
| 4 | Positive | Negative | 51.95% | ❌ |
| 5 | Negative | Negative | 50.70% | ✅ |
| 6 | Tricky (Positive) | Positive | 50.65% | ✅ |
