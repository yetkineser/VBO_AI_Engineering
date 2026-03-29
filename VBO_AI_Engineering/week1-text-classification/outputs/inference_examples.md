# Inference Examples — Week 1 Homework

**Model:** distilbert-base-uncased | **Device:** mps | **Date:** 2026-03-29 20:24

---

## Example 1 (Expected: Positive)

> "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat the entire time."

- **Prediction:** Positive  ✅
- **Confidence:** 99.73%

## Example 2 (Expected: Negative)

> "Terrible film. The plot made no sense, the dialogue was awful, and I wanted to leave after 20 minutes."

- **Prediction:** Negative  ✅
- **Confidence:** 99.12%

## Example 3 (Expected: Ambiguous)

> "An average movie. Some parts were enjoyable but overall it felt a bit too long and predictable."

- **Prediction:** Negative  🔍
- **Confidence:** 93.93%

## Example 4 (Expected: Positive)

> "One of the best films I have ever seen. A masterpiece of storytelling with brilliant performances from the entire cast."

- **Prediction:** Positive  ✅
- **Confidence:** 99.72%

## Example 5 (Expected: Negative)

> "I was really disappointed. The trailer looked promising but the movie itself was boring and poorly directed."

- **Prediction:** Negative  ✅
- **Confidence:** 99.17%

## Example 6 (Expected: Tricky (Positive))

> "I expected this movie to be absolutely awful, but I was dead wrong. It completely blew me away and I could not stop watching."

- **Prediction:** Negative  🔍
- **Confidence:** 95.07%

---

## Summary Table

| # | Expected | Prediction | Confidence | Correct? |
|---|----------|-----------|------------|----------|
| 1 | Positive | Positive | 99.73% | ✅ |
| 2 | Negative | Negative | 99.12% | ✅ |
| 3 | Ambiguous | Negative | 93.93% | 🔍 (subjective) |
| 4 | Positive | Positive | 99.72% | ✅ |
| 5 | Negative | Negative | 99.17% | ✅ |
| 6 | Tricky (Positive) | Negative | 95.07% | ❌ (tricky) |
