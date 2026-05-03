# Week 5 — Feedback and Fix

## Instructor Feedback (Alican Payaslı, 2026-04-29)

The original feedback was given in Turkish. Quoted verbatim below, with an English summary underneath.

> Ellerine sağlık Yetkin, her hafta olduğu gibi yine dört dörtlük bir çalışma. İstenilenin de ötesinde teslimler. Sadece şurada küçük bir eksik var:
>
> **Error Handling:**
> - Try/except yok, bir satırın LLM çağrısında hata olursa tüm pipeline çöker ve geri kalan ticketlar işlenmez. Satır bazında try/except + opsiyonel retry iyi olur. (-1p)
>
> Teşekkür ederiz

**English summary:** great submission overall, going beyond what was asked. One small gap: there is no try/except around the LLM call. If a single row fails, the whole pipeline crashes and the remaining tickets are not processed. A per-row try/except plus an optional retry would fix this. (-1 point)

## What Was Fixed

### The Problem
In the original `main.py`, the `agent.invoke(...)` call sat **bare** inside the `for` loop. If a single row raised an error (rate limit, timeout, network drop, Pydantic validation failure), the whole pipeline would crash and the rest of the tickets would never run. Crashing after 47 of 50 rows means starting from scratch.

### The Fix: Per-Row try/except + Tenacity Retry

A two-layer error-handling strategy was added:

1. **Inner layer (retry on transient errors):** `tenacity` retries up to 3 times with exponential backoff (2s → 4s → 8s).
2. **Outer layer (per-row try/except):** if all 3 retries fail, the error is appended to `errors.jsonl` and the loop continues with the next row.

### Smart Retry: Skip ValidationError

A `pydantic.ValidationError` happens when the model output does not match the schema — re-sending the same prompt will produce the same error. With `retry_if_not_exception_type(ValidationError)` we fail fast in that case instead of waiting through three useless retries.

### New Output File: `errors.jsonl`
For every failed row:
```json
{"customer_id": "CUST-XXX", "error_type": "TimeoutError", "error_message": "..."}
```

Sample stdout progress:
```
[3/8] FAIL CUST-003: TransientLLMError — Connection timeout
  → logged to errors.jsonl, continuing.
...
Done. 7/8 successful, 1 errors.
```

## The Lesson

**In production-style pipelines, every external API call inside a loop must be wrapped in per-row try/except plus retry.** A single transient error in the middle of a batch will crash the pipeline, the partial results disappear, and the run has to start over. Per-row isolation plus retry shrinks the failure radius — successful rows are kept, and only the rows that genuinely fail get skipped.

The fix lives in [main.py](main.py); `tenacity>=8.0` was added to `pyproject.toml`.
