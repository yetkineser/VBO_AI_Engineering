# Week 5 Homework — LangChain Structured Output ➜ JSONL File

**Duration:** ~2 hours
**Python:** 3.12, `langchain>=1.2.0`
**LLM:** OpenRouter (Gemini / GPT-4o-mini / any) — same setup as class
**Rule:** Do **not** commit your API key. Use `.env` only.

---

## What you already know (from class)

In `main5_structured_output.py` we built a tiny agent that extracts a flat `ContactInfo` object from a single hardcoded sentence:

```python
agent = create_agent(model=model, tools=[search],
                     response_format=ToolStrategy(ContactInfo))
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(result["structured_response"])
```

This homework is **one step harder**:
- The Pydantic schema has a **nested model** and **`Literal` (enum) fields**.
- The input comes from a **CSV file** — loop over rows instead of one hardcoded string.
- The validated results are **appended to a JSONL file** (one JSON object per line) so the run is reproducible on disk.

No database. No Docker. No SQLModel. Just: CSV in → LLM → Pydantic → JSONL out.

---

## Objective

Read `support_tickets_minimal.csv`. For each row, send the ticket text to an LLM agent and get back a **strict structured object** validated by Pydantic. Write every validated record to:

1. **`output.jsonl`** — append one JSON line per ticket.
2. **stdout** — pretty-print the same JSON so you can watch progress.

---

## Target schema (must match exactly)

**Why this shape?** Imagine you work at a telco and thousands of support tickets arrive every day as free text. To route them automatically you need three pieces of information: *what kind of problem* (`issue_type`), *how urgent* (`urgency`), *where did it come from* (`channel`). You also want to pull specific facts out of the text — a disputed amount, a device name, a move-address request — so downstream systems can act without a human re-reading the ticket; that's `entities`. `summary` is a one-liner for dashboards, and `status_suggestion` is the initial triage state. The "must match exactly" rule exists because this JSON is the **contract** between the LLM and the systems that consume it — if the field names or enum values drift, the next pipeline step breaks.

**Why two classes instead of one flat model?** The top-level fields (`issue_type`, `urgency`, `channel`, `status_suggestion`) are **classification decisions** — every ticket must have exactly one value for each, chosen from a fixed set. The `Entities` fields are **extracted facts** — they are all optional, they may or may not appear in the text, and the list will grow over time (imagine adding `contract_number`, `phone_number`, `city`...). Separating them makes that distinction explicit at the type level: classification is mandatory and enum-constrained, entity extraction is best-effort and nullable. It also mirrors how real NLP pipelines are built — a small required "header" plus a flexible "bag of extracted facts" underneath.

```python
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Entities(BaseModel):
    amount: Optional[float] = Field(default=None, description="Numeric amount, e.g., 49.99")
    invoice_period: Optional[str] = None
    ticket_id: Optional[str] = None
    device: Optional[str] = None
    address_move: Optional[bool] = None

class TicketExtraction(BaseModel):
    source_id: str                     # the CSV row's customer id, e.g. "CUST-001"
    issue_type: Literal["billing", "technical", "account", "general"]
    urgency: Literal["low", "medium", "high"]
    channel: Literal["phone", "email", "chat", "unknown"]
    entities: Entities
    summary: str
    status_suggestion: Literal["open", "in_progress", "resolved"]
```

**Rules**
- Use Pydantic models (no `TypedDict`).
- No extra keys. Enforce enums.
- If a nested field is missing / unknown → `None`.
- If a top-level enum is unclear → pick the closest option (for `channel`, use `"unknown"`).

---

## Tasks

### 1. Setup
We use **uv** for environment + package management (same as the rest of the bootcamp).

### 2. Build
Write `main.py` so that:

```bash
uv run python main.py /path/to/support_tickets_minimal.csv
```

reads every row of the CSV, sends the ticket text to the agent, and produces `output.jsonl` next to the script.

### 3. Deliverables
- `main.py`
- `output.jsonl` — one JSON object per line, one line per CSV row, matching the schema exactly
- `README.md` — one paragraph + the command to run it

---

## Expected `output.jsonl`

One JSON object per line. Every line must parse with `json.loads` and validate against `TicketExtraction`. Example:

```json
{"source_id":"CUST-001","issue_type":"billing","urgency":"high","channel":"phone","entities":{"amount":200.0,"invoice_period":null,"ticket_id":null,"device":null,"address_move":null},"summary":"Faturamda 200 TL fazla ücret var, acil düzeltilsin.","status_suggestion":"open"}
{"source_id":"CUST-004","issue_type":"technical","urgency":"high","channel":"unknown","entities":{"amount":null,"invoice_period":null,"ticket_id":null,"device":"modem","address_move":null},"summary":"Dün akşamdan beri internet çok yavaş...","status_suggestion":"open"}
```

Required per line: all top-level fields present, enums within allowed values, `entities` always an object (never null), `source_id` matches the CSV row's customer id.
