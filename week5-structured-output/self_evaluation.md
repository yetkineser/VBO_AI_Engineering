# Self Evaluation — Week 5 Structured Output Homework

## 1. Deliverables Checklist

| Requirement | Status | Notes |
|---|---|---|
| `main.py` exists | PASS | Single-file script, ~145 lines |
| `output.jsonl` exists | PASS | 8 lines, one per CSV row |
| `README.md` exists | PASS | Includes description, setup, run command, Mermaid diagrams |
| Uses `uv` for environment management | PASS | `pyproject.toml` present, runs via `uv run python main.py` |
| API key not committed | PASS | `.env` in `.gitignore`, only `.env.example` is tracked |
| No database, no Docker | PASS | Pure CSV → LLM → JSONL pipeline |

---

## 2. Schema Compliance

### 2.1 Pydantic Models (not TypedDict)

| Requirement | Status | Notes |
|---|---|---|
| Uses `pydantic.BaseModel` | PASS | Both `Entities` and `TicketExtraction` extend `BaseModel` |
| Nested model (`Entities` inside `TicketExtraction`) | PASS | `entities: Entities` field present |
| `Literal` (enum) fields | PASS | `issue_type`, `urgency`, `channel`, `status_suggestion` all use `Literal` |
| Optional fields default to `None` | PASS | All `Entities` fields are `Optional[...] = None` |
| Field names match spec exactly | PASS | `source_id`, `issue_type`, `urgency`, `channel`, `entities`, `summary`, `status_suggestion` |
| No extra keys | PASS | Pydantic strict mode prevents extra fields by default |

### 2.2 Enum Values Match Spec

| Field | Allowed Values | Implemented |
|---|---|---|
| `issue_type` | `billing`, `technical`, `account`, `general` | PASS |
| `urgency` | `low`, `medium`, `high` | PASS |
| `channel` | `phone`, `email`, `chat`, `unknown` | PASS |
| `status_suggestion` | `open`, `in_progress`, `resolved` | PASS |

---

## 3. Pipeline Behavior

| Requirement | Status | Notes |
|---|---|---|
| Reads CSV file from command-line argument | PASS | `sys.argv[1]` used as path |
| Loops over every row | PASS | `csv.DictReader` + `for` loop over all rows |
| Sends ticket text to LLM agent | PASS | Uses `agent.invoke()` with system + user messages |
| Gets structured response via `response_format` | PASS | `response_format=TicketExtraction` in `create_react_agent` |
| Writes to `output.jsonl` (one JSON per line) | PASS | `model_dump_json()` + newline per row |
| Pretty-prints to stdout | PASS | `json.dumps(..., indent=2)` with progress counter |
| `output.jsonl` written next to the script | PASS | `Path(__file__).parent / "output.jsonl"` |

---

## 4. Output Validation (line by line)

Each line in `output.jsonl` was checked against three criteria:
- **Format**: valid JSON, all required fields present, enums within allowed values, `entities` is an object
- **source_id**: matches CSV `customer_id`
- **Extraction accuracy**: does the LLM output match what the ticket text actually says?

### CUST-001

- **Ticket**: "faturamda 200 TL fazla ücret var. Telefonla aradım... Acil düzeltilmesini istiyorum."
- **Expected**: `issue_type=billing`, `channel=phone`, `urgency=high`, `amount=200.0`
- **Got**: `issue_type=technical`, `channel=email`, `urgency=medium`, `amount=null`, `ticket_id=REF-12345`
- **Format**: PASS
- **Accuracy**: FAIL — LLM hallucinated `REF-12345` (not in text), misclassified as `technical`/`email`, missed the 200 TL amount

### CUST-002

- **Ticket**: "E-posta ile daha önce de yazmıştım, abonelik paketimi değiştirmek istiyorum."
- **Expected**: `issue_type=account`, `channel=email`, `urgency=low/medium`
- **Got**: `issue_type=account`, `channel=email`, `urgency=medium`
- **Format**: PASS
- **Accuracy**: PASS

### CUST-003

- **Ticket**: "Chat üzerinden yazıyorum. 49.99 TL'lik ek hizmet bedeli faturama yansımamış"
- **Expected**: `issue_type=billing`, `channel=chat`, `amount=49.99`
- **Got**: `issue_type=billing`, `channel=chat`, `amount=49.99`
- **Format**: PASS
- **Accuracy**: PASS

### CUST-004

- **Ticket**: "internet çok yavaş, modem ışıkları sürekli yanıp sönüyor. Acil çözüm lazım!"
- **Expected**: `issue_type=technical`, `channel=unknown`, `urgency=high`, `device=modem`
- **Got**: `issue_type=technical`, `channel=unknown`, `urgency=high`, `device=modem`
- **Format**: PASS
- **Accuracy**: PASS — matches the homework example output exactly

### CUST-005

- **Ticket**: "Taşınma nedeniyle adresimi güncellemem gerekiyor... Mail olarak bilgi bekliyorum."
- **Expected**: `issue_type=account`, `channel=email`, `address_move=true`
- **Got**: `issue_type=account`, `channel=email`, `address_move=true`
- **Format**: PASS
- **Accuracy**: PASS

### CUST-006

- **Ticket**: "TKT-88123 numaralı talebim hâlâ çözülmedi. İnternet bağlantım 3 gündür kopuyor."
- **Expected**: `issue_type=technical`, `urgency=high`, `ticket_id=TKT-88123`
- **Got**: `issue_type=technical`, `urgency=high`, `channel=unknown`, `ticket_id=TKT-88123`
- **Format**: PASS
- **Accuracy**: PASS

### CUST-007

- **Ticket**: "genel bir bilgi almak istiyorum. Fiber altyapı... Chat'ten soruyorum."
- **Expected**: `issue_type=general`, `channel=chat`, `urgency=low`
- **Got**: `issue_type=general`, `channel=chat`, `urgency=low`
- **Format**: PASS
- **Accuracy**: PASS

### CUST-008

- **Ticket**: "Telefon ile aradım, Ocak 2024 faturamda 150 TL hatalı yansıyan tutar var."
- **Expected**: `issue_type=billing`, `channel=phone`, `amount=150.0`, `invoice_period=Ocak 2024`
- **Got**: `issue_type=technical`, `channel=email`, `amount=null`, `ticket_id=INC-12345`, `device=modem`
- **Format**: PASS
- **Accuracy**: FAIL — LLM hallucinated `INC-12345` and `modem` (not in text), misclassified as `technical`/`email`, missed 150 TL and Ocak 2024

---

## 5. Summary

### What works well

- **Schema and format are fully correct**: every line in `output.jsonl` is valid JSON, validates against `TicketExtraction`, and has all required fields with proper enum values. The Pydantic structured output contract is enforced.
- **Pipeline architecture is solid**: CSV in → agent → Pydantic validation → JSONL out, exactly as specified.
- **6 out of 8 tickets** are extracted accurately (CUST-002 through CUST-007).
- **All deliverables** (`main.py`, `output.jsonl`, `README.md`) are present and match the spec.
- **Environment management** uses `uv` with `pyproject.toml` as required.

### What could be improved (cloud model)

- **CUST-001 and CUST-008 have extraction errors**: the LLM (Gemini 2.0 Flash) hallucinated entities that don't exist in the text and misclassified `issue_type` and `channel`. This is a model quality issue, not a code issue — the structured output format is enforced correctly, but the content filled into that format was wrong.
- **LangGraph deprecation warning**: `create_react_agent` from `langgraph.prebuilt` is deprecated in favor of `langchain.agents.create_agent`. This is cosmetic and does not affect functionality, but should be updated for future-proofing.

### How local models fixed it

After observing the cloud model's failures, the pipeline was re-run with two local models via Ollama (`--local` flag). Both local models correctly extracted CUST-001 and CUST-008 without any hallucinations. See `model_comparison.md` for the full ticket-by-ticket breakdown.

- **Llama 3.2 (3B)**: 7.5/8 accuracy — only minor miss was `channel=phone` for CUST-004 (should be `unknown`)
- **Qwen 2.5 (14B)**: 8/8 accuracy — best overall, correct on every ticket including channel detection

### Scores (by model)

| Category | Max | Gemini Flash | Llama 3.2 | Qwen 2.5 |
|---|---|---|---|---|
| Deliverables complete | 3 | 3 | 3 | 3 |
| Schema matches spec exactly | 5 | 5 | 5 | 5 |
| Pipeline works end-to-end | 5 | 5 | 5 | 5 |
| Output format (valid JSONL) | 8 | 8 | 8 | 8 |
| Extraction accuracy | 8 | 6 | 7.5 | 8 |
| Code quality and documentation | 5 | 5 | 5 | 5 |
| **Total** | **34** | **32** | **33.5** | **34** |
