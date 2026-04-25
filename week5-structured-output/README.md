# Week 5 — Structured Output Agent

Reads Turkish telecom support tickets from a CSV file, sends each one to an LLM agent via LangChain/LangGraph, and extracts structured triage data (issue type, urgency, channel, entities) validated by Pydantic. Results are written to `output.jsonl` — one JSON object per line.

## Pipeline Overview

```mermaid
flowchart LR
    A["support_tickets_minimal.csv"] -->|csv.DictReader| B["main.py"]
    B -->|row by row| C["LangGraph ReAct Agent"]
    C -->|OpenRouter API| D["LLM\n(Gemini Flash)"]
    D -->|structured response| E["Pydantic\nValidation"]
    E -->|model_dump_json| F["output.jsonl"]
    E -->|pretty print| G["stdout"]
```

## Data Flow per Ticket

```mermaid
flowchart TD
    subgraph Input
        CSV["CSV Row\ncustomer_id + ticket_text"]
    end

    subgraph Agent
        SYS["System Prompt\n(triage rules)"]
        USR["User Message\n(customer_id + text)"]
        LLM["LLM Call\nvia OpenRouter"]
        SYS --> LLM
        USR --> LLM
    end

    subgraph Validation
        RAW["Raw LLM Response"]
        TE["TicketExtraction\n(Pydantic model)"]
        ENT["Entities\n(nested model)"]
        TE --- ENT
    end

    CSV --> USR
    LLM --> RAW
    RAW -->|"response_format=TicketExtraction"| TE
```

## Schema Structure

```mermaid
classDiagram
    class TicketExtraction {
        +str source_id
        +Literal issue_type
        +Literal urgency
        +Literal channel
        +Entities entities
        +str summary
        +Literal status_suggestion
    }

    class Entities {
        +float? amount
        +str? invoice_period
        +str? ticket_id
        +str? device
        +bool? address_move
    }

    TicketExtraction *-- Entities

    class issue_type_enum {
        billing
        technical
        account
        general
    }

    class urgency_enum {
        low
        medium
        high
    }

    class channel_enum {
        phone
        email
        chat
        unknown
    }

    class status_enum {
        open
        in_progress
        resolved
    }

    TicketExtraction .. issue_type_enum
    TicketExtraction .. urgency_enum
    TicketExtraction .. channel_enum
    TicketExtraction .. status_enum
```

## Setup

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

## Run

```bash
uv run python main.py support_tickets_minimal.csv
```

Output is written to `output.jsonl` and printed to stdout.

## Project Structure

```
week5-structured-output/
├── main.py                         # Agent pipeline script
├── support_tickets_minimal.csv     # Input: 8 Turkish telco tickets
├── output.jsonl                    # Output: one JSON object per line
├── homework.md                     # Assignment description
├── learning.md                     # Concepts & learning resources
├── pyproject.toml                  # uv project config
├── .env.example                    # API key template
└── .gitignore
```
