"""
Week 5 — Structured Output Agent
Reads support tickets from a CSV, extracts structured data via LLM,
and writes validated results to output.jsonl.

Per-row try/except + tenacity retry for transient errors so a single
failing ticket does not crash the whole pipeline (feedback: Alican
Payaslı, 2026-04-29).

Usage:
    uv run python main.py support_tickets_minimal.csv              # OpenRouter (default)
    uv run python main.py support_tickets_minimal.csv --local      # Ollama local model
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Entities(BaseModel):
    amount: Optional[float] = Field(default=None, description="Numeric amount, e.g., 49.99")
    invoice_period: Optional[str] = None
    ticket_id: Optional[str] = None
    device: Optional[str] = None
    address_move: Optional[bool] = None


class TicketExtraction(BaseModel):
    source_id: str
    issue_type: Literal["billing", "technical", "account", "general"]
    urgency: Literal["low", "medium", "high"]
    channel: Literal["phone", "email", "chat", "unknown"]
    entities: Entities
    summary: str
    status_suggestion: Literal["open", "in_progress", "resolved"]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a support-ticket triage assistant for a Turkish telecom company.
Given a customer ID and ticket text, extract structured information.

Rules:
- issue_type: billing | technical | account | general
- urgency: low | medium | high
- channel: phone | email | chat | unknown (infer from text clues like \
"telefonla aradım" → phone, "e-posta/mail" → email, "chat" → chat)
- entities.amount: extract numeric monetary amounts if mentioned
- entities.invoice_period: extract billing period if mentioned (e.g. "Ocak 2024")
- entities.ticket_id: extract ticket/reference numbers if mentioned (e.g. "TKT-88123")
- entities.device: extract device names if mentioned (e.g. "modem", "router")
- entities.address_move: true if the customer mentions moving/address change
- summary: one-line Turkish summary of the issue
- status_suggestion: open (new/unresolved), in_progress (being handled), resolved
- If a field is not mentioned or unclear, use null for optional fields
- source_id must match the provided customer ID exactly
"""


def build_agent(local: bool = False):
    """Create a LangGraph ReAct agent with structured output."""
    if local:
        model = ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    else:
        model = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            temperature=0,
        )
    agent = create_react_agent(
        model=model,
        tools=[],
        response_format=TicketExtraction,
    )
    return agent


# ---------------------------------------------------------------------------
# Resilient LLM invocation
#
# Retry up to 3 times on transient errors (rate limit / timeout / network).
# Skip retry for ValidationError — re-prompting the same input will fail again.
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type(ValidationError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def invoke_with_retry(agent, messages: list[dict]) -> TicketExtraction:
    result = agent.invoke({"messages": messages})
    return result["structured_response"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_csv(csv_path: str, local: bool = False):
    """Read CSV, send each row to the agent, write JSONL output."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    output_file = "output_local.jsonl" if local else "output.jsonl"
    output_path = Path(__file__).parent / output_file
    errors_path = Path(__file__).parent / "errors.jsonl"
    agent = build_agent(local=local)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} tickets from {csv_path.name}\n")

    success_count = 0
    error_count = 0

    with open(output_path, "w", encoding="utf-8") as out, \
         open(errors_path, "w", encoding="utf-8") as err:

        for i, row in enumerate(rows, 1):
            customer_id = row["customer_id"]
            ticket_text = row["ticket_text"]

            prompt = (
                f"Customer ID: {customer_id}\n"
                f"Ticket text: {ticket_text}"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                extraction: TicketExtraction = invoke_with_retry(agent, messages)
                extraction.source_id = customer_id

                out.write(extraction.model_dump_json() + "\n")
                success_count += 1

                print(f"[{i}/{len(rows)}] OK   {customer_id}")
                print(json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2))
                print()

            except Exception as e:
                error_count += 1
                error_record = {
                    "customer_id": customer_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(rows)}] FAIL {customer_id}: {type(e).__name__} — {e}")
                print(f"  → logged to {errors_path.name}, continuing.\n")

    print(f"Done. {success_count}/{len(rows)} successful, {error_count} errors.")
    print(f"Output: {output_path}")
    if error_count > 0:
        print(f"Errors: {errors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Output Agent")
    parser.add_argument("csv_path", help="Path to support_tickets_minimal.csv")
    parser.add_argument("--local", action="store_true", help="Use local Ollama model (qwen2.5:14b)")
    args = parser.parse_args()
    process_csv(args.csv_path, local=args.local)
