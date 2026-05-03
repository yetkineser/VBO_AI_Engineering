"""
HR RAG Chatbot — CLI entrypoint.

Subcommands:
    ingest  --profile {openrouter,ollama}   Build the vector store.
    chat    --model   {gemini,deepseek,ollama}    Interactive REPL.
    test    --model   {gemini,deepseek,ollama}    Smoke + memory test, JSONL out.

Default profile / model = the homework spec defaults
(OpenRouter embeddings + Gemini 2.5 Flash Lite chat).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from document_loader import load_and_chunk
from rag_agent import PROFILES, build_agent, chat, open_checkpointer
from vector_store import create_vector_store

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "hr_documents_pack" / "initial_docs"
RESULTS_DIR = Path(__file__).parent / "results"

TEST_QUESTIONS = [
    "What is the company's leave policy?",
    "How many vacation days do employees get?",
    "What are the steps in the offboarding process?",
    "What are the IT security requirements for new employees?",
    "What is the performance review process?",
    "How do I submit travel expenses for reimbursement?",
]

MEMORY_CHECK = [
    ("What is the leave policy?",      "first turn — establishes topic"),
    ("What about sick leave?",          "follow-up: must resolve 'sick leave' against prior topic"),
    ("How many days exactly?",          "follow-up: must answer with sick-leave day count"),
]


def cmd_ingest(args) -> None:
    chunks = load_and_chunk(DOCS_DIR)
    if not chunks:
        logger.error("No chunks produced. Aborting ingestion.")
        sys.exit(1)
    create_vector_store(chunks, profile=args.profile)
    logger.info("Done. Index lives at ./chroma_db/.")


def cmd_chat(args) -> None:
    thread_id = str(uuid.uuid4())
    print(f"Model profile: {args.model}    Session thread_id: {thread_id}")
    print("Type 'exit' or Ctrl-C to quit.\n")
    with open_checkpointer() as checkpointer:
        agent = build_agent(checkpointer, profile=args.model)
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            try:
                reply = chat(agent, query, thread_id)
            except Exception as e:
                print(f"[error] {type(e).__name__}: {e}\n")
                continue
            print(f"Bot: {reply}\n")


def cmd_test(args) -> None:
    """Run the 6 smoke questions + 3-turn memory check, write a JSONL log."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"test_{args.model}.jsonl"
    print(f"=== Running test with model={args.model} ===")
    print(f"Logging to {out_path}")

    with open_checkpointer() as checkpointer, open(out_path, "w", encoding="utf-8") as out:
        agent = build_agent(checkpointer, profile=args.model)

        print("\n--- Smoke test (6 standalone questions) ---")
        for i, q in enumerate(TEST_QUESTIONS, 1):
            thread_id = f"smoke-{args.model}-{i}-{uuid.uuid4().hex[:6]}"
            t0 = time.time()
            try:
                reply = chat(agent, q, thread_id)
                ok, err = True, None
            except Exception as e:
                reply, ok, err = "", False, f"{type(e).__name__}: {e}"
            elapsed = round(time.time() - t0, 2)
            record = {
                "model": args.model,
                "section": "smoke",
                "index": i,
                "thread_id": thread_id,
                "question": q,
                "answer": reply,
                "ok": ok,
                "error": err,
                "latency_s": elapsed,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\n[{i}/{len(TEST_QUESTIONS)}] {q}   ({elapsed}s)")
            print(f"   {reply if ok else '[ERROR] ' + err}")

        print("\n--- Memory test (3 turns on one thread_id) ---")
        memory_thread = f"memory-{args.model}-{uuid.uuid4().hex[:6]}"
        for i, (q, note) in enumerate(MEMORY_CHECK, 1):
            t0 = time.time()
            try:
                reply = chat(agent, q, memory_thread)
                ok, err = True, None
            except Exception as e:
                reply, ok, err = "", False, f"{type(e).__name__}: {e}"
            elapsed = round(time.time() - t0, 2)
            record = {
                "model": args.model,
                "section": "memory",
                "index": i,
                "thread_id": memory_thread,
                "question": q,
                "note": note,
                "answer": reply,
                "ok": ok,
                "error": err,
                "latency_s": elapsed,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"\nYou:  {q}    ({note}, {elapsed}s)")
            print(f"Bot:  {reply if ok else '[ERROR] ' + err}")

    print(f"\nDone — results in {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HR RAG Chatbot CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Load + embed HR docs into Chroma")
    ing.add_argument("--profile", choices=["openrouter", "ollama"], default="openrouter",
                     help="Embedding backend (openrouter = spec default)")

    ch = sub.add_parser("chat", help="Interactive chat with short-term memory")
    ch.add_argument("--model", choices=list(PROFILES), default="gemini")

    te = sub.add_parser("test", help="Run smoke + memory tests")
    te.add_argument("--model", choices=list(PROFILES), default="gemini")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "chat": cmd_chat, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
