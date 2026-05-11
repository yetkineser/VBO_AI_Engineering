"""Interactive client that talks to the database MCP server through an LLM agent.

Workflow per turn:
    user question → create_agent → (list_tables / get_table_schema / validate_query
    / execute_query) → final answer.

The agent decides which tools to call and in what order. We only set the rules
(SELECT-only, always cite the SQL) in the system prompt.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")
MODEL_ID = os.getenv("MODEL_ID", "openai:google/gemini-2.5-flash-lite")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

SYSTEM_PROMPT = """You are a careful read-only database analyst for the Chinook
sample database (music store). Tables are lowercase, snake_case:
album, artist, customer, employee, genre, invoice, invoice_line,
media_type, playlist, playlist_track, track. Columns are also snake_case
(album_id, artist_id, billing_country, total, ...).

Always follow this workflow:
1. If you do not yet know the schema, call `list_tables` first.
2. Call `get_table_schema` for the tables that look relevant.
3. Write a single SELECT (or WITH ... SELECT) and call `validate_query` first.
4. Only if validation passes, call `execute_query`. Use a sensible LIMIT.
5. Summarize the result in 1–3 sentences and quote the exact SQL you ran.

Hard rules:
- Never call `execute_query` with anything other than SELECT or WITH.
- Never invent rows. If `execute_query` returns row_count = 0, say so.
- Always include the SQL in a fenced code block in your final answer.
- If the user asks for a write/update/delete, refuse and explain why.
"""


async def run() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is not set. Copy .env.example to .env first.")

    client = MultiServerMCPClient(
        {
            "database": {
                "transport": "streamable_http",
                "url": MCP_URL,
            }
        }
    )

    print(f"[client] connecting to MCP at {MCP_URL} ...")
    tools = await client.get_tools()
    print(f"[client] discovered {len(tools)} tools: {[t.name for t in tools]}")

    model = init_chat_model(
        MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )

    checkpointer = MemorySaver()
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    print("\nReady. Ask a question about the Chinook database (or type 'exit').\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = result["messages"][-1].content
            print(f"\nbot> {answer}\n")
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(run())
