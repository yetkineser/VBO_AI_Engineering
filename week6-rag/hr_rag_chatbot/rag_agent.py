"""
RAG agent with short-term memory.

Three model profiles, selectable via the `profile` argument:

  - "gemini"   : OpenRouter embeddings + google/gemini-2.5-flash-lite chat
                 (homework spec default)
  - "deepseek" : OpenRouter embeddings + deepseek/deepseek-chat
  - "ollama"   : nomic-embed-text + qwen2.5:14b, all local

All profiles share the same retriever tool, system prompt, agent shape, and
PostgresSaver checkpointer — only the model strings differ. This isolates
the variable being studied (chat model quality) for the model_comparison.md
write-up.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver

from vector_store import get_retriever

logger = logging.getLogger(__name__)
load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

PROFILES = {
    "gemini":   {"chat": "openai:google/gemini-2.5-flash-lite", "embed_profile": "openrouter"},
    "deepseek": {"chat": "openai:deepseek/deepseek-chat",        "embed_profile": "openrouter"},
    "ollama":   {"chat": f"openai:{OLLAMA_CHAT_MODEL}",          "embed_profile": "ollama"},
}

SYSTEM_PROMPT = """\
You are an HR assistant for VBO. Answer questions about company HR policies
using ONLY the content returned by the `retrieve_hr_docs` tool.

Rules:
- Always call `retrieve_hr_docs` for any question about HR policies, benefits,
  procedures, or employee rules.
- Answer in 2-3 sentences. Be concise.
- ALWAYS cite the source by appending: [Source: <file_name>] using the file_name
  from the retrieved chunk's metadata.
- If multiple files are relevant, list each on its own [Source: ...] line.
- If retrieval returns nothing relevant, say: "I don't know — this isn't covered
  in the HR documents." Do NOT invent answers.
- For follow-up questions (e.g., "what about X?", "how many of those?"), use
  the previous turn's topic to expand the query before retrieving.
"""


# Module-level mutable so the @tool function knows which collection to query.
# Set by build_agent() before the first invoke.
_ACTIVE_EMBED_PROFILE = "openrouter"


@tool(response_format="content")
def retrieve_hr_docs(query: str) -> str:
    """Retrieve relevant HR document chunks for a query.

    Args:
        query: A natural-language question about HR policies, benefits, or
            procedures. Expand follow-up references ("it", "that policy")
            into a self-contained query before calling.

    Returns:
        Top-k chunks formatted as "[file_name] chunk_text", joined with
        blank lines. Empty string if nothing was retrieved.
    """
    retriever = get_retriever(profile=_ACTIVE_EMBED_PROFILE, k=4)
    docs = retriever.invoke(query)
    if not docs:
        return ""
    parts = []
    for d in docs:
        file_name = d.metadata.get("file_name", "unknown")
        page = d.metadata.get("page_number")
        header = f"[{file_name}" + (f" p.{page}" if page not in (None, "") else "") + "]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def _build_model(profile: str):
    cfg = PROFILES[profile]
    if profile == "ollama":
        return init_chat_model(
            cfg["chat"],
            api_key="ollama",
            base_url=OLLAMA_BASE,
            temperature=0,
        )
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing — set it in .env.")
    return init_chat_model(
        cfg["chat"],
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        temperature=0,
    )


def build_agent(checkpointer, profile: str = "gemini"):
    """Construct the agent for a given model profile."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}. Choose from {list(PROFILES)}.")
    global _ACTIVE_EMBED_PROFILE
    _ACTIVE_EMBED_PROFILE = PROFILES[profile]["embed_profile"]
    return create_agent(
        model=_build_model(profile),
        tools=[retrieve_hr_docs],
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )


@contextmanager
def open_checkpointer():
    """Open a PostgresSaver scoped to a `with` block.

    Calls setup() on entry — it is idempotent so it is safe to run every time.
    """
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError(
            "DB_URI missing — set it to a Postgres connection string in .env, "
            "e.g. postgresql://user:pass@localhost:5432/hr_rag?sslmode=disable"
        )
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()
        yield checkpointer


def chat(agent, query: str, thread_id: str) -> str:
    """Send one user turn through the agent and return the assistant's reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    return result["messages"][-1].content
