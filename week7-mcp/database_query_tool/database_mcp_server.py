"""MCP server that exposes the Chinook database as five read-only tools.

Run with:
    python database_mcp_server.py

The server listens on http://0.0.0.0:8000/mcp using the streamable-http
transport (JSON-RPC over HTTP + Server-Sent Events). Any MCP-aware client
can discover and call its tools — no per-client glue code required.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://chinook:chinook@localhost:5432/chinook",
)
SERVER_HOST = os.getenv("MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("MCP_PORT", "8000"))
DEFAULT_ROW_LIMIT = int(os.getenv("MCP_DEFAULT_LIMIT", "100"))
MAX_ROW_LIMIT = int(os.getenv("MCP_MAX_LIMIT", "1000"))

mcp = FastMCP("chinook-database", host=SERVER_HOST, port=SERVER_PORT)
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
    "CREATE", "GRANT", "REVOKE", "REPLACE", "MERGE", "CALL", "EXEC",
    "ATTACH", "DETACH", "COPY",
)


def _strip_sql_comments(query: str) -> str:
    """Remove -- line comments and /* ... */ block comments."""
    no_block = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return no_line


def _is_select_only(query: str) -> tuple[bool, str]:
    """Return (allowed, reason). Reject anything that is not a single read query."""
    cleaned = _strip_sql_comments(query).strip().rstrip(";").strip()
    if not cleaned:
        return False, "empty query"
    if ";" in cleaned:
        return False, "multiple statements are not allowed"
    upper = cleaned.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "only SELECT or WITH (CTE) queries are allowed"
    padded = " " + upper + " "
    for kw in _FORBIDDEN_KEYWORDS:
        if f" {kw} " in padded:
            return False, f"forbidden keyword detected: {kw}"
    return True, "ok"


@mcp.tool()
async def list_tables() -> str:
    """List every table the database exposes, comma-separated and sorted."""
    insp = inspect(engine)
    tables = sorted(insp.get_table_names())
    return ", ".join(tables) if tables else "<no tables>"


@mcp.tool()
async def get_table_schema(table_names: str) -> str:
    """Return the column definitions for one or more comma-separated tables.

    Example call: table_names="Album, Artist".
    """
    requested = [t.strip() for t in table_names.split(",") if t.strip()]
    if not requested:
        return "Error: pass at least one table name."

    insp = inspect(engine)
    available = set(insp.get_table_names())
    out: list[str] = []
    for name in requested:
        if name not in available:
            out.append(f"-- {name}: not found")
            continue
        cols = insp.get_columns(name)
        col_lines = [f"  {c['name']} {c['type']}" for c in cols]
        pk = insp.get_pk_constraint(name).get("constrained_columns") or []
        if pk:
            col_lines.append(f"  PRIMARY KEY ({', '.join(pk)})")
        for fk in insp.get_foreign_keys(name):
            col_lines.append(
                f"  FOREIGN KEY ({', '.join(fk['constrained_columns'])}) "
                f"REFERENCES {fk['referred_table']}({', '.join(fk['referred_columns'])})"
            )
        out.append(f"CREATE TABLE {name} (\n" + ",\n".join(col_lines) + "\n);")
    return "\n\n".join(out)


@mcp.tool()
async def validate_query(query: str) -> str:
    """Check whether a query is safe to run via execute_query (no execution)."""
    allowed, reason = _is_select_only(query)
    return "OK — query is read-only and looks safe." if allowed else f"REJECTED — {reason}."


@mcp.tool()
async def execute_query(query: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Run a single read-only SELECT/WITH query and return up to `limit` rows as JSON.

    Anything other than a single SELECT or WITH statement is rejected before the
    database ever sees it. The hard cap is set by MCP_MAX_LIMIT (default 1000).
    """
    allowed, reason = _is_select_only(query)
    if not allowed:
        return f"Error: {reason}."

    safe_limit = max(1, min(int(limit), MAX_ROW_LIMIT))
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.mappings().fetchmany(safe_limit)
            payload: list[dict[str, Any]] = [dict(r) for r in rows]
        return json.dumps(
            {"row_count": len(payload), "rows": payload},
            default=str,
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_database_info() -> str:
    """Return engine, driver, sanitized URL, and table count for the database."""
    insp = inspect(engine)
    sanitized = engine.url.set(password="***") if engine.url.password else engine.url
    return (
        f"dialect: {engine.dialect.name}\n"
        f"driver: {engine.dialect.driver}\n"
        f"url: {sanitized}\n"
        f"table_count: {len(insp.get_table_names())}"
    )


if __name__ == "__main__":
    print(f"[mcp] starting on http://{SERVER_HOST}:{SERVER_PORT}/mcp")
    print(f"[mcp] database: {engine.url.set(password='***') if engine.url.password else engine.url}")
    mcp.run(transport="streamable-http")
