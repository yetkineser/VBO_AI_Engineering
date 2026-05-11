"""End-to-end smoke test for the database MCP server.

This is not a unit test — it talks to a running server and a running database.
Start both before running:

    docker compose up -d
    python database_mcp_server.py   # in another shell
    python test_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")

EXPECTED_TOOLS = {
    "list_tables",
    "get_table_schema",
    "execute_query",
    "validate_query",
    "get_database_info",
}


def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def _text(result) -> str:
    """Coerce an MCP tool result (list of content blocks or plain string) to text."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(result)


async def main() -> int:
    client = MultiServerMCPClient(
        {"database": {"transport": "streamable_http", "url": MCP_URL}}
    )
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}

    failures = 0

    print("\n== 1. Tool discovery ==")
    missing = EXPECTED_TOOLS - by_name.keys()
    if not _check("all expected tools exposed", not missing, f"missing={sorted(missing)}"):
        failures += 1

    print("\n== 2. get_database_info ==")
    info = _text(await by_name["get_database_info"].ainvoke({}))
    print(info)
    if not _check("dialect line present", "dialect:" in info):
        failures += 1

    print("\n== 3. list_tables ==")
    tables = _text(await by_name["list_tables"].ainvoke({}))
    print(tables)
    if not _check("album table found", "album" in tables):
        failures += 1

    print("\n== 4. get_table_schema(album) ==")
    schema = _text(await by_name["get_table_schema"].ainvoke({"table_names": "album"}))
    print(schema)
    if not _check("schema starts with CREATE TABLE", schema.lstrip().startswith("CREATE TABLE")):
        failures += 1

    print("\n== 5. validate_query — good ==")
    msg = _text(await by_name["validate_query"].ainvoke({"query": "SELECT 1"}))
    print(msg)
    if not _check("plain SELECT accepted", msg.startswith("OK")):
        failures += 1

    print("\n== 6. validate_query — bad (DDL) ==")
    msg = _text(await by_name["validate_query"].ainvoke({"query": "DROP TABLE album"}))
    print(msg)
    if not _check("DROP rejected", msg.startswith("REJECTED")):
        failures += 1

    print("\n== 7. validate_query — bad (multi-statement) ==")
    msg = _text(await by_name["validate_query"].ainvoke({"query": "SELECT 1; SELECT 2"}))
    print(msg)
    if not _check("multi-statement rejected", msg.startswith("REJECTED")):
        failures += 1

    print("\n== 8. execute_query — good ==")
    res = _text(
        await by_name["execute_query"].ainvoke(
            {"query": "SELECT title FROM album LIMIT 3", "limit": 3}
        )
    )
    print(res)
    try:
        payload = json.loads(res)
        ok = isinstance(payload.get("rows"), list) and payload["row_count"] == len(payload["rows"])
    except Exception:
        ok = False
    if not _check("returned valid JSON with rows", ok):
        failures += 1

    print("\n== 9. execute_query — rejected (DELETE) ==")
    res = _text(await by_name["execute_query"].ainvoke({"query": "DELETE FROM album"}))
    print(res)
    if not _check("DELETE rejected before reaching DB", res.startswith("Error:")):
        failures += 1

    print("\n== 10. execute_query — rejected (multi-statement) ==")
    res = _text(
        await by_name["execute_query"].ainvoke({"query": "SELECT 1; DROP TABLE album"})
    )
    print(res)
    if not _check("piggyback DROP rejected", res.startswith("Error:")):
        failures += 1

    print(f"\n=== Summary: {10 - failures}/10 checks passed ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
