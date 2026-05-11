# Week 7 — Extras and Where to Go Next

## What is already done (so this page does not repeat it)

The homework spec is met:

- **Five tools** exposed by the MCP server over `streamable-http`
- **LangChain client** that discovers the tools through `MultiServerMCPClient`
- **Smoke test**: 10/10 checks pass against MCP + Postgres
- **Beyond-spec deliverables**:
  - [`compare_models.py`](database_query_tool/compare_models.py) — benchmark harness for three local Ollama models
  - [`model_comparison.md`](model_comparison.md) — 3-model comparison **plus** Claude Opus 4.7 LLM-as-judge scoring

The list below is *next ideas* — what would move the project from "homework
MVP" toward something you would actually let a colleague run against real
data. Read it as a **roadmap, not a checklist**. Each item is a small,
finite project that unlocks one named capability.

---

## ⭐ The critical trio for production

If the three sections below were the only ones I had time for, I would still
pick **these three**. They turn "the regex catches DELETE" into a stack a
security review would actually pass. Each item is a one-evening task; together
they are the difference between a demo and a service.

### 1. A read-only database role at the Postgres layer

Right now the only thing stopping a DELETE is the application-level
`validate_query` regex plus the system prompt. That is a single trust
boundary. Production-grade defense in depth puts a **second wall inside the
database**: a Postgres role with `GRANT SELECT` only on the Chinook schema.
Then even a bug in `_is_select_only` cannot do damage.

#### What to do

```sql
CREATE ROLE chinook_reader LOGIN PASSWORD 'chinook_reader';
GRANT CONNECT ON DATABASE chinook TO chinook_reader;
GRANT USAGE ON SCHEMA public TO chinook_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO chinook_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO chinook_reader;
```

Then change the MCP server's `DATABASE_URL` to use `chinook_reader`. The
application code does not change — but now even `psql` from the same
container cannot drop a table.

#### Reading

- [PostgreSQL docs — `GRANT`](https://www.postgresql.org/docs/current/sql-grant.html)
- [Predefined roles (`pg_read_all_data`)](https://www.postgresql.org/docs/current/predefined-roles.html)
- [OWASP — Database Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Database_Security_Cheat_Sheet.html)

---

### 2. Statement timeout — block runaway SELECTs at the database level

The SELECT-only guard stops writes, but it does not stop **expensive
reads**. A query like `SELECT pg_sleep(60)` or a six-way Cartesian join
will tie up a worker for as long as Postgres lets it. The fix is one line
of Postgres config: a server-side timeout that aborts any statement that
runs longer than N seconds.

#### What to do

Per-session, in the engine URL:

```python
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args={"options": "-c statement_timeout=5000"},  # milliseconds
)
```

Or globally, on the Postgres role:

```sql
ALTER ROLE chinook_reader SET statement_timeout = '5s';
```

The role-level setting is the safer pattern — it survives even if the
application forgets to set the timeout on a new connection. Pair this with
the read-only role from #1 and your worst case becomes "a single 5-second
query", not "the whole agent stalls".

#### Reading

- [PostgreSQL — `statement_timeout`](https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-STATEMENT-TIMEOUT)
- [PostgreSQL — `idle_in_transaction_session_timeout`](https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-IDLE-IN-TRANSACTION-SESSION-TIMEOUT) — same idea, for transactions left open
- [SQLAlchemy — passing `options` to libpq](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#psycopg2-connect-arguments)
- [Citus — what happens when a Postgres query runs forever](https://www.citusdata.com/blog/2017/09/29/what-happens-when-a-postgres-query-runs-forever/) — concrete failure stories

---

### 3. Audit log — one JSON line per tool call

Observability (item #9) covers traces. **Audit** is different: a permanent,
append-only record of *what the agent did to the database*, regardless of
whether anything went wrong. For a system that talks to user data this is
non-negotiable in most regulated environments.

The minimum useful schema, per tool call:

```json
{
  "ts": "2026-05-11T13:42:01.847Z",
  "tool": "execute_query",
  "thread_id": "abc-123",
  "input": {"query": "SELECT title FROM album LIMIT 3", "limit": 3},
  "query_hash": "sha256:8a3f…",
  "outcome": "accepted",
  "rejection_reason": null,
  "latency_ms": 47,
  "row_count": 3,
  "error": null
}
```

#### What to do

Wrap each `@mcp.tool()` in a thin decorator that records start/finish to
a JSONL file (or a Kafka topic, or an append-only Postgres table — same
shape, different sink):

```python
import hashlib, json, time
from pathlib import Path

AUDIT_LOG = Path("/var/log/chinook-mcp/audit.jsonl")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

def _hash(query: str) -> str:
    return "sha256:" + hashlib.sha256(query.encode()).hexdigest()[:16]

def audit(tool_name: str):
    def deco(fn):
        async def wrapper(*args, **kwargs):
            t0 = time.time()
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "tool": tool_name,
                "input": kwargs if kwargs else {"args": [str(a)[:200] for a in args]},
            }
            if "query" in kwargs:
                entry["query_hash"] = _hash(kwargs["query"])
            try:
                result = await fn(*args, **kwargs)
                entry["outcome"] = "rejected" if isinstance(result, str) and result.startswith("Error") else "accepted"
                entry["latency_ms"] = int((time.time() - t0) * 1000)
                return result
            except Exception as e:
                entry["outcome"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
                entry["latency_ms"] = int((time.time() - t0) * 1000)
                raise
            finally:
                with AUDIT_LOG.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
        return wrapper
    return deco

@mcp.tool()
@audit("execute_query")
async def execute_query(query: str, limit: int = 100) -> str:
    ...
```

Keep the **query hash**, not the full query, if any of the data is
sensitive — the hash lets you correlate calls without leaking PII into the
audit log. Rotate the file daily and ship it to S3/object storage for
retention.

#### Reading

- [NIST 800-53 — Audit and Accountability (AU)](https://nvd.nist.gov/800-53/Rev5/family/Audit%20and%20Accountability) — the canonical reference for what an audit log should capture
- [OWASP — Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html) — what to log, what *not* to log
- [The Twelve-Factor App — XI. Logs](https://12factor.net/logs) — treat logs as event streams, not files
- [Vector by Datadog](https://vector.dev/) — open-source log/audit shipper, ships JSONL to almost anything
- [PostgreSQL `pgaudit` extension](https://www.pgaudit.org/) — if you'd rather have the database itself emit the audit trail

---

## 🛡️ Security & Correctness

The critical trio above fixes the biggest gaps. The two items below close the
remaining "obvious flaws" — a smarter SQL guard, and an auth layer for the
day the server stops being a localhost-only toy.

### 4. Replace the regex SELECT-only guard with `sqlglot`

The current `_is_select_only` strips comments and scans for forbidden
keywords. It will reject *valid* queries like:

```sql
SELECT * FROM album WHERE title = 'DROP TABLE artist'
```

…because the literal string `DROP` triggers the keyword scan. A proper fix
is a real **SQL parser**: parse the statement into an AST, then check the
AST for write nodes.

#### What to do

```python
import sqlglot
from sqlglot import expressions as exp

def is_read_only(query: str) -> bool:
    try:
        stmts = sqlglot.parse(query, read="postgres")
    except sqlglot.errors.ParseError:
        return False
    if len(stmts) != 1:
        return False
    root = stmts[0]
    if not isinstance(root, (exp.Select, exp.With)):
        return False
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.AlterTable, exp.Create)
    return not any(root.find_all(*forbidden))
```

#### Reading

- [`sqlglot` docs](https://sqlglot.com/sqlglot.html)
- [Why parser beats regex for SQL — sqlglot blog](https://sqlglot.com/sqlglot/parser.html)
- [Tree-sitter grammars for SQL](https://github.com/tree-sitter/tree-sitter-sql) — an alternative if `sqlglot` ever stops fitting

---

### 5. Add auth to the MCP server (Bearer token, then OAuth)

The current server trusts every caller on `localhost`. Once it leaves the
laptop, you need to know **who** is connecting. The MCP spec deliberately
does not mandate one auth scheme, so the two practical choices are:

- **Bearer token in the `Authorization` header** — simplest. Validate per
  request, rotate manually.
- **OAuth 2.1** — proper for multi-tenant. Lets you scope tokens per user.

#### What to do (Bearer, ~30 lines)

Wrap the FastMCP server in a Starlette middleware that checks
`Authorization: Bearer <token>` against an env var. Reject 401 if missing
or wrong. Tailscale ACLs can replace this if you trust the network layer.

#### Reading

- [MCP Architecture — Authentication discussion](https://modelcontextprotocol.io/docs/concepts/architecture)
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1)
- [FastAPI security utilities](https://fastapi.tiangolo.com/tutorial/security/) — the same patterns work in any Starlette app
- [Tailscale ACLs as auth substitute](https://tailscale.com/kb/1018/acls/)

---

## ⚙️ DX & Operations

Security buys you trust; operations buys you sleep. The three items below
shave the friction off the daily loop — one command to bring everything up,
a container that restarts itself, and lint that catches mistakes before
they ship.

### 6. One-command lifecycle — `run_all.sh` to bring up the whole stack

Today you need three terminals: Docker, the MCP server, the client (or
tests). A single wrapper script that brings each piece up, checks it is
healthy, and tails the logs is a small DX win that pays back every time
you sit down to work on the project.

#### What to do

```bash
#!/usr/bin/env bash
# run_all.sh — bring up Chinook + MCP server, then drop into the client.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] starting Chinook database…"
./setup_db.sh

echo "[2/3] starting MCP server in the background…"
source .venv/bin/activate
python database_mcp_server.py > /tmp/mcp-server.log 2>&1 &
MCP_PID=$!
trap "echo '[cleanup] stopping MCP server'; kill $MCP_PID 2>/dev/null" EXIT

echo "[wait] for MCP /mcp to answer 200…"
until curl -fs -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/mcp \
       -H "Content-Type: application/json" \
       -H "Accept: application/json, text/event-stream" \
       -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1"}}}' \
       2>/dev/null | grep -q "^200$"; do sleep 1; done
echo "[ok] MCP server ready"

echo "[3/3] dropping into client…"
python database_query_client.py
```

The trap line is the critical bit — when the client exits (clean or via
Ctrl-C), the MCP server is killed automatically. No orphan processes.

#### Reading

- [Bash strict mode (`set -euo pipefail`)](http://redsymbol.net/articles/unofficial-bash-strict-mode/) — why these flags belong in every script
- [`trap` for cleanup on exit](https://www.gnu.org/software/bash/manual/html_node/Signals.html)
- [Docker Compose `depends_on` with `condition: service_healthy`](https://docs.docker.com/compose/compose-file/05-services/#depends_on) — the same idea, in YAML, when you graduate from bash
- [Foreman / honcho](https://github.com/nickstenning/honcho) — Procfile-based process orchestrators if `run_all.sh` ever outgrows bash

---

### 7. Containerize the MCP server (Dockerfile + healthcheck)

Right now you run `python database_mcp_server.py` by hand. For an
always-on home server on a Mac Mini (see [[local-llm-hardware]]) you want
the server in a container with a healthcheck so launchd / docker compose
can restart it after crashes.

#### What to do

```dockerfile
FROM python:3.13-slim
RUN useradd -m -u 1000 mcp
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY database_mcp_server.py .
USER mcp
EXPOSE 8000
HEALTHCHECK CMD curl -fs http://localhost:8000/mcp || exit 1
CMD ["python", "database_mcp_server.py"]
```

#### Reading

- [Docker multi-stage Python builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker Compose healthchecks](https://docs.docker.com/compose/compose-file/05-services/#healthcheck)
- [Kubernetes probes (when you outgrow Compose)](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

---

### 8. Pre-commit hooks: ruff, mypy, pytest on every change

Personal projects rot fast without lint and type checks. Adding three hooks
takes 10 minutes and pays back forever:

- **ruff** — linter and formatter in one, ~100x faster than flake8
- **mypy** — static types catch ~30% of bugs before they run
- **pytest** — turn `test_mcp_server.py` into proper pytest tests

#### Reading

- [pre-commit framework](https://pre-commit.com/)
- [ruff docs](https://docs.astral.sh/ruff/)
- [mypy quickstart](https://mypy.readthedocs.io/en/stable/getting_started.html)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) — for async tests

---

## 📡 Telemetry & Audit

Audit (item #3) tells you *what happened*. Telemetry tells you *what is
happening right now* and *what it cost*. Together they cover the "I need to
answer a question about this system" axis.

### 9. Observability: structured logs + OpenTelemetry traces

When the agent makes a wrong SQL call at 3 AM, you need to answer two
questions: *what query did it run* and *what did the database return*. A
plain `print` is not enough. The minimum production setup:

- One JSON line per tool call written to disk
- One OpenTelemetry span per agent turn, nested per tool call
- A LangSmith or Phoenix UI to replay runs

#### What to do

```python
# In database_mcp_server.py
import structlog
log = structlog.get_logger()

@mcp.tool()
async def execute_query(query: str, limit: int = 100) -> str:
    log.info("execute_query.start", query=query[:200], limit=limit)
    ...
    log.info("execute_query.done", row_count=len(payload), duration_ms=...)
```

Then ship `OTEL_EXPORTER_OTLP_ENDPOINT=...` and traces flow into Jaeger,
Honeycomb, Grafana Tempo, or LangSmith.

#### Reading

- [OpenTelemetry for Python](https://opentelemetry.io/docs/languages/python/) — vendor-neutral
- [LangSmith documentation](https://docs.smith.langchain.com/) — LangChain-native
- [Phoenix by Arize](https://docs.arize.com/phoenix) — open-source alternative
- [`structlog` docs](https://www.structlog.org/en/stable/) — structured logging in Python
- [Charity Majors — observability is a verb](https://charity.wtf/2020/03/03/observability-a-3-year-retrospective/)

---

### 10. Cost and latency tracking per agent turn

If this agent ever ships, you will want to know "what does the average
question cost?" and "what is the p95 latency?". Both are easy to add:

- Wrap each model call with a `time.time()` delta
- For OpenRouter / Anthropic, the API returns `usage` with token counts
- For Ollama, count tokens manually with `tiktoken` (approximate but cheap)
- Aggregate into a `runs.jsonl` and roll up with `pandas`

#### Reading

- [Anthropic — token counting in responses](https://docs.anthropic.com/en/api/messages)
- [OpenRouter generation API](https://openrouter.ai/docs/api-reference/generation) — has usage + cost fields
- [LangSmith — cost tracking built-in](https://docs.smith.langchain.com/observability/how_to_guides/log_token_usage)
- [Honeycomb's latency-percentile primer](https://docs.honeycomb.io/concepts/observability/percentiles/)

---

## 🔌 Reach & Integration

The five items below are not about hardening what exists — they are about
**reaching further**. A second MCP server, a different MCP client, a web
frontend, persistent memory across sessions, and a streaming UX. Each one
roughly doubles where this code can run.

### 11. Hook up a second MCP server (filesystem) and route between them

`MultiServerMCPClient` is named "Multi" for a reason — it can fan out to
many servers and let the agent decide which one to call. Add the official
filesystem MCP server alongside the database one, and the agent gains
"summarize this CSV in `~/Downloads/` then save the SQL to a file" without
any new code in the database server.

#### What to do

```python
client = MultiServerMCPClient({
    "database": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/yetkineser/Desktop"],
        "transport": "stdio",
    },
})
```

The agent then sees tools from both servers and picks the right one per
turn. Watch for naming collisions (e.g. both servers exposing `list`).

#### Reading

- [Reference MCP servers (filesystem, GitHub, Slack, Git, …)](https://github.com/modelcontextprotocol/servers)
- [`langchain-mcp-adapters` multi-server section](https://github.com/langchain-ai/langchain-mcp-adapters#multi-server-example)
- [Pieter Levels-style "MCP-first" architecture thread](https://modelcontextprotocol.io/quickstart/user)

---

### 12. Connect the MCP server to Claude Desktop and Cursor

The same server that this LangChain client uses also works in **Claude
Desktop** and **Cursor** with zero changes — they speak MCP natively. The
trick is the transport: those clients prefer `stdio`, so you launch a
second `mcp.run(transport="stdio")` mode and point the client's config at
the binary.

#### What to do

In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chinook": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/database_mcp_server_stdio.py"]
    }
  }
}
```

Then ask Claude Desktop: *"how many albums are in the chinook database?"*
— it uses your tools, locally, with no API.

#### Reading

- [Claude Desktop MCP setup](https://modelcontextprotocol.io/quickstart/user)
- [Cursor MCP integration](https://docs.cursor.com/context/model-context-protocol)
- [MCP transports — `stdio` vs `streamable-http`](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)

---

### 13. Streaming responses (token-by-token)

Right now the agent returns its full answer in one block at the end. For
long SQL summaries this feels slow. Streaming makes latency feel half as
long even when the total time is identical. Both LangGraph and `create_agent`
support `astream` natively.

#### What to do

```python
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": question}]},
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="messages",
):
    print(chunk[0].content, end="", flush=True)
```

#### Reading

- [LangChain streaming how-to](https://python.langchain.com/docs/how_to/streaming/)
- [LangGraph `astream` modes](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [Server-Sent Events explained](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

---

### 14. A web frontend with Streamlit or Chainlit

A CLI is fine for testing; real users need a chat window. Two paths from
short to clean:

- **Streamlit** (~30 lines of Python) — fastest path from script to webapp,
  good enough for a demo or an internal tool
- **Chainlit** — purpose-built for LLM chat, has streaming + history +
  step visualization built in

#### Reading

- [Streamlit docs](https://docs.streamlit.io/)
- [Chainlit docs](https://docs.chainlit.io/)
- [LangServe](https://python.langchain.com/docs/langserve/) — serve LangChain runnables as REST APIs (different shape: API not UI)

---

### 15. Persistent short-term memory with `PostgresSaver`

The current client uses `MemorySaver` — conversation history dies on
restart. Week 6 already showed how to swap in `PostgresSaver`; reusing that
pattern here gives the database agent multi-turn references that survive
across sessions ("the query I just ran", "those artists").

#### What to do

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(model=..., tools=..., checkpointer=checkpointer, ...)
```

This is **the same setup we used in Week 6** — see [`week6-rag/hr_rag_chatbot/rag_agent.py`](../week6-rag/hr_rag_chatbot/rag_agent.py).

#### Reading

- [LangGraph persistence concepts](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [`PostgresSaver` reference](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver)

---

## 📊 Evaluation & Quality

[`model_comparison.md`](model_comparison.md) already covers the basics — three
models, a custom rubric, Claude Opus 4.7 as judge. The two items below
upgrade that effort to something you could publish or reuse across projects.

### 16. Real benchmark suite — Spider 2 or BIRD instead of three handpicked questions

Three questions and a manual rubric got us a signal. A standardized
benchmark gives us a *number you can compare across the field*. For SQL
agents specifically, the two benchmarks that matter in 2026:

- **[Spider 2](https://spider2-sql.github.io/)** — 600+ real enterprise SQL
  scenarios across 213 databases. The current de-facto standard.
- **[BIRD](https://bird-bench.github.io/)** — 12,751 question–SQL pairs over
  95 large-scale databases. Harder than Spider 2, more realistic schemas.

Running 50 BIRD examples through our MCP agent and computing
**execution accuracy** + **valid efficiency score** would be a small but
publishable artifact.

#### Reading

- [Spider 2 leaderboard](https://spider2-sql.github.io/) — see where open
  models actually land
- [BIRD paper (Li et al. 2023)](https://arxiv.org/abs/2305.03111)
- [Text-to-SQL survey 2024](https://arxiv.org/abs/2406.08426)

---

### 17. LLM-as-judge ensemble (multiple judges, Bradley-Terry aggregation)

The `model_comparison.md` already uses Claude Opus 4.7 as judge. The
production-grade version uses **multiple judges** and aggregates with a
preference model:

```
9 answers × 3 judges (Opus + GPT-5 + Gemini-2.5-Pro) = 27 scores
→ Bradley-Terry / Elo aggregation
→ a single ranking with confidence intervals
```

This needs an OpenRouter key (one API key, three models). Cost for our
9-answer comparison: pennies.

#### Reading

- [G-Eval paper (Liu et al. 2023)](https://arxiv.org/abs/2303.16634) —
  CoT-based LLM judging, the canonical reference
- [Prometheus 2 (Kim et al. 2024)](https://github.com/prometheus-eval/prometheus-eval) — open-weight judge model
- [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) — automated
  pairwise judge with Bradley-Terry aggregation
- [LMArena leaderboard](https://lmarena.ai/) — what the field is doing in
  the same shape
- [Inspect AI (UK AISI)](https://inspect.ai-safety-institute.org.uk/) — the
  evaluation framework that is becoming a 2026 standard

---

## 🚀 Learning Projects

Everything above improves *this* MCP server. The item below is about
**leaving this codebase behind** — taking the pattern and building your own
server on a domain that actually matters to your daily workflow.

### 18. The big one — write your own MCP server for a non-trivial domain

The five tools in this project are a teaching exercise. The natural next
project is a server that you would actually use day to day:

- **[[ai-daily-brief]] tools** — `summarize_brief`, `mark_done`, `flag_topic`
- **Knowledge graph tools** — `query_graph`, `add_concept_link` on top of
  the [[perfecto-inference]] / graphify Neo4j graph
- **Personal finance tools** — `list_transactions`, `search_merchant`,
  `monthly_summary` over a SQLite extract of bank CSVs

Each is a 1-2 day project that turns one of your existing workflows into
something an LLM (Claude Desktop, Cursor, this LangChain agent) can use
from one prompt away.

#### Reading

- [MCP server quickstart (build your own)](https://modelcontextprotocol.io/quickstart/server)
- [Awesome MCP — community servers for inspiration](https://github.com/punkpeye/awesome-mcp-servers)
- [Anthropic's MCP introduction post](https://www.anthropic.com/news/model-context-protocol)

---

## Where this list could grow

When the homework deadline passes and there is time to play, three threads
are particularly worth following because they connect to other things in
the perfecto_brain wiki:

| Direction | Wiki cross-link | Why it matters |
|---|---|---|
| Build a knowledge-graph MCP server | [[perfecto-inference]], graphify-out | turns the static wiki into a queryable resource |
| Run on a dedicated Mac Mini | [[local-llm-hardware]] | always-on inference for [[ai-daily-brief]] and friends |
| Multi-judge LLM-as-judge harness | item #17 above | re-usable across every benchmark you ever write |

The point of the list is not to do everything. The point is to know
*which extra* maps onto *which capability you would gain* — so the next
time you have a free Saturday afternoon, the choice is obvious.
