# Week 5 — What This Homework Teaches and Why It Matters

## Why Are We Doing This?

In real-world AI engineering, raw LLM output is rarely useful on its own. Applications need **structured, validated data** they can store, route, and act on — not free-form text. This homework bridges that gap: you take messy, unstructured ticket text and turn it into clean JSON that downstream systems can consume without human intervention.

This is one of the most common patterns in production AI: **LLM as a structured extraction engine**. Whether you're building a triage pipeline, a data enrichment service, or an automated form-filler, the core loop is the same — text in, validated schema out.

## Key Concepts to Learn

### 1. Pydantic for LLM Output Validation

Pydantic models act as a **contract** between your LLM and the rest of your system. Instead of hoping the model returns the right JSON shape, you define the schema up front and let Pydantic reject anything that doesn't conform.

- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Pydantic Field Types and Validators](https://docs.pydantic.dev/latest/concepts/fields/)
- [Using `Literal` for enum-like constraints](https://docs.pydantic.dev/latest/concepts/types/#literal)

### 2. Nested Models vs Flat Schemas

This homework uses a nested `Entities` model inside `TicketExtraction`. This is intentional — it teaches you to separate **classification fields** (fixed, required, enum-constrained) from **extraction fields** (optional, nullable, growing over time). This mirrors how production NLP pipelines are structured.

- [Pydantic Nested Models](https://docs.pydantic.dev/latest/concepts/models/#nested-models)

### 3. LangChain Structured Output

LangChain provides multiple strategies to force an LLM to return data that matches a Pydantic schema. The two main approaches are:

- **ToolStrategy**: wraps your Pydantic model as a fake "tool" so the model is forced to call it with the right arguments. Works with any model that supports tool/function calling.
- **ProviderStrategy**: uses the provider's native structured output feature (e.g., OpenAI's `response_format`).

Resources:
- [LangChain Structured Output Guide](https://python.langchain.com/docs/how_to/structured_output/)
- [LangGraph ReAct Agent with response_format](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)

### 4. LangGraph Agents

`create_react_agent` from LangGraph builds a ReAct-style agent that can use tools and, when given a `response_format`, adds a final node that forces the output into your schema. Even without tools (like in this homework), the agent pattern is useful because it standardizes the invoke/response interface.

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Prebuilt Agents](https://langchain-ai.github.io/langgraph/reference/prebuilt/)

### 5. OpenRouter as an LLM Gateway

OpenRouter provides a unified API that lets you swap between models (Gemini, GPT-4o-mini, Claude, etc.) without changing your code — just change the model string. This is useful for cost optimization and comparing model performance.

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter with LangChain](https://openrouter.ai/docs/frameworks/langchain)

### 6. JSONL as an Output Format

JSONL (JSON Lines) is a simple format where each line is a valid JSON object. It's widely used in ML/data pipelines because it's append-friendly, streamable, and easy to process line by line — unlike a single JSON array, you don't need to load the entire file into memory.

- [JSON Lines Specification](https://jsonlines.org/)

### 7. CSV Processing in Python

Reading structured input from CSV files is a fundamental data engineering skill. Python's built-in `csv.DictReader` gives you each row as a dictionary, making it easy to access columns by name.

- [Python csv module docs](https://docs.python.org/3/library/csv.html)

## What You Should Be Able to Do After This

- Define Pydantic models with nested structures, optional fields, and Literal enums
- Use LangChain/LangGraph to get validated structured output from any LLM
- Build a simple batch processing pipeline: read input → call LLM → validate → write output
- Understand why structured output matters for production AI systems
- Swap LLM providers through OpenRouter without changing application logic
