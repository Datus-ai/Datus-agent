<p align="center">
  <strong>Datus — Open-Source Data Engineering Agent</strong>
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blueviolet?style=for-the-badge" alt="License"></a>
  <a href="https://datus.ai"><img src="https://img.shields.io/badge/Website-5A0FC8?style=for-the-badge" alt="Website"></a>
  <a href="https://docs.datus.ai/"><img src="https://img.shields.io/badge/Docs-654FF0?style=for-the-badge" alt="Docs"></a>
  <a href="https://docs.datus.ai/getting_started/Quickstart/"><img src="https://img.shields.io/badge/Quick%20Start-3423A6?style=for-the-badge" alt="Quick Start"></a>
  <a href="https://docs.datus.ai/release_notes/"><img src="https://img.shields.io/badge/Release%20Notes-092540?style=for-the-badge" alt="Release Notes"></a>
  <a href="https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg"><img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack"></a>
</p>

---

## What is Datus?

**Datus** is an open-source data engineering agent that builds **evolvable context** for your data system — turning natural language into accurate SQL through domain-aware reasoning, semantic search, and continuous learning.

Data engineering is shifting from "building tables and pipelines" to "delivering scoped, domain-aware agents for analysts and business users." Datus makes that shift concrete.

![Datus Architecture](docs/assets/datus_architecture.svg)

## Key Features

### Build Evolvable Context, Not Static Pipelines

Traditional data engineering ends at data delivery. Datus goes further — it builds a **living knowledge base** that captures schema metadata, reference SQL, semantic models, metrics, and domain knowledge into a unified context layer. This context is what makes LLM-generated SQL accurate and trustworthy, and it improves with every interaction through a continuous learning loop. → [Contextual Data Engineering](https://docs.datus.ai/getting_started/contextual_data_engineering/)

### From Exploration to Domain-Specific Agents

Datus provides a complete journey for data engineers: start with a **Claude-Code-like CLI** to explore your data interactively, use [Plan Mode](https://docs.datus.ai/cli/plan_mode/) to review before executing, and build up context over time. When a domain matures, package it into a **Subagent** — a scoped chatbot with curated context, tools, and business rules — and deliver it to analysts via web, API, or MCP. → [Subagent docs](https://docs.datus.ai/subagent/introduction/)

### Metrics and Semantic Layer

Go beyond raw SQL with pluggable **semantic adapters**. Define business metrics in YAML via [MetricFlow](https://docs.datus.ai/metricflow/introduction/) integration, and let Datus generate SQL from metric queries — bridging the gap between business language and database dialect. Use [Dashboard Copilot](https://docs.datus.ai/getting_started/dashboard_copilot/) to turn existing BI dashboards into conversational analytics. → [Semantic Adapters docs](https://docs.datus.ai/adapters/semantic_adapters/)

### Measure and Improve

Built-in evaluation framework supporting **BIRD** and **Spider 2.0-Snow** datasets. Benchmark your agent's SQL accuracy, compare configurations, and track improvements as context evolves. → [Benchmark docs](https://docs.datus.ai/benchmark/benchmark_manual/)

### Open Platform

- **10+ LLM providers** (OpenAI, Claude, Gemini, DeepSeek, Qwen, Kimi, OpenRouter, and more) with per-node model assignment — mix models within a single workflow
- **11 databases** — Built-in SQLite & DuckDB, plus pluggable adapters for PostgreSQL, MySQL, Snowflake, StarRocks, ClickHouse, and more
- **MCP Protocol** — Both an MCP server (exposing Datus tools to Claude Desktop, Cursor, etc.) and an MCP client (consuming external tools via `.mcp` in the CLI). → [MCP docs](https://docs.datus.ai/integration/mcp/)
- **Skills** — Extend Datus with [agentskills.io](https://agentskills.io)-style packaged tools, configurable permissions, and marketplace support. → [Skills docs](https://docs.datus.ai/integration/skills/)

## Getting Started

### Install

**Requirements:** Python >= 3.12

```bash
pip install datus-agent
datus-agent init
```

`datus-agent init` walks you through configuring your LLM provider, database connection, and knowledge base. For detailed guidance, see the [Quickstart Guide](https://docs.datus.ai/getting_started/Quickstart/).

### Four Ways to Use Datus

| Interface | Command | Use Case |
|-----------|---------|----------|
| **CLI** (Interactive REPL) | `datus-cli --namespace demo` | Data engineers exploring data, building context, creating subagents |
| **Web Chatbot** (Streamlit) | `datus-cli --web --namespace demo` | Analysts chatting with subagents via browser (`http://localhost:8501`) |
| **API Server** (FastAPI) | `datus-api --namespace demo` | Applications consuming data services via REST (`http://localhost:8000`) |
| **MCP Server** | `datus-mcp --namespace demo` | MCP-compatible clients (Claude Desktop, Cursor, etc.) |

> **Tip:** Use `datus-cli --print --namespace demo` for JSON streaming to stdout — useful for piping into other tools.

## Architecture

### Workflow Engine

Datus uses a configurable **node-based workflow engine**. Each workflow is a plan of nodes executed in sequence, parallel, or as sub-workflows:

```yaml
workflow:
  plan: planA
  planA:
    - schema_linking     # Find relevant tables
    - parallel:          # Run in parallel
      - generate_sql     # SQL generation
      - reasoning        # Chain-of-thought reasoning
    - selection          # Pick the best result
    - execute_sql        # Run the query
    - output             # Format and return
```

### Node Types

| Category | Nodes |
|----------|-------|
| **Core** | `schema_linking`, `generate_sql`, `execute_sql`, `reasoning`, `reflect`, `output` |
| **Agentic** | `chat`, `explore`, `gen_semantic_model`, `gen_metrics`, `gen_ext_knowledge`, `gen_sql_summary`, `gen_skill`, `gen_table`, `compare` |
| **Control Flow** | `parallel`, `selection`, `subworkflow` |
| **Utility** | `date_parser`, `doc_search`, `fix` |

### RAG Knowledge Base

The knowledge base is powered by **LanceDB** and organizes context into multiple layers:

- **Schema Metadata** — Table and column descriptions, relationships
- **Reference SQL** — Curated query examples with summaries
- **Reference Templates** — Parameterized Jinja2 SQL templates for stable, reusable queries
- **Semantic Models** — Business logic and metric definitions
- **Metrics** — Executable business metrics via semantic layer integration
- **External Knowledge** — Domain rules and concepts beyond raw schema
- **Platform Docs** — Ingested from GitHub repos, websites, or local files

Build the knowledge base with:

```bash
datus-agent bootstrap-kb --namespace demo --components metadata,reference_sql,ext_knowledge
```

## Configuration

Datus is configured via `agent.yml`. Run `datus-agent init` to generate a starter config, or see [`conf/agent.yml.example`](conf/agent.yml.example) for all options.

| Section | Purpose |
|---------|---------|
| `agent.models` | LLM provider definitions (API keys, model IDs, base URLs) |
| `agent.nodes` | Per-node model assignment and tuning parameters |
| `agent.namespace` | Database connections (SQLite, DuckDB, Snowflake, etc.) |
| `agent.storage` | Embedding models, vector DB, and RAG configuration |
| `agent.workflow` | Execution plans with sequential, parallel, and sub-workflow steps |
| `agent.agentic_nodes` | Configuration for agentic nodes (semantic model gen, metrics gen) |
| `agent.document` | Platform documentation sources (GitHub repos, websites, local files) |

API keys are injected via environment variables using `${ENV_VAR}` syntax.

## Supported LLM Providers

| Provider | Type | Notes |
|----------|------|-------|
| OpenAI | `openai` | GPT-4o, GPT-4, etc. |
| Anthropic Claude | `claude` | Direct API |
| Google Gemini | `gemini` | Gemini 2.0+ |
| DeepSeek | `deepseek` | DeepSeek-Chat, DeepSeek-Coder |
| Alibaba Qwen | `qwen` | Qwen series |
| Moonshot Kimi | `kimi` | Kimi models |
| MiniMax | `minimax` | MiniMax models |
| GLM (Zhipu) | `glm` | GLM-4 series |
| OpenAI Codex | `codex` | OAuth-based Codex models (gpt-5.3-codex, o3-codex) |
| OpenRouter | `openrouter` | 300+ models via a single API key |

**Embedding models:** OpenAI, Sentence-Transformers, FastEmbed, Hugging Face.

Per-node model assignment lets you use different providers for different workflow steps (e.g., a cheaper model for schema linking, a stronger model for SQL generation).

## Supported Databases

| Database | Type | Package |
|----------|------|---------|
| SQLite | `sqlite` | Built-in |
| DuckDB | `duckdb` | Built-in |
| PostgreSQL | `postgresql` | [`datus-postgresql`](https://github.com/Datus-ai/Datus-adapters) |
| MySQL | `mysql` | [`datus-mysql`](https://github.com/Datus-ai/Datus-adapters) |
| Snowflake | `snowflake` | [`datus-snowflake`](https://github.com/Datus-ai/Datus-adapters) |
| StarRocks | `starrocks` | [`datus-starrocks`](https://github.com/Datus-ai/Datus-adapters) |
| ClickHouse | `clickhouse` | [`datus-clickhouse`](https://github.com/Datus-ai/Datus-adapters) |
| ClickZetta | `clickzetta` | [`datus-clickzetta`](https://github.com/Datus-ai/Datus-adapters) |
| Hive | `hive` | [`datus-hive`](https://github.com/Datus-ai/Datus-adapters) |
| Spark | `spark` | [`datus-spark`](https://github.com/Datus-ai/Datus-adapters) |
| Trino | `trino` | [`datus-trino`](https://github.com/Datus-ai/Datus-adapters) |

See [Database Adapters documentation](https://docs.datus.ai/adapters/db_adapters/) for details.

## How It Works

![How It Works](docs/assets/how_it_works.svg)

**Explore** — Chat with your database, test queries, and ground prompts with `@table` or `@file` references.

```bash
datus-cli --namespace demo
/Check the top 10 banks by assets lost @table duckdb-demo.main.bank_failures
```

**Build Context** — Generate semantic models, import SQL history, define metrics. Each piece becomes reusable context for future queries.

```bash
/gen_semantic_model xxx        # Generate semantic model from tables
/gen_sql_summary               # Index SQL history for retrieval
```

**Create a Subagent** — Package mature context into a scoped, domain-aware chatbot with curated tools and business rules.

```bash
.subagent add mychatbot        # Create a new subagent
```

**Deliver** — Serve the subagent to analysts via web (`localhost:8501/?subagent=mychatbot`), REST API, or MCP — with feedback collection (upvotes, issue reports) built in.

**Measure** — Run benchmarks against BIRD or Spider 2.0-Snow to track SQL accuracy as context evolves.

**Iterate** — Analyst feedback loops back: engineers fix SQL, add rules, refine semantic models, and extend with Skills or MCP tools. The agent gets more accurate over time.

→ [End-to-end tutorial](https://docs.datus.ai/getting_started/Datus_tutorial/) · [CLI docs](https://docs.datus.ai/cli/introduction/) · [Knowledge Base docs](https://docs.datus.ai/knowledge_base/introduction/) · [Subagent docs](https://docs.datus.ai/subagent/introduction/)

## Development

```bash
uv sync                                           # Install dependencies
uv run pytest tests/unit_tests/ -q                # Run CI tests (no external deps)
uv run ruff format . && uv run ruff check --fix . # Lint & format
```

Enable `--save_llm_trace` on CLI commands or set `save_llm_trace: true` per model in `agent.yml` to persist LLM inputs/outputs for debugging. → [LLM Trace docs](https://docs.datus.ai/training/llm_trace_usage/)

See [CLAUDE.md](CLAUDE.md) for full development conventions, architecture patterns, and testing rules.

## License

[Apache 2.0](LICENSE)
