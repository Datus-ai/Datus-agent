<p align="center">
  <strong>Datus · Open-Source Data Engineering Agent</strong>
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blueviolet?style=for-the-badge" alt="License"></a>
  <a href="https://datus.ai"><img src="https://img.shields.io/badge/Website-5A0FC8?style=for-the-badge" alt="Website"></a>
  <a href="https://docs.datus.ai/"><img src="https://img.shields.io/badge/Docs-654FF0?style=for-the-badge" alt="Docs"></a>
  <a href="https://docs.datus.ai/latest/getting_started/Quickstart/"><img src="https://img.shields.io/badge/Quick%20Start-3423A6?style=for-the-badge" alt="Quick Start"></a>
  <a href="https://docs.datus.ai/latest/release_notes/"><img src="https://img.shields.io/badge/Release%20Notes-092540?style=for-the-badge" alt="Release Notes"></a>
  <a href="https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg"><img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack"></a>
</p>

<p align="center">
  English | <a href="README.zh.md">简体中文</a>
</p>

---

## What is Datus?

**Datus** is the open-source data engineering agent for the modern data stack: one agent that connects your warehouse, catalog, semantic layer, and BI, grounded in an **evolvable context engine** your team owns.

Copilots answer questions; Datus executes data work end to end. It plans and writes SQL, validates the results, authors semantic models and metrics, and generates pipelines, reports, and dashboards. All of it runs on context the agent keeps: your schemas, reference SQL, and business rules, refined with every interaction. An agent without evolvable context is a stranger every Monday. Datus remembers what your team taught it.

The journey is concrete: explore your data in a Claude-Code-like CLI, build context into a living knowledge base and semantic layer, package mature domains into scoped subagents, and deliver them to analysts through the web chatbot, REST API, MCP, Slack/Feishu, or VS Code.

![Datus Architecture](docs/assets/datus_architecture.svg)

## Key Features

### An Evolvable Context Engine, Not Static Pipelines

NL2SQL tools hallucinate joins and metrics because they see your database cold. Datus builds a **context engine** that combines tree-structured business domains with vector retrieval. It captures schema metadata, reference SQL, parameterized SQL templates, semantic models, metrics, and domain knowledge in one layer your team owns. `/init` scans your project into an `AGENTS.md` inventory with file-based knowledge and memory stores; `/build-kb` builds the vector-indexed knowledge base on top, scoped by file, table, datasource, or business domain. This context is what makes agent-generated SQL accurate and trustworthy, and every query, correction, and domain rule feeds it back. → [Contextual Data Engineering](https://docs.datus.ai/latest/getting_started/contextual_data_engineering/)

### Metrics and Semantic Layer: MetricFlow and OSI

Go beyond raw SQL with pluggable semantic adapters. Author business metrics in [MetricFlow](https://docs.datus.ai/latest/metricflow/introduction/) YAML or as [OSI (Open Semantic Interchange)](https://docs.datus.ai/latest/adapters/osi_semantic_adapter/) spec-aligned documents, executed through MetricFlow or the native Rust-based [Dosi engine](docs/adapters/dosi_semantic_adapter.md). Datus generates them from your schema and SQL history, including cumulative, rolling-window, and period-over-period time metrics. The AskMetrics subagent answers KPI, trend, and attribution questions directly from the metric layer instead of raw SQL, and [Dashboard Copilot](https://docs.datus.ai/latest/getting_started/dashboard_copilot/) turns existing BI dashboards into conversational analytics. → [Semantic Adapters docs](https://docs.datus.ai/latest/adapters/semantic_adapters/)

### From Exploration to Domain-Specific Subagents

Vertical agents win by mastering a domain's context, not by wrapping a bigger model. Start in the interactive CLI: chat with your database, ground prompts with `@table` / `@file` references, press `Tab` to cycle between chat, SQL, and bash input modes (or run agent tools and plugin CLIs directly with `!`), review multi-step work with [Plan Mode](https://docs.datus.ai/latest/cli/plan_mode/) before executing, and let automatic session compaction, memory, and `/resume` keep long engagements coherent. When a domain matures, open the `/agent` manager and package it into a **subagent**: a scoped chatbot with curated context, tools, and business rules. Deliver it via web, API, MCP, or IM, so the numbers analysts get in Slack match what leadership sees in dashboards. → [Subagent docs](https://docs.datus.ai/latest/subagent/introduction/)

### Data Engineering Automation

Built-in subagents cover the engineering work around the SQL: cross-database migration, ETL and job generation, wide-table generation from JOIN SQL, and scheduler orchestration through the Airflow adapter. BI adapters for Superset and Grafana let the agent read and write real dashboards. → [Built-in subagents](https://docs.datus.ai/latest/subagent/builtin_subagents/) · [Scheduler adapters](https://docs.datus.ai/latest/adapters/scheduler_adapters/) · [BI adapters](https://docs.datus.ai/latest/adapters/bi_adapters/)

### Visual Reports and Dashboards

Generate self-contained HTML reports (KPI cards, charts, tables, narrative) and interactive dashboards straight from chat. Dashboard filters re-run live against your database through the local web server, without a SaaS backend, and every artifact supports section-by-section refinement. → [Visual report](https://docs.datus.ai/latest/subagent/gen_visual_report/) · [Visual dashboard](https://docs.datus.ai/latest/subagent/gen_visual_dashboard/)

### Enterprise-Ready Governance

Datus ships with permission profiles (`normal` / `auto` / `dangerous`, switchable per request), fine-grained SQL permissions by statement class, command-level bash allow/deny rules backed by an OS-level sandbox, a request-scoped SQL policy framework for row-level rewriting, a read-only multi-tenant configuration mode, and configurable tracing to Langfuse, LangSmith, Datadog, Braintrust, or any OTLP collector. → [SQL policy](https://docs.datus.ai/latest/configuration/sql_policy/) · [Observability](https://docs.datus.ai/latest/develop/observability/)

### Open Platform

- **10+ LLM providers**: OpenAI, Claude, Gemini, DeepSeek, Qwen, Kimi, OpenRouter, and more, plus subscription auth (Claude subscription, OpenAI Codex OAuth) and coding-plan providers. Per-node model assignment mixes models within a single workflow.
- **13 databases**: built-in SQLite & DuckDB plus pluggable adapters for PostgreSQL, MySQL, Snowflake, StarRocks, ClickHouse, Doris, and more.
- **MCP protocol**: both an MCP server (exposing Datus tools to Claude Desktop, Cursor, etc.) and an MCP client (consuming external tools via `/mcp` in the CLI). → [MCP docs](https://docs.datus.ai/latest/integration/mcp/)
- **Skills and plugins**: extend Datus with [agentskills.io](https://agentskills.io)-style packaged skills and marketplace support, or ship a full extension as a plugin with a declarative `datus-plugin.yml` manifest and install/pack/export CLI. → [Skills docs](https://docs.datus.ai/latest/skills/introduction/) · [Plugin docs](https://docs.datus.ai/latest/plugin/introduction/)

### Measure and Improve

Datus ships an evaluation framework for the BIRD and Spider 2.0-Snow datasets. Benchmark your agent's SQL accuracy, compare configurations, and track improvements as context evolves. → [Benchmark docs](https://docs.datus.ai/latest/benchmark/benchmark_manual/)

## Getting Started

### Install

**Requirements:** Linux or macOS. Python 3.12 is installed automatically when you use the one-liner.

#### One-liner (Linux / macOS)

Stable install from PyPI:

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

This creates a dedicated venv at `~/.datus/venv`, installs `datus-agent` from PyPI into it, and drops `datus`, `datus-cli`, `datus-api`, `datus-mcp`, `datus-agent`, `datus-gateway`, and `datus-pip` shims into `~/.local/bin`. Open a new shell (or `source ~/.zshrc`) to pick up PATH, then run `datus` to launch the REPL: use `/model` to configure an LLM, `/datasource` to add a datasource, and (optionally) `/init` to generate `AGENTS.md` for the current project.

To install additional Python packages into the global venv later, use `datus-pip install <package>`. To upgrade Datus itself, run `datus upgrade` (or `datus upgrade --check` to look without installing).

Dev install from GitHub source (picks up unreleased changes):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
# or pin to a branch / tag / commit
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
```

Pin a PyPI version (stable installer only):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.3.9 sh
```

Other variables supported by both installers: `DATUS_HOME` (default `~/.datus`), `DATUS_BIN_DIR` (default `~/.local/bin`), `DATUS_FORCE=1` to recreate the venv, `DATUS_NO_MODIFY_PATH=1` to skip shell rc edits.

#### Manual install

```bash
pip install datus-agent
datus
```

After the REPL starts, run `/model` to configure an LLM, `/datasource` to add a datasource, and (optionally) `/init` to generate `AGENTS.md` for the current project. For detailed guidance, see the [Quickstart Guide](https://docs.datus.ai/latest/getting_started/Quickstart/).

### Interfaces

The examples below use a datasource named `demo`. Create one first with `/datasource` in the REPL; the built-in DuckDB sample database is offered as a default.

| Interface | Command | Use Case |
|-----------|---------|----------|
| **CLI** (interactive REPL) | `datus --datasource demo` | Data engineers exploring data, building context, creating subagents |
| **Web Chatbot** (FastAPI + React) | `datus --web --datasource demo` | Analysts chatting with subagents via browser (`http://localhost:8501`) |
| **REST API** (FastAPI) | `datus-api --datasource demo` | Applications consuming data services via REST (`http://localhost:8000`) |
| **MCP Server** | `datus-mcp --datasource demo` | MCP-compatible clients (Claude Desktop, Cursor, etc.) |
| **IM Gateway** | `datus-gateway` | Analysts talking to subagents in Slack or Feishu/Lark → [Gateway docs](https://docs.datus.ai/latest/gateway/introduction/) |
| **VS Code** (Datus Studio) | connects to `datus --web` | Catalog explorer, chat panel, SQL results & AI charts in the IDE → [Datus Studio docs](https://docs.datus.ai/latest/vscode_extension/introduction/) |

> **Tip:** Print mode streams JSON message lines to stdout for scripting and CI: `datus -p "your question" --datasource demo`. Add `--resume <session_id>` for multi-turn runs and `--plan-mode` to auto-confirm generated plans.

## How It Works

![How It Works](docs/assets/how_it_works.svg)

**Explore**: chat with your database, test queries, and ground prompts with `@table` or `@file` references.

```bash
datus --datasource demo
Check the top 10 banks by assets lost @table duckdb-demo.main.bank_failures
```

**Build Context**: scan the project, generate semantic models and metrics, and index SQL history. Each piece becomes reusable context for future queries.

```bash
/init            # Project inventory (AGENTS.md) + knowledge & memory stores
/bootstrap       # TUI: crawl schema, import reference SQL, generate semantic models & metrics
/build-kb        # Build the vector knowledge base, optionally scoped to files/tables/domains
```

**Create a Subagent**: open the unified agent manager and package mature context into a scoped, domain-aware chatbot with curated tools and business rules.

```bash
/agent           # Create and manage subagents in the TUI
```

**Deliver**: serve the subagent to analysts via web (`localhost:8501/?subagent=mychatbot`), REST API, MCP, or Slack/Feishu, with feedback collection (upvotes, issue reports) built in.

**Measure**: run benchmarks against BIRD or Spider 2.0-Snow to track SQL accuracy as context evolves.

**Iterate**: analyst feedback loops back. Engineers fix SQL, add rules, refine semantic models, and extend with Skills, plugins, or MCP tools. The agent gets more accurate over time.

→ [End-to-end tutorial](https://docs.datus.ai/latest/getting_started/contextual_data_engineering/#part-2--hands-on-tutorial-california-schools) · [CLI docs](https://docs.datus.ai/latest/cli/introduction/) · [Knowledge Base docs](https://docs.datus.ai/latest/knowledge_base/introduction/) · [Subagent docs](https://docs.datus.ai/latest/subagent/introduction/)

## Architecture

### Workflow Engine

Under the agentic chat surface, Datus uses a configurable **node-based workflow engine**. Benchmark and batch runs use it directly, and you can compose custom plans with sequential, parallel, and sub-workflow steps:

```yaml
workflow:
  plan: planA
  planA:
    - schema_linking     # Find relevant tables
    - parallel:          # Run in parallel
      - gen_sql          # SQL generation
      - reasoning        # Chain-of-thought reasoning
    - selection          # Pick the best result
    - execute_sql        # Run the query
    - output             # Format and return
```

### Node Types

| Category | Nodes |
|----------|-------|
| **Core** | `schema_linking`, `execute_sql`, `reasoning`, `reflect`, `output`, `fix`, `date_parser`, `doc_search` |
| **Agentic** | `chat`, `gen_sql`, `explore`, `semantic` (semantic model generation), `sql_summary`, `search_metrics`, `ask_metrics`, `compare`, `gen_table`, `gen_job`, `gen_skill`, `gen_report`, `gen_visual_report`, `gen_visual_dashboard`, `gen_dashboard`, `scheduler`, `feedback` |
| **Control Flow** | `parallel`, `selection`, `subworkflow`, `hitl` |

### RAG Knowledge Base

The knowledge base runs on **LanceDB** by default, with pluggable vector and relational backends (including PostgreSQL-backed storage) and optional full-text search for metadata retrieval. It organizes context in layers:

- **Schema Metadata**: table and column descriptions, relationships
- **Reference SQL**: curated query examples with summaries
- **Reference Templates**: parameterized Jinja2 SQL templates for stable, reusable queries
- **Semantic Models**: business logic and metric definitions (MetricFlow or OSI)
- **Metrics**: executable business metrics via semantic layer integration
- **Platform Docs**: ingested from GitHub repos, websites, or local files

Build it interactively with `/bootstrap` and `/build-kb`, or in batch:

```bash
datus-agent bootstrap-kb --datasource demo --components metadata reference_sql
```

## Configuration

Datus is configured via `agent.yml`. Launch `datus` and use `/model` plus `/datasource` to populate it interactively, or copy [`conf/agent.yml.example`](conf/agent.yml.example) and edit it by hand.

LLM access uses a two-tier model: **`agent.providers`** holds credentials for known providers (the model catalog comes from `conf/providers.yml`, and `/model` switches models without YAML edits), while **`agent.models`** defines custom or self-hosted endpoints. The active selection persists per project in `./.datus/config.yml` as `target: { provider: ..., model: ... }`.

| Section | Purpose |
|---------|---------|
| `agent.providers` | Credentials for built-in LLM providers (model catalog from `providers.yml`) |
| `agent.models` | Custom / self-hosted LLM endpoint definitions |
| `agent.nodes` | Per-node model assignment and tuning parameters |
| `agent.services.datasources` | Database connections (SQLite, DuckDB, Snowflake, etc.) |
| `agent.storage` | Embedding models, vector DB, and RAG configuration |
| `agent.workflow` | Execution plans with sequential, parallel, and sub-workflow steps |
| `agent.agentic_nodes` | Configuration for agentic nodes (semantic model gen, metrics gen) |
| `agent.skills` | Skill directories and permissions |
| `agent.document` | Platform documentation sources (GitHub repos, websites, local files) |

API keys are injected via environment variables using `${ENV_VAR}` syntax.

## Supported LLM Providers

| Provider | Type | Notes |
|----------|------|-------|
| OpenAI | `openai` | GPT-4o, GPT-4.1, etc. |
| Anthropic Claude | `claude` | Direct API |
| Claude Subscription | `claude_subscription` | OAuth via a Claude Pro/Max subscription |
| OpenAI Codex | `codex` | OAuth via a ChatGPT subscription |
| Google Gemini | `gemini` | Gemini 2.0+ |
| DeepSeek | `deepseek` | DeepSeek-Chat, DeepSeek-Coder |
| Alibaba Qwen | `qwen` | Qwen series |
| Moonshot Kimi | `kimi` | Kimi models |
| MiniMax | `minimax` | MiniMax models |
| GLM (Zhipu) | `glm` | GLM-4 series |
| OpenRouter | `openrouter` | 300+ models via a single API key |
| Coding plans | `alibaba_coding`, `bigmodel_coding`, `zai_coding`, `minimax_coding`, `kimi_coding` | Reuse vendor coding-plan subscriptions |

**Embedding models:** OpenAI, Sentence-Transformers, FastEmbed, Hugging Face.

Per-node model assignment lets you use different providers for different workflow steps (e.g., a cheaper model for schema linking, a stronger model for SQL generation).

## Supported Databases

| Database | Type | Package |
|----------|------|---------|
| SQLite | `sqlite` | Built-in |
| DuckDB | `duckdb` | Built-in |
| PostgreSQL | `postgresql` | [`datus-postgresql`](https://github.com/Datus-ai/datus-db-adapters) |
| MySQL | `mysql` | [`datus-mysql`](https://github.com/Datus-ai/datus-db-adapters) |
| Snowflake | `snowflake` | [`datus-snowflake`](https://github.com/Datus-ai/datus-db-adapters) |
| StarRocks | `starrocks` | [`datus-starrocks`](https://github.com/Datus-ai/datus-db-adapters) |
| ClickHouse | `clickhouse` | [`datus-clickhouse`](https://github.com/Datus-ai/datus-db-adapters) |
| ClickZetta | `clickzetta` | [`datus-clickzetta`](https://github.com/Datus-ai/datus-db-adapters) |
| Apache Doris | `doris` | [`datus-doris`](https://github.com/Datus-ai/datus-db-adapters) |
| Hologres | `hologres` | [`datus-hologres`](https://github.com/Datus-ai/datus-db-adapters) |
| Hive | `hive` | [`datus-hive`](https://github.com/Datus-ai/datus-db-adapters) |
| Spark | `spark` | [`datus-spark`](https://github.com/Datus-ai/datus-db-adapters) |
| Trino | `trino` | [`datus-trino`](https://github.com/Datus-ai/datus-db-adapters) |

See [Database Adapters documentation](https://docs.datus.ai/latest/adapters/db_adapters/) for details.

## Development

```bash
uv sync                                                                    # Install dependencies
uv run python ci/run-pr-tests.py upstream/main                             # PR CI harness (no external deps)
uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/  # Lint & format
```

Enable `--save_llm_trace` on CLI commands or set `save_llm_trace: true` per model in `agent.yml` to persist LLM inputs/outputs for debugging, or configure `agent.observability.tracing` to export run traces to Langfuse, LangSmith, Datadog, Braintrust, or an OTLP collector. → [LLM Trace docs](https://docs.datus.ai/latest/training/llm_trace_usage/) · [Observability docs](https://docs.datus.ai/latest/develop/observability/)

See [CLAUDE.md](CLAUDE.md) for full development conventions, architecture patterns, and testing rules.

## License

[Apache 2.0](LICENSE)
