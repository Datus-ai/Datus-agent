<p align="center">
  <strong>Datus · Open-Source Data Engineering Agent</strong>
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/datus-agent/"><img src="https://img.shields.io/pypi/v/datus-agent?logo=pypi&logoColor=white&color=654FF0" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <a href="https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg"><img src="https://img.shields.io/badge/Slack-join%20chat-4A154B?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTUuMDQyIDE1LjE2NWEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyIDIuNTIzQTIuNTI4IDIuNTI4IDAgMCAxIDAgMTUuMTY1YTIuNTI3IDIuNTI3IDAgMCAxIDIuNTIyLTIuNTJoMi41MnYyLjUyek02LjMxMyAxNS4xNjVhMi41MjcgMi41MjcgMCAwIDEgMi41MjEtMi41MiAyLjUyNyAyLjUyNyAwIDAgMSAyLjUyMSAyLjUydjYuMzEzQTIuNTI4IDIuNTI4IDAgMCAxIDguODM0IDI0YTIuNTI4IDIuNTI4IDAgMCAxLTIuNTIxLTIuNTIydi02LjMxM3pNOC44MzQgNS4wNDJhMi41MjggMi41MjggMCAwIDEtMi41MjEtMi41MkEyLjUyOCAyLjUyOCAwIDAgMSA4LjgzNCAwYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIxIDIuNTIydjIuNTJIOC44MzR6TTguODM0IDYuMzEzYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIxIDIuNTIxIDIuNTI4IDIuNTI4IDAgMCAxLTIuNTIxIDIuNTIxSDIuNTIyQTIuNTI4IDIuNTI4IDAgMCAxIDAgOC44MzRhMi41MjggMi41MjggMCAwIDEgMi41MjItMi41MjFoNi4zMTJ6TTE4Ljk1NiA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMSAyLjUyMi0yLjUyMUEyLjUyOCAyLjUyOCAwIDAgMSAyNCA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMiAyLjUyMWgtMi41MjJWOC44MzR6TTE3LjY4OCA4LjgzNGEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMyAyLjUyMSAyLjUyNyAyLjUyNyAwIDAgMS0yLjUyLTIuNTIxVjIuNTIyQTIuNTI3IDIuNTI3IDAgMCAxIDE1LjE2NSAwYTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIzIDIuNTIydjYuMzEyek0xNS4xNjUgMTguOTU2YTIuNTI4IDIuNTI4IDAgMCAxIDIuNTIzIDIuNTIyQTIuNTI4IDIuNTI4IDAgMCAxIDE1LjE2NSAyNGEyLjUyNyAyLjUyNyAwIDAgMS0yLjUyLTIuNTIydi0yLjUyMmgyLjUyek0xNS4xNjUgMTcuNjg4YTIuNTI3IDIuNTI3IDAgMCAxLTIuNTItMi41MjMgMi41MjYgMi41MjYgMCAwIDEgMi41Mi0yLjUyaDYuMzEzQTIuNTI3IDIuNTI3IDAgMCAxIDI0IDE1LjE2NWEyLjUyOCAyLjUyOCAwIDAgMS0yLjUyMiAyLjUyM2gtNi4zMTN6Ii8%2BPC9zdmc%2B" alt="Slack"></a>
</p>

<p align="center">
  <a href="https://datus.ai">Website</a> ·
  <a href="https://docs.datus.ai/">Docs</a> ·
  <a href="https://docs.datus.ai/latest/getting_started/Quickstart/">Quick Start</a> ·
  <a href="https://dosi.datus.ai/">Dosi</a> ·
  <a href="https://docs.datus.ai/latest/release_notes/">Release Notes</a>
</p>

<p align="center">
  English | <a href="README.zh.md">简体中文</a>
</p>

---

**Datus** is the open-source data engineering agent for the modern data stack: one agent that connects your warehouse, catalog, semantic layer, and BI, grounded in an evolvable context engine your team owns.

Datus handles SQL authoring and validation, semantic model and metric construction, and the generation of pipelines, reports, and dashboards. Every run and every correction settles into context, which steadily raises the accuracy of its output. The whole stack stays open and flexible: databases, BI, schedulers, LLMs, and your team's own tools all connect through standard interfaces.

## Architecture

![Datus Architecture](docs/assets/datus_architecture.svg)

The diagram reads top to bottom: who uses Datus, what the agent is made of, and what it connects to.

- **Three entry points, by role**: data engineers work in [Datus-CLI](https://docs.datus.ai/latest/cli/introduction/) to explore data and build assets; analysts ask through [Datus-Chat](https://docs.datus.ai/latest/web_chatbot/introduction/) on the web, in Slack/Feishu, or in VS Code, and their feedback flows back into the agent; other agents and applications consume [Datus-API](https://docs.datus.ai/latest/API/introduction/) over REST and MCP.
- **The agent core**: [subagents](https://docs.datus.ai/latest/subagent/introduction/) package curated context, tools, and rules for one business domain, and [skills](https://docs.datus.ai/latest/skills/introduction/) add packaged tools. Underneath sits the [context engine](https://docs.datus.ai/latest/knowledge_base/introduction/): metadata, metrics, reference SQL, knowledge, and local files, retrieved through business-domain trees plus vector search, with [storage](https://docs.datus.ai/latest/configuration/storage/) on embedded LanceDB and SQLite and PostgreSQL for teams that share context.
- **Connected systems**: LLM providers, data warehouses, the [Dosi](https://dosi.datus.ai/) semantic layer, job schedulers, BI tools, and MCP servers and clients, reached through adapters and through [plugins](https://docs.datus.ai/latest/plugin/introduction/) that bring third-party and in-house tools into the agent.

## Features

### Semantic layer

- **Automated semantic modeling**: the agent reads your database schema and SQL history, then generates [OSI](https://dosi.datus.ai/) semantic models and metric definitions, with no hand-written YAML.
- **[Dosi](https://dosi.datus.ai/) execution engine**: compiles one semantic model into SQL for 13+ database dialects, and ships as an independent program you can also run as a CLI, REST server, or MCP server.
- **Metric Q&A and attribution**: [AskMetrics](https://docs.datus.ai/latest/subagent/ask_metrics/) answers business questions from metric definitions instead of improvising SQL, and when a metric moves, dimension attribution locates which dimension drove the change.

### Agent and context

- **Sharper with use**: the [context engine](https://docs.datus.ai/latest/getting_started/contextual_data_engineering/) gathers schemas, reference SQL, and business rules, and writes every correction back, so later answers keep getting more accurate.
- **[Subagent](https://docs.datus.ai/latest/subagent/introduction/) delivery**: curate context, tools, and rules for one domain, package them as a dedicated chatbot, and serve it to analysts over web, API, MCP, Slack/Feishu, or VS Code.
- **Data engineering automation**: [built-in subagents](https://docs.datus.ai/latest/subagent/builtin_subagents/) handle cross-database migration, ETL job generation, and wide-table builds, with [Airflow](https://docs.datus.ai/latest/adapters/scheduler_adapters/) orchestration and Superset/Grafana dashboard read-write.
- **Report and dashboard generation**: produce self-contained [HTML reports and interactive dashboards](https://docs.datus.ai/latest/subagent/gen_visual_report/) straight from chat, previewed locally with no SaaS backend.

### Openness and governance

- **Open ecosystem**: adapters for [15 databases](https://docs.datus.ai/latest/adapters/db_adapters/), 10+ LLM providers, and an [MCP](https://docs.datus.ai/latest/integration/mcp/) server and client.
- **External integrations**: the [plugin](https://docs.datus.ai/latest/plugin/introduction/) framework connects third-party platforms and in-house tools to the agent; one `datus-plugin.yml` manifest declares CLI commands, skills, and prompt context, with per-project activation.
- **[Skills](https://docs.datus.ai/latest/skills/introduction/)**: packaged tools following the agentskills.io convention, installable from a marketplace.
- **Enterprise governance**: tiered permission profiles, statement-level [SQL authorization](https://docs.datus.ai/latest/configuration/sql_policy/) with AI pre-review, bash confined to an OS-level sandbox, and [traces](https://docs.datus.ai/latest/develop/observability/) exportable to any OTLP platform.

## Quickstart

Linux or macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

Open a new shell and run `datus`, then:

1. `/model` to configure an LLM
2. `/datasource` to add a datasource
3. `/init` (optional) to scan the current project

Manual install works too: `pip install datus-agent` (Python 3.12+); more install options are covered in the [Quickstart](https://docs.datus.ai/latest/getting_started/Quickstart/). When `pip` spends minutes backtracking through `litellm` releases (versions up to 0.3.9 are affected), `uv` resolves the same set in seconds: `pip install uv && uv pip install datus-agent --system`. The [end-to-end tutorial](https://docs.datus.ai/latest/getting_started/contextual_data_engineering/#part-2-hands-on-tutorial-california-schools) demonstrates the full flow on a sample dataset. Configuration has two levels: a global `agent.yml` for the main settings, and a per-project `.datus/config.yml` for overrides such as the active model and default datasource (see the [configuration docs](https://docs.datus.ai/latest/configuration/introduction/)).

## Interfaces

The examples below use a datasource named `demo`; create one first with `/datasource`.

| Interface | Command | Use Case |
|-----------|---------|----------|
| **CLI** (interactive REPL) | `datus --datasource demo` | Data engineers exploring data, building context, creating subagents |
| **Web Chatbot** (FastAPI + React) | `datus --web --datasource demo` | Analysts chatting with subagents via browser (`http://localhost:8501`) |
| **REST API** (FastAPI) | `datus-api --datasource demo` | Applications consuming data services via REST (`http://localhost:8000`) |
| **MCP Server** | `datus-mcp --datasource demo` | MCP-compatible clients (Claude Desktop, Cursor, etc.) |
| [**IM Gateway**](https://docs.datus.ai/latest/gateway/introduction/) | `datus-gateway` | Analysts talking to subagents in Slack or Feishu/Lark |
| [**VS Code**](https://docs.datus.ai/latest/vscode_extension/introduction/) (Datus Studio) | connects to `datus --web` | Catalog explorer, chat panel, SQL results & AI charts in the IDE |

> **Tip:** Print mode streams JSON to stdout for scripting and CI: `datus -p "your question" --datasource demo`.

## Development

### Developing Datus

Start here to work on Datus itself: install dependencies with uv, then run the PR test harness and format checks before submitting.

```bash
uv sync                                                                    # Install dependencies
uv run python ci/run-pr-tests.py upstream/main                             # PR CI harness (no external deps)
uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/  # Lint & format
```

See [CLAUDE.md](CLAUDE.md) for development conventions, architecture patterns, and testing rules.

### Developing a plugin

Extending Datus does not require touching its core: declare CLI commands, skills, and prompt context in a `datus-plugin.yml` manifest, then pack it for distribution and per-project activation. The [plugin development guide](https://docs.datus.ai/latest/plugin/development/) walks through the full flow.

## License

[Apache 2.0](LICENSE)
