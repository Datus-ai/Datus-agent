# Subagent Guide

## Overview

Subagents are specialized AI assistants in Datus. They share the same project configuration as the main chat agent, but run with their own prompt, tool surface, session, and optional scoped context.

A subagent can be:

- A built-in system subagent such as `gen_sql`, `explore`, or `scheduler`
- A custom subagent defined in `agent.agentic_nodes` in `agent.yml`

## What a Subagent Includes

A subagent can have:

- **Dedicated system prompt**: separate prompt template and prompt version
- **Custom tools**: native tools, MCP tools, skills, and node-specific rules
- **Scoped context**: optional limits for tables, metrics, and reference SQL
- **Independent session**: separate conversation history from the main chat node
- **Delegation policy**: optional `task()` access to other subagents via the `subagents` field

## Built-in Subagents

The current built-in set comes from `SYS_SUB_AGENTS` in code:

1. `explore`: read-only schema, knowledge, and file exploration
2. `gen_sql`: specialized SQL generation
3. `gen_report`: structured report generation
4. `gen_semantic_model`: MetricFlow semantic model generation
5. `gen_metrics`: MetricFlow metric generation
6. `gen_sql_summary`: SQL summary generation
7. `gen_ext_knowledge`: business knowledge extraction
8. `gen_table`: interactive table creation
9. `gen_job`: data pipeline jobs (single-database ETL AND cross-database migration)
10. `gen_skill`: skill creation and optimization
11. `gen_dashboard`: BI dashboard creation and management
12. `scheduler`: Airflow job lifecycle management

See [Built-in subagents](./builtin_subagents.md) for details.

## Custom Subagents

Custom subagents are configured under `agent.agentic_nodes`.

The unified agent TUI (`/agent` or `/subagent`) Custom-tab wizard currently creates `gen_sql`-style or `gen_report`-style custom subagents. If you want to alias more specialized node classes such as `explore`, `gen_table`, `gen_skill`, `gen_dashboard`, or `scheduler`, edit `agent.yml` manually.

Example:

```yaml
agent:
  agentic_nodes:
    finance_report:
      node_class: gen_report
      model: claude
      system_prompt: finance_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "Finance reporting assistant"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      subagents: explore, gen_sql
      max_turns: 30
      scoped_context:
        datasource: finance
        tables: mart.finance_daily, mart.finance_budget
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup
      rules:
        - Prefer existing finance metrics before writing new SQL
```

Notes:

- `node_class` defaults to `gen_sql` if omitted
- When writing `scoped_context` manually, set `datasource` explicitly
- `subagents` controls which task types this node may delegate to

## How to Use Subagents

### Method 1: CLI Slash Command

Start the CLI:

```bash
datus --database production
```

Then launch a subagent with `/[name]`:

```text
/gen_metrics Generate a revenue metric from this SQL: SELECT SUM(revenue) FROM orders
/finance_report Analyze quarter-over-quarter revenue changes
```

### Method 2: Web Interface

Start the web interface:

```bash
datus --web --database production
```

Open a specific subagent directly:

```bash
datus --web --database production --subagent finance_report
```

Direct URLs also work:

```text
http://localhost:8501/?subagent=gen_metrics
http://localhost:8501/?subagent=finance_report
```

### Method 3: Subagent as Tool (`task()`)

The main chat agent can delegate complex work to specialized subagents through the `task()` tool.

```mermaid
graph LR
    A[User Question] --> B[Chat Agent]
    B --> C{Needs delegation?}
    C -->|No| D[Direct response]
    C -->|Yes| E["task(type=...)"]
    E --> F[Specialized subagent]
    F --> G[Result returned]
    G --> D
```

Important behavior:

- `chat` defaults to `subagents: "*"` and can delegate to all discoverable subagents
- Most other agentic nodes default to `subagents: explore`
- Setting `subagents` to an empty value disables the `task()` tool
- Subagent nodes do not get their own nested `task()` tool; delegation depth is capped at two levels

### Common Task Types

| Type | Purpose |
|------|---------|
| `explore` | Gather schema, sample data, knowledge, or file context |
| `gen_sql` | Generate SQL with deeper multi-step reasoning |
| `gen_report` | Produce structured reports and analysis |
| `gen_semantic_model` | Generate MetricFlow semantic models |
| `gen_metrics` | Generate MetricFlow metrics |
| `gen_sql_summary` | Summarize SQL into reusable knowledge |
| `gen_ext_knowledge` | Extract business knowledge from question-SQL pairs |
| `gen_table` | Create tables interactively |
| `gen_job` | Build data pipeline jobs (single-database ETL or cross-database migration) |
| `gen_skill` | Create or optimize skills |
| `gen_dashboard` | Create or manage BI dashboards |
| `scheduler` | Submit or operate Airflow jobs |
| Custom names | Any discoverable custom subagent defined in `agent.yml` |
