---
name: init
description: Initialize project workspace — generate AGENTS.md with architecture, directory map, services, and artifacts
tags:
  - init
  - workspace
  - project
version: 1.0.0
user_invocable: true
---

# Project Initialization

You are initializing a project workspace. Your goal is to generate an `AGENTS.md` file that describes this project's architecture, directory structure, services, and data assets at a high level.

**IMPORTANT**: AGENTS.md is an overview document, NOT a data dictionary. Keep it concise. Do NOT list every table — summarize databases by category and table count. The agent can use `list_tables` and `search_table` at runtime to explore details.

Follow these steps in order:

## Step 1: Gather Project Context

1. Use `ask_user` to ask: **"What is the goal of this project? Describe it in 1-2 sentences."**
2. Use `filesystem_tools` to list the current directory structure (scan top 3 levels, skip hidden dirs and `__pycache__`/`node_modules`/.venv`)
3. Read `README.md` if it exists (first 3000 chars)

## Step 2: Select Services

1. The system context includes configured databases from `agent.yml`. Present them to the user.
2. Use `ask_user` to ask: **"Which of these services should be included in the project? You can also describe additional services (e.g., Airflow scheduler, Superset BI) that aren't configured yet."**
   - Show the available databases as choices
   - Allow free-text input for additional services
3. For each selected database, use `db_tools.list_tables()` to get a table overview. **Do NOT include all tables in AGENTS.md** — just count them and categorize.

## Step 3: Generate AGENTS.md

Based on the gathered context, generate a markdown document with these sections:

### # {project_name}
One-line project description based on what the user told you.

### ## Architecture
Brief architecture description. Include:
- The data flow or system stack
- How services connect
- An ASCII diagram if the project is complex enough

### ## Directory Map
A table with columns: **Directory | Purpose | Key Entry Point | Consumer**

Cover the main directories found in the scan.

### ## Services
A table with columns: **Service | Type | Connection | Description**

Include both configured databases and any additional services the user mentioned.

### ## Data Assets
**Do NOT list every table.** Instead, provide a high-level summary per database:

- Database name, type, and total table count
- Categorize tables by domain (e.g., "Core: 8 tables — blocks, transactions, logs...", "Analytics: 4 tables — DEX trades, NFT trades...")
- Mention 3-5 key representative tables per category, not all of them
- Note: "Use `list_tables` and `search_table` to explore table details at runtime"

Example format:
```
### ethereum_iceberg (StarRocks, 15 tables)

| Category | Count | Key Tables | Description |
|----------|-------|------------|-------------|
| Core | 8 | blocks, transactions, logs, traces | Raw blockchain data |
| Derived | 3 | logs_decoded, traces_decoded | Decoded contract events |
| Analytics | 4 | dex_trades, nft_trades | Pre-built analytical datasets |

> Use `list_tables` and `search_table` tools to explore schema details.
```

For databases with many tables (>50), group by schema or naming pattern.

### ## Recommended Tools
For each configured service, list the tools the agent should use to interact with it at runtime. Match tools to service types:

| Service Type | Recommended Tools |
|-------------|-------------------|
| Database (sqlite, duckdb, snowflake, starrocks, mysql, postgresql) | `list_tables`, `search_table`, `describe_table`, `read_query` |
| BI Tool (superset) | `bi_tools.*` |
| Scheduler (airflow) | (future) |
| Semantic Layer (metricflow, dbt, cube) | `search_metrics`, `search_semantic_model` |
| Knowledge Base | `search_documents`, `search_historical_sql` |

Only include rows for services that are actually configured or mentioned by the user.

### ## Artifacts
Describe data artifacts, configs, or outputs this project produces:
- Data catalogs, semantic models
- SQL files, reference queries
- Reports, dashboards
- API schemas, config files

## Step 4: Write File

1. If `AGENTS.md` already exists in the current directory, use `ask_user` to ask: **"AGENTS.md already exists. Overwrite it?"**
2. Use `filesystem_tools.write_file` to write the generated content to `./AGENTS.md`
3. Tell the user the file has been created and they can edit it to refine.

## Important Notes

- **Keep it concise** — AGENTS.md is a project overview, not documentation for every table/column
- If you can't determine something (e.g., architecture), use placeholder comments like `<!-- Describe your architecture here -->`
- The ASCII diagram should be simple and readable
- Use the project directory name as the project name unless the README suggests otherwise
- For large databases (hundreds/thousands of tables), categorize and summarize — never enumerate all tables
