# Customized Subagents

## Overview

Use `/subagent` to manage custom subagents stored under `agent.agentic_nodes` in `agent.yml`.

The current CLI supports:

- `add`: create a custom subagent with the interactive wizard
- `list`: list configured custom subagents
- `update <agent_name>`: edit an existing custom subagent
- `remove <agent_name>`: delete a custom subagent

Built-in system subagents from `SYS_SUB_AGENTS` are reserved and cannot be removed or edited with `/subagent`.

## What the Wizard Creates

The wizard writes two things:

1. A new entry under `agent.agentic_nodes`
2. A prompt template file named like `{agent_name}_system_{prompt_version}.j2`

The wizard currently supports two custom node styles:

- `gen_sql` (default)
- `gen_report`

If you want advanced aliases such as `explore`, `gen_table`, `gen_skill`, `gen_dashboard`, or `scheduler`, edit `agent.yml` manually.

## Wizard Fields

### Step 1: Basic Info

- `system_prompt`: the subagent name and config key
- `node_class`: `gen_sql` or `gen_report`
- `agent_description`: short description shown in previews and task-tool descriptions

### Step 2: Tools and MCP

You must select at least one native tool or one MCP tool.

- Native tools are stored as comma-separated category or method patterns such as `db_tools`, `semantic_tools.*`, or `context_search_tools.list_subject_tree`
- MCP selections are stored as comma-separated server or `server.tool` entries

### Step 3: Scoped Context

The wizard supports scoped values for:

- `tables`
- `metrics`
- `sqls`

When the wizard saves the config, it also records the current database as `scoped_context.datasource`.

### Step 4: Rules

Rules are stored as a string list under `rules` and appended to the final system prompt.

## Command Reference

### `/subagent add`

Starts the interactive wizard and creates a new custom subagent.

```bash
/subagent add
```

![Add subagent](../assets/add_subagent.png)

### `/subagent list`

Lists configured custom subagents.

```bash
/subagent list
```

The table currently shows:

- `Name`
- `Scoped Context`
- `Scoped KB`
- `Tools`
- `MCPs`
- `Rules`

`Scoped KB` is a legacy display column and is usually `—` for new configurations.

The list is filtered by the current database when a subagent has scoped context.

![List subagent](../assets/list_subagents.png)

### `/subagent update <agent_name>`

Loads the existing config into the wizard and saves changes back to `agent.yml`.

```bash
/subagent update finance_report
```

![Update subagent](../assets/update_subagent.png)

### `/subagent remove <agent_name>`

Deletes the config entry and its generated prompt template.

```bash
/subagent remove finance_report
```

## Example Output

A generated config typically looks like this:

```yaml
agent:
  agentic_nodes:
    finance_report:
      system_prompt: finance_report
      node_class: gen_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "Finance reporting assistant"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      mcp: ""
      rules:
        - Prefer existing finance metrics before generating new SQL
      scoped_context:
        datasource: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup
```

## Scoped Context Semantics

Current code no longer builds a separate scoped knowledge-base directory for each subagent.

Instead:

- scoped context is stored in `agentic_nodes.<name>.scoped_context`
- Datus applies filters against the shared global storage at query time
- database tools may also narrow their visible table surface based on the active subagent

That means:

- there is no `/subagent bootstrap` command in the current CLI
- `scoped_kb_path` is deprecated and not persisted for newly saved configs
- global knowledge still needs to be populated separately with `datus-agent bootstrap-kb`

## Advanced Manual Configuration

The wizard covers the common `gen_sql` and `gen_report` cases. For more advanced setups, edit `agent.yml` directly.

Example:

```yaml
agent:
  agentic_nodes:
    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      max_turns: 30
```

See [Subagent Guide](./introduction.md) and [Built-in subagents](./builtin_subagents.md) for the supported node classes and runtime behavior.
