# Agent

`agent.agentic_nodes` is where Datus configures the chat node, built-in subagents, and your custom subagents.

It applies to:

- built-in agentic nodes such as `chat`, `explore`, `gen_sql`, `gen_report`, `gen_dashboard`, and `scheduler`
- custom subagents created with `.subagent`
- advanced manual aliases that point a custom name at a built-in node class

For LLM provider entries (`agent.models`) and the default selector (`agent.target`), see **[Models](models.md)**.

## Common fields

The runtime currently reads these commonly used fields from `agentic_nodes` entries:

- `model`: provider key from `agent.models`
- `system_prompt`: subagent name / prompt template base name
- `node_class`: node implementation to use, such as `gen_sql`, `gen_report`, `explore`, `gen_table`, `gen_skill`, `gen_dashboard`, or `scheduler`
- `prompt_version`, `prompt_language`
- `agent_description`
- `tools`, `mcp`, `skills`
- `rules`
- `max_turns`
- `workspace_root`
- `scoped_context`
- `subagents`
- `semantic_adapter` for semantic-model and metrics agents
- `bi_platform` for dashboard agents
- `scheduler_service` for scheduler agents

`scoped_kb_path` is deprecated. New configs use shared global storage with query-time filters instead of per-subagent scoped KB directories.

## `subagents` delegation

`subagents` controls whether the node exposes the `task()` tool and which subagent types it may delegate to.

- `subagents: "*"` — allow all discoverable subagents except self
- `subagents: explore, gen_sql` — allow only a named subset
- blank or omitted:
    - `chat` defaults to `*`
    - most other agentic nodes default to `explore`
    - explicitly setting an empty value disables delegation

## `scoped_context`

`scoped_context` limits what the subagent should see from shared metadata and knowledge:

```yaml
scoped_context:
  namespace: finance
  tables: mart.finance_daily, mart.finance_budget
  metrics: finance.revenue.daily_revenue
  sqls: finance.revenue.region_rollup
```

When writing YAML manually, set `namespace` explicitly. The `.subagent` wizard fills it from the current database automatically.

## Example

```yaml
agent:
  agentic_nodes:
    chat:
      model: claude
      max_turns: 50
      subagents: "*"

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
        namespace: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup

    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    semantic_metrics:
      node_class: gen_metrics
      model: claude
      semantic_adapter: metricflow
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      scheduler_service: airflow_prod
      max_turns: 30
```
