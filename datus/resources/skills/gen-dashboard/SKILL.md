---
name: gen-dashboard
description: Create, update, and manage BI dashboards with guided multi-step workflow. Use when user asks to create dashboards, add charts, visualize data, or manage BI platform assets.
tags:
  - dashboard
  - bi
  - visualization
  - superset
  - grafana
version: 1.0.0
user_invocable: true
---

# BI Dashboard Management

You are helping the user create, update, or inspect BI dashboards. Delegate the actual BI operations to the `gen_dashboard` subagent via `task()`.

## When to Use This Skill

- User asks to **create a dashboard** or **visualize data**
- User asks to **list, inspect, or modify existing dashboards**
- User asks to **add charts** to a dashboard
- User asks to **create datasets** in the BI platform

## Workflow

### Step 1: Clarify Requirements

Before delegating, gather enough context. Use `ask_user` if needed:

- **What data** should be visualized? (tables, metrics, SQL query)
- **What chart types** are appropriate? (bar, line, pie, table, big_number)
- **Dashboard name** and description
- **Target platform** if multiple are configured

### Step 2: Delegate to gen_dashboard Subagent

Use the `task` tool to delegate:

```
task(type="gen_dashboard", prompt="<detailed request>", description="<short summary>")
```

**For creating a dashboard**, include in the prompt:
- The data source (table names or SQL query)
- Desired charts with types and metrics
- Dashboard title and description

**For listing/inspecting**, keep the prompt simple:
- "List all dashboards"
- "Show details of dashboard ID 42"
- "List charts in dashboard 'Sales Overview'"

### Step 3: Report Results

After the subagent completes:
- Present the dashboard ID/URL to the user
- Offer to make modifications (add more charts, change types)
- For modifications, call `task(type="gen_dashboard")` again with the update request

## Dashboard Creation Workflow (Reference)

The gen_dashboard subagent follows this multi-step workflow internally:

1. **write_query** — Execute SQL on the source DB and write results to the dashboard DB (if materialization needed)
2. **list_bi_databases** + **create_dataset** — Register the data table in the BI platform
3. **create_chart** — Create visualizations with appropriate chart type, metrics, and dimensions
4. **create_dashboard** — Create the dashboard container (or use an existing one)
5. **add_chart_to_dashboard** — Assemble charts into the dashboard

## Chart Type Guide

| Chart Type | Best For |
|------------|----------|
| `bar` | Comparing categories, rankings |
| `line` | Trends over time, time series |
| `pie` | Part-to-whole relationships (< 7 categories) |
| `table` | Detailed data, exact values |
| `big_number` | Single KPI metrics, totals |
| `scatter` | Correlation between two metrics |

## Tips

- For complex dashboards with multiple charts, describe all charts in a single prompt — the subagent handles the full workflow
- If the user provides a SQL query, suggest using `write_query` to materialize the results first
- Always confirm the dashboard name and chart selections before creating
