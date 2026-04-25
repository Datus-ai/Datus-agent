---
name: superset-dashboard
description: Create, view, and manage Superset dashboards with charts and datasets
tags:
  - superset
  - dashboard
  - BI
  - visualization
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_dashboard
---

# Superset Dashboard Skill

Use this skill to create, update, or inspect dashboards in Apache Superset.

The data you need is already in a table inside a Superset-connected database —
`gen_dashboard` does not move data. Your job: register the dataset, build
charts, assemble the dashboard, validate.

## Dashboard Creation Workflow

Follow these steps **in order**. Each step depends on the output of the
previous one.

### Step 1: Locate the BI database

```python
list_bi_databases()
```

Match the orchestrator-provided `bi_database_name` against the returned
`name`, and pick the corresponding `id` — that's your `database_id` for every
`create_dataset` call below. If no match, bail with a structured error naming
the missing DB.

### Step 2: Register Dataset (`create_dataset`)

Register a table or SQL query as a Superset dataset.

**Physical dataset** (table already exists in the BI database):
```python
create_dataset(name="target_table", database_id="<from step 1>")
```

**Virtual dataset** (aggregated/transformed view over the physical table):
```python
create_dataset(name="view_name", database_id="<from step 1>", sql="SELECT ... FROM target_table")
```

- Returns `dataset_id` — save it for Step 3.
- IMPORTANT: `database_id` must be a Superset BI database connection ID from
  `list_bi_databases()`. Source connector IDs and Superset BI database IDs are
  separate identifiers.

### Step 3: Create Charts (`create_chart`)

Create visualization charts referencing the dataset.

```python
create_chart(
    chart_type="bar",        # bar, line, pie, table, big_number, scatter
    title="Chart Title",
    dataset_id="<from step 2>",
    metrics="revenue,COUNT(order_id)",
    x_axis="date_column",
    dimensions="category"
)
```

**Metrics format:**
- Plain column name defaults to `SUM(column)`: `"revenue"` -> `SUM(revenue)`
- Explicit aggregation: `"AVG(price)"`, `"MAX(amount)"`, `"MIN(cost)"`, `"COUNT(id)"`
- Multiple metrics comma-separated: `"revenue,COUNT(order_id),AVG(price)"`

**For `big_number` charts:**
- Use a single metric: `metrics="AVG(activity_count)"`
- No `x_axis` or `dimensions` needed.

### Step 4: Create Dashboard (`create_dashboard`)

```python
create_dashboard(title="Dashboard Title", description="Optional description")
```

Returns `dashboard_id` — save it for Step 5.

### Step 5: Add Charts to Dashboard (`add_chart_to_dashboard`)

```python
add_chart_to_dashboard(chart_id="<from step 3>", dashboard_id="<from step 4>")
```

Repeat for each chart.

### Step 6: Finish and Let Validation Run

After assembling the dashboard, finish the run and return the created IDs.
`bi-validation` is a validator skill invoked automatically by
`ValidationHook.on_end`; do not call `load_skill("bi-validation")` or try to
run validator checks manually.

Publish is complete when the creation calls succeed and the dashboard / chart /
dataset identifiers are known. The framework validates reachability and wiring
after the agent run ends. Return the `dashboard_id`,
`dataset_id`, and `chart_ids`. Treat follow-up layout or chart rewiring as a
separate update request.

## Viewing & Querying

| Action | Tool | Notes |
|--------|------|-------|
| List dashboards | `list_dashboards(search="keyword")` | Filter by keyword |
| Get dashboard details | `get_dashboard(dashboard_id="...")` | Full info including charts |
| List charts in dashboard | `list_charts(dashboard_id="...")` | All charts with config |
| Get chart details | `get_chart(chart_id="...")` | Full chart definition and query details |
| Get chart data | `get_chart_data(chart_id="...", limit=10)` | Query result rows for numeric validation |
| List datasets | `list_datasets()` | All registered datasets |
| List BI databases | `list_bi_databases()` | Database connections in Superset |

## Updating

- **Update dashboard**: `update_dashboard(dashboard_id, title="New Title", description="New desc")`
- **Update chart**: `update_chart(chart_id, title="...", chart_type="...", metrics="...", x_axis="...", description="...")`
- Use updating for an explicit change request or a concrete validation
  mismatch. Keep existing published resources when the publish checks pass.

## Deleting

**MUST confirm with user before any deletion.**

- `delete_dashboard(dashboard_id="...")`
- `delete_chart(chart_id="...")`
- `delete_dataset(dataset_id="...")`

## Important Rules

1. **Data movement is outside scope.** If the target table doesn't exist in
   the BI database, stop and return a structured error naming the missing
   table. The caller must prepare or refresh data separately with `gen_job`
   or `scheduler` before retrying dashboard creation.
2. **Use a Superset BI database ID** for `create_dataset` — obtain it from
   `list_bi_databases`, not from a source connector.
3. **Language**: Match the user's language (Chinese input -> Chinese output).
4. **Multiple charts**: Create separate datasets for charts needing different data shapes.
