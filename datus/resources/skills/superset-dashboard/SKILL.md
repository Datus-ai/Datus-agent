---
name: superset-dashboard
description: Create, view, and manage Superset dashboards with charts and datasets
tags:
  - superset
  - dashboard
  - BI
  - visualization
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_dashboard
---

# Superset Dashboard Skill

This skill defines the workflow for creating and managing dashboards in Apache Superset.

## Dashboard Creation Workflow

Follow these steps **in order**. Each step depends on the output of the previous one.

### Step 1: Materialize Data (`write_query`) When Needed

Use `write_query` only when you need to run analytical SQL on the **source database** and materialize the result into **Superset's own database**.

```python
write_query(sql="SELECT ... FROM source_table ...", table_name="materialized_table_name")
```

- The SQL runs on the source (datasource) database via the active connector.
- Results are written as a physical table in Superset's dataset database.
- Returns `database_id` (when resolvable) — save it for Step 2. If not returned, use `list_bi_databases()` to find the correct Superset database ID.
- If the target table already exists in a Superset-connected BI database, or a virtual dataset is sufficient, skip this step and go directly to Step 2.

### Step 2: Register Dataset (`create_dataset`)

Register a table or SQL query as a Superset dataset.

**Physical dataset** (table created by `write_query` or already present in a Superset-connected BI database):
```python
create_dataset(name="materialized_table_name", database_id="<from step 1 or list_bi_databases>")
```

**Virtual dataset** (aggregated/transformed view):
```python
create_dataset(name="view_name", database_id="<from step 1 or list_bi_databases>", sql="SELECT ... FROM materialized_table_name")
```

- Returns `dataset_id` — save it for Step 3.
- IMPORTANT: `create_dataset.database_id` must be a Superset BI database connection ID from `write_query` or `list_bi_databases()`. Do NOT assume a source connector ID is valid here.

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

### Step 6: Validate the Published Dashboard (`bi-validation`)

After assembling the dashboard, load `bi-validation` and verify:

- the dashboard exists and is reachable via `get_dashboard`
- the expected charts appear via `list_charts`
- each chart is inspected via `get_chart`
- chart titles, types, metrics, x-axis fields, dimensions, and dataset wiring match the intended configuration
- each chart is checked with `get_chart_data` to confirm it runs without backend errors
- key numeric values are compared where expected results or tolerances are available

Do not report success until this validation pass completes.

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
- If the wrong `dimensions` or `dataset_id` was used, recreate the chart instead of relying on `update_chart`.

## Deleting

**MUST confirm with user before any deletion.**

- `delete_dashboard(dashboard_id="...")`
- `delete_chart(chart_id="...")`
- `delete_dataset(dataset_id="...")`

## Important Rules

1. **Use `write_query` only when needed** to materialize source-side results into a Superset-accessible table.
2. **Use a Superset BI database ID** for `create_dataset` — obtain it from `write_query` or `list_bi_databases`, not from a source connector.
3. **Choose the correct chain**:
   - materialized flow: `write_query` -> `database_id` -> `create_dataset` -> `dataset_id` -> `create_chart`
   - direct BI flow: `list_bi_databases` -> `create_dataset(... or sql=...)` -> `dataset_id` -> `create_chart`
4. **Language**: Match the user's language (Chinese input -> Chinese output).
5. **Multiple charts**: Create separate datasets for charts needing different data shapes.
