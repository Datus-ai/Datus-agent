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
---

# Superset Dashboard Skill

This skill defines the workflow for creating and managing dashboards in Apache Superset.

## Dashboard Creation Workflow

Follow these steps **in order**. Each step depends on the output of the previous one.

### Step 1: Materialize Data (`write_query`)

Run the analytical SQL on the **source database** and write results to **Superset's own database**.

```python
write_query(sql="SELECT ... FROM source_table ...", table_name="materialized_table_name")
```

- The SQL runs on the source (namespace) database via the active connector.
- Results are written as a physical table in Superset's dataset database.
- Returns `database_id` (when resolvable) — save it for Step 2. If not returned, use `list_bi_databases()` to find the correct ID.

### Step 2: Register Dataset (`create_dataset`)

Register the materialized table as a Superset dataset.

**Physical dataset** (table created by `write_query`):
```python
create_dataset(name="materialized_table_name", database_id="<from step 1>")
```

**Virtual dataset** (aggregated/transformed view):
```python
create_dataset(name="view_name", database_id="<from step 1>", sql="SELECT ... FROM materialized_table_name")
```

- Returns `dataset_id` — save it for Step 3.
- IMPORTANT: Use the `database_id` returned by `write_query` or from `list_bi_databases()`. This is the BI platform's own database, NOT the source database.

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

## Viewing & Querying

| Action | Tool | Notes |
|--------|------|-------|
| List dashboards | `list_dashboards(search="keyword")` | Filter by keyword |
| Get dashboard details | `get_dashboard(dashboard_id="...")` | Full info including charts |
| List charts in dashboard | `list_charts(dashboard_id="...")` | All charts with config |
| List datasets | `list_datasets()` | All registered datasets |
| List BI databases | `list_bi_databases()` | Database connections in Superset |

## Updating

- **Update dashboard**: `update_dashboard(dashboard_id, title="New Title", description="New desc")`
- **Update chart**: `update_chart(chart_id, title="...", chart_type="...", metrics="...")`

## Deleting

**MUST confirm with user before any deletion.**

- `delete_dashboard(dashboard_id="...")`
- `delete_chart(chart_id="...")`
- `delete_dataset(dataset_id="...")`

## Important Rules

1. **Never skip `write_query`** — Superset charts query the BI platform's own database, not the source database directly.
2. **Never use the source database ID** for `create_dataset` — always use the ID from `write_query` or `list_bi_databases`.
3. **Chain outputs**: `write_query` -> `database_id` -> `create_dataset` -> `dataset_id` -> `create_chart`.
4. **Language**: Match the user's language (Chinese input -> Chinese output).
5. **Multiple charts**: Create separate datasets for charts needing different data shapes.
