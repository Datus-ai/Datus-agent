---
name: grafana-dashboard
description: Create, view, and manage Grafana dashboards with panels and datasources
tags:
  - grafana
  - dashboard
  - BI
  - visualization
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_dashboard
---

# Grafana Dashboard Skill

Use this skill to create, update, or inspect dashboards in Grafana.

The data you need is already in a table inside a Grafana datasource
database — `gen_dashboard` does not move data. Your job: build panels against
that table, assemble the dashboard, validate.

## Key Differences from Superset

- Grafana charts (panels) embed SQL queries directly — no separate "dataset" concept.
- Panels belong to dashboards — `dashboard_id` is **required** for `create_chart`.
- The datasource is auto-resolved from the BI platform's `dataset_db` config.
- `update_chart` is **not supported** — delete and recreate the panel instead.

## Dashboard Creation Workflow

Follow these steps **in order**.

### Step 1: Confirm the Serving Datasource Contract

Grafana does not expose `list_bi_databases`. Use `get_bi_serving_target()` when
available to confirm the configured serving datasource, or proceed with the
configured `dataset_db.bi_database_name`; `create_chart` resolves the Grafana
datasource automatically. If the datasource cannot be resolved, bail with a
structured error. Do NOT try to auto-provision.

### Step 2: Create Dashboard (`create_dashboard`)

Create the dashboard first — Grafana requires `dashboard_id` to add panels.

```python
create_dashboard(title="Dashboard Title", description="Optional description")
```

Returns `dashboard_id` — save it for Step 3.

### Step 3: Create Charts (`create_chart`)

Create panels with SQL that queries the **existing table** in the
Grafana-registered datasource.

```python
create_chart(
    chart_type="line",           # bar, line, pie, table, big_number, scatter
    title="Chart Title",
    sql="SELECT date_col AS time, value_col FROM target_table ORDER BY date_col",
    dashboard_id="<from step 2>"
)
```

**CRITICAL RULES for the `sql` parameter:**
- The SQL must query the table already present in the Grafana datasource
  database. Do NOT reference source-warehouse names — Grafana's datasource
  doesn't reach the source.
- For time series charts: alias the time column as `time` (e.g., `SELECT date_col AS time, ...`).
- For table charts: use descriptive column aliases (e.g., `SELECT date AS "Date", count AS "Count"`).
- Always include `ORDER BY` for time series data.
- The datasource is automatically resolved — do not worry about datasource configuration.

### Multiple Charts

Repeat Step 3 for each chart. All panels use the same `dashboard_id`.

If a chart needs a different data shape and the table isn't available, stop
and return a structured error listing the missing table. The caller must
prepare or refresh that data separately before retrying.

### Step 4: Finish and Let Validation Run

After creating the dashboard and panels, finish the run and return the created
IDs. `bi-validation` is a validator skill invoked automatically by
`ValidationHook.on_end`; do not call `load_skill("bi-validation")` or try to
run validator checks manually.

Publish is complete when the creation calls succeed and the dashboard / panel
identifiers are known. The framework validates reachability and wiring after
the agent run ends.

## Viewing & Querying

| Action | Tool | Notes |
|--------|------|-------|
| List dashboards | `list_dashboards(search="keyword")` | Filter by keyword |
| Get dashboard details | `get_dashboard(dashboard_id="...")` | Full info including panels |
| List charts in dashboard | `list_charts(dashboard_id="...")` | All panels with SQL |
| Get chart details | `get_chart(chart_id="...", dashboard_id="...")` | Full panel metadata; dashboard_id required in Grafana |
| List datasources | `list_datasets()` | Grafana datasources |

## Updating

- **Update dashboard**: `update_dashboard(dashboard_id, title="New Title", description="New desc")`
- **Update chart**: Not supported in Grafana. Delete the chart and create a new one instead.

## Deleting

**MUST confirm with user before any deletion.**

- `delete_dashboard(dashboard_id="...")`
- `delete_chart(chart_id="...")`

## Important Rules

1. **Data movement is outside scope.** If the target table doesn't exist in
   the Grafana datasource, stop and return a structured error naming the
   missing table. The caller must prepare or refresh data separately with
   `gen_job` or `scheduler` before retrying dashboard creation.
2. **Chart SQL must reference tables the Grafana datasource can see** — never
   use source-warehouse table names in `create_chart(sql=...)`.
3. **`dashboard_id` is required** for `create_chart` — create the dashboard before creating charts.
4. **Time series**: Alias the time column as `time` for Grafana to recognize it.
5. **Language**: Match the user's language (Chinese input -> Chinese output).
6. **Stat/big_number panels**: Use a single aggregation query like `SELECT COUNT(*) AS value FROM table`.
