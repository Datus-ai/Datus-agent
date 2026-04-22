---
name: bi-validation
description: Validate BI dataset, chart, and dashboard changes after publish by checking object creation, refresh success, and key metric consistency against expected tolerances
tags:
  - bi
  - dashboard
  - publish
  - validation
  - metrics
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_dashboard
---

# BI Validation

Use this skill after publishing BI changes. It assumes BI platform tools exist and can create or inspect datasets, charts, and dashboards.

## When to use this skill

Activate when you need to:

- verify a dataset or chart was created successfully
- verify a dashboard update is wired correctly
- compare refreshed BI metrics against expected values or tolerances
- block rollout when BI outputs drift materially

## Core workflow

1. Identify the target dashboard by ID.
2. Call `get_dashboard` to confirm it exists and retrieve the full chart list.
3. For **EVERY** chart on the dashboard, call `get_chart` and inspect its configuration. Do not validate only a subset.
4. Verify `chart_type`, `metrics`, `x_axis`, `dimensions`, and `dataset_id` against the intended design for each chart.
5. When `get_chart_data` is supported on the active platform, call it for **every** chart to confirm the chart returns data without backend errors.
6. Compare key numeric values when expected results or tolerances are available. If exact expectations are not available, still treat successful `get_chart_data` execution as a required runtime check.
7. If a chart is wrong, choose a remediation that matches the available tools:
   - use `update_chart` only for fields the tool actually supports, such as `title`, `chart_type`, `metrics`, `x_axis`, `description`, and platform-specific SQL where applicable
   - if the problem is `dimensions`, `dataset_id`, or any unsupported wiring change, recreate the chart/panel or report failure
   - on Grafana, prefer recreate-over-update because `update_chart` is not supported
8. Return a compact pass / fail report covering **every** chart.

**IMPORTANT**: Every chart on the dashboard must pass configuration inspection with `get_chart`. Data validation with `get_chart_data` is mandatory when that tool is supported on the active platform. If `get_chart_data` is unavailable, mark the data check as unsupported / N/A and validate configuration plus reference SQL or known expectations instead.

## Platform notes

- Superset: confirm dashboard reachability, chart count, chart type, metric expressions, group-by dimensions, and runtime query success. Use `get_chart` and `get_chart_data` for every chart. Compare exact numeric values where expectations are known.
- Grafana: confirm dashboard reachability, panel count, panel type, datasource wiring, and panel SQL against the materialized tables. Use `get_chart` with `dashboard_id` for every panel. `get_chart_data` is not available in Grafana yet, so configuration validation is mandatory and the data check should be reported as unsupported / N/A unless a separate reference query is available.

## Configuration inspection checklist (per chart)

When calling `get_chart` for each chart, verify these fields:

| Field | Check |
|-------|-------|
| `chart_type` | Matches intended visualization (bar, pie, big_number, line, table, etc.) |
| `metrics` | Correct aggregation expressions (COUNT, SUM, AVG, etc.) |
| `x_axis` | Correct column for the horizontal axis (bar/line charts) |
| `dimensions` | Correct grouping columns (pie charts, grouped bar charts) |
| `dataset_id` | Points to the correct dataset |

If `title`, `chart_type`, `metrics`, `x_axis`, `description`, or platform-specific SQL is wrong, fix it with `update_chart` when the tool supports that change. If `dimensions`, `dataset_id`, or unsupported wiring is wrong, recreate the chart/panel or report failure.

## Publish verification checklist

- [references/publish-checklist.md](references/publish-checklist.md)

## Output expectations

For each chart, report:

| Column | Description |
|--------|-------------|
| Chart ID | The chart identifier |
| Chart Name | Human-readable title |
| Config Check | PASS if `get_chart` fields match design, FAIL + reason otherwise |
| Data Check | PASS if `get_chart_data` succeeds and expected values match when available; `N/A` when the platform does not support `get_chart_data`; FAIL + reason otherwise |

Final summary: total charts, passed, failed, overall PASS/FAIL decision.
