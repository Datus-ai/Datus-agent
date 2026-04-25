---
name: bi-validation
description: BI validator driven by ValidationHook — inspects every dashboard, chart, and dataset delivered in the run, verifies config quality (chart type, metrics, dimensions, dataset wiring) and data presence via get_chart_data when supported
tags:
  - bi
  - dashboard
  - publish
  - validation
  - metrics
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_dashboard
kind: validator
severity: blocking
mode: llm
targets: []   # empty = match all; body filters to dashboard / chart / dataset types
---

# BI Validation

Driven by `ValidationHook.on_end` for `gen_dashboard` runs. The hook passes a
`SessionTarget` containing every `DashboardTarget` / `ChartTarget` /
`DatasetTarget` the run delivered. Iterate them and run the checks below.

## Target shape

You receive `SessionTarget.targets` — loop over each entry and dispatch on
`type`:

- `dashboard` → inspect chart list via `get_dashboard` / `list_charts`
- `chart` → inspect config via `get_chart`; if supported, run `get_chart_data`
- `dataset` → verify fields via `get_dataset`

Layer A (the builtin hook) has already confirmed each resource **exists**
and is reachable — skip existence checks and focus on config correctness
and data presence.

## Publish success criteria

A dashboard publish is complete when:

- `get_dashboard` returns the dashboard
- the expected chart ids / names are present in `get_dashboard` or `list_charts`
- every chart can be inspected with `get_chart`
- supported `get_chart_data` calls return data or a valid empty result without backend errors
- known values match expected results or tolerances

When these criteria pass, return PASS and stop. Treat additional styling or
structural changes as a separate request unless a concrete failing check is
present.

## Core workflow

For every target in the session:

1. **Dashboard targets**: call `get_dashboard(dashboard_id)` to retrieve the
   chart list. For each chart on the dashboard call `get_chart` and inspect
   config (step 2). When `get_chart_data` is supported, validate data (step 3).
2. **Chart targets (standalone or from step 1)**: verify `chart_type`,
   `metrics`, `x_axis`, `dimensions`, `dataset_id` against the intended design.
3. **Data presence**: when `get_chart_data` is supported, call it on every chart
   to confirm the chart returns data without backend errors. Compare numeric
   values against expected tolerances when expectations are available.
4. **Dataset targets**: call `get_dataset(dataset_id)` to verify schema / SQL /
   column definitions.
5. Remediation planning:
   - only propose or perform a follow-up change for a specific failed
     configuration or data check
   - prefer the smallest supported update for fields the platform can modify
     (`title`, `chart_type`, `metrics`, `x_axis`, `description`, platform SQL)
   - for unsupported wiring changes, report the exact mismatch and required
     follow-up action for the main agent / user
   - if the publish success criteria already pass, return PASS instead of
     planning follow-up changes
6. Return a compact pass / fail report covering **every** target.

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

If `title`, `chart_type`, `metrics`, `x_axis`, `description`, or platform-specific SQL is wrong, fix it with `update_chart` when the tool supports that change. If `dimensions`, `dataset_id`, or unsupported wiring is wrong, report the mismatch and the required follow-up action.

## Publish verification checklist

After a BI publish, run this checklist:

1. Call `get_dashboard` to confirm the dashboard is reachable and to retrieve
   the full chart list.
2. For **every** chart on the dashboard:
   - call `get_chart` to inspect configuration (`chart_type`, `metrics`,
     `x_axis`, `dimensions`, `dataset_id`)
   - verify each configuration field matches the intended design
   - if `get_chart_data` is supported on the active platform, call it to
     confirm the chart returns data without backend errors
   - verify key numeric values match expected results or tolerances when
     those expectations are available
   - if `get_chart_data` is not supported, record the data check as
     unsupported / N/A and rely on configuration inspection plus reference SQL
     or known expectations
3. If any chart fails validation:
   - use `update_chart` only for fields the tool actually supports
   - recreate the chart or panel when the issue is `dimensions`, `dataset_id`,
     or another unsupported wiring change
4. Report both absolute and relative differences when possible.
5. Block rollout when any chart fails configuration or supported data
   validation.

Do **not** skip any charts. `get_chart_data` is required for every chart only
on platforms that actually expose it.

## Output expectations

For each chart, report:

| Column | Description |
|--------|-------------|
| Chart ID | The chart identifier |
| Chart Name | Human-readable title |
| Config Check | PASS if `get_chart` fields match design, FAIL + reason otherwise |
| Data Check | PASS if `get_chart_data` succeeds and expected values match when available; `N/A` when the platform does not support `get_chart_data`; FAIL + reason otherwise |

Final summary: total charts, passed, failed, overall PASS/FAIL decision.
