---
name: gen-dashboard
description: Create, update, or inspect BI dashboards by delegating to the gen_dashboard subagent. Use when the user asks to visualize data, build dashboards, add charts, or manage BI platform assets.
tags:
  - dashboard
  - bi
  - visualization
version: 1.1.0
user_invocable: true
---

# BI Dashboard Management

You are helping the user create, update, or inspect BI dashboards on the
configured BI platform. Dashboard creation assumes the required data already
exists in the BI platform's registered database. Data preparation is a separate
`gen_job` / `scheduler` step.

## When to Use This Skill

- User asks to **create a dashboard** or **visualize data**
- User asks to **list, inspect, or modify existing dashboards**
- User asks to **add charts** to a dashboard
- User asks to **create datasets** in the BI platform

## Workflow

### Step 1 — Capture the request

Pull together what the user wants. Use `ask_user` only for missing
critical pieces:

- **What table or SQL dataset** to visualize in the BI-registered database
- **Time range / granularity** (e.g. "last 90 days, daily")
- **Chart types** if they expressed a preference; otherwise let the
  subagent choose
- **Dashboard title** and target platform if multiple are configured

If the user only described raw source data that still needs to be prepared,
do not delegate to `gen_dashboard` yet. Ask them to run the data-preparation
step first or delegate that separate request to `gen_job` / `scheduler`.

### Step 2 — Delegate to gen_dashboard

```
task(type="gen_dashboard",
     prompt="<full request: data concept, time range, chart hint, title>",
     description="<one-line summary>")
```

`gen_dashboard` owns the BI asset workflow:

1. Resolves the BI platform database / datasource.
2. Registers the existing table or SQL dataset.
3. Builds chart / dashboard assets against the BI platform.
4. Finishes the BI asset creation flow; `bi-validation` runs automatically through `ValidationHook.on_end`.

### Step 3 — Report results

When the subagent returns:

- Surface the dashboard ID / URL.
- Offer to make modifications via another `task(type="gen_dashboard", ...)`
  call.
- For modifications (add a chart, retitle, etc.), pass the dashboard
  identifier in the new prompt.

## Hard rules

- **Never move data yourself.** It is not this skill's job. If data needs to
  be created or refreshed, handle that as a separate `gen_job` / `scheduler`
  request before dashboard creation.
- **Never explore the source DB on the user's behalf** before delegating.
  If the user has not specified the serving table or SQL dataset clearly,
  ask them — do not run `read_query` to guess.

## Chart Type Hints (pass-through to subagent)

| Chart Type | Best For |
|------------|----------|
| `bar` | Comparing categories, rankings |
| `line` | Trends over time, time series |
| `pie` | Part-to-whole relationships (< 7 categories) |
| `table` | Detailed data, exact values |
| `big_number` | Single KPI metrics, totals |
| `scatter` | Correlation between two metrics |

If the user did not specify, let the subagent choose based on the provided
serving table or SQL dataset.
