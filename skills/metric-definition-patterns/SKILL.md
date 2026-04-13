---
name: metric-definition-patterns
description: Define and review analytical metrics for marts and intermediate tables, including safe division, windowed metrics, ratios, retention, and aggregation semantics
tags:
  - metrics
  - marts
  - aggregation
  - analytics
  - data-engineering
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Metric Definition Patterns

Use this skill when implementing analytical metrics in intermediate or marts SQL. The goal is to keep business calculations stable, explicit, and contract-aligned.

## When to use this skill

Activate when writing SQL that includes:

- ratios or percentages
- rolling / windowed aggregates
- funnel metrics
- retention or adoption metrics
- scoring or weighted summaries

## Rules of thumb

- Name metrics explicitly and consistently with the contract.
- Prefer deriving metrics from well-defined upstream grains.
- Use safe division for ratios.
- Do not silently change precision by casting to integers unless the contract says so.
- Distinguish between:
  - counts
  - sums
  - averages
  - rates
  - weighted scores

## Common failure modes

- integer truncation in averages or rates
- using the wrong denominator
- mixing entity grain and event grain in one calculation
- using non-deterministic dates in time-window metrics
- leaking helper columns into the final output

## Working style

Before writing a complex metric table:

1. Identify the target grain.
2. List each metric with its numerator / denominator / grouping logic.
3. Build helper CTEs at a stable grain.
4. Assemble the final metrics only after the intermediate grains are correct.

For concrete reminders, read [references/patterns.md](references/patterns.md).

