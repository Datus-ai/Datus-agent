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

1. Identify the target dataset, chart, or dashboard.
2. Publish or refresh the BI object.
3. Confirm the BI object exists and is reachable.
4. Compare a small set of key metrics against expected values or tolerances.
5. Return a compact pass / fail report.

## Checklist

- Publish verification checklist: [references/publish-checklist.md](references/publish-checklist.md)

Use the checklist to drive direct BI verification with existing BI tools. Keep the metric set small and write down the expected values or tolerances in the working notes for the current task.

## Output expectations

- target BI object
- publish / refresh status
- observed metrics
- expected values or tolerances
- pass / fail decision
