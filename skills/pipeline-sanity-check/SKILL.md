---
name: pipeline-sanity-check
description: Sanity-check layered SQL pipelines by verifying upstream availability, table grain assumptions, output completeness, and likely cascading failure points before or after generation
tags:
  - data-engineering
  - pipeline
  - validation
  - dependency
  - dags
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Pipeline Sanity Check

Use this skill when reviewing or validating a layered SQL pipeline before execution, after generation, or when debugging failures in downstream tables.

## When to use this skill

Activate when you need to answer questions like:

- Are all required upstream tables available?
- Does this table depend on same-layer outputs that are not materialized yet?
- Is the final output missing required columns?
- Which failures are root causes versus cascaded failures?

## Sanity-check workflow

1. Confirm the target table and layer.
2. Enumerate direct upstream dependencies.
3. Classify each dependency:
   - upstream materialized and queryable
   - declared but missing
   - same-layer and execution-order dependent
4. Check whether the output contract is complete.
5. Separate root-cause failures from cascade failures.

## What this skill is good at

- debugging pipeline execution order issues
- spotting incomplete upstream state
- explaining why one failure propagates to many tables
- checking whether a downstream mismatch is likely caused by an earlier broken node

## What this skill is not

- not a generic SQL optimization guide
- not a data profiling skill
- not a replacement for table-level contract validation

For a practical checklist, read [references/checks.md](references/checks.md).

