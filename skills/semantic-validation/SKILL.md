---
name: semantic-validation
description: Validate semantic layer changes by checking semantic config health, metric discoverability, queryability, and consistency between metric results and warehouse expectations
tags:
  - semantic-layer
  - metrics
  - validation
  - data-engineering
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Semantic Validation

Use this skill after changing semantic models or metrics. Reuse existing semantic tools instead of inventing custom checks.

## Preferred tools

- `validate_semantic`
- `list_metrics`
- `get_dimensions`
- `query_metrics`

## Core workflow

1. Run `validate_semantic`.
2. Confirm the expected metrics are visible in `list_metrics`.
3. Query a small set of key metrics with representative dimensions and time filters.
4. Compare semantic results against warehouse expectations or a direct SQL baseline.
5. Return a compact pass / fail report and stop rollout if semantic results drift materially.

## Checklist

Use the core workflow steps above as the checklist. No external reference document is needed — the preferred tools provide the verification surface directly.

## Output expectations

- semantic validation status
- missing or broken metrics
- query results for key checks
- mismatch summary against expected values or SQL baseline
