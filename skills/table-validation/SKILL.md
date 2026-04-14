---
name: table-validation
description: Validate warehouse tables after DDL or writes using existing database tools for schema checks, row-count gates, null ratios, ranges, accepted values, regex checks, and duplicate detection
tags:
  - data-engineering
  - validation
  - schema
  - data-quality
  - sql
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Table Validation

Use this skill when a table has been created or written and you need to verify both:

- schema contract
- post-write data quality

This skill assumes you can use existing database tools such as `describe_table`, `get_table_ddl`, and `read_query`. Prefer direct validation queries over prose reasoning.

## When to use this skill

Activate when you need to validate one or more of:

- object existence
- missing / extra columns
- type or nullability mismatches
- row-count gates
- null ratios
- numeric ranges
- accepted values
- regex / format rules
- uniqueness / duplicate keys

## Core workflow

1. Confirm the exact target table and expected grain.
2. Run schema checks first.
3. Run row-count and cheap aggregate checks next.
4. Run duplicate and format checks after the cheap gates pass.
5. Return a compact pass / fail report with observed values and failing sample filters when useful.

## Checklist

- Validation sequence: [references/checklist.md](references/checklist.md)

Use the checklist to decide which queries to run with existing tools. Do not wait for prebuilt templates; write the smallest useful validation SQL directly with `read_query`.

## Output expectations

At minimum, return:

- target table
- executed checks
- observed values
- pass / fail status
- blocking issues that must be fixed before downstream use
