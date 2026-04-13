---
name: data-schema-profiling
description: Profile raw or staged tables for data engineering tasks by inspecting schema, sampling risky columns, testing casts, and summarizing findings for downstream SQL generation or contract validation
tags:
  - data-engineering
  - schema
  - profiling
  - explore
  - staging
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Data Schema Profiling

Use this skill when a data engineering task depends on understanding what a table actually contains before writing transformations. This skill is especially useful for raw ingestion, staging contracts, and unknown timestamp/string formats.

## When to use this skill

Activate for tasks that require any of:

- Inspecting a new source table before writing SQL
- Verifying whether contract types match the observed source types
- Deciding how to cast dates / timestamps / numerics safely
- Producing a short profile that another SQL-generation step will consume

Do not use this skill for general repository exploration. This skill is for **table-centric profiling**.

## Core workflow

1. Identify the exact target tables before exploring.
2. Use schema inspection first.
3. Sample only columns that are likely to be risky:
   - timestamps / dates
   - ids with possible formatting drift
   - free-text categorical columns used in CASE logic
4. Run a small number of cast probes when type conversion is relevant.
5. Summarize findings as facts, not recommendations.

## Profiling priorities

- Start with `describe_table`
- Sample columns rather than full rows when possible
- Prefer lightweight `COUNT`, `COUNT(DISTINCT)`, and `TRY_CAST` checks
- Avoid broad exploration outside the explicitly scoped tables

## Output expectations

The useful output of profiling is a compact fact set:

- observed source types
- risky columns
- cast success/failure observations
- obvious nullability or formatting surprises

If you need examples of what to inspect, read [references/checklist.md](references/checklist.md).

