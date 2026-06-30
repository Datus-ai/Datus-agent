---
name: semantic-sql-history-profiler
description: Optional semantic-model profiling workflow that mines historical SQL and bounded column distributions before YAML authoring
tags:
  - semantic-model
  - sql-history
  - profiling
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
---

# Semantic SQL History Profiler

Use this workflow when the skill is loaded because the user provided historical SQL, success-story SQL, or explicitly asked for profiling. Once loaded, run the profiler before semantic YAML authoring.

## Workflow

1. Call `profile_semantic_model_evidence` before writing semantic model YAML.
   - When historical SQL is provided inline, pass every provided SQL statement via `sql_entries_json` or `sql_queries`; do not choose only representative examples.
   - Use `query_text` only when direct SQL text is unavailable and existing reference SQL must be searched.
   - Use `profile_mode="sql_only"` when the user wants quick generation.
   - Use `profile_mode="lightweight"` when sampled field distributions are helpful.
   - Use `profile_mode="deep"` only when the user explicitly allows a slower exploration.
   - Set conservative bounds such as `max_tables`, `max_columns_per_table`, `top_n`, and `max_profile_seconds`.

2. Use the evidence to decide the model shape:
   - Join relationships from historical SQL become identifier/entity hints.
   - Group-by and filter fields are dimension candidates.
   - Aggregate expressions and numeric profiles are measure candidates.
   - Min/max values, percentiles, and null/fill rates help describe numeric ranges and data quality.
   - Date spans, freshness, and duration profiles help identify usable time columns and common lifecycle intervals.
   - Top values and distinct ratios help detect enum-like categorical columns.
   - Referential coverage and join fanout hints help judge relationship reliability.
   - Common filter templates help capture reusable row-selection semantics without copying long SQL.

3. Put useful distribution evidence into YAML descriptions while keeping them readable:
   - Start with the DDL comment or stable business meaning.
   - Add a compact distribution note when it helps downstream generation:
     - numeric fields: include observed min/max, p50/p90, or null rate when material.
     - date/time fields: include observed span, freshness, or paired duration when useful.
     - low-cardinality categorical fields: include distinct count and representative stable values.
     - enum-like fields: include the full stable code mapping when available.
     - relationship hints: mention low referential coverage or fanout only when it affects join semantics.
     - filter templates: mention common equality/range/text-search/function filters only when backed by history.
   - Convert raw evidence into concise semantic phrasing. Prefer "Order status, 4 distinct values; common values include paid/refund" over dumping profiler JSON.
   - Do not include SQL snippets longer than a short operator/function hint, and do not paste entire top-N lists or long filter examples.
   - Prefer omitting a field over writing a low-confidence or very verbose description.

4. Treat profiling evidence as non-exhaustive.
   - Sampled top values and min/max values are hints, not hard constraints.
   - If evidence conflicts with DDL comments or validation, prefer DDL comments and validated schema.

5. Validate and publish exactly as in the active semantic-model authoring workflow.
