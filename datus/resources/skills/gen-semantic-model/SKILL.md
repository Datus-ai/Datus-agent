---
name: gen-semantic-model
description: Generate MetricFlow semantic models from database tables with validation and Knowledge Base publishing
tags:
  - semantic-model
  - metricflow
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# Generate Semantic Model Skill

Create production-ready MetricFlow semantic model YAML for one or more database tables, validate it, and publish it to the Knowledge Base.

## Workflow

1. **Understand target tables**
   - Identify the table or tables from the user request.
   - Use `describe_table`, `get_table_ddl`, and relationship tools as needed.
   - Use `ask_user` only when a critical modeling choice cannot be inferred.

2. **Model columns**
   - Choose one primary time dimension when a reliable time column exists.
   - Define `type: TIME` only for a physical DATE/TIME/TIMESTAMP column, or for a SQL expression / `sql_query` alias that is guaranteed to return a DATE/TIME/TIMESTAMP value.
   - Do not mark numeric surrogate keys such as `*_date_sk`, `*_date_key`, `*_dt_key`, or integer YYYYMMDD keys as `type: TIME`; model them as identifiers or categorical dimensions unless you explicitly convert them to a real date.
   - If a fact table uses a calendar/date dimension to derive its business date, prefer a `sql_query` data source that joins the date dimension, selects the real date column with a clear alias, and uses that alias as the primary time dimension and measure `agg_time_dimension`.
   - Define identifiers for primary keys and join keys.
   - Define measures only for reusable aggregations.
   - Define dimensions for grouping/filtering fields.
   - Do not define the same column/name under both `identifiers` and `dimensions`. Use identifiers for
     primary/join keys and dimensions for grouping/filtering fields.
   - Use `expr: "1"` for row-count measures with `agg: COUNT`.
   - For measures, use `agg` for the aggregation type; do not add a `type` field to measure entries.

3. **Write YAML**
   - Save files under the semantic model directory shown in the system prompt.
   - Preferred path shape: `subject/semantic_models/<current_datasource>/{table_name}.yml`.
   - Use paths relative to the filesystem sandbox root.
   - For multiple related tables, write all relevant semantic model files before validation.
   - Keep each YAML document to one MetricFlow object type. Semantic model generation should write `data_source:` documents; do not put a top-level `metrics:` list beside `data_source:` in the same document.
   - If explicit metric definitions are needed, write them through the metrics generation workflow as separate `metric:` documents.

4. **Validate and fix**
   - Call `validate_semantic(scope="semantic_model")`.
   - If validation fails, use `edit_file` to fix the YAML and call `validate_semantic` again.
   - Repeat until `validate_semantic` succeeds.

5. **Publish**
   - After validation succeeds, call `end_semantic_model_generation` with all generated semantic model file paths.
   - This publishes the validated semantic models to the Knowledge Base.
   - If you miss this tool call, the host will use the final JSON `semantic_model_files` to validate and publish before reporting success.
   - Validation passing is the publish gate; no additional approval step is needed.

## Rules

- Do not publish before `validate_semantic` succeeds.
- After validation succeeds, prefer publishing directly through `end_semantic_model_generation`; final JSON `semantic_model_files` is the host fallback.
- Do not manually write Knowledge Base summary files.
- Keep YAML focused on semantic model definitions; avoid markdown or explanatory prose in YAML files.
- Keep MetricFlow document boundaries valid: one top-level object type per YAML document.
- Use existing semantic models when present; edit them only when needed for the requested metrics or relationships.
