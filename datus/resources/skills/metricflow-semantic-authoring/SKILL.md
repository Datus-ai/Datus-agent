---
name: metricflow-semantic-authoring
description: MetricFlow semantic model authoring specification — YAML structure, field classification, validation, and Knowledge Base publishing
tags:
  - semantic-model
  - metricflow
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# MetricFlow Semantic Authoring

Author production-ready MetricFlow semantic model YAML for one or more database tables, validate it, and publish it to the Knowledge Base.

## Column analysis

- Use `analyze_column_usage_patterns(table_name)` to discover how columns are used in historical SQL — filter operators, functions, and actual filter examples. This is strongly recommended for every table unless profiler evidence already covers it.
- When profiler evidence from `profile_semantic_model_evidence` is available, use it as the primary source for join hints, commonly filtered/grouped dimensions, aggregate candidates, and concise usage hints; use `analyze_column_usage_patterns` only as a fallback or to fill a narrow gap.

## Field classification rules

Classify each column by actual data type and analytical usage, not by display convenience alone:

- Put DECIMAL/NUMERIC/INTEGER/FLOAT/DOUBLE or equivalent fields in `measures` when they represent measured quantities, scores, rates, prices, amounts, durations, counts, or percentages and historical SQL uses them in arithmetic, range predicates, numeric ordering, or numeric aggregates such as `AVG`, `SUM`, `MIN`, `MAX`, `STD`, `VARIANCE`, `CORR`, or `COVAR`.
- Do not model an aggregatable numeric business value as a categorical dimension just because it is selected, displayed, filtered, sorted, or appears in a WHERE clause. A DECIMAL field used by `AVG(<field>)` should be a measure with an appropriate aggregate such as `AVG`.
- Put text/enums/labels/booleans and numeric-coded category fields in `dimensions` when their values identify groups or labels and averaging or summing the raw value has no business meaning. Explain code semantics (e.g. "status: 1=Active, 2=Inactive") in the description.
- Put primary keys, foreign keys, unique entity IDs, and opaque IDs in `identifiers`. Do not classify identifiers as measures merely because their physical storage type is integer.
- Do not define the same column/name under both `identifiers` and `dimensions`.
- If schema type and SQL usage appear to conflict, prefer concrete SQL usage plus column comments/profiler evidence, and ask only when the modeling choice is business-critical.
- Define measures only for reusable aggregations. Use `expr: "1"` for row-count measures with `agg: COUNT`. For measures, use `agg` for the aggregation type; do not add a `type` field to measure entries.

## Time dimension correctness

- Define `type: TIME` only for a physical DATE/TIME/TIMESTAMP column, or for a `sql_query` alias / SQL expression that is guaranteed to return a DATE/TIME/TIMESTAMP value.
- Do NOT mark numeric surrogate keys such as `*_date_sk`, `*_date_key`, `*_dt_key`, or integer YYYYMMDD keys as `type: TIME`. Model them as identifiers or categorical dimensions unless you explicitly convert them to a real date.
- If a fact table derives its business date by joining a calendar/date dimension, prefer a `sql_query` data source that joins the date dimension, selects the real date column with a clear alias, and uses that alias as the primary time dimension.
- `agg_time_dimension` on measures must point to that real date/time dimension, not a numeric surrogate key.
- Include a primary time dimension only when a reliable DATE/TIME/TIMESTAMP column or expression exists; never force one.

## Descriptions

Populate `description` fields for ALL measures, dimensions, and identifiers by COMBINING all available information:

- Start with DDL comments and **preserve their original language — DO NOT TRANSLATE** (Chinese comments stay Chinese).
- Append usage patterns and filter examples from `analyze_column_usage_patterns`.
- Append compact observed distribution evidence from `profile_semantic_model_evidence` when available: null rate, numeric ranges and percentiles, date spans/freshness/duration, distinct counts, representative stable values, referential coverage, common filter templates, and enum/code mappings.
- Describe what a column means first, then add concise observed evidence. Avoid long SQL snippets or procedural query instructions.
- ALWAYS wrap `description` values in double quotes (`"`). Escape special characters: `"` → `\"`, `\` → `\\`.

Example:

```yaml
- name: status
  type: CATEGORICAL
  expr: status
  description: "Order status; commonly filtered with =/IN; representative values include paid/refund"
```

## Multi-table workflow

When the request covers multiple tables:

1. Batch retrieve DDL with `get_multiple_tables_ddl`.
2. Discover join relationships with `analyze_table_relationships`. The tool uses a three-tier strategy: DDL foreign keys (high confidence), historical SQL JOIN patterns (medium), column name inference (low).
3. Generate one YAML file per table. Use the `entity` field to reference other tables in **singular form** (`customer`, not `customers`); `type: PRIMARY` for the table's primary key, `type: FOREIGN` for columns that join to other tables. Linked identifiers share the same `name` (one PRIMARY, one FOREIGN).
4. Write ALL files before validation, then validate them together.

## File paths

- Save files under `subject/semantic_models/<current_datasource>/{table_name}.yml`, relative to the filesystem sandbox root (use the exact prefix shown in the system prompt Workspace section).
- If a semantic model already exists, update it with `edit_file` instead of rewriting.

## Validate and publish

1. Call `validate_semantic(scope="semantic_model")`. If validation fails, fix the YAML with `edit_file` and validate again; repeat until it passes. Common errors:
   - PostgreSQL column case sensitivity: wrap uppercase column names in double quotes (e.g., `expr: '"SP_POP_TOTL"'`)
   - Column not found: check column names match the DDL exactly
   - Duplicate semantic element name: remove the column from either `identifiers` or `dimensions`
   - Invalid YAML syntax: check indentation and quoting
2. Only after validation succeeds, publish via `end_semantic_model_generation` with all generated file paths. Do not publish before validation passes; do not manually write Knowledge Base summary files.

## Document structure rules

- A semantic model file defines `data_source:` document(s). Do NOT add a top-level `metrics:` list to the same YAML document — MetricFlow requires exactly one top-level object type per YAML document. Explicit `metric:` documents belong to the metrics workflow.
- Do not rely on `create_metric: true` for persisted metrics; runtime metrics are not part of the Knowledge Base metric catalog.
- All `name` fields follow `^[a-z][a-z0-9_]*[a-z0-9]$` (snake_case, no double underscores).
- Choose exactly ONE of `sql_table` (schema-qualified) or `sql_query` (databases without schema, or custom joins).
- Preserve original language everywhere (Chinese text remains Chinese) for optimal vector search.

## MetricFlow semantic model structure specification

```yaml
data_source:
  # === Required Fields ===
  name: string (required)             # Data source name, pattern: ^[a-z][a-z0-9_]*[a-z0-9]$

  # === Optional Metadata Fields ===
  description: string                 # Data source description
  display_name: string                # Display name
  owners:                             # List of owners
    - email@domain.com
  tier: string|integer                # Data tier

  # === Data Source Definition (Choose ONE) ===
  sql_table: schema.table_name        # For databases with schema support (PostgreSQL, Snowflake, Redshift, BigQuery)
  # OR
  sql_query: |                        # For databases without schema (SQLite, DuckDB) or custom queries
    SELECT * FROM table_name
    WHERE condition = 'value'

  # === Core Components ===
  measures:                           # Measure definitions (array)
    - name: string (required)         # Measure name
      agg: enum (required)            # SUM|MIN|MAX|AVERAGE|COUNT_DISTINCT|COUNT|PERCENTILE|MEDIAN|SUM_BOOLEAN
      description: string             # Description - put extracted comments here
      expr: string|integer|boolean    # Expression, defaults to column name
      agg_time_dimension: string      # Aggregation time dimension
      agg_params:                     # Aggregation parameters (for PERCENTILE)
        percentile: number
        use_discrete_percentile: boolean
        use_approximate_percentile: boolean
      create_metric: boolean          # Runtime MetricFlow metric only; not persisted to the Knowledge Base metric catalog
      create_metric_display_name: string
      non_additive_dimension:         # Non-additive dimension (snapshot/balance measures)
        name: string
        window_choice: MIN|MAX
        window_groupings: [string]

  dimensions:                         # Dimension definitions (array)
    - name: string (required)
      type: enum (required)           # CATEGORICAL|TIME
      description: string             # Description - put extracted comments/enums here
      expr: string|boolean
      is_partition: boolean
      type_params:                    # Required for TIME type
        is_primary: boolean           # Exactly one primary time dimension per data_source
        time_granularity: enum (required)  # DAY|WEEK|MONTH|QUARTER|YEAR
        time_format: string
        validity_params:              # For SCD Type 2
          is_start: boolean
          is_end: boolean

  identifiers:                        # Identifier definitions (array)
    - name: string (required)
      type: enum (required)           # PRIMARY|UNIQUE|FOREIGN|NATURAL
      description: string
      expr: string|boolean
      entity: string                  # Associated entity name (singular form)
      role: string
      identifiers:                    # Composite identifiers
        - name: string
          expr: string|boolean
          ref: string

  # === Mutability Configuration ===
  mutability:
    type: enum (required)             # IMMUTABLE|APPEND_ONLY|FULL_MUTATION|DS_APPEND_ONLY
    type_params:
      min: string
      max: string
      update_cron: string
      along: string
```

## PostgreSQL column name case sensitivity (CRITICAL for PostgreSQL)

- PostgreSQL converts unquoted identifiers to lowercase. If a column was created with quotes (e.g., `"SP_POP_TOTL"`), it retains uppercase and MUST be quoted when queried.
- In `expr` fields, wrap uppercase column names with double quotes: wrong `expr: SP_POP_TOTL`, correct `expr: '"SP_POP_TOTL"'`.
- Check the DDL: if column names contain uppercase letters, they likely need quoting. This applies to measures, dimensions, and identifiers `expr` fields.

## Example

```yaml
data_source:
  name: my_transactions
  description: Transaction data with customer and order details
  owners:
    - data-team@company.com

  sql_table: analytics.transactions

  measures:
    - name: total_amount
      agg: SUM
      expr: transaction_amount
    - name: transaction_count
      agg: SUM
      expr: "1"
    - name: unique_customers
      agg: COUNT_DISTINCT
      expr: customer_id

  dimensions:
    - name: transaction_date
      type: TIME
      type_params:
        is_primary: true
        time_granularity: DAY
    - name: payment_method
      type: CATEGORICAL
    - name: is_refund
      type: CATEGORICAL
      expr: "CASE WHEN amount < 0 THEN 'Yes' ELSE 'No' END"
      description: "Refund status (1:Refunded, 0:Normal)" # Keep original enums in description

  identifiers:
    - name: transaction
      type: PRIMARY
      expr: transaction_id
    - name: customer
      type: FOREIGN
      expr: customer_id
    - name: order
      type: FOREIGN
      expr: order_id

  mutability:
    type: APPEND_ONLY
```
