---
name: gen-metrics
description: Generate MetricFlow metrics from natural language business descriptions
tags:
  - metrics
  - metricflow
version: "1.2.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_metrics
---

# Generate Metrics Skill

Guide the user through metric generation using natural language business descriptions.

## Phase 0: Discovery — Scan Existing Assets

If the user prompt already includes `Existing Metric Catalog JSON`, treat it as the Phase 0 catalog loaded by the bootstrap host. Do not call `list_metrics()` just to rediscover existing metrics. Otherwise, before anything else, call `list_metrics()` to get all metrics already in the knowledge base. Build an existing metric catalog JSON array with each metric's exact `name`, `type`, `description` when available, `subject_path` when available, and structured compatibility fields such as base measures, dimensions, entities, and semantic model when available. Use this throughout the remaining phases to:
- **Skip redundant work** — don't recreate metrics that already exist
- **Reuse existing measures** — reference measures from existing models instead of creating duplicates
- **Detect conflicts** — warn the user if a proposed metric name collides with an existing one
- **Enable derived/ratio metrics** — know which metrics can serve as building blocks for more complex definitions

Metric names are datasource-local business keys. Within the current datasource, every persisted `metric.name` must be unique; `subject_tree` is only a display/classification path and must not be used to create a second metric with the same name. If a proposed name already exists:
- Reuse or skip it when it has the same business definition.
- In interactive mode, ask the user before replacing or renaming a conflicting definition.
- In batch/bootstrap mode, infer a more specific English snake_case business name from the SQL/question/table context when the same name would mean a different calculation.

Only inspect and edit semantic model YAML files under the current datasource directory shown in the system prompt, such as `subject/semantic_models/<current_datasource>/...`. Do not reuse or sync YAML files from sibling datasource directories; those files are outside the active MetricFlow adapter scope.

## Phase 1: Understand Intent

Analyze the user's request and confirm the generation scope before proceeding. When `ask_user` is available, call it to confirm the metric name(s), business meaning, and calculation logic. When `ask_user` is not available (for example workflow or batch mode), infer from the provided SQL/request and stop only if the scope is materially ambiguous.

### Input Mode Detection

- **Single mode**: User describes one metric or provides one SQL → follow Step 1a–1d below
- **Batch mode**: User provides multiple SQL queries (pasted directly, or a CSV file path containing `question` + `sql` columns) → follow Step 1-batch below

### Single Mode: Step 1a–1d

**Step 1a: Inspect the table** — Call `describe_table(table_name)` to understand the columns and types. Optionally call `read_query` to sample data.

**Step 1b: Ask for reference SQL (optional)** — When `ask_user` is available, use it to ask:
> "Do you have any existing SQL queries for this table that show the aggregations you care about? You can paste them here, or skip if not available."

When `ask_user` is not available, skip this question and infer SQL/aggregation context from the user's request, attached files, or discovered query/table evidence. If that is not enough, stop and explain the missing information instead of calling `ask_user`.

If the user provides SQL, parse it to extract:
- Final business output expressions (e.g., `SUM(amount) / COUNT(DISTINCT user_id) AS arppu` → candidate metric `arppu`)
- Aggregation functions + columns that the final metric depends on (e.g., `SUM(amount)` → candidate measure `total_amount`, `COUNT(*)` → candidate measure `record_count`)
- GROUP BY columns → recommended dimensions
- WHERE conditions → potential metric constraints

If the provided SQL contains no metric-producing output, keep filter-only or detail-query evidence as filters, dimensions, segments, or view evidence instead of generating fake metrics.

If the user skips, proceed to Step 1c using only table structure and the user's description.

**Step 1c: Propose metric candidates** — Based on the table structure, reference SQL (if provided), and user's request, identify potential metric scenarios. See "Metric type detection rules" below.

**Step 1d: Confirm scope** — when `ask_user` is available, call it to confirm and present proposed metrics with `multi_select: true` (see Step 1-batch-d for format). If `ask_user` is not available, proceed with the confirmed/inferred scope from the input.

### Batch Mode: Step 1-batch

**Step 1-batch-a: Parse SQL queries**
- The input may contain multiple SQL queries in various forms:
  - **Direct paste**: multiple SQL statements in the prompt
  - **File path**: user provides a path — call `read_file` to load it, then parse by file type:
    - `.sql`: split by `;` or blank-line separators to extract individual statements
    - `.csv` / `.tsv`: identify the SQL column by header name (common names: `sql`, `query`, `SQL`, `statement`) or by content heuristic (column values contain SQL keywords like `SELECT`, `FROM`, `GROUP BY`). The description/question column is any remaining text column. If column roles are ambiguous, call `ask_user` when available to confirm which column is SQL; otherwise stop and explain the missing column mapping.
    - Other formats: call `ask_user` when available to clarify the file structure before proceeding; otherwise stop and explain the supported file formats or required structure.
- Parse all SQL queries from the input
- Call `describe_table` for each unique table found in the SQL queries

**Step 1-batch-b: Mine metric candidates from SQL ASTs**

If the user prompt already includes `Precomputed Metric Candidate Plan JSON`, use that as the result of batch metric candidate mining for the current SQL batch. Do not call `analyze_metric_candidates_from_history` again unless the supplied JSON is malformed or clearly insufficient. Otherwise, call `analyze_metric_candidates_from_history` with all parsed SQL queries and `existing_metric_catalog_json` from Phase 0. Use its output to preserve final business metric expressions and their dependencies:

1. **Preserve final output metrics** — SQL aliases and final SELECT expressions are the primary metric candidates.
2. **Keep base measures as dependencies** — base measures support the final metric but do not replace it.
3. **Deduplicate by business metric** — merge repeated aliases/normalized expressions across SQL files while preserving source evidence.
4. **Separate non-metrics** — filter-only/detail SQL belongs in `non_metric_evidence`, not metric YAML.
5. **Respect modeling classifications** — if `query_classification` is `metric_plus_derived_datasource` or `derived_datasource_recommendations` is non-empty, do not generate a direct metric from `blocked_direct_metric_candidates`; first model the recommended `sql_query` data source or materialized view, then define metrics on that data source.
6. **Choose business-safe names** — if a candidate has `requires_name_translation: true`, treat `name` as a technical fallback only. Also inspect every `source_alias`: when the alias appears generated or lacks business meaning, do not use it as the final MetricFlow name. In interactive mode, ask the user to confirm if the business meaning is unclear; in batch/bootstrap mode, infer a clear English snake_case name from the SQL expression, question, table/column context, and external knowledge without stopping.
7. **Preserve SQL literal values** — if `literal_mappings` is present, keep the literal `value` exactly as it appears in SQL predicates/CASE/sql_query output. Only MetricFlow object names may be translated or normalized.
8. **Preserve SQL time grain** — if `time_grain_evidence` is present, expose an equivalent time dimension in any derived data source. Do not replace a projected date such as `CURDATE() AS part_dt` or `DATE(create_time) AS part_dt` with raw `create_time` as the primary time dimension.
   Define `type: TIME` only for physical DATE/TIME/TIMESTAMP columns or SQL expressions / `sql_query` aliases guaranteed to return DATE/TIME/TIMESTAMP values. Numeric surrogate keys such as `*_date_sk`, `*_date_key`, `*_dt_key`, or integer YYYYMMDD keys must be identifiers or categorical dimensions unless converted to a real date.
9. **Preserve post-aggregation constraints** — if `post_aggregation_constraints` is present, keep each HAVING/post-aggregation condition as a query constraint, metric usage note, or later derived data source. Do not silently drop it or push it into a base measure.
10. **Cross-reference with Phase 0** — remove any candidate that already exists in the knowledge base with the same definition; rename candidates whose name collides with a different existing definition.
11. **Separate derived metrics** — treat `derived_metric_candidates` as second-stage metrics over existing metrics. Do not mix them into base semantic model or measure generation.
12. **Ignore passthrough references** — entries in `identity_metric_references` show existing metrics selected without new business formula; do not generate new metrics for them.
13. **Honor alias mappings** — when the plan includes `metric_aliases` or `source_alias_mappings`, use the `canonical_name` as the generated/reused metric name. Do not generate a second metric just because another SQL uses a different alias for the same formula.

**Step 1-batch-c: Business metric principle**

From N SQL queries, propose a focused set of business metrics. Ask yourself for each candidate:
- Is this a final output a business user would recognize as a KPI?
- Are its base measures complete enough to validate and dry-run?
- Should the evidence be a metric, or only a filter/dimension/segment/view definition?
- Does the tool say the metric depends on a ranked/windowed CTE or other derived data source? If yes, generate the derived data source first instead of forcing a direct metric.
- Are SQL literals, output time grain, and HAVING/post-aggregation constraints preserved from the tool evidence?

**Step 1-batch-d: Confirm with the user when possible**
- When `ask_user` is available, present the mined business metric candidates as **options** with `multi_select: true`
- Pass `questions` as an actual array argument, not a JSON string. Example tool arguments:
  ```json
  {
    "questions": [
      {
        "title": "Metrics",
        "question": "I analyzed N SQL queries and identified the following metric candidates. Select which ones to generate:",
        "options": ["paid_arppu - SUM(paid_amount) / COUNT(DISTINCT user_id)", "gross_margin_rate - (SUM(revenue) - SUM(cost)) / SUM(revenue)"],
        "multi_select": true
      }
    ]
  }
  ```
- Clearly show how many SQL queries were analyzed, how many metric candidates were extracted, and which candidates were skipped as non-metric evidence.
- When `ask_user` is not available, proceed with the mined metrics only if the input makes the scope unambiguous; otherwise stop and explain what needs to be provided.

### Metric type detection rules

1. **Simple counting + filter**: "How many completed orders" → conditional measure in the semantic model + `measure_proxy` metric referencing that measure by string
2. **Aggregation + filter**: "Total revenue from premium customers" → conditional measure in the semantic model + `measure_proxy` metric referencing that measure by string
3. **Ratio**: "Order completion rate", "Refund rate", "Revenue share", "Revenue per user" → `ratio` type
4. **Expression**: "Gross profit", "Gross margin rate" → `expr` type combining measures
5. **Derived**: "ROAS over existing revenue and ad_spend metrics" → `derived` type combining metrics
6. **Cumulative**: "Running total of revenue", "MTD sales", "Year-to-date signups" → `cumulative` type

Detection keywords:
- "running total", "MTD", "YTD", "cumulative", "to-date" → cumulative
- "rate", "ratio", "percentage of", "share of" → ratio
- "per", "divided by", "average ... per" → ratio or expr depending on expression shape
- "list all...", "show me the..." → not a metric, better suited for `gen_sql`

**IMPORTANT**: Do NOT proceed to Phase 2 with materially ambiguous scope. Use `ask_user` when available; otherwise stop and explain what information is needed.

## Phase 2: Ensure Semantic Model Exists

For each table involved in the metric:

### 2a. Check Existing Model

1. Call `check_semantic_object_exists(name="{table_name}", kind="table")` to check if a semantic model exists.
2. **If the semantic model exists:**
   - Use `read_file` to read the existing semantic model YAML
   - Verify that it contains the measures and dimensions needed for this metric
   - If missing measures/dimensions, use `edit_file` to add them, then `validate_semantic`

### 2b. Create Missing Model

If the semantic model is missing, follow the `gen-semantic-model` workflow when that skill is available. In brief: inspect table structure with `describe_table`, discover joins with `analyze_table_relationships` when multiple tables are involved, use `analyze_column_usage_patterns` for likely measures and dimensions, write the semantic model YAML under the semantic model directory shown in the system prompt, then run `validate_semantic` and fix issues until it passes before continuing.

### 2c. Multi-Table / JOIN SQL Modeling

When the metric involves multiple tables (detected from JOIN in SQL or user description), choose the modeling strategy based on SQL complexity:

**Strategy A: Identifier-based JOIN (default — use when possible)**

Use when: simple equi-JOIN between 2-3 tables via foreign keys, ≤ 2 JOIN hops.

- Each table gets its own `data_source` with `sql_table`
- Tables are linked via matching `identifiers` (same `name`, one PRIMARY, one FOREIGN)
- Use `analyze_table_relationships` results to set up correct identifier linkages
- Example: `orders.customer_id` (FOREIGN) links to `customers.customer_id` (PRIMARY) — both identifiers share `name: customer`
- MetricFlow engine automatically resolves the JOIN path at query time

**Strategy B: `sql_query` pre-joined data source (complex cases)**

Use when: non-equi JOINs, > 2 hop joins, subqueries, LATERAL/CROSS joins, complex ON conditions, or window functions in the JOIN.

- Create a single `data_source` with `sql_query` containing the pre-joined SQL
- Flatten the result: measures and dimensions reference the output columns directly
- Example:
  ```yaml
  data_source:
    name: order_customer_summary
    sql_query: >
      SELECT o.order_id, o.amount, o.order_date,
             c.name as customer_name, c.segment
      FROM schema.orders o
      JOIN schema.customers c ON o.customer_id = c.id
    measures:
      - name: total_revenue
        agg: SUM
        expr: amount
    dimensions:
      - name: customer_name
        type: CATEGORICAL
      - name: order_date
        type: TIME
        type_params:
          is_primary: true
          time_granularity: DAY
  ```
- Trade-off: dimensions from the pre-joined query are NOT reusable by other data sources (no identifier linkage). Only use this when Strategy A cannot handle the complexity.

**Decision rule**: Default to Strategy A. Switch to Strategy B only if the JOIN cannot be expressed as simple identifier matching (e.g., composite keys, non-equi conditions, 3+ hop joins, or subquery-based logic).

## Phase 3: Generate and Validate

**File paths**: All `write_file` / `edit_file` / `read_file` calls use paths relative to the filesystem sandbox root. Always use the semantic model directory shown in the system prompt so subsequent reads find the file. For example:
- Semantic model: `subject/semantic_models/<current_datasource>/{table_name}.yml`
- Metric file: `subject/semantic_models/<current_datasource>/metrics/{table_name}_metrics.yml`

Bare filenames are silently normalized by the host, but the prefixed form is preferred for clarity. Absolute paths are also tolerated.
Do not read, edit, or pass `metric_file` / `semantic_model_files` paths from another datasource directory such as `subject/semantic_models/other_datasource/...`.

1. **Check existing**: Call `check_semantic_object_exists(name="{metric_name}", kind="metric")` for each metric confirmed in Phase 1. If it already exists, inform the user and skip it.

2. **Update semantic model safely**: When a semantic model YAML already exists, read it first and preserve all existing identifiers, measures, dimensions, `sql_table`, and `sql_query`. Add only missing measures/dimensions needed by the new metric. Never rewrite an existing semantic model with a smaller subset of columns.

3. **Write metric YAML**: Use `write_file` to save each metric definition to `subject/semantic_models/<current_datasource>/metrics/{table_name}_metrics.yml`. If the metrics YAML already exists, read it first and preserve all existing `metric:` entries; append only missing new metrics. Do not rewrite the file with a smaller subset, and do not delete existing metric YAML entries just because they already exist in the KB catalog.
   - For `measure_proxy`, keep `type_params.measure` as a string measure name.
   - For filtered metrics, add a dedicated conditional measure to the semantic model first, then reference that measure from the metric YAML.
   - Each generated metric must be an explicit named top-level `metric:` YAML document. Do not emit unnamed `metric:` blocks or wrap metrics inside another object.

4. **Validate (MUST PASS)**: Call `validate_semantic` to check the metric YAML.
   - If validation fails, fix errors with `edit_file` and retry until it **passes**.
   - **Do NOT proceed to Phase 4 until validation passes.** No exceptions.

5. **Dry-run SQL**: Call `query_metrics(metrics=["{metric_name}"], dry_run=True)` to generate the SQL.
   - If the source SQL groups by dimensions or a time grain, also dry-run the generated metric set with matching `dimensions` / `time_granularity` from that source query.
   - Use `get_dimensions` to find exact generated dimension names; if a grouped source dimension cannot be queried, fix the semantic model joins/dimensions and retry.
   - Collect the SQL into a dict: `{"{metric_name}": "SELECT ..."}`

## Phase 4: Batch Sync to Knowledge Base

After all generated metrics have passed validation and dry-run:
- Collect all generated metrics and their dry-run SQLs into `metric_sqls_json`
- You MUST call `end_metric_generation(metric_file, semantic_model_files, metric_sqls_json)` **ONCE** to sync them to Knowledge Base while you can still fix publish errors
- `semantic_model_files` must include every semantic model file newly created or updated for this batch. If one metric file contains metrics backed by multiple tables, include all affected semantic model files.
- Do not rely on the final JSON host fallback. The host fallback is only a last-resort guard when the tool call was accidentally missed.
- If no metrics were generated, do NOT call `end_metric_generation`

Phase 1 confirms the generation scope; validation plus dry-run are the acceptance gate before syncing.

## Common Pitfalls (MUST avoid)

1. **Explicit metric files**: Write explicit metric YAML files under the semantic model directory's `metrics/` subdirectory instead of relying on `create_metric: true`. Runtime-generated metrics are not part of the persisted metric catalog.

2. **Metric name must match measure name**: For a `measure_proxy` metric, the metric name should typically equal the measure name (or be a clear derivative). The `type_params.measure` must exactly match a measure name from the semantic model. Do NOT invent unrelated names (e.g., measure `activity_count` → metric name should be `activity_count`, NOT `total_activity_count` or `activity_count_metric`).

3. **Filtered metrics**: Model reusable filter logic as a conditional measure in the semantic model, such as `expr: "CASE WHEN status = 'completed' THEN 1 ELSE 0 END"` with `agg: SUM`, then write `type_params.measure: completed_order_count` in the metric YAML.

4. **Check before creating**: ALWAYS call `check_semantic_object_exists(name="{metric_name}", kind="metric")` before writing a new metric. If the metric already exists, skip it.

5. **Do not use subject path to disambiguate metric names**: `revenue` under `Finance` and `revenue` under `Sales` are still the same datasource-local metric name. If the calculations differ, choose clearer names such as `finance_revenue` and `sales_revenue`.

6. **Verify names after validation**: After `validate_semantic` succeeds and the adapter reloads, call `list_metrics` to see the exact metric names available. Use these exact names when calling `query_metrics`.

7. **Every metric needs explicit YAML**: Whether it's a simple aggregation, filtered variant, ratio, expr, derived, or cumulative — write a `metric:` entry in the metrics YAML file so it can be persisted and discovered later.

8. **Derived metrics are second-stage**: Generate non-derived metrics first, validate them, refresh the metric catalog with `list_metrics`, then generate `derived_metric_candidates` only when every referenced metric exists in the refreshed catalog or was generated earlier in the same batch.

## Important Rules

- **Phase 1**: Confirm which metrics to generate before proceeding. Use `ask_user` when it is available.
- **Validation MUST pass** — always call `validate_semantic` and ensure it passes before proceeding to the next phase. If it fails, fix and retry until it passes.
- **Sync automatically after validation** — once validation and dry-run pass, call `end_metric_generation` without another user confirmation. The final JSON `metric_file` is only a last-resort fallback.
- **COUNT agg must use `expr: "1"`** — never use `expr: {column}` with COUNT (use COUNT_DISTINCT for that).
- For ratio metrics, both numerator and denominator measures must exist in the semantic model.
- For expr metrics, all referenced measures must exist in the semantic model.
- For derived metrics, all referenced metrics must already be defined, the expression must not be a single metric passthrough, and the dependency graph must not contain cycles.
- For cumulative metrics, the measure must exist and a primary time dimension must be defined.
- Use consistent naming: metric names in snake_case, measure names matching the semantic model.
- Every metric data_source needs a primary time dimension when a reliable DATE/TIME/TIMESTAMP column or expression exists. Do not force a primary TIME dimension from numeric surrogate keys; join/convert to a real date first.
- Measure names must be globally unique across all data sources.
- For snapshot/balance data, always add `non_additive_dimension` to prevent incorrect time aggregation.
- **Keep files scoped** — only write semantic model YAML and metric YAML files. Sync metrics through `end_metric_generation`; the final JSON `metric_file` is only a last-resort fallback.
