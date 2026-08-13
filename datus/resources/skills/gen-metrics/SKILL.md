---
name: gen-metrics
description: Generate MetricFlow metrics from natural language business descriptions
tags:
  - metrics
  - metricflow
version: "1.3.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_metrics
---

# Generate Metrics Skill

Guide the user through metric generation using natural language business descriptions.

## Phase 0: Read the evidence

Use SQL supplied with the request directly. For an explicitly named readable workspace SQL file, call `read_file` once. Interpret the complete SQL together with the user's business intent, the live target YAML, and the metric catalog when deciding reuse, datasets, and expressions.

Only inspect and edit semantic model YAML files under the current datasource directory shown in the system prompt, such as `subject/semantic_models/<current_datasource>/...`. Do not reuse or sync YAML files from sibling datasource directories; those files are outside the active MetricFlow adapter scope.

## Phase 1: Understand Intent

Analyze the user's request and confirm the generation scope before proceeding. When `ask_user` is available, call it to confirm the metric name(s), business meaning, and calculation logic. When `ask_user` is not available (for example workflow or batch mode), infer from the provided SQL/request and stop only if the scope is materially ambiguous.

### Input Mode Detection

- **Single mode**: User describes one metric or provides one SQL → follow Step 1a–1d below
- **Batch mode**: The current task directly contains multiple SQL queries or explicitly names a readable workspace SQL file → follow Step 1-batch below

### Single Mode: Step 1a–1d

**Step 1a: Inspect the table** — Call `describe_table(table_name)` to understand the columns and types. Optionally call `execute_sql(sql="SELECT * FROM <table> LIMIT 5")` to sample data.

**Step 1b: Ask for reference SQL (optional)** — When `ask_user` is available, use it to ask:
> "Do you have any existing SQL queries for this table that show the aggregations you care about? You can paste them here, or skip if not available."

When `ask_user` is not available, skip this question and infer SQL/aggregation context from the user's request, attached files, or discovered query/table evidence. If that is not enough, stop and explain the missing information instead of calling `ask_user`.

If the user provides SQL, use it to identify:
- Final business output expressions (e.g., `SUM(amount) / COUNT(DISTINCT user_id) AS arppu` → candidate metric `arppu`)
- Aggregation functions + columns that the final metric depends on (e.g., `SUM(amount)` → candidate measure `total_amount`, `COUNT(*)` → candidate measure `record_count`)
- GROUP BY columns → recommended dimensions
- WHERE conditions → potential metric constraints

If the provided SQL contains no metric-producing output, keep filter-only or detail-query evidence as filters, dimensions, segments, or view evidence instead of generating fake metrics.

If the user skips, proceed to Step 1c using only table structure and the user's description.

**Step 1c: Propose metric candidates** — Based on the table structure, reference SQL (if provided), and user's request, identify potential metric scenarios. See "Metric type detection rules" below.

**Step 1d: Confirm scope** — when `ask_user` is available, call it to confirm and present proposed metrics with `multi_select: true` (see Step 1-batch-d for format). If `ask_user` is not available, proceed with the confirmed/inferred scope from the input.

### Batch Mode: Step 1-batch

**Step 1-batch-a: Collect SQL queries**
- SQL statements may be pasted directly or stored in a workspace SQL file explicitly named by the user.
- For a named workspace SQL file, the parent may pass its contents, or this agent may call `read_file` once.
- Read every complete SQL statement in the request or file as evidence for one coherent authoring decision.
- Call `describe_table` for each unique table found in the SQL queries

**Step 1-batch-b: Interpret reusable semantics**

1. Identify the reusable business metrics and the base calculations they depend on.
2. Reuse an existing metric only when its calculation, dataset, time/window behavior, and business meaning match.
3. By default, model reusable datasets, dimensions, relationships, and base metrics; keep literal filters, grouping, ordering, and result layout at query time.
4. Choose a query-backed dataset when the user establishes the result itself as a durable reusable asset or explicitly asks for faithful one-query reproduction.
5. After any artifact correction, rerun semantic validation before publication.

**Step 1-batch-c: Business metric principle**

From N SQL queries, propose a focused set of business metrics. Ask yourself for each candidate:
- Is this a final output a business user would recognize as a KPI?
- Are its base measures complete enough to validate?
- Should the evidence be a metric, or only a filter/dimension/segment/view definition?
- Is this alias only a supporting count/sum used by another final output? If yes, create or reuse the measure but do not publish a separate metric for it.
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

If the semantic model is missing, follow the `metricflow-semantic-authoring` workflow when that skill is available. In brief: call `inspect_semantic_sources` with all required physical tables, then use the live schemas, request-SQL field usage, and relationship candidates to write semantic model YAML under the directory shown in the system prompt. Run `validate_semantic` and fix issues until it passes before continuing.

### 2c. Multi-Table / JOIN SQL Modeling

When the metric involves multiple tables (detected from JOIN in SQL or user description), choose the modeling strategy based on SQL complexity:

**Strategy A: Identifier-based JOIN (default — use when possible)**

Use when: simple equi-JOIN between 2-3 tables via foreign keys, ≤ 2 JOIN hops.

- Each table gets its own `data_source` with `sql_table`
- Tables are linked via matching `identifiers` (same `name`, one PRIMARY, one FOREIGN)
- Use `inspect_semantic_sources.relationships` to set up correct identifier linkages
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
    sql_query: |
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

**Decision rule**: Default to Strategy A. Use it when the join can be represented as identifier-level keys (single-column or derived expressions). Use Strategy B for composite multi-column equi-joins unless they are represented in source SQL as a derived key expression, and for non-equi conditions, 3+ hop joins, or subquery-based logic.

## Phase 3: Generate and Validate

**File paths**: All `write_file` / `edit_file` / `read_file` calls use paths relative to the filesystem sandbox root. Always use the semantic model directory shown in the system prompt so subsequent reads find the file. For example:
- Semantic model: `subject/semantic_models/<current_datasource>/{table_name}.yml`
- Metric file: `subject/semantic_models/<current_datasource>/metrics/{table_name}_metrics.yml`

Bare filenames are silently normalized by the host, but the prefixed form is preferred for clarity. Absolute paths are also tolerated.
Do not read, edit, or pass `metric_file` / `semantic_model_files` paths from another datasource directory such as `subject/semantic_models/other_datasource/...`.

1. **Check existing**: Call `check_semantic_object_exists(name="{metric_name}", kind="metric")` for each metric confirmed in Phase 1. If it already exists, inform the user and skip it.

2. **Write metric YAML**: Use `write_file` to save each metric definition to `subject/semantic_models/<current_datasource>/metrics/{table_name}_metrics.yml`.
   - For `measure_proxy`, keep `type_params.measure` as a string measure name.
   - For filtered metrics, add a dedicated conditional measure to the semantic model first, then reference that measure from the metric YAML.
   - Each generated metric must be an explicit named top-level `metric:` YAML document. Do not emit unnamed `metric:` blocks or wrap metrics inside another object.

3. **Validate (MUST PASS)**: Call `validate_semantic` to check the metric YAML.
   - If validation fails, fix errors with `edit_file` and retry until it **passes**.
   - **Do NOT proceed to Phase 4 until validation passes.** No exceptions.

## Phase 4: Batch Sync to Knowledge Base

After all generated metrics have passed validation:
- For each generated metric file, call `publish_metrics(metric_file)` once to sync it to Knowledge Base while you can still fix publish errors.
- Do not rely on the final JSON host fallback. The host fallback is only a last-resort guard when the tool call was accidentally missed.
- If no metrics were generated, do NOT call `publish_metrics`

Phase 1 confirms the generation scope; validation is the semantic acceptance gate before syncing.

## Common Pitfalls (MUST avoid)

1. **Explicit metric files**: Write explicit metric YAML files under the semantic model directory's `metrics/` subdirectory instead of relying on `create_metric: true`. Runtime-generated metrics are not part of the persisted metric catalog.

2. **Metric name must match measure name**: For a `measure_proxy` metric, the metric name should typically equal the measure name (or be a clear derivative). The `type_params.measure` must exactly match a measure name from the semantic model. Do NOT invent unrelated names (e.g., measure `activity_count` → metric name should be `activity_count`, NOT `total_activity_count` or `activity_count_metric`).

3. **Filtered metrics**: Model reusable filter logic as a conditional measure in the semantic model, such as `expr: "CASE WHEN status = 'completed' THEN 1 ELSE 0 END"` with `agg: SUM`, then write `type_params.measure: completed_order_count` in the metric YAML.

4. **Check before creating**: ALWAYS call `check_semantic_object_exists(name="{metric_name}", kind="metric")` before writing a new metric. If the metric already exists, skip it.

5. **Verify the artifact after validation**: Ensure each `metric_file` points to validated metric YAML before calling `publish_metrics(metric_file)`. Do not pass output bindings.

6. **Every metric needs explicit YAML**: Whether it's a simple aggregation, filtered variant, ratio, expr, derived, or cumulative — write a `metric:` entry in the metrics YAML file so it can be persisted and discovered later.

7. **Derived metrics are second-stage**: Generate and validate input metrics first. Author a derived metric only when every referenced metric exists in the live target or was generated earlier in the same run.

8. **Support measures are not always metrics**: Add support measures needed for ratios, expressions, filters, and validation, but do not publish each support measure as a separate metric unless it is itself a requested/final business KPI.

## MetricFlow Metric Structure Reference

**measure_proxy** (simple aggregation):
```yaml
metric:
  name: {metric_name}
  description: "{description}"
  type: measure_proxy
  type_params:
    measure: {measure_name}
  locked_metadata:
    tags:
      - "{category}"
      - "subject_tree: {domain}/{layer1}/{layer2}"
```

For a filtered metric, define a dedicated conditional measure in the semantic model and keep the metric's `type_params.measure` as a string:
```yaml
data_source:
  name: orders
  measures:
    - name: completed_order_count
      description: "Completed order count"
      agg: SUM
      expr: "CASE WHEN status = 'completed' THEN 1 ELSE 0 END"
---
metric:
  name: completed_order_count
  description: "Completed order count"
  type: measure_proxy
  type_params:
    measure: completed_order_count
```

**ratio** (ratio of two measures):
```yaml
metric:
  name: {metric_name}
  description: "{description}"
  type: ratio
  type_params:
    numerator: {measure_or_metric_name}
    denominator: {measure_or_metric_name}
  locked_metadata:
    tags:
      - "subject_tree: {domain}/{layer1}/{layer2}"
```

**expr** (expression combining measures):
```yaml
metric:
  name: {metric_name}
  description: "{description}"
  type: expr
  type_params:
    measures:
      - measure_a
      - measure_b
    expr: "{expression}"  # e.g. "(measure_a - measure_b) / measure_a"
  locked_metadata:
    tags:
      - "subject_tree: {domain}/{layer1}/{layer2}"
```

**derived** (expression combining existing metrics):
```yaml
metric:
  name: {metric_name}
  description: "{description}"
  type: derived
  type_params:
    metrics:
      - name: metric_a
        # Optional: period-over-period comparison
        alias: metric_a_prev
        offset_window: 1 week    # compare to 1 week ago (WoW)
      - name: metric_b
        offset_to_grain: month   # compare to start of current month (MTD)
    expr: "{expression}"  # e.g. "metric_a / metric_a_prev"
  locked_metadata:
    tags:
      - "subject_tree: {domain}/{layer1}/{layer2}"
```

Period-over-period example — a MoM SQL whose final output is `metric_a_mom_delta` should publish a fixed MoM delta metric, not a query-time compare instruction and not a previous-value helper unless that helper is itself the final requested output:
```yaml
metric:
  name: metric_a_mom_delta
  description: "{metric_a month-over-month delta description}"
  type: derived
  type_params:
    metrics:
      - name: metric_a
      - name: metric_a
        alias: metric_a_prev
        offset_window: 1 month
    expr: "metric_a - metric_a_prev"
```

**cumulative** (running total over time):
```yaml
metric:
  name: {metric_name}
  description: "{description}"
  type: cumulative
  type_params:
    measure: {measure_name}
    # Use ONE of:
    window: {time_window}        # rolling window, e.g. "7 days", "1 month"
    grain_to_date: month|year    # MTD/YTD - resets at grain boundary
  locked_metadata:
    tags:
      - "subject_tree: {domain}/{layer1}/{layer2}"
```

Ordinary period-over-period SQL (`LAG`, previous period, DoD/WoW/MoM/QoQ/YoY, delta, or rate) is fixed long-term metric evidence when it is a final business output: monthly YoY is distinct from weekly YoY, MoM rate is distinct from MoM delta, and previous-period value is distinct from a rate. Publish a previous-period metric only when it is itself the requested final output.

## Important Rules

- **Phase 1**: Confirm which metrics to generate before proceeding. Use `ask_user` when it is available.
- **Validation MUST pass** — always call `validate_semantic` and ensure it passes before proceeding to the next phase. If it fails, fix and retry until it passes.
- **Sync automatically after validation** — call `publish_metrics` without another user confirmation; the final JSON `metric_file` is only a last-resort fallback.
- **COUNT agg must use `expr: "1"`** — never use `expr: {column}` with COUNT (use COUNT_DISTINCT for that).
- For ratio metrics, both numerator and denominator measures must exist in the semantic model.
- For expr metrics, all referenced measures must exist in the semantic model.
- For derived metrics, all referenced metrics must already be defined, the expression must not be a single metric passthrough, and the dependency graph must not contain cycles.
- For cumulative metrics, the measure must exist and a primary time dimension must be defined.
- Use consistent naming: metric names in snake_case, measure names matching the semantic model.
- Every metric data_source needs a primary time dimension when a reliable DATE/TIME/TIMESTAMP column or expression exists. Do not force a primary TIME dimension from numeric surrogate keys; join/convert to a real date first.
- Measure names must be globally unique across all data sources.
- For snapshot/balance data, always add `non_additive_dimension` to prevent incorrect time aggregation.
- **Keep files scoped** — only write semantic model YAML and metric YAML files. Sync metrics through `publish_metrics`; the final JSON `metric_file` is only a last-resort fallback.
