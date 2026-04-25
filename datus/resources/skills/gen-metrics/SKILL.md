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

Before anything else, call `list_metrics()` to get all metrics already in the knowledge base. Note their names, types, and associated measures. Use this throughout the remaining phases to:
- **Skip redundant work** — don't recreate metrics that already exist
- **Reuse existing measures** — reference measures from existing models instead of creating duplicates
- **Detect conflicts** — warn the user if a proposed metric name collides with an existing one
- **Enable derived/ratio metrics** — know which metrics can serve as building blocks for more complex definitions

## Phase 1: Understand Intent (MANDATORY ask_user)

Analyze the user's request, then **ALWAYS call `ask_user`** to confirm before proceeding. This phase supports two input modes:

### Input Mode Detection

- **Single mode**: User describes one metric or provides one SQL → follow Step 1a–1d below
- **Batch mode**: User provides multiple SQL queries (pasted directly, or a CSV file path containing `question` + `sql` columns) → follow Step 1-batch below

### Single Mode: Step 1a–1d

**Step 1a: Inspect the table** — Call `describe_table(table_name)` to understand the columns and types. Optionally call `read_query` to sample data.

**Step 1b: Ask for reference SQL (optional)** — Use `ask_user` to ask:
> "Do you have any existing SQL queries for this table that show the aggregations you care about? You can paste them here, or skip if not available."

If the user provides SQL, parse it to extract:
- Aggregation functions + columns (e.g., `SUM(amount)` → candidate measure `total_amount`, `COUNT(*)` → candidate measure `record_count`)
- GROUP BY columns → recommended dimensions
- WHERE conditions → potential metric constraints

If the provided SQL contains no aggregation patterns (no SUM, COUNT, AVG, MAX, MIN, etc.), inform the user and proceed as if no SQL was provided.

If the user skips, proceed to Step 1c using only table structure and the user's description.

**Step 1c: Propose metric candidates** — Based on the table structure, reference SQL (if provided), and user's request, identify potential metric scenarios. See "Metric type detection rules" below.

**Step 1d: MUST call `ask_user`** to confirm — present proposed metrics with `multi_select: true` (see Step 1-batch-d for format).

### Batch Mode: Step 1-batch

**Step 1-batch-a: Parse SQL queries**
- The input may contain multiple SQL queries in various forms:
  - **Direct paste**: multiple SQL statements in the prompt
  - **File path**: user provides a path — call `read_file` to load it, then parse by file type:
    - `.sql`: split by `;` or blank-line separators to extract individual statements
    - `.csv` / `.tsv`: identify the SQL column by header name (common names: `sql`, `query`, `SQL`, `statement`) or by content heuristic (column values contain SQL keywords like `SELECT`, `FROM`, `GROUP BY`). The description/question column is any remaining text column. If column roles are ambiguous, call `ask_user` to confirm which column is SQL.
    - Other formats: call `ask_user` to clarify the file structure before proceeding
- Parse all SQL queries from the input
- Call `describe_table` for each unique table found in the SQL queries

**Step 1-batch-b: Extract and deduplicate aggregation patterns**

Scan ALL SQL queries and extract every aggregation pattern (`SUM(col)`, `COUNT(*)`, `AVG(col)`, etc.). Then **strictly deduplicate** to identify only **core base metrics**:

1. **Group by (aggregation_function, column)** — e.g., all `SUM(amount)` across different queries map to ONE candidate metric `total_amount`
2. **Discard detail queries** — SQL without any aggregation (pure SELECT/JOIN) is not a metric source; skip silently
3. **Discard filtered variants** — `SUM(amount) WHERE status='paid'` and `SUM(amount) WHERE region='US'` are NOT separate metrics; they are filters on the same core metric `total_amount`. Only propose the unfiltered base metric.
4. **Do NOT generate derived metrics** — ratio, expression, or cumulative metrics should NOT be auto-proposed from batch SQL. Only propose them if the user explicitly describes them (e.g., "conversion rate", "average order value"). The goal is the minimal set of reusable base measures.
5. **Cross-reference with Phase 0** — remove any candidate that already exists in the knowledge base

**Step 1-batch-c: Core metric principle**

From N SQL queries, propose at most a **small set of core metrics** (typically fewer than N). Ask yourself for each candidate:
- Is this a **unique aggregation** not covered by another candidate? If not, skip.
- Is this a **base metric** (simple aggregation on a column) or a **derivative** (ratio, expression combining other metrics)? Only propose base metrics by default.
- Would a business user recognize this as a **standalone KPI**? If it's just an intermediate calculation, skip.

**Step 1-batch-d: MUST call `ask_user`** to confirm with the user:
- Present the deduplicated core metrics as **options** with `multi_select: true`
- Pass `questions` as an actual array argument, not a JSON string. Example tool arguments:
  ```json
  {
    "questions": [
      {
        "title": "Metrics",
        "question": "I analyzed N SQL queries and identified the following core metrics. Select which ones to generate:",
        "options": ["total_revenue - SUM(amount) on orders", "order_count - COUNT on orders"],
        "multi_select": true
      }
    ]
  }
  ```
- Clearly show how many SQL queries were analyzed and how many core metrics were extracted
- If the user wants additional derived/ratio metrics beyond the core set, they can request them after the base metrics are created

### Metric type detection rules

1. **Simple counting + filter**: "How many completed orders" → conditional measure in the semantic model + `measure_proxy` metric referencing that measure by string
2. **Aggregation + filter**: "Total revenue from premium customers" → conditional measure in the semantic model + `measure_proxy` metric referencing that measure by string
3. **Ratio**: "Order completion rate", "Conversion rate" → `ratio` type
4. **Derived/Expression**: "Average order value", "Revenue per user" → `expr` type combining metrics
5. **Cumulative**: "Running total of revenue", "MTD sales", "Year-to-date signups" → `cumulative` type
6. **Conversion**: "Signup-to-purchase conversion", "Trial-to-paid funnel" → `conversion` type

Detection keywords:
- "running total", "MTD", "YTD", "cumulative", "to-date" → cumulative
- "conversion", "funnel", "from X to Y" → conversion
- "rate", "ratio", "percentage of", "share of" → ratio
- "per", "divided by", "average ... per" → derived/expr
- "list all...", "show me the..." → not a metric, better suited for `gen_sql`

**IMPORTANT**: Do NOT proceed to Phase 2 without user confirmation from `ask_user`.

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

1. **Check existing**: Call `check_semantic_object_exists(name="{metric_name}", kind="metric")` for each metric confirmed in Phase 1. If it already exists, inform the user and skip it.

2. **Write metric YAML**: Use `write_file` to save each metric definition to `subject/semantic_models/<current_datasource>/metrics/{table_name}_metrics.yml`.
   - For `measure_proxy`, keep `type_params.measure` as a string measure name.
   - For filtered metrics, add a dedicated conditional measure to the semantic model first, then reference that measure from the metric YAML.

3. **Validate (MUST PASS)**: Call `validate_semantic` to check the metric YAML.
   - If validation fails, fix errors with `edit_file` and retry until it **passes**.
   - **Do NOT proceed to Phase 4 until validation passes.** No exceptions.

4. **Dry-run SQL**: Call `query_metrics(metrics=["{metric_name}"], dry_run=True)` to generate the SQL.
   - Collect the SQL into a dict: `{"{metric_name}": "SELECT ..."}`

## Phase 4: Batch Sync to Knowledge Base

After all generated metrics have passed validation and dry-run:
- Collect all generated metrics and their dry-run SQLs into `metric_sqls_json`
- Call `end_metric_generation(metric_file, semantic_model_file, metric_sqls_json)` **ONCE** to sync them to Knowledge Base
- If no metrics were generated, do NOT call `end_metric_generation`

Phase 1 confirms the generation scope; validation plus dry-run are the acceptance gate before syncing.

## Common Pitfalls (MUST avoid)

1. **Explicit metric files**: Write explicit metric YAML files under the semantic model directory's `metrics/` subdirectory instead of relying on `create_metric: true`. Runtime-generated metrics are not part of the persisted metric catalog.

2. **Metric name must match measure name**: For a `measure_proxy` metric, the metric name should typically equal the measure name (or be a clear derivative). The `type_params.measure` must exactly match a measure name from the semantic model. Do NOT invent unrelated names (e.g., measure `activity_count` → metric name should be `activity_count`, NOT `total_activity_count` or `activity_count_metric`).

3. **Filtered metrics**: Model reusable filter logic as a conditional measure in the semantic model, such as `expr: "CASE WHEN status = 'completed' THEN 1 ELSE 0 END"` with `agg: SUM`, then write `type_params.measure: completed_order_count` in the metric YAML.

4. **Check before creating**: ALWAYS call `check_semantic_object_exists(name="{metric_name}", kind="metric")` before writing a new metric. If the metric already exists, skip it.

5. **Verify names after validation**: After `validate_semantic` succeeds and the adapter reloads, call `list_metrics` to see the exact metric names available. Use these exact names when calling `query_metrics`.

6. **Every metric needs explicit YAML**: Whether it's a simple aggregation, filtered variant, ratio, derived, cumulative, or conversion — write a `metric:` entry in the metrics YAML file so it can be persisted and discovered later.

## Important Rules

- **Phase 1**: MUST call `ask_user` to confirm which metrics to generate before proceeding.
- **Validation MUST pass** — always call `validate_semantic` and ensure it passes before proceeding to the next phase. If it fails, fix and retry until it passes.
- **Sync automatically after validation** — once validation and dry-run pass, call `end_metric_generation` without another user confirmation.
- **COUNT agg must use `expr: "1"`** — never use `expr: {column}` with COUNT (use COUNT_DISTINCT for that).
- For ratio metrics, both numerator and denominator measures must exist in the semantic model.
- For derived metrics, all referenced metrics must already be defined.
- For cumulative metrics, the measure must exist and a primary time dimension must be defined.
- For conversion metrics, both base and conversion measures must reference the same entity.
- Use consistent naming: metric names in snake_case, measure names matching the semantic model.
- Every data_source MUST have a primary time dimension (`type: TIME` with `is_primary: true`).
- Measure names must be globally unique across all data sources.
- For snapshot/balance data, always add `non_additive_dimension` to prevent incorrect time aggregation.
- **Keep files scoped** — only write semantic model YAML and metric YAML files. Sync metrics through `end_metric_generation`.
