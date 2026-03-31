---
name: define-metric
description: Interactively define MetricFlow metrics from natural language business descriptions
tags: "metrics, metricflow, interactive"
version: "1.1.0"
user_invocable: false
disable_model_invocation: false
---

# Define Metric Skill

Guide the user through interactive metric definition using natural language business descriptions.

## Phase 1: Understand Intent (MANDATORY ask_user)

Analyze the user's request, inspect the table structure, then **ALWAYS call `ask_user`** to confirm before proceeding.

**Step 1a: Inspect the table** — Call `describe_table(table_name)` to understand the columns and types. Optionally call `read_query` to sample data.

**Step 1b: Ask for reference SQL (optional)** — Use `ask_user` to ask:
> "Do you have any existing SQL queries for this table that show the aggregations you care about? You can paste them here, or skip if not available."

If the user provides SQL, parse it to extract:
- Aggregation functions + columns (e.g., `SUM(amount)` → candidate measure `total_amount`, `COUNT(*)` → candidate measure `record_count`)
- GROUP BY columns → recommended dimensions
- WHERE conditions → potential metric constraints
- Use these patterns as the primary source for metric candidates

If the provided SQL contains no aggregation patterns (no SUM, COUNT, AVG, MAX, MIN, etc. — e.g., it's a simple SELECT or detail query), inform the user:
> "The SQL you provided doesn't contain aggregation patterns (SUM, COUNT, AVG, etc.), so it can't be used as a reference for metric definition. I'll proceed based on the table structure and your description instead."
Then proceed as if no SQL was provided.

If the user skips, proceed to Step 1c using only table structure and the user's description.

**Step 1c: Propose metric candidates** — Based on the table structure, reference SQL (if provided), and user's request, identify potential metric scenarios:

1. **Simple counting + filter**: "How many completed orders" → `measure_proxy` with `constraint`
2. **Aggregation + filter**: "Total revenue from premium customers" → `measure_proxy` with `constraint`
3. **Ratio**: "Order completion rate", "Conversion rate" → `ratio` type
4. **Derived/Expression**: "Average order value", "Revenue per user" → `expr` type combining metrics
5. **Cumulative**: "Running total of revenue", "MTD sales", "Year-to-date signups" → `cumulative` type
6. **Conversion**: "Signup-to-purchase conversion", "Trial-to-paid funnel" → `conversion` type

**Detection rules:**
- Keywords "running total", "MTD", "YTD", "cumulative", "to-date" → cumulative metric
- Keywords "conversion", "funnel", "from X to Y" → conversion metric
- Keywords "rate", "ratio", "percentage of", "share of" → ratio metric
- Keywords "per", "divided by", "average ... per" → derived/expr metric
- If the description sounds like a detail query ("list all...", "show me the..."), inform the user this is better suited for SQL generation (`gen_sql`), not metric definition.

**Step 1d: MUST call `ask_user`** to confirm with the user:
- Present the proposed metrics as **options** with `multi_select: true`, so the user can select multiple metrics using checkboxes (Space to toggle, Enter to confirm)
- Example: `ask_user(questions=[{"question": "Select metrics to generate", "options": ["total_orders - COUNT", "total_revenue - SUM(amount)", ...], "multi_select": true}])`
- In the same or a follow-up `ask_user` call, ask for any missing information:
  - What is the business meaning?
  - What is the calculation logic (numerator/denominator for ratios)?
  - For cumulative: What is the accumulation window (7 days, 1 month, unbounded)?
  - For conversion: What are the base and conversion events? What is the conversion window?

**IMPORTANT**: Do NOT proceed to Phase 2 without user confirmation from `ask_user`.

## Phase 2: Ensure Semantic Model Exists

For each table involved in the metric:

### 2a. Check Existing Model

1. Call `check_semantic_object_exists(object_name="{table_name}", kind="table")` to check if a semantic model exists.
2. **If the semantic model exists:**
   - Use `read_file` to read the existing semantic model YAML
   - Verify that it contains the measures and dimensions needed for this metric
   - If missing measures/dimensions, use `edit_file` to add them, then `validate_semantic`

### 2b. Create Missing Model

If the semantic model is missing, use the analysis tools to build a high-quality model:

1. **Gather table structure:**
   - Call `describe_table(table_name)` to get column names and types
   - If multiple tables are involved, call `analyze_table_relationships(tables=[...])` to discover foreign key relationships and JOIN patterns
   - Call `analyze_column_usage_patterns(table_name)` to understand how columns are typically queried and filtered in historical SQL

2. **Use `ask_user` to confirm column roles:**
   - Which columns should be **measures** (aggregatable numeric columns)?
   - Which columns should be **dimensions** (grouping/filtering columns)?
   - Which column is the **primary time dimension** (required — every data_source must have exactly one)?
   - For multi-table scenarios: which columns are **primary key** and **foreign keys** linking to other tables?

3. **Generate semantic model YAML following these rules:**

   **Identifiers (optional for single-table, required for multi-table):**
   - Identifiers are only needed when joining multiple data sources or for entity-based metrics (e.g., conversion)
   - For single-table metrics, identifiers can be omitted entirely — do NOT ask the user for a primary key if the metric only involves one table
   - When needed: use `type: PRIMARY` for the main key, `type: FOREIGN` for columns referencing other tables
   - Also available: `type: UNIQUE` for natural keys, `type: NATURAL` for business keys

   **Measures:**
   - `agg: COUNT` MUST include `expr: "1"` (counts rows, not a column value)
   - `agg: COUNT_DISTINCT` uses `expr: {column}` (counts distinct values of a column)
   - `agg: SUM|AVERAGE|MIN|MAX` uses `expr: {column}`
   - `agg: PERCENTILE` requires `percentile: 0.95` (or other value) in `agg_params`
   - `agg: MEDIAN` is shorthand for PERCENTILE(0.5)
   - Measure names MUST be globally unique across ALL data sources
   - Do NOT use `create_metric: true` — always write explicit metric YAML files instead (see Common Pitfalls)

   **Dimensions:**
   - `type: TIME` dimensions MUST include `type_params` with `time_granularity`
   - Exactly ONE time dimension must have `type_params.is_primary: true`
   - `type: CATEGORICAL` for all non-time dimensions

   **non_additive_dimension** (for snapshot/balance metrics):
   - If a measure represents a point-in-time value (account balance, inventory count, active users), add:
     ```yaml
     non_additive_dimension:
       name: {time_dimension_name}
       window_choice: min|max  # max = latest snapshot, min = earliest
     ```
   - This prevents incorrect aggregation across time (e.g., summing daily balances)

4. Save with `write_file` (use relative path like `{table_name}.yml`) → `validate_semantic` (MUST pass before continuing) → `end_semantic_model_generation`

### 2c. Multi-Table Entity Modeling

When the metric spans multiple tables:
- Each table gets its own `data_source` in the semantic model
- Tables are linked via matching `identifiers` (same `name`, one PRIMARY, one FOREIGN)
- Use `analyze_table_relationships` results to set up correct identifier linkages
- Example: `orders.customer_id` (FOREIGN) links to `customers.customer_id` (PRIMARY) — both identifiers share `name: customer`

## Phase 3: Generate and Validate

**File paths**: All `write_file` / `edit_file` / `read_file` calls use **relative paths** within the workspace directory (shown in system prompt as `semantic_model_dir`). Never use absolute paths. For example, use `table_name.yml` for semantic models and `metrics/table_name_metrics.yml` for metrics.

1. **Check existing**: Call `check_semantic_object_exists(name="{metric_name}", kind="metric")` for each metric confirmed in Phase 1. If it already exists, inform the user and skip it.

2. **Write metric YAML**: Use `write_file` to save each metric definition to `metrics/{table_name}_metrics.yml`.

3. **Validate (MUST PASS)**: Call `validate_semantic` to check the metric YAML.
   - If validation fails, fix errors with `edit_file` and retry until it **passes**.
   - **Do NOT proceed to Phase 4 until validation passes.** No exceptions.

4. **Dry-run SQL**: Call `query_metrics(metrics=["{metric_name}"], dry_run=True)` to generate the SQL.
   - Collect the SQL into a dict: `{"{metric_name}": "SELECT ..."}`

**Do NOT call `end_metric_generation` yet** — proceed to Phase 4 for user review.

## Phase 4: User Review — One by One (MANDATORY ask_user)

After generating and validating, present each metric to the user **one at a time** via `ask_user`.

**For each metric, present:**
- **Metric name** and **type**
- **YAML content** (the actual metric definition)
- **Generated SQL** from dry-run
- **Subject tree** classification

**Ask the user to choose ONE of:**
1. **Confirm** — approve this metric
2. **Modify** — specify what to change
3. **Reject** — remove this metric

**Based on user response:**

- **If Confirm**: Mark as approved, proceed to the next metric.

- **If Modify**: Use `edit_file` to modify the metric YAML based on user feedback → re-validate → dry-run SQL again → present updated results to user via `ask_user` again (loop until user confirms or rejects this metric). Then proceed to the next metric.

- **If Reject**: Use `edit_file` to remove this metric entry from the YAML file. Proceed to the next metric.

## Phase 5: Batch Sync to Knowledge Base

After ALL metrics have been reviewed one by one:
- Collect all approved metrics and their dry-run SQLs into `metric_sqls_json`
- Call `end_metric_generation(metric_file, semantic_model_file, metric_sqls_json)` **ONCE** to sync all approved metrics to Knowledge Base
- If no metrics were approved (all rejected), do NOT call `end_metric_generation`

**IMPORTANT**: NEVER call `end_metric_generation` without explicit user approval from Phase 4.

## Common Pitfalls (MUST avoid)

1. **Do NOT use `create_metric: true`**: Never set `create_metric: true` on measures. Always write explicit metric YAML files in `metrics/` directory. Reason: `create_metric: true` only creates metrics at MetricFlow runtime — they are NOT synced to the Knowledge Base (vector DB). Only explicit `metric:` YAML entries get imported.

2. **Metric name must match measure name**: For a `measure_proxy` metric, the metric name should typically equal the measure name (or be a clear derivative). The `type_params.measure` must exactly match a measure name from the semantic model. Do NOT invent unrelated names (e.g., measure `activity_count` → metric name should be `activity_count`, NOT `total_activity_count` or `activity_count_metric`).

3. **Check before creating**: ALWAYS call `check_semantic_object_exists(name="{metric_name}", kind="metric")` before writing a new metric. If the metric already exists, skip it.

4. **Verify names after validation**: After `validate_semantic` succeeds and the adapter reloads, call `list_metrics` to see the exact metric names available. Use these exact names when calling `query_metrics`.

5. **Every metric needs explicit YAML**: Whether it's a simple aggregation, filtered variant, ratio, derived, cumulative, or conversion — always write a `metric:` entry in the metrics YAML file. This is the only way metrics get synced to the Knowledge Base.

## Important Rules

- **Phase 1**: MUST call `ask_user` to confirm which metrics to generate before proceeding.
- **Phase 4**: MUST call `ask_user` to let user review generated results before syncing to DB.
- **Validation MUST pass** — always call `validate_semantic` and ensure it passes before proceeding to the next phase. If it fails, fix and retry until it passes.
- **COUNT agg must use `expr: "1"`** — never use `expr: {column}` with COUNT (use COUNT_DISTINCT for that).
- For ratio metrics, both numerator and denominator measures must exist in the semantic model.
- For derived metrics, all referenced metrics must already be defined.
- For cumulative metrics, the measure must exist and a primary time dimension must be defined.
- For conversion metrics, both base and conversion measures must reference the same entity.
- Use consistent naming: metric names in snake_case, measure names matching the semantic model.
- Every data_source MUST have a primary time dimension (`type: TIME` with `is_primary: true`).
- Measure names must be globally unique across all data sources.
- For snapshot/balance data, always add `non_additive_dimension` to prevent incorrect time aggregation.
- **Do NOT create extra files** — only write semantic model YAML and metric YAML files. Do NOT manually create knowledge base documents, summaries, or any other files. The ONLY way to sync metrics to the Knowledge Base is via `end_metric_generation`.
