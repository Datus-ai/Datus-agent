---
name: osi-metrics-authoring
description: OSI core schema metric authoring specification — metric expression shape, Datus extension hints, window/period-over-period semantics, and skip gates
tags:
  - metrics
  - osi
version: "1.1.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_metrics
---

# OSI Metrics Authoring

ADD metrics to an existing strict OSI (Open Semantic Interchange) core semantic model, capturing each metric's business meaning from SQL queries or natural language.

CRITICAL BOUNDARY: You author **OSI core semantics only**. You do NOT write MetricFlow YAML, `measure_proxy`, `type_params`, `measures:`, `ratio`, `cumulative`, or any execution-engine syntax. The Datus OSI compiler infers backing measures, picks the backend metric kind, and lowers to the execution engine. Do NOT write legacy MetricFlow `metric:` blocks.

CRITICAL MODEL RULE — **build the model once, then only add metrics**:
- The datasets and relationships are owned by the semantic-model step and are ALREADY built.
- The semantic model file is the durable OSI domain document. Load it, preserve existing datasets and relationships, and append metrics under `semantic_model[0].metrics`.
- A given logical dataset has one canonical definition shared by all metrics. Reuse that dataset by name in each metric's DATUS `dataset` hint.

The OSI expression dialect and target semantic model file for the current run are shown in the system prompt Workspace section — use those exact values (`<osi_dialect>` below stands for that dialect).

## What you produce

Metric definitions inside a valid OSI core document:

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: <target_model_name>
    datasets:
      - name: <existing_dataset_name>
        source: <existing_source>
        primary_key: [<existing_primary_key>]
        fields: [...]                         # preserve existing dataset definitions
    relationships: [...]                       # preserve existing relationships
    metrics:
      - name: <metric_name>                    # globally unique snake_case
        description: "<business definition>"
        ai_context:
          instructions: "<how AI should use this metric, including grain, conditions, time field, and join caveats>"
        expression:
          dialects:
            - dialect: <osi_dialect>
              expression: "COUNT(DISTINCT id)" # aggregate business expression; no OVER/LAG/RANK
        custom_extensions:
          - vendor_name: DATUS
            data: '{"dataset":"<existing_dataset_name>","time_dimension":"<date_column>","subject_path":["<domain>","<layer1>","<layer2>"],"format":"0.00%","unit":"<unit>"}'
```

Datus execution hints such as `dataset`, `time_dimension`, `metric_kind`, `inputs`, `numerator`, `denominator`, `window`, `grain_to_date`, `window_aggregation`, `offset_window`, `period_over_period`, `subject_path`, `format`, and `unit` MUST be encoded in the metric's DATUS `custom_extensions[].data` JSON string. They are not OSI core top-level metric fields.

## Authoring rules

1. **Reference, don't rebuild.** Every metric's DATUS `dataset` hint must reference an existing dataset of the semantic model. If a needed dataset, field, time field, or relationship is missing, do not invent a conflicting one; note it and minimally update the semantic model only if truly absent.
2. **Aggregates**: write the natural business expression in OSI core `expression.dialects[0].expression`, e.g. `COUNT(DISTINCT order_id)`, `SUM(amount)`, `AVG(score)`.
3. **Conditional aggregates**: keep the CASE inside the expression, e.g. `COUNT(DISTINCT CASE WHEN <condition> THEN id END)`. Preserve literal values exactly.
4. **Conditional semantics**: encode metric-specific business conditions inside the OSI core metric expression, e.g. `COUNT(DISTINCT CASE WHEN status = 'paid' THEN id END)`. Do NOT create a separate dataset for a metric-only condition. Fixed logical dataset scope belongs in the dataset `source` query plus `description`/`ai_context`. Do NOT bury query-time date ranges into the metric; time ranges are query parameters.
5. **Ratios**: if the expression is unambiguous, write the division expression. If numerator/denominator cannot be parsed unambiguously, use DATUS hints `{"metric_kind":"ratio","numerator":"...","denominator":"..."}`.
6. **Time-window metrics — do NOT simplify away.**
   - A window/cumulative/offset decoration over an existing metric defines a NEW standalone metric, NOT a reference to its base. Aggregate windows such as `SUM(x) OVER (... UNBOUNDED PRECEDING ...)`, `AVG(x) OVER (ROWS BETWEEN n PRECEDING ...)`, `MIN(x)`/`MAX(x) OVER (...)`, and `LAG(x) OVER (...)` each yield a new metric (e.g. `running_x`, `moving_n_x`, `previous_period_x`) even when the base metric `x` is already published. Never skip such a candidate as "already covered by the base metric".
   - Rolling / cumulative: the OSI core metric expression is the base aggregate itself plus DATUS hints `window` or `grain_to_date`.
   - If `analyze_metric_candidates_from_history` returns a cumulative/window candidate with `window`, `grain_to_date`, `window_aggregation`, `base_metric_name`, or `time_grain`, preserve those structured fields when authoring the metric.
   - Every metric with `window` or `grain_to_date` MUST include `window_aggregation` in the DATUS extension JSON. This tells execution how to combine ordered base-period values. Allowed values are `sum`, `avg`, `min`, `max`, `count`, and `row_count`. Use `row_count` only when the business meaning is the number of rows or periods in the window, not when counting business entities.
     ```yaml
     metrics:
       - name: revenue_l7d
         expression:
           dialects:
             - dialect: <osi_dialect>
               expression: "SUM(amount)"
         custom_extensions:
           - vendor_name: DATUS
             data: '{"dataset":"orders","time_dimension":"order_date","window":"7 days","window_aggregation":"sum","subject_path":["sales","revenue","trailing"]}'
     ```
   - Period-over-period (`LAG() OVER`, previous period, DoD/WoW/MoM/QoQ/YoY): publish reusable comparison outputs as fixed, standalone metrics. A comparison output is a business metric such as YoY rate, YoY delta, MoM rate, MoM delta, WoW ratio, or a previous-period value when that shifted value is the primary reusable business result on its own. Author the OSI expression as the base aggregate expression, and put the fixed comparison semantics in the DATUS `period_over_period` extension. When a SQL result presents current value, previous-period value, and comparison in one answer, publish the comparison metric as the reusable metric and describe current/previous values as comparison context computed from the same base aggregate.
     A monthly YoY SQL over revenue should publish one fixed monthly YoY metric:
     ```yaml
     metrics:
       - name: revenue_month_yoy
         description: "Monthly year-over-year revenue growth rate"
         expression:
           dialects:
             - dialect: <osi_dialect>
               expression: "SUM(amount)"
         custom_extensions:
           - vendor_name: DATUS
             data: '{"dataset":"orders","time_dimension":"order_date","period_over_period":{"time_grain":"month","offset_window":"1 year","calculation":"percent_change"},"subject_path":["sales","revenue","growth"],"format":"0.00%","unit":"%"}'
     ```
7. **Joins**: to group or slice by another table, use the existing semantic-model relationships. If the link is truly absent, add one OSI core relationship under the semantic model object:
   ```yaml
   relationships:
     - name: <fact_dataset>_to_<dimension_dataset>
       from: <fact_or_many_side_dataset>
       to: <dimension_or_one_side_dataset>
       from_columns: [<foreign_key_column>]
       to_columns: [<primary_or_unique_key_column>]
   ```
   Never put `relationships` inside a dataset. Do NOT use non-core fields such as `from_dataset`, `from_identifier`, `to_dataset`, `to_identifier`, `join_on`, `from_column`, or `to_column`.
8. **Not metrics**: detail/list queries (`SELECT DISTINCT ...`) and window/ranking (`ROW_NUMBER()`, `RANK() OVER`, TopN per group) are NOT metrics. SKIP them. Do NOT create a dataset/view here and never force them into a metric. If the discovery tool returns `metric_generation_skips` or rank-like `derived_datasource_recommendations`, treat that SQL as skipped.
9. Use clear English `snake_case` metric names; metric names must be globally unique. Every metric MUST include `description` and `ai_context`. Put the business definition in `description`; put LLM-facing usage guidance in `ai_context.instructions`, including grain, metric-specific conditions, time field, and join caveats.
10. **Subject classification (required).** Every metric MUST carry a `subject_path` in its DATUS extension, encoded as an ordered `[domain, layer1, layer2]` list (e.g. `["sales","revenue","growth"]`). Choose the classification exactly as instructed by the **Subject Classification** section of the system prompt — same required categories, same reuse-or-create rule, same `{domain}/{layer1}/{layer2}` hierarchy the MetricFlow path uses; the only difference is the carrier (a DATUS `subject_path` list here, a `locked_metadata.tags` entry there).

## Hard skip gate

Before writing any YAML, classify the source SQL:

- If the SQL has no aggregate output (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, ratio, rolling/cumulative aggregate, etc.) and primarily returns row-level fields (`SELECT DISTINCT ac_name, ac_code, ...`, list/detail/ranking), then it is not a metric request.
- For non-metric SQL, do not invent convenience metrics such as `*_count`, `*_avg_duration`, `*_max_sr`, or `*_min_sr` just because the table contains those columns.
- For non-metric SQL, do not call `write_file`, `validate_semantic`, `query_metrics`, or `end_metric_generation`. Report `status: "skipped"` with `metric_file: null` in the final JSON.
- For TopN per group / ranking-window SQL (`ROW_NUMBER`, `RANK`, `DENSE_RANK`) do the same: skip metric generation. A derived dataset/view may be a future modeling task, but it is outside this `gen_metrics` run.

## Workflow notes

- FIRST, load the existing model from the target semantic model file to learn dataset names, fields, time fields, and relationships. Call `list_metrics()` to see metrics that already exist.
- For provided SQL/history, call `analyze_metric_candidates_from_history` before writing files. Use `direct_metric_candidates` for base metrics and fixed `period_over_period` final metrics; use `derived_metric_candidates` only for second-stage OSI metrics that explicitly depend on published input metrics. If it returns `metric_generation_skips`, skip those SQLs instead of writing metric YAML.
- Candidate-plan compliance: every candidate in `direct_metric_candidates` and `derived_metric_candidates` (including cumulative, rolling-window, and period_over_period candidates) MUST end this run either published via `end_metric_generation` or listed in your final `output` with a concrete blocker such as a validation failure. "Covered by an existing base metric" is never a valid blocker for a cumulative/window/period_over_period candidate.
- Reference, reconcile, reuse: point each metric's DATUS `dataset` hint at an existing dataset. If a metric with the same meaning already exists (`check_semantic_object_exists(name, kind='metric')`), reuse/skip it. "Same meaning" requires the same aggregation AND the same window/offset semantics: a base aggregate never covers its cumulative/rolling/period-over-period variants, so `running_x`/`moving_x`/`previous_x` candidates must still be published when only `x` exists. For a derived metric, make sure its input metrics already exist.
- From SQL: find the table (FROM), aggregate expression(s), and business conditions vs query-time ranges. Anchor the metric on the aggregated table's existing dataset; encode metric-specific conditions with CASE inside the metric expression.
- When a required business input is missing or ambiguous, ASK for the business semantics; do not guess.
- Call `validate_semantic(scope="all")` after writing OSI metrics and fix errors until it passes.
- Call `query_metrics(metrics=[...], dry_run=True)` for every generated metric name before publishing.
- After validation and dry-run pass, call `end_metric_generation(metric_file="<target model file>", semantic_model_file="<target model file>", metric_sqls_json="<dry-run SQL JSON>")`.
