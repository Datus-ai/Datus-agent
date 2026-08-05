---
name: osi-metrics-authoring
description: OSI core schema metric authoring specification — metric expression shape, Datus extension hints, window/period-over-period semantics, and skip gates
tags:
  - metrics
  - osi
version: "1.3.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_metrics
---

# OSI Metrics Authoring

Create, update, or explicitly delete metrics in an existing strict OSI (Open Semantic Interchange) core semantic model, capturing each metric's business meaning from SQL queries or natural language.

CRITICAL BOUNDARY: You author **OSI core semantics only**. You do NOT write MetricFlow YAML, `measure_proxy`, `type_params`, `measures:`, `ratio`, `cumulative`, or any execution-engine syntax. The Datus OSI compiler infers backing measures, picks the backend metric kind, and lowers to the execution engine. Do NOT write legacy MetricFlow `metric:` blocks.

CRITICAL MODEL RULE — **reuse the model, then make only narrow changes required by the requested metrics**:
- Start from the datasets and relationships built by the semantic-model step.
- Use `upsert_osi_datasets` only when a requested metric needs a missing field or a normal/query-backed dataset. Pass the complete final dataset object because the tool replaces that named dataset while preserving all other model content.
- Use `delete_osi_datasets` only when the corrected final plan no longer needs a named dataset. Never infer relationship or metric cascade deletions.
- Use `upsert_osi_metrics` to create or update metrics and `delete_osi_metrics` to remove explicitly requested names under `semantic_model[0].metrics`.
- General filesystem writes and relationship/model-metadata edits remain out of scope. If the target model is missing, stop and report that `gen_semantic_model` must run first.
- A given logical dataset has one canonical definition shared by all metrics. Reuse that dataset by name in each metric's DATUS `dataset` hint.

The OSI expression dialect is shown in the system prompt Workspace section. Bind one existing target with `bind_osi_semantic_model_target` before writing metrics. A path or model name supplied in the request is only a selection hint; if it is absent or does not bind, call `list_existing_osi_semantic_models` and compare the live candidates using SQL tables, dataset metadata, and business meaning (`<osi_dialect>` below stands for the active dialect).

## What you produce

Metric definitions inside a valid OSI core document:

```yaml
version: <osi_version>            # use the exact version from the existing semantic model file
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
              expression: "COUNT(DISTINCT <existing_dataset_name>.id)" # aggregate business expression; no OVER/LAG/RANK
        custom_extensions:
          - vendor_name: DATUS
            data: '{"time_dimension":"<date_column>","subject_path":["<domain>","<layer1>","<layer2>"],"format":"0.00%","unit":"<unit>"}'
```

Qualify every column in the metric expression with its **dataset name** (`SUM(orders.amount)`, not `SUM(amount)`). The compiler resolves the metric's dataset from the qualifier and strips it before execution, so a DATUS `dataset` hint is only needed when the expression references no qualified columns (e.g. some derived metrics).

Datus execution hints such as `dataset`, `time_dimension`, `metric_kind`, `inputs`, `numerator`, `denominator`, `window`, `grain_to_date`, `window_aggregation`, `offset_window`, `period_over_period`, `subject_path`, `format`, and `unit` MUST be encoded in the metric's DATUS `custom_extensions[].data` JSON string. They are not OSI core top-level metric fields.

## Authoring rules

1. **Reference, then repair narrowly when required.** Every metric must anchor on a dataset of the bound semantic model — via qualified column names in its expression, or via a DATUS `dataset` hint. If a required field or dataset is missing, add it with `upsert_osi_datasets`. Prefer a query-backed dataset when the requested output cannot be represented by normal fields and metric expressions; do not use one merely to bake a query-time filter into a new metric. If a required relationship is missing and a query-backed dataset cannot preserve the semantics, report the prerequisite because this workflow does not edit relationships.
2. **Aggregates**: write the natural business expression in OSI core `expression.dialects[0].expression`, qualifying columns with the owning dataset name, e.g. `COUNT(DISTINCT orders.order_id)`, `SUM(orders.amount)`, `AVG(reviews.score)`. Every referenced row-level column must be declared in the owning dataset's `fields`; aggregated-only columns, including business keys used by `COUNT(DISTINCT ...)`, are plain fields without a `dimension:` block.
3. **Conditional aggregates**: after a condition has been classified as part of the durable metric definition, keep the CASE inside the expression, e.g. `COUNT(DISTINCT CASE WHEN <condition> THEN id END)`. Preserve literal values exactly.
4. **Filter ownership — classify before authoring.** A predicate in a source SQL `WHERE` is a query-time filter by default; do not automatically copy every predicate into a metric expression. Promote it to a metric-specific business condition only when durable intent is established: the user explicitly requests a named filtered KPI, the aggregate already contains `CASE WHEN` or an equivalent filter, or an established business/catalog definition requires the predicate on every use. A final output alias (`AS <name>`) that names the filtered concept is supporting evidence, not proof by itself; likewise, a natural-language question that merely describes one query slice does not by itself define a new metric.
   - Equality, range, or membership predicates over ordinary queryable dimensions default to query-time slicing. Reuse the base metric rather than creating one metric per dimension value.
   - Time bounds, ad hoc entity selections, sampling/debug predicates, and access-control conditions remain query-time or policy-layer concerns. `HAVING` remains a post-aggregation constraint, and `JOIN ... ON` belongs in relationships or dataset modeling.
   - If durable filtered-metric intent is absent or ambiguous, reuse the base metric with the source predicate as query context; ask when possible instead of inventing a filtered metric.
   - For a confirmed filtered metric, encode the condition inside each affected aggregate, e.g. `COUNT(DISTINCT CASE WHEN status = 'paid' THEN id END)`. Do NOT create a separate dataset for a metric-only condition. Fixed logical dataset scope belongs in the dataset `source` query plus `description`/`ai_context`. Use a query-backed dataset only when the confirmed output semantics cannot be represented by normal fields, relationships, and metric expressions. Do NOT bury query-time date ranges into the metric; time ranges are query parameters.
5. **Ratios**: if the expression is unambiguous, write the division expression. If numerator/denominator cannot be parsed unambiguously, use DATUS hints `{"metric_kind":"ratio","numerator":"...","denominator":"..."}`.
6. **Time-window metrics — do NOT simplify away.**
   - A window/cumulative/offset decoration over an existing metric defines a NEW standalone metric, NOT a reference to its base. Aggregate windows such as `SUM(x) OVER (... UNBOUNDED PRECEDING ...)`, `AVG(x) OVER (ROWS BETWEEN n PRECEDING ...)`, `MIN(x)`/`MAX(x) OVER (...)`, and `LAG(x) OVER (...)` each yield a new metric (e.g. `running_x`, `moving_n_x`, `previous_period_x`) even when the base metric `x` is already published. Never skip such a candidate as "already covered by the base metric".
   - Rolling / cumulative: the OSI core metric expression is the base aggregate itself plus DATUS hints `window` or `grain_to_date`.
   - Infer `window`, `grain_to_date`, `window_aggregation`, base metric, and time grain from the original SQL plus the live semantic model; the compact plan does not freeze these implementation details.
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
7. **Joins**: to group or slice by another table, use an existing relationship. If the link is absent, use a query-backed dataset when it can preserve the exact output semantics; otherwise report that `gen_semantic_model` must add the relationship.
8. **Not metrics**: final detail/list fields and positional ranking outputs (`ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`) are not metric outputs. Skip outputs marked `non_metric`; do not infer metrics from them.
9. Use clear English `snake_case` metric names; metric names must be globally unique. Every metric MUST include `description` and `ai_context`. Put the business definition in `description`; put LLM-facing usage guidance in `ai_context.instructions`, including grain, metric-specific conditions, time field, and join caveats.
10. **Subject classification (required).** Every metric MUST carry a `subject_path` in its DATUS extension, encoded as an ordered `[domain, layer1, layer2]` list (e.g. `["sales","revenue","growth"]`). Choose the classification exactly as instructed by the **Subject Classification** section of the system prompt — same required categories, same reuse-or-create rule, same `{domain}/{layer1}/{layer2}` hierarchy the MetricFlow path uses; the only difference is the carrier (a DATUS `subject_path` list here, a `locked_metadata.tags` entry there).

## Hard skip gate

Before binding a target or writing YAML, classify the source SQL:

- This gate applies to metric creation requests, not an explicit request to delete named existing metrics. A named deletion remains in scope even when accompanying SQL is non-aggregate context.

- If the SQL has no aggregate output (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, ratio, rolling/cumulative aggregate, etc.) and primarily returns row-level fields (`SELECT DISTINCT ac_name, ac_code, ...`, list/detail/ranking), then it is not a metric request.
- For non-metric SQL, do not invent convenience metrics such as `*_count`, `*_avg_duration`, `*_max_sr`, or `*_min_sr` just because the table contains those columns.
- For a non-metric creation request that is not an explicit named deletion, do not bind a target and do not call `upsert_osi_metrics`, `validate_semantic`, `query_metrics`, or `publish_metrics`. Report `status: "skipped"`, `skip_reason: "not_a_metric"`, and `metric_file: null` in the final JSON.
- For TopN per group / ranking-window SQL, mark positional ranking outputs as non-metrics unless the business request explicitly treats them as KPIs. Define metric outputs against a query-backed dataset when required, creating or correcting it with `upsert_osi_datasets`.

## Workflow notes

- For requests classified as metrics, bind an existing model target before writes. Try an exact YAML path or model name from the request, then fall back to the complete live inventory. Compare candidates using physical SQL tables, dataset names/sources/descriptions, and business meaning; read plausible files when necessary. Bind only one unique target. If several remain plausible, ask the user when possible or return the selection-required blocker. Load the bound file to learn dataset names, fields, time fields, relationships, and existing metrics; inspect live catalog state when reuse is relevant.
- For provided SQL/history, consume the compact plan returned by `prepare_sql_modeling_plan`. Treat metric-role `outputs` as the current requested results. If SQL semantics, the bound model, compilation, or warehouse feedback reveals a bad role, expression, generated query, dimension qualification, or time grain, correct the complete plan with `update_sql_modeling_plan` before publishing.
- Reconcile by stable output identity: every metric-role `outputs[].output_id` must be published, reused, or reported with a concrete blocker. Multiple equivalent outputs may bind to the same metric name; retain every binding so the source evidence remains traceable. Inspect the bound YAML to decide whether a normal metric or a query-backed dataset is appropriate, and use the narrow dataset tools when the required final dataset is missing or incomplete.
- Keep every `queryability_contract`. Each contract is one complete grouping combination, not a list of independent dimension checks. Qualify its dimensions to the exact names exposed by the bound semantic model; do not remove a contract to bypass a failure.
- Reference and reconcile: point each metric's DATUS `dataset` hint at an existing dataset. "Same meaning" requires the same aggregation AND the same window/offset semantics: a base aggregate never covers its cumulative/rolling/period-over-period variants, so `running_x`/`moving_x`/`previous_x` candidates must still be published when only `x` exists. For a derived metric, make sure its input metrics already exist.
- Every requested metric candidate must pass through `upsert_osi_metrics`, validation, and publish even when the bound YAML already contains the correct definition. Reuse that exact metric object unchanged; the upsert is a byte-preserving no-op but records the exact publish scope. This makes retries repair a prior run that wrote YAML but stopped before KB sync. Existing metrics are not a `skipped` outcome.
- Delete only when the user explicitly asks to remove a named metric. Call `delete_osi_metrics(path="<target model file>", metric_names=["<name>"])`. Missing names are a successful `already_absent` result so a retry can still clean stale Knowledge Base rows. Do not invent cascade rules: validate the resulting model, then use the validation result and user intent to decide whether to delete another metric, restore one with `upsert_osi_metrics`, or edit a dependent definition.
- From SQL: find the table (FROM), aggregate expression(s), and classify every predicate using **Filter ownership** above. Anchor the metric on the aggregated table's existing dataset; encode only confirmed metric-specific conditions with CASE inside the metric expression.
- When a required business input is missing or ambiguous, ASK for the business semantics; do not guess.
- Call `upsert_osi_metrics(path="<target model file>", metrics_json="<JSON array of complete OSI metric objects>")` once per coherent metric batch. This tool preserves datasets and relationships, appends new metrics, and replaces existing metrics by name.
- Call `upsert_osi_datasets(path="<target model file>", datasets_json="<JSON array of complete OSI dataset objects>")` to add or replace only datasets required by this metric request. Call `delete_osi_datasets` only for named datasets removed by the corrected final plan. Both tools preserve unrelated datasets, relationships, metrics, and model metadata.
- Call `validate_semantic(semantic_model_name="<bound semantic model name>")` after changing OSI metrics. When any deletion occurred, pass `scope="semantic_model"`; this keeps real model and metric errors while tolerating the valid no-metrics state. Fix errors with the appropriate `delete_osi_metrics` or `upsert_osi_metrics` call until validation passes. Do not pass a custom `checks` subset.
- After validation passes, call `publish_metrics(metric_file="<target model file>", metric_output_bindings=[{"output_id":"...","metric_name":"..."}])`. For each planner-owned `queryability_contract`, the tool queries all bound metrics with the contract's complete dimension list and optional time grain, then requires semantic compilation and a live warehouse dry-run before syncing. Correct the artifact or compact plan and retry on failure. Return the same binding array in final JSON. For deletion-only publication, omit bindings and no metric dry-run is required because no metric is being authored.
