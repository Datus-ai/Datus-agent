---
name: dosi-metrics-authoring
description: Strict OSI metric authoring using Dosi native D-TIME, D-FILL, and D-WINDOW semantics
tags:
  - metrics
  - osi
  - dosi
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_metrics
---

# Dosi Native Metrics Authoring

Create, update, or explicitly delete metrics in an existing strict OSI core semantic model. Dosi executes this document directly in Rust using the DATUS extension profile below.

## Model and mutation boundary

- Bind one existing model with `bind_osi_semantic_model_target` before writing metrics. If no model exists, report that `gen_semantic_model` must run first.
- Reuse existing datasets and relationships. Use `upsert_osi_datasets` only for a metric-required missing field or dataset; it replaces that named dataset, so pass its complete final object.
- Use `upsert_osi_metrics` for all metric creates/updates and `delete_osi_metrics` only for explicitly requested names. Preserve unrelated model content.
- Every physical column used by a metric expression must already be a field on its owning dataset. Qualified dataset columns are preferred: `SUM(orders.amount)`, not `SUM(amount)`.
- The OSI dialect and version come from the bound document/system context. Datus-only keys belong in a `vendor_name: DATUS` extension whose `data` is a JSON string.

## Base metrics

- Write aggregates as a single natural aggregate expression such as `COUNT(DISTINCT orders.order_id)`, `SUM(orders.amount)`, or `AVG(reviews.score)`.
- Keep confirmed durable metric conditions inside the aggregate with `CASE WHEN`. Ordinary query slicing, ad-hoc entity filters, time bounds, access policies, `HAVING`, and join predicates are not metric definitions.
- A ratio/expression metric may use its natural expression when the model can infer its inputs. Use DATUS `dataset` only to resolve otherwise unattributable expressions such as `COUNT(*)` in a multi-dataset model.
- Every metric must have a globally unique `snake_case` name, `description`, `ai_context.instructions`, and a three-level DATUS `subject_path`.
- Model metrics as aggregate business values. Keep detail/list and positional `ROW_NUMBER`/`RANK`/`DENSE_RANK` outputs in query or reporting workflows.

Base metric example:

```yaml
- name: revenue
  description: "Total order revenue"
  ai_context:
    instructions: "Use order_date as business time; query at day or coarser grains."
  expression:
    dialects:
      - dialect: <osi_dialect>
        expression: "SUM(orders.amount)"
  custom_extensions:
    - vendor_name: DATUS
      data: '{"v":"1.2","time_dimension":"orders.order_date","subject_path":["sales","revenue","total"],"unit":"USD"}'
```

## Native D-WINDOW

A window metric is a new standalone metric. Use exactly one plain base aggregate as its OSI expression and put one structured `window` object in the metric's DATUS payload.

For every SQL plan output carrying `native_window`, bind that output to a standalone metric with the matching D-WINDOW `type`, function/offset, periods, and calculation. Use `base_expression` as the base aggregate, qualify it against the reusable dataset fields, and apply the supplied business time. Reuse an existing window metric when its base aggregate, time axis, and complete window contract match.

D-WINDOW obtains its time axis from metric-level `time_dimension`, otherwise from the unique dataset primary time field. When the axis is ambiguous, ask for the intended business time or repair D-TIME. The time field's `time_granularity` is its stored/native grain. The requested query grain remains a runtime argument.

Supported forms:

```json
{"v":"1.2","window":{"type":"rolling","function":"avg","periods":3}}
{"v":"1.2","window":{"type":"rolling","function":"sum","periods":7,"require_full_window":true}}
{"v":"1.2","window":{"type":"cumulative","function":"sum"}}
{"v":"1.2","window":{"type":"cumulative","function":"sum","reset":"year"}}
{"v":"1.2","window":{"type":"pop","offset":"1 month","calculation":"percent_change"}}
{"v":"1.2","window":{"offset":{"count":1,"granularity":"year"},"calculation":"delta"}}
```

- Frame functions: `sum`, `avg`, `min`, `max`, `count`. `count` counts buckets in the frame, not source entities.
- Rolling `periods` is the number of time buckets including the current bucket. Use `require_full_window: true` when incomplete leading frames should return NULL.
- Cumulative `reset` may be `week`, `month`, `quarter`, or `year`; omit it for an all-time running value.
- Offset calculations: `value`, `delta`, `percent_change`, or `ratio`; omitted `calculation` defaults to `percent_change`.
- Period-over-period query grain is not frozen by the model. Record required business grain in `description`/`ai_context.instructions` and keep a queryability contract at that grain.
- Final D-WINDOW outputs retain NULL when no prior bucket or full frame exists.

Complete rolling example:

```yaml
- name: revenue_rolling_3_month_avg
  description: "Average monthly revenue over the current and prior two months"
  ai_context:
    instructions: "Query with metric_time at month grain; leading partial windows are allowed."
  expression:
    dialects:
      - dialect: <osi_dialect>
        expression: "SUM(orders.amount)"
  custom_extensions:
    - vendor_name: DATUS
      data: '{"v":"1.2","time_dimension":"orders.order_date","window":{"type":"rolling","function":"avg","periods":3},"subject_path":["sales","revenue","trailing"],"unit":"USD"}'
```

## Queryability and publication

- Keep every planner `queryability_contract` as one complete grouping combination. For time-series/window metrics, the query input is the reserved dimension `metric_time` plus a separate `time_granularity`; use the engine's suffixed result name only as an output/order key.
- Publish each requested rolling, cumulative, previous-value, delta, ratio, or growth result as its own standalone metric, alongside the base metric when requested.
- Re-upsert an already-correct requested metric so retries can complete validation and KB publication.
- After mutation, call `validate_semantic` through Dosi and fix every native issue. Then call `publish_metrics` with every requested output binding. Publication dry-runs every required queryability contract; fix the model or contract until each one succeeds.
- For deletion-only publication, omit metric output bindings. Missing requested delete names are an idempotent success.

For existing-model migration, use the one-shot `datus-dosi-migrate` checker and apply the conversions it reports as safe.
