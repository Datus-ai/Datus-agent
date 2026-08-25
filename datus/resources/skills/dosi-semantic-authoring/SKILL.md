---
name: dosi-semantic-authoring
description: Dosi native OSI dataset, relationship, metric, and structured-window authoring contract
tags:
  - semantic-model
  - metrics
  - osi
  - dosi
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - semantic_modeling
  - gen_semantic_model
  - gen_metrics
---

# Dosi Semantic Authoring

Author the active Dosi semantic model as strict OSI core YAML. Use this skill for native document authoring rules; use the active adapter specification and native validation as the exact document and DATUS-extension contract. The node prompt owns target selection, result-set strategy, mutation order, validation, and synchronization.

## Model reusable semantics

- Keep one `semantic_model` per file and stable `snake_case` names. Preserve unrelated content; an upsert replaces the complete same-named object.
- Bind a dataset to a qualified physical table or a complete reusable SELECT. Declare every referenced physical column as a field with the active OSI dialect.
- Mark time fields with `dimension: {is_time: true}`. Keep other fields available as dimensions.
- For a dataset bound to a physical table, use `primary_key` to transcribe a source-declared physical primary key, and `unique_keys` to transcribe source-declared unique constraints and indexes. For a query-backed dataset, a source key holds only if the query preserves it: a one-to-many join repeats it, so validate against the result, not the base table. Declare a key the source does not declare only after full-table validation shows those columns are non-null and duplicate-free; a stated grain, a query pattern, or one partition is not evidence.
- Give a field a `label` when its column name is not what a reader would call it.
- Give a dataset `ai_context.instructions` when its grain or intended use does not follow from the description, and give a field `ai_context.synonyms` when users ask for it by a name the column does not carry. Leave both out otherwise: restating the description dilutes what a reader can act on.
- Define model-level relationships with aligned `from_columns` and `to_columns`; bind the target columns to one complete verified key.

## Author DATUS extensions

Put each Dosi-only key in the owning object's `custom_extensions` entry. Encode `data` as one JSON-object string and stamp it with the runtime `<datus_extension_version>`.

| Carrier | Supported keys |
|---|---|
| Dataset | `time_dimension` |
| Time field | `time_granularity` |
| Relationship | `join_type` |
| Metric | `dataset`, `time_dimension`, `fill_nulls_with`, `window`, `subject_path`, `unit`, `format` |

- Use `time_dimension` to resolve the business time when inference is ambiguous; qualify metric-level references when field names collide.
- Use `time_granularity` for the field's stored grain and `join_type` for `left` or `inner` relationship behavior.
- Use metric `dataset` to attribute an otherwise unbound aggregate such as `COUNT(*)`.
- Give each business metric a description, `ai_context.instructions`, and a three-level `subject_path`.

```yaml
- name: revenue
  description: Total order revenue
  ai_context: {instructions: Use order_date as business time.}
  expression: {dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount)}]}
  custom_extensions:
    - vendor_name: DATUS
      data: '{"v":"<datus_extension_version>","time_dimension":"orders.order_date","subject_path":["sales","revenue","total"],"unit":"USD"}'
```

## Author metrics and windows

- Express a base metric with its natural aggregate, ratio, or arithmetic expression. Put a durable metric condition inside its aggregate with `CASE WHEN`.
- Express each window result as a standalone metric whose OSI expression is one plain aggregate. Put the derivation in one structured `window` object.
- Choose the window form from the intended calculation:

| Intent | Form |
|---|---|
| Period comparison or following-period value | `pop` or general `offset` |
| Trailing buckets | `rolling` |
| Running or period-to-date value | `cumulative` |
| Explicit aggregate/statistical frame | general `frame` |
| Ranking or distribution | `rank` |
| First, last, or nth value | `value` |

- Derive time, query grain, ordering, partition, and frame from the requested analytic meaning. Treat query grain as a runtime argument.
- Use `order.by` values `time` or `value`. Use partition modes `query_dimensions`, `query_dimensions_except`, `time_bucket`, or `none`; qualify excluded fields.
- Supply `buckets` for `ntile`, `n` for `nth_value`, and a second plain aggregate metric for covariance or correlation.
- Reuse a window metric only when its base aggregate, time axis, calculation, ordering, partition, and frame all match.
- Preserve meaningful window nulls for missing comparison buckets or incomplete required frames.

Validate the final model with the native Dosi parser/compiler after the last mutation.
