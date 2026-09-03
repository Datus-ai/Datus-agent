---
name: dosi-semantic-authoring
description: Dosi native OSI dataset, relationship, and metric authoring guidance
tags:
  - semantic-model
  - metrics
  - osi
  - dosi
version: "1.1.0"
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

## Choose DATUS metric capabilities

Put Dosi-only metadata in the owning object's DATUS `custom_extensions` entry. Encode `data` as one JSON-object string and stamp it with the runtime `<datus_extension_version>`. The injected active DATUS extension specification is authoritative for supported carriers, keys, exact shapes, enums, constraints, and examples; never invent a field from this conceptual guide.

- Prefer a plain base metric when one aggregate or arithmetic expression completely represents the business meaning.
- Use a derived filter metric when the business concept narrows one reusable base metric. Use a derived compose metric only when the result combines two or more reusable metrics. Author and validate every referenced base metric first; do not inline its calculation again or create a one-input passthrough.
- Use a structured window metric for period comparison, rolling, cumulative, ranking, distribution, or framed statistical calculations. Keep the underlying OSI expression as the plain aggregate described by the active contract.
- Use a parameterized metric only when different callers must supply a bounded runtime business input to the same reusable definition. Stable policy belongs in the metric itself. Declare each parameter's type, default, and allowed values or bounds according to the active contract.
- Use explicit measure metadata only when the metric needs a stable engine-facing measure identity or behavior that cannot be inferred from its OSI expression.
- Combine capabilities only when the active contract explicitly permits their keys and dependencies on the same carrier. If the requested capability is absent from that contract, report it as unsupported by the installed engine instead of approximating it in YAML.

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

## Author base and window metrics

- Express a base metric with its natural aggregate, ratio, or arithmetic expression. Put a durable metric condition inside its aggregate with `CASE WHEN`.
- Express each window result as a standalone metric whose OSI expression is one plain aggregate. Put the derivation in one structured `window` object.
- Choose the window family from the intended calculation and use the exact form advertised by the active contract.

- Derive time, query grain, ordering, partition, and frame from the requested analytic meaning. Treat query grain as a runtime argument.
- Reuse a window metric only when its base aggregate, time axis, calculation, ordering, partition, and frame all match.
- Preserve meaningful window nulls for missing comparison buckets or incomplete required frames.

Validate the final model with the native Dosi parser/compiler after the last mutation.

For a parameterized metric, inspect its `param_schema` in `list_metrics`, then verify query behavior with `query_metrics(params={...}, dry_run=True)`. Exercise the default and meaningful enum/boundary or list-valued cases; never invent undeclared parameter names. Native validation proves the definition compiles, while this optional query check proves a user-requested binding shape.
