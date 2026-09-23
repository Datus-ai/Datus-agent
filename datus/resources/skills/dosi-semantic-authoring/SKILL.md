---
name: dosi-semantic-authoring
description: Dosi native OSI dataset, relationship, and metric authoring guidance
tags:
  - semantic-model
  - metrics
  - osi
  - dosi
version: "1.4.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - semantic_modeling
  - gen_semantic_model
  - gen_metrics
---

# Dosi Semantic Authoring

Author the active Dosi semantic model as strict OSI core YAML. Use this skill for native document authoring rules; use the active adapter specification and native validation as the exact document and DATUS-extension contract. The node prompt owns target selection, mutation order, validation, and synchronization.

## Model reusable semantics

- Keep one `semantic_model` per file and stable `snake_case` names. Preserve unrelated content; an upsert replaces the complete same-named object.
- Prefer binding a dataset to a qualified physical table, and declare every referenced physical column as a field with the active OSI dialect. Do not create an unused physical dataset solely as a placeholder for query-backed SQL. A dataset is useful when it supplies requested fields or relationships; when metrics are in scope, keep directly expressible metrics on their physical datasets instead of relocating them wholesale to query-backed SQL.
- Treat source SQL as evidence for reusable semantics rather than a required persisted result shape. Keep request-specific time ranges, grouping selections, ordering, and result layout as query-time concerns unless they define durable business semantics.
- When metrics are in scope, inspect the active contract before creating a query-backed dataset and decompose the requested result into the smallest supported native metric DAG. Define and validate reusable leaf metrics on physical datasets first, then reuse them through supported relationships, conformed dimensions, filter, compose, and window metrics. One dimensioned metric DAG may start from multiple physical datasets; it does not need one shared SQL-backed dataset.
- A SELECT that joins several facts and aggregates them to a common reporting grain is not a durable reusable row set merely because it makes later metrics easier to author. Do not use query-backed SQL only to align common dimensions, give downstream metrics one dataset, precompute a supported aggregate or derived metric, or avoid a relationship or conformance shape that the active contract supports.
- Use a complete reusable SELECT only for independently reusable row-set semantics, such as a cohort, mapping, deduplication, or grain transformation, or for a residual transformation unavailable in the active contract. During planning, treat capabilities advertised by the active contract as available and plan the smallest native DAG without waiting for pre-write validation. Independently reusable row-set transformations may be query-backed in the initial plan. For analytic fallback, first author and validate the planned native DAG; only after a concrete blocking result may a revised plan move the unsupported residual into SQL. Name the exact missing capability or blocking validation result in the dataset description; convenience, uncertainty, or a missing key alone is not evidence of a capability gap. Even when a residual query-backed dataset is necessary, keep independently useful leaf metrics on their physical datasets. A datasets-only run must not encode out-of-scope metric calculations in a query-backed dataset.
- Mark time fields with `dimension: {is_time: true}`. For other fields, let the engine infer their role unless model evidence supports an explicit dimension declaration under the active contract.
- Use source DDL as the only evidence for new key declarations. For a physical table, transcribe its declared physical primary key into `primary_key` and its declared unique constraints or whole-table unique indexes on plain columns into `unique_keys`. Preserve each complete composite key and its declared column order. Partial or expression indexes do not establish a whole-table key on their named columns. ClickHouse `PRIMARY KEY`/`ORDER BY` and StarRocks/Doris `DUPLICATE KEY` are sort keys, not uniqueness declarations.
- Do not execute data queries to discover or verify keys, including full-table NULL/duplicate checks. Samples, approximate distinct counts, column names, SQL JOINs, and stated grain are not substitutes for DDL key declarations.
- If the DDL is unavailable or declares no usable key, leave the key undeclared and continue modeling fields, datasets, and independent metrics. Do not block the whole request or ask to scan the table to fill the gap. For a query-backed dataset, retain a DDL-declared source key only when the query provably preserves it; a one-to-many join can repeat it. Otherwise leave the key undeclared without scanning the source or query result.
- Give a field a `label` when its column name is not what a reader would call it.
- Give a dataset `ai_context.instructions` when its grain or intended use does not follow from the description, and give a field `ai_context.synonyms` when users ask for it by a name the column does not carry. Leave both out otherwise: restating the description dilutes what a reader can act on.
- Define ordinary model-level relationships with aligned `from_columns` and `to_columns`; bind the target columns to one complete DDL-declared key that holds at the target dataset's grain. When the active contract supports deliberate fact-to-fact conformance, use its cardinality semantics and aggregate-before-join behavior instead of inventing target uniqueness, and create such an edge only when a requested cross-fact metric needs it. If neither a keyed relationship nor supported conformance applies, omit the relationship and continue with independent datasets and metrics. For a requested cross-dataset result, isolate the minimal query-backed residual allowed above or explain why the result is unsupported. Do not invent a key to make a relationship or metric compile.

## Choose DATUS metric capabilities

Put Dosi-only metadata in the owning object's DATUS `custom_extensions` entry. Encode `data` as one JSON-object string and stamp it with the runtime `<datus_extension_version>`. The injected active DATUS extension specification is authoritative for supported carriers, keys, exact shapes, enums, constraints, and examples; never invent a field from this conceptual guide.

- Prefer a plain base metric when one aggregate or arithmetic expression completely represents the business meaning.
- Use a derived filter metric when the business concept narrows one reusable base metric. Do not specialize otherwise identical metric definitions for literal dimension-member values when query-time filtering or grouping preserves the same business meaning; specialize only when the request or semantics genuinely differ. Use a derived compose metric for reusable scalar arithmetic over one or more existing metrics. Author and validate every referenced metric first; do not inline its calculation again or create an identity passthrough.
- Use a structured window metric for period comparison, rolling, cumulative, ranking, distribution, or framed statistical calculations. Author its expression, optional base, and window payload exactly as the active contract requires.
- Use a parameterized metric only when different callers must supply a bounded runtime business input to the same reusable definition. Stable policy belongs in the metric itself. Declare each parameter's type, default, and allowed values or bounds according to the active contract.
- Use explicit measure metadata only when the metric needs a stable engine-facing measure identity or behavior that cannot be inferred from its OSI expression.
- Combine capabilities only when the active contract explicitly permits their keys and dependencies on the same carrier. If the requested capability is absent from that contract, report it as unsupported by the installed engine instead of approximating it in YAML.
- Keep the helper DAG minimal: retain a helper only when another requested metric needs it under the active contract or it is independently reusable. Reuse one helper when its base metric, grain, partition, ordering, and frame match; do not clone a helper chain for literal dimension values when one dimensioned chain can express the same semantics. Do not mechanically expose every calculation step as a business metric.
- Keep the complete metric set minimal: every authored metric must represent a requested business concept, be required to express another requested metric under the active contract, or be independently reusable.

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
- Express each window result as a standalone metric and put the derivation in one structured `window` object. Follow the active contract for whether its expression is a plain aggregate or a basic-mode approximation and whether its base may reference another metric.
- Choose the window family from the intended calculation and use the exact form advertised by the active contract.

- Derive time, query grain, ordering, partition, and frame from the requested analytic meaning. Treat query grain as a runtime argument.
- Reuse a window metric only when its base aggregate, time axis, calculation, ordering, partition, and frame all match.
- Preserve meaningful window nulls for missing comparison buckets or incomplete required frames.

For a parameterized metric, inspect its `param_schema` in `list_metrics`, then verify query behavior with `query_metrics(params={...}, dry_run=True)`. Exercise the default and meaningful enum/boundary or list-valued cases; never invent undeclared parameter names. Native validation proves the definition compiles, while this optional query check proves a user-requested binding shape.
