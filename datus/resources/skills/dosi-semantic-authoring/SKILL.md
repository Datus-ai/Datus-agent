---
name: dosi-semantic-authoring
description: Dosi native OSI dataset, relationship, and metric authoring guidance
tags:
  - semantic-model
  - metrics
  - osi
  - dosi
version: "1.7.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - semantic_modeling
---

# Dosi Semantic Authoring

Use this skill for modeling decisions. The node prompt owns request scope, target selection, mutation order, verification and synchronization. The active engine owns executable syntax and constraints.

## Read the active contract

The appended OSI shapes retain all core members and enums; they are reference shapes, not a complete model to paste. Required members are root `version`/`semantic_model`, model `name`/`datasets`, dataset `name`/`source`, field and metric `name`/`expression`, and relationship `name`/`from`/`to`/`from_columns`/`to_columns`. An expression contains `dialects` entries with `dialect` and `expression`. Use the datasource's OSI dialect and reject unknown core keys.

The appended DATUS directory lists every active capability, its carriers and when to use it. Before authoring a capability with `schema_ref`, use `read_file` to read its full rules and examples from the generated authoring reference directory below. The directory entry gives its filename, such as `capability-window.yaml` or `capability-derive.yaml`; referenced definitions are included. Follow `related_references` when combining capabilities, such as windows with parameters. Read `osi-core.yaml` for full core details or `datus.yaml` for the complete extension contract when needed. An unread definition is not an unsupported capability.

Put Dosi metadata in at most one `custom_extensions` entry per object, with `vendor_name: DATUS` and `data` encoding one JSON object. Use runtime version `<datus_extension_version>` and the directory's `requires` rules. Never invent extension keys or infer support from an older example. Native validation remains authoritative.

## Sources and relationships

- Default to qualified physical tables or views. Declare every referenced source column as a field. Physical datasets, single-table aggregates, ratios and windows need no primary or unique key.
- Declare keys only from source DDL: preserve complete physical primary keys and whole-table unique constraints/indexes on plain columns, including composite order. Partial/expression indexes, sort keys, names, samples, JOINs and stated grain do not establish uniqueness. Never query data to discover or verify keys. Without DDL evidence, omit the key and continue independent modeling.
- For ordinary `many_to_one`, pair `from_columns`/`to_columns` in order and require a complete DDL-declared target key. DATUS `many_to_many` instead pairs conformed fact dimensions after separate aggregation; it requires compatible dimension meaning/encoding, not unique target rows. Read `cardinality` before using it; it cannot replace an arbitrary row-level JOIN.
- Try native fields, relationships, metric composition and windows before a query-backed dataset. Use the smallest query only for a concrete required row-set transformation those capabilities cannot express faithfully, or an explicit user requirement for a query source. Identify affected outputs and the gap or user requirement; preserve grain, retained rows, filters and NULLs. Missing keys, shared ranking populations, convenient preaggregation or a request to reproduce SQL do not justify SQL sources by themselves. Keep independent physical inputs and metrics. A SELECT establishes no new uniqueness; retain a source DDL key only when the query provably preserves it. Explain unsupported outputs and continue the rest.

## Fields and grain

- Distinguish original calculation grain, dataset row grain and query grouping. A queryable field does not prove a stored result can be reaggregated across it. Preserve business time and stored encoding; `metric_time` must resolve to that intended axis.
- Review live fields against requested analysis and SQL lineage, including group keys and projected attributes. GROUP BY is not the complete dimension inventory; field names or types alone do not decide roles, and not every field is a dimension. Mark business time with `dimension: {is_time: true}` and use the directory's `is_dimension` policy for grouping availability. Report meaningful omissions or uncertainty.
- `dimension.is_time` is independent of datatype; temporal datatypes default to a time role unless explicitly opted out. Omit unknown `datatype`; use `Opaque` with extensions for known nonportable types. `ai_context` accepts text or an object with instructions, synonyms and examples. Add field labels/synonyms or dataset instructions when they clarify names, grain or use; avoid restating descriptions.

## Metric definitions

- Use a plain aggregate, ratio or arithmetic expression for a base metric, with durable conditions in `CASE WHEN`. Use a derived filter for a governed subset; compose two or more reusable metrics rather than copying their calculations or making one-input passthroughs. Define dependencies first. Use explicit measure metadata only when engine-facing identity/behavior requires it.
- Use structured windows for comparisons, rolling, cumulative, rank, distribution or framed calculations. A window without a referenced base uses one plain aggregate as its OSI expression; follow the active rules for a referenced base and nested stages. Preserve time axis, fixed levels, denominator/ranking population, partition, ordering, frame, ties and semantic NULLs. Reuse a window only when all these agree.
- Parameterize only bounded business inputs that callers need to vary; keep stable policy in the definition. Read `params` and the capability using it together. Use declared defaults/domains and the node's checks for meaningful bindings. Combine capabilities only as their rules permit.
- Resolve ambiguous business time with `time_dimension`; use `time_granularity` for stored grain and metric `dataset` for otherwise unbound aggregates such as `COUNT(*)`. Give business metrics descriptions, useful `ai_context.instructions` and a three-level `subject_path`.

## Author advanced metrics without trial models

Every metric requires an explicit `expression`; `derive` and `window` do not request SQL generation. Define dependencies first or in the same batch. Read returned errors and the relevant capability reference rather than creating temporary probe metrics to discover syntax. Follow the node workflow for writing and verifying the model.

- **Base:** `expression` uses declared `dataset.field` columns and aggregates. Metric names are not dataset columns; `MAX(other_metric)` is not a reference to another metric.
- **Compose without windows:** `derive.expr` uses metric names; `expression` expands those names to their aggregate SQL. Both must preserve exactly the same filters, DISTINCT, arithmetic, zero-denominator and NULL behavior. Adding `NULLIF` or `COALESCE` to only one representation is a different definition. Missing, empty or names-only SQL does not request automatic generation.
- **Window:** put the ranking/frame in `window`, never `OVER(...)` in `expression`. With `window.base`, use that base's basic-mode expression as required by the active contract. `partition.exclude` lists dimensions that each query must include; it does not remove them from the result grouping.
- **Compose after windows:** `derive.expr` references the window metric names, so ranking/scoring executes after aggregation. Its mandatory SQL is a basic-mode approximation, not a replacement for the window calculation; follow stage-tier rules. Preserve the full business formula in `derive.expr` and query the final derived metric itself.

The following complete model shows the base → composition → ranking → scoring path. Read the active `derive`, `window` and `time` references before adapting it. The example's ratio deliberately uses unguarded division in both representations; choose any required zero/NULL policy consistently, rather than adding a guard to only the SQL. The score's SQL replaces the rank with its basic base value; Dosi executes the referenced rank through `derive`. Query its ranks/scores with `orders.market` and the intended time grain so the ranking population is retained.

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: sales_patterns
    datasets:
      - name: orders
        source: analytics.orders
        fields:
          - name: month
            expression: {dialects: [{dialect: <osi_dialect>, expression: month}]}
            dimension: {is_time: true}
            custom_extensions:
              - vendor_name: DATUS
                data: '{"v":"<datus_extension_version>","requires":["time"],"time":{"format":"%Y-%m","storage":"string","granularity":"month"}}'
          - name: market
            expression: {dialects: [{dialect: <osi_dialect>, expression: market}]}
          - name: order_id
            expression: {dialects: [{dialect: <osi_dialect>, expression: order_id}]}
          - name: amount
            expression: {dialects: [{dialect: <osi_dialect>, expression: amount}]}
    metrics:
      - name: revenue
        expression: {dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount)}]}
      - name: order_count
        expression: {dialects: [{dialect: <osi_dialect>, expression: COUNT(DISTINCT orders.order_id)}]}
      - name: average_order_value
        expression: {dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount) / COUNT(DISTINCT orders.order_id)}]}
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"<datus_extension_version>","derive":{"type":"compose","expr":"revenue / order_count"}}'
      - name: revenue_rank
        expression: {dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount)}]}
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"<datus_extension_version>","window":{"base":"revenue","rank":{"function":"rank","order":{"by":"value","direction":"desc"},"partition":{"mode":"query_dimensions_except","exclude":["orders.market"]}}}}'
      - name: market_count
        expression: {dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount)}]}
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"<datus_extension_version>","window":{"base":"revenue","frame":{"function":"count","scope":"partition","partition":{"mode":"query_dimensions_except","exclude":["orders.market"]}}}}'
      - name: revenue_score
        expression: {dialects: [{dialect: <osi_dialect>, expression: "CASE WHEN SUM(orders.amount) <= 3 THEN 90 WHEN SUM(orders.amount) >= SUM(orders.amount) - 2 THEN 0 ELSE 40 END"}]}
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"<datus_extension_version>","derive":{"type":"compose","expr":"CASE WHEN revenue_rank <= 3 THEN 90 WHEN revenue_rank >= market_count - 2 THEN 0 ELSE 40 END"}}'
```

The score's basic SQL is the mechanical expansion required by validation: `SUM(orders.amount) >= SUM(orders.amount) - 2` does not encode ranking. Runtime ranking and scoring come from `derive.expr` and its referenced windows.

## Identity and source semantics

- Keep one model per file and stable `snake_case` names. Upserts replace complete same-named objects: read before updating and preserve unrelated properties. Preserve an existing metric's host-managed `uid` verbatim, including a rename; omit it for new metrics or those without one. Never generate or copy a UID, or use `locked_metadata.uid` as its source.
- Preserve aggregate/DISTINCT semantics, complete Boolean logic, CASE constants, NULL versus zero, weighting, denominators and arithmetic/rounding order. Follow UNION branches and ALL/DISTINCT. Keep JOIN ON, WHERE, conditional aggregates, HAVING and post-window filtering at their original stages, including LEFT JOIN retained populations and zero-result cohorts.
- Follow upstream evidence for passthrough calculations and their original grain. Numeric types/samples or missing producer SQL cannot establish a definition. Unknown UDFs/templates and missing dependencies are specific gaps. Reuse a metric only when its definition agrees. Use the node's verification workflow to distinguish compilation, execution and numerical equivalence.

## Minimal physical model

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: sales
    datasets:
      - name: orders
        source: analytics.orders
        fields:
          - name: amount
            expression:
              dialects: [{dialect: <osi_dialect>, expression: amount}]
    metrics:
      - name: revenue
        description: Total order revenue
        ai_context: {instructions: Sum order amounts at the requested grouping.}
        expression:
          dialects: [{dialect: <osi_dialect>, expression: SUM(orders.amount)}]
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"<datus_extension_version>","subject_path":["sales","revenue","total"],"unit":"USD"}'
```
