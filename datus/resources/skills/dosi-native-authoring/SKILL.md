---
name: dosi-native-authoring
description: Dosi native DATUS extension profile for strict OSI semantic-model authoring
tags:
  - semantic-model
  - osi
  - dosi
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# Dosi Native Semantic-Model Authoring

Author strict OSI core YAML for the active `dosi` semantic adapter. Dosi validates and executes the document directly; it does not load the Python OSI adapter or lower through MetricFlow.

## Document and dataset contract

- Keep exactly one `semantic_model` object per YAML file. Give the model, every dataset, field, relationship, and metric a stable `snake_case` name.
- A dataset `source` is either a qualified physical table or a complete SELECT query. Declare every physical column referenced by expressions as a field on its owning dataset.
- Every field expression uses the active datasource's OSI dialect. Add `dimension: {is_time: true}` only to time fields; other fields remain queryable dimensions without inventing adapter-specific roles.
- Declare `primary_key` only from source metadata or an explicit contract. Add `unique_keys` only after full-table uniqueness and non-nullness checks.
- Declare relationships once at model level with `from`, `to`, ordered `from_columns`, and ordered `to_columns`. Both column lists have the same length, and the target list matches one complete target key.
- Preserve unrelated datasets, relationships, metrics, descriptions, AI context, and extensions when editing an existing model. Use the narrow upsert/delete tools instead of rewriting the whole file.

## Extension envelope

- Put every Dosi-only key in the owning object's `custom_extensions` entry with `vendor_name: DATUS`.
- `data` is a JSON string containing one object. Stamp every non-empty DATUS payload with the current canonical string version: `"v":"1.2"`.
- Use the Dosi keys defined below inside that DATUS extension envelope.
- Add `requires` when a capability explicitly marks a key as fail-closed.

## Native semantic-model keys

- **D-TIME, dataset:** when a dataset has more than one `dimension: {is_time: true}` field, add `{"v":"1.2","time_dimension":"<field>"}` to the dataset. With exactly one time field, Dosi infers it, but an explicit declaration is allowed when it clarifies business time.
- **D-TIME, metric:** a metric may override its datasets with `time_dimension`; prefer a qualified name such as `activities.start_date` when multiple datasets expose the same field name.
- **D-GRAIN, field:** put `{"v":"1.2","time_granularity":"day|week|month|quarter|year"}` on each time field whose native stored grain is known. This is the storage grain, not a fixed query grain.
- **D-JOIN, relationship:** add `join_type: "left"|"inner"` to the relationship's DATUS payload only when the desired unmatched-row behavior is known. `left` preserves unmatched facts in a NULL dimension group; `inner` drops them. The default is `left`.
- **D-DATASET, metric:** add `dataset` when a base aggregate such as `COUNT(*)` cannot be attributed from qualified expression columns.
- **D-FILL, metric:** add numeric `fill_nulls_with` when missing branch groups must be represented by that number. Window outputs retain NULL for cases such as “no prior/comparable bucket.”

## SQL plan window handoff

- Treat an output carrying `native_window` as a request for a native D-WINDOW metric. Build or reuse a dataset that exposes the fields required by `base_expression`, its business `time_dimension`, and the source grouping dimensions.
- Keep that dataset reusable at the base row or base-period grain. The metrics phase applies the structured `native_window` contract to the standalone final metric.

Example relationship and time declarations:

```yaml
datasets:
  - name: orders
    source: orders
    custom_extensions:
      - vendor_name: DATUS
        data: '{"v":"1.2","time_dimension":"order_date"}'
    fields:
      - name: order_date
        expression: {dialects: [{dialect: <osi_dialect>, expression: order_date}]}
        dimension: {is_time: true}
        custom_extensions:
          - vendor_name: DATUS
            data: '{"v":"1.2","time_granularity":"day"}'
relationships:
  - name: orders_to_customers
    from: orders
    to: customers
    from_columns: [customer_id]
    to_columns: [customer_id]
    custom_extensions:
      - vendor_name: DATUS
        data: '{"v":"1.2","join_type":"inner"}'
```

After every mutation, use `validate_semantic` against the active Dosi adapter. Its native parser/compiler is authoritative; successful YAML parsing alone is not execution validation.
