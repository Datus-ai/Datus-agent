---
name: osi-semantic-authoring
description: OSI core schema semantic model authoring specification — dataset shape, Datus extension hints, relationships, and validation
tags:
  - semantic-model
  - osi
version: "1.1.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# OSI Semantic Authoring

Describe tables as strict **OSI (Open Semantic Interchange) core schema** documents plus Datus business hints.

CRITICAL BOUNDARY: You author **OSI core semantics only**. You do NOT write MetricFlow `data_source:`, `measures:`, `identifiers:`, `agg_time_dimension`, `create_metric`, or any execution-engine YAML. The Datus OSI compiler lowers OSI core documents to the configured backend.

The OSI expression dialect, target semantic model name, and target semantic model file for the current run are shown in the system prompt Workspace section — use those exact values.

## What you produce

One valid OSI core document for the current business domain / semantic model scope (`<osi_dialect>` stands for the OSI expression dialect from the system prompt):

```yaml
version: 0.2.0.dev0
semantic_model:
  - name: <target_model_name>
    datasets:
      - name: <dataset_name>
        source: <physical_table_view_or_query_name>
        description: "<human-readable business purpose and grain of this dataset>"
        ai_context:
          instructions: "<how AI should use this dataset for analytics, including grain, time field, common row-selection conditions or groupings, and join caveats>"
          synonyms: ["<business term>", "<alternate table name>"]
          examples: ["<question this dataset can answer>"]
        primary_key: [<primary_key_column>]
        fields:
          - name: <date_or_timestamp_column>
            expression:
              dialects:
                - dialect: <osi_dialect>
                  expression: <date_or_timestamp_column>
            dimension:
              is_time: true
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"time","time_granularity":"day"}'
          - name: <categorical_business_column>
            expression:
              dialects:
                - dialect: <osi_dialect>
                  expression: <categorical_business_column>
            dimension:
              is_time: false
            description: "<business meaning of the column>"
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"categorical"}'
          - name: <numeric_business_column>
            expression:
              dialects:
                - dialect: <osi_dialect>
                  expression: <numeric_business_column>
            dimension:
              is_time: false
            description: "<business meaning of the measured numeric value>"
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"numeric"}'
    relationships:
      - name: <fact_dataset>_to_<dimension_dataset>
        from: <fact_or_many_side_dataset>
        to: <dimension_or_one_side_dataset>
        from_columns: [<foreign_key_column>]
        to_columns: [<primary_or_unique_key_column>]
```

## Authoring rules

1. **Root schema is fixed.** Root keys are only `version` and `semantic_model`. `semantic_model` is a list. Do NOT write top-level `datasets:`, `relationships:`, or `metrics:`.
2. **Use OSI core dataset shape.** Dataset `source` is a string, not `{table: ...}`. Dataset columns are `fields`, not `dimensions`. Field expressions are `expression.dialects[]`, not `expr`. Use the exact OSI expression dialect from the system prompt in every `expression.dialects[].dialect`.
3. **Datus-only hints go into `custom_extensions`.** OSI core does not have top-level dataset `time_dimension`, field `type`, field `granularity`, metric `dataset`, metric `window`, etc. Put those Datus execution hints in `custom_extensions: [{vendor_name: DATUS, data: '<JSON object string>'}]`.
4. **Semantic model boundary.** One OSI `semantic_model` represents the current business domain / metric domain. Put all related logical datasets needed by the provided SQL history in this semantic model, with relationships declared once under the semantic model object.
5. **Canonical logical datasets.** A dataset is a logical dataset backed by a table, view, or query. For the same source and row grain, create one canonical dataset that metrics can reference by name. Create a separate dataset only when the logical row grain or fixed business scope is genuinely different. Include the full logical dataset: primary key, primary time field, and all business-meaningful fields.
6. **Dataset description and AI context are required for every dataset.**
   - `description` is for humans: one concise sentence describing the business entity/event represented by the dataset and its row grain.
   - `ai_context` is for AI use: include `instructions`, optional `synonyms`, and optional `examples`. The instructions should explain when to use this dataset, the row grain, the primary time field, important row-selection or grouping columns, and relationship caveats. Keep it generic and derived from schema/comments/history; do not write scenario-specific examples unless they come from provided history.
7. **Primary / unique keys** -> OSI core `primary_key: [<column>]`. Use real columns only.
8. **Time dimension**: mark the primary date/time field with `dimension.is_time: true` and Datus extension `{"type":"time","time_granularity":"day|week|month|quarter|year"}`. Point it at a real date/time column, never a numeric surrogate key.
9. **Fields**: include every business-meaningful column the user may group or slice by. Populate `description` for ALL non-obvious fields by combining the column comment, sample values, profiler evidence, and external knowledge. Useful profiler evidence may include null rate, numeric ranges/percentiles, date span/freshness/duration, distinct counts, representative stable values, common filter templates, and relationship coverage/fanout. Wrap descriptions in double quotes and keep each description concise.
10. **Field type classification must follow actual data type and analytical usage.** The DATUS `custom_extensions.data.type` value is not a display hint; it is the semantic field type.
    - Use `{"type":"numeric"}` for DECIMAL/NUMERIC/INTEGER/FLOAT/DOUBLE or equivalent fields that represent measured quantities, scores, rates, prices, amounts, durations, counts, percentages, or other arithmetic values.
    - Keep a field `numeric` when historical SQL uses it in arithmetic, range predicates, numeric ordering, or numeric aggregates such as `AVG`, `SUM`, `MIN`, `MAX`, `STD`, `VARIANCE`, `CORR`, or `COVAR`. A numeric field does not become categorical just because it is selected, displayed, filtered, or grouped.
    - Use `{"type":"categorical"}` for text/enums/labels/booleans and code fields whose values are categories, not quantities. Numeric-coded categories such as status codes, product type codes, or channel codes can be categorical when averaging or summing the code would have no business meaning; explain the code semantics in the description.
    - Use `{"type":"identifier"}` for primary keys, foreign keys, unique entity IDs, or opaque IDs. Do not classify identifiers as numeric merely because their physical storage type is integer.
    - Use `{"type":"time","time_granularity":"..."}` only for real date/time values as described above.
    - If evidence conflicts, prefer the physical column type plus concrete SQL usage. A DECIMAL field used by `AVG(<field>)` or numeric threshold filters must be `numeric`; an integer status code used only as labels/groups should be `categorical`.
11. **One column, one type model-wide.** A given column name must carry the SAME `type` everywhere it appears across ALL datasets and in `primary_key`. Do not model the same column as `identifier` in one place and `categorical`/`dimension` in another. Decide its role once from its strongest signal: a key/foreign-key/join column is an `identifier` even when it also appears in a `GROUP BY`; a code/label column is `categorical`. If `validate_semantic` reports a column "used as multiple types", fix every occurrence to a SINGLE type and re-validate — never toggle it back and forth between validation attempts.
12. **Do not model columns no query uses.** Include a field only when the provided SQL selects, filters, groups, joins, or aggregates by it, or it is the dataset primary key or primary time field. A key column present in the DDL but never referenced by any provided SQL should be a plain `identifier` (or the dataset `primary_key`) consistently — do not guess whether it is a dimension. When in doubt about an unused column, omit it rather than introduce an ambiguous type.
13. **Relationships** live inside the semantic model object, never inside a dataset. Use OSI core fields `from`, `to`, `from_columns`, `to_columns`. Do NOT use non-core fields such as `from_dataset`, `from_identifier`, `to_dataset`, `to_identifier`, `join_on`, `from_column`, or `to_column`.
14. Do NOT add metrics in the semantic-model step. Metrics are added by the metrics workflow under `semantic_model[0].metrics`.
15. Preserve literal values and column names exactly; do not invent columns. Keep column comments in their original language — do not translate.

## Workflow notes

- Write OSI core YAML under the semantic model directory shown in the system prompt only, at the target semantic model file path.
- Inspect the table schema and comments; map columns to keys, the time field, business fields, and relationships.
- When a critical modeling choice is ambiguous (which column is the grain, which is the primary time dimension), ASK before generating.
- Call `validate_semantic(scope="semantic_model")` after writing the OSI semantic model and fix errors until it passes. The Datus OSI compiler validates against the OSI core schema, then lowers to the configured execution backend.
- After validation passes, call `end_semantic_model_generation(semantic_model_files=[...])`. In OSI mode this syncs OSI datasets to the Knowledge Base without using MetricFlow YAML.
