# Semantic Modeling

`semantic_modeling` is the Dosi-only authoring agent for datasets, relationships, and metrics. It selects or creates one business-domain semantic model, edits the YAML source of truth, validates the completed model, and fully reconciles the YAML into the Knowledge Base.

```text
/semantic_modeling Model orders and customers, then define revenue and order-count metrics.
```

Existing MetricFlow and OSI projects remain queryable but are query-only. To author an existing OSI project, first change its semantic type to Dosi, then use `semantic_modeling` to repair and validate the existing YAML in place. MetricFlow YAML migration is not supported.

The current supported warehouse scope is StarRocks, Snowflake, PostgreSQL,
Doris, and Hologres. New models are grouped by business domain unless the user
explicitly requests another grouping.

The retired `gen_semantic_model` and `gen_metrics` names remain reserved for configuration compatibility. They are hidden from agent discovery and direct invocation reports an error that recommends `semantic_modeling`.
Custom agent entries whose legacy `node_class` or `type` is `gen_semantic_model`
or `gen_metrics` transparently use `semantic_modeling` on Dosi projects.

For `bootstrap-kb`, the historical component names are compatibility aliases:

- `--components semantic_model` runs `semantic_modeling` with datasets-only scope. It may update datasets, fields, relationships, and model metadata, but existing metrics are protected from changes.
- `--components metrics` and `--components semantic_modeling` run the full workflow.
- Combining these components executes the workflow once; full scope wins when `metrics` or `semantic_modeling` is present.

Existing YAML import and `refresh-profile` remain non-LLM operations. They do not reactivate either retired subagent.
