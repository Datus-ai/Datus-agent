# `gen_semantic_model` (Retired)

`gen_semantic_model` is retained only for configuration compatibility. It is hidden from subagent discovery, and direct invocation returns an error recommending [`semantic_modeling`](semantic_modeling.md).

## Replacement

Use `semantic_modeling` in a Dosi project to create or update datasets, fields, relationships, model metadata, and metrics. The workflow edits Dosi YAML, validates the selected model, and reconciles the YAML source of truth with the Knowledge Base.

```text
Model the orders and customers datasets and their relationship. @Agent semantic_modeling
```

MetricFlow and OSI projects remain queryable but are query-only for semantic changes. To modify an existing OSI project, first change its semantic type to Dosi, then use `semantic_modeling` to repair and validate the existing YAML. MetricFlow YAML migration is not supported.

## Bootstrap compatibility

On a Dosi project, `bootstrap-kb --components semantic_model` runs `semantic_modeling` with datasets-only scope. It may change datasets, fields, relationships, and model metadata, but it preserves every existing metric definition. Existing YAML import and profile parsing remain available for supported legacy files.

See [Semantic Modeling](semantic_modeling.md) and [Semantic Models](../knowledge_base/semantic_model.md) for the supported workflow.
