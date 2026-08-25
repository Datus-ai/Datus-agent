# `gen_metrics` (Retired)

`gen_metrics` is retained only for configuration compatibility. It is hidden from subagent discovery, and direct invocation returns an error recommending [`semantic_modeling`](semantic_modeling.md).

## Replacement

Use `semantic_modeling` in a Dosi project to author metrics together with their required datasets and relationships. The workflow validates the completed Dosi model, checks generated metrics, and reconciles the YAML source of truth with the Knowledge Base.

```text
Define revenue and order-count metrics from this SQL evidence. @Agent semantic_modeling
```

MetricFlow and OSI projects retain execution and query support for existing metrics, but they cannot author new metrics. To modify existing OSI YAML, first change the project's semantic type to Dosi and then use `semantic_modeling`. MetricFlow YAML migration is not supported.

## Bootstrap compatibility

On a Dosi project, `bootstrap-kb --components metrics` runs the full `semantic_modeling` workflow. Existing YAML import remains available for supported legacy files. Combining `metrics` with `semantic_model` or `semantic_modeling` still performs one full authoring run.

See [Semantic Modeling](semantic_modeling.md) and [Metrics](../knowledge_base/metrics.md) for the supported workflow.
