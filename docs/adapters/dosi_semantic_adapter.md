# Dosi Semantic Adapter

The Dosi adapter executes OSI semantic models directly in the native Rust
engine. Its authoring, validation, catalog loading, and execution paths are
provided by `datus-semantic-dosi`; it does not load the Python OSI adapter or
lower models to MetricFlow.

Dosi is the built-in default semantic adapter and supports both
`semantic_modeling` authoring and the complete query surface. When no semantic
adapter is configured, interactive Datus startup installs and selects it.

## Install

For local development, a published `dosi-engine` package is not required. From
the `Datus-agent` checkout in the standard sibling-repository layout, install
the engine binding and adapter sources into the same environment:

```bash
uv pip install -e ../osi-engine/crates/dosi-py
uv pip install -e ../datus-semantic-adapter/datus-semantic-core
uv pip install -e ../datus-semantic-adapter/datus-semantic-dosi
```

The adapter still declares the logical `dosi-engine` dependency, but the local
editable install satisfies it directly from source. A released engine version
is needed only for registry-based installation and formal delivery.

## Configure

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
    semantic_layer:
      dosi: {}  # optional: Dosi is selected when this section is empty
        # semantic_model_path: /absolute/path/to/model.yaml
```

Datus supplies the active datasource connection and defaults
`semantic_models_path` to `subject/semantic_models/<datasource>/`. The adapter
selects the only YAML or JSON model in that directory. Configure
`semantic_model_path` when more than one model file exists.

## Supported Semantics

Dosi executes aggregate, ratio, and expression metrics; dimensions;
many-to-one relationships; composite joins; query-backed datasets; filters;
day-through-year time grains; `time_dimension` primary axes; relationship
`join_type`; metric `fill_nulls_with`; and structured windows. Datus-ext 1.3
supports period comparison, rolling, cumulative, rank/distribution,
first/last/nth value, backward or forward offset, and explicit statistical
frames over row- or value-based ranges.

## Authoring and Validation

Selecting `dosi` activates strict OSI authoring plus a Dosi-native extension
profile. `validate_semantic`, `list_metrics`, `get_dimensions`, and
`query_metrics` are routed through the native engine. For time queries, pass
`metric_time` as the dimension and the grain separately; a returned column such
as `metric_time__<grain>` is an output/order key.

The unified `semantic_modeling` workflow lists valid and repairable existing
models separately. It binds a valid target or plans the same repairable model
in place, applies the requested changes, validates the exact final artifact,
and reconciles that artifact to the Knowledge Base.
