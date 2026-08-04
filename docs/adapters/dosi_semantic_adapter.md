# Dosi Semantic Adapter

The Dosi adapter executes OSI semantic models directly in the native Rust
engine. It uses the same strict OSI authoring workflow as the existing OSI
adapter, but does not lower models to MetricFlow.

## Install

```bash
pip install datus-semantic-dosi
```

This command also installs the required `dosi-engine` wheel. The `/services`
Semantic tab performs the same installation automatically when `dosi` is
selected.

## Configure

```yaml
agent:
  services:
    datasources:
      warehouse:
        type: duckdb
        uri: /absolute/path/to/warehouse.db
    semantic_layer:
      dosi:
        default: true
        # semantic_model_path: /absolute/path/to/model.yaml
```

Datus supplies the active datasource connection and defaults
`semantic_models_path` to `subject/semantic_models/<datasource>/`. The adapter
selects the only YAML or JSON model in that directory. Configure
`semantic_model_path` when more than one model file exists.

## Supported Semantics

Dosi currently executes aggregate, ratio, and expression metrics; dimensions;
many-to-one relationships; composite joins; query-backed datasets; filters;
and day through year time grains. Its planner rejects ambiguous join paths and
fan-out-prone queries rather than silently double-counting.

Derived metrics, cumulative or rolling windows, metric offsets, and
period-over-period metrics remain on the OSI + MetricFlow path until Dosi gains
equivalent execution support.

## Authoring and Validation

Selecting `dosi` still activates Datus's OSI semantic-model and metric
authoring skills. `validate_semantic`, `list_metrics`, `get_dimensions`, and
`query_metrics` are routed through the Dosi adapter. The authored OSI document
remains the source of truth.
