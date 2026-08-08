# Dosi Semantic Adapter

The Dosi adapter executes OSI semantic models directly in the native Rust
engine. Its authoring, validation, catalog loading, and execution paths are
provided by `datus-semantic-dosi`; it does not load the Python OSI adapter or
lower models to MetricFlow.

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
      dosi:
        default: true
        # semantic_model_path: /absolute/path/to/model.yaml
```

Datus supplies the active datasource connection and defaults
`semantic_models_path` to `subject/semantic_models/<datasource>/`. The adapter
selects the only YAML or JSON model in that directory. Configure
`semantic_model_path` when more than one model file exists.

## Supported Semantics

Dosi executes aggregate, ratio, and expression metrics; dimensions;
many-to-one relationships; composite joins; query-backed datasets; filters;
day-through-year time grains; D-TIME primary axes; D-JOIN/D-FILL behavior; and
native D-WINDOW period-over-period, rolling, and cumulative metrics. D-WINDOW
in datus-ext 1.2 uses one plain aggregate as its base expression and declares
the window derivation on top of it.

## Authoring and Validation

Selecting `dosi` activates strict OSI authoring plus a Dosi-native extension
profile. `validate_semantic`, `list_metrics`, `get_dimensions`, and
`query_metrics` are routed through the native engine. For time queries, pass
`metric_time` as the dimension and the grain separately; a returned column such
as `metric_time__<grain>` is an output/order key.

Before switching existing Python-OSI models, run the one-shot checker. It is
read-only by default; `--write` converts only unambiguous cases, validates with
the installed native engine, creates a sibling `.bak`, and refuses ambiguous
files:

```bash
datus-dosi-migrate subject/semantic_models/<datasource>
datus-dosi-migrate subject/semantic_models/<datasource> --write
```
