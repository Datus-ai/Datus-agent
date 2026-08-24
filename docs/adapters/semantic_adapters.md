# Semantic Adapters

Datus Agent uses semantic adapters to connect semantic authoring, validation,
discovery, and querying to a concrete semantic layer implementation. Dosi is
the default and the only authoring adapter; MetricFlow and OSI remain available
as explicitly configured query-compatibility adapters.

This page is the adapter overview. For adapter-specific behavior, use:

- [MetricFlow Semantic Adapter](metricflow_semantic_adapter.md)
- [OSI Semantic Adapter](osi_semantic_adapter.md)
- [Dosi Semantic Adapter](dosi_semantic_adapter.md)

## Overview

All three semantic adapters provide a unified query interface for:

- listing executable metrics
- discovering dimensions for a metric
- querying metric values
- validating existing semantic assets

Dosi additionally provides the authoring lifecycle for:

- validating authored semantic assets before publishing
- syncing validated semantic assets into the Datus Knowledge Base

Three adapters are currently supported:

| Adapter | Package | Source format | Execution backend | Datus mode |
|---------|---------|---------------|-------------------|------------|
| Dosi | `datus-semantic-dosi` | strict OSI core YAML + DATUS custom extensions | Native Dosi engine | Default; authoring + query |
| MetricFlow | `datus-semantic-metricflow` | MetricFlow YAML | MetricFlow | Explicit; query-only |
| OSI | `datus-semantic-osi[metricflow]` | strict OSI core YAML + DATUS custom extensions | MetricFlow | Explicit; query-only |

The adapters share a query interface, but no longer share the authoring surface:

- Dosi authors strict OSI-compatible YAML and compiles, plans, and executes it directly in the Rust engine.
- MetricFlow loads existing MetricFlow YAML for validation, discovery, and querying.
- OSI loads existing OSI core YAML, compiles it to Datus Semantic IR, and lowers it to MetricFlow for querying.

## Architecture

```text
datus-agent
├── Semantic tools
│   ├── list_metrics
│   ├── get_dimensions
│   ├── query_metrics
│   └── validate_semantic
│
├── SemanticAdapterRegistry
│
└── Adapter packages
    ├── datus-semantic-metricflow
    │   └── MetricFlowAdapter
    ├── datus-semantic-osi
    │   └── DatusOSIAdapter
    └── datus-semantic-dosi
        └── DosiAdapter
```

Adapters are discovered through Python entry points under `datus.semantic_adapters`.

## Configuration

Configure semantic adapters under `agent.services.semantic_layer` in `agent.yml`.

```yaml
agent:
  services:
    semantic_layer:
      dosi:
        default: true

      metricflow: {}  # optional query compatibility

      osi: {}         # optional query compatibility; install osi[metricflow]
```

The key under `services.semantic_layer` must equal the adapter type, for example `metricflow`, `osi`, or `dosi`. If a `type:` field is present, it must match the key.
The selected semantic adapter is global. Legacy node-level `semantic_adapter` and `authoring_format` fields are ignored.

See [Semantic Layer Configuration](../configuration/semantic_layer.md) for selection rules, defaults, and project-level pins.

## Core Interface

All semantic adapters implement these methods:

| Method | Purpose |
|--------|---------|
| `list_metrics(path, limit, offset)` | List executable metrics. |
| `get_dimensions(metric_name, path)` | Return dimensions that can be used with a metric. |
| `query_metrics(metrics, dimensions, ...)` | Query metrics or render SQL with `dry_run=True`. |
| `validate_semantic(scope)` | Validate semantic assets and backend compatibility. |

Optional semantic-model methods include `get_semantic_model()` and `list_semantic_models()`.

## Choosing an Adapter

Use MetricFlow when:

- you already have MetricFlow YAML
- your team maintains those assets outside Datus
- you need Datus query surfaces to keep using them during migration

Use OSI when:

- you want the authored source to follow OSI core schema
- you want Datus-specific execution hints isolated in `custom_extensions`
- you already have OSI assets that must remain queryable through MetricFlow

Use Dosi when:

- you are starting new semantic authoring or changing existing models
- you want the strict OSI authoring workflow
- aggregate, ratio, and expression metrics cover the model
- you want native join planning, fan-out protection, and execution without MetricFlow

## Implementing a Custom Adapter

Implement a semantic adapter by extending `BaseSemanticAdapter` and registering it through an entry point:

```toml
[project.entry-points."datus.semantic_adapters"]
myservice = "datus_semantic_myservice:register"
```

Required methods:

| Method | Return Type |
|--------|-------------|
| `list_metrics()` | `List[MetricDefinition]` |
| `get_dimensions()` | `List[DimensionInfo]` |
| `query_metrics()` | `QueryResult` |
| `validate_semantic()` | `ValidationResult` |
