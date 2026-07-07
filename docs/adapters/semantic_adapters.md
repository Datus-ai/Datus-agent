# Semantic Adapters

Datus Agent uses semantic adapters to connect metric generation, validation, discovery, and querying to a concrete semantic layer implementation.

This page is the adapter overview. For adapter-specific behavior, use:

- [MetricFlow Semantic Adapter](metricflow_semantic_adapter.md)
- [OSI Semantic Adapter](osi_semantic_adapter.md)

## Overview

Semantic adapters provide a unified interface for:

- listing executable metrics
- discovering dimensions for a metric
- querying metric values
- validating semantic assets before publishing
- syncing validated semantic assets into the Datus Knowledge Base

Two adapters are currently supported:

| Adapter | Package | Authoring format | Execution backend | Status |
|---------|---------|------------------|-------------------|--------|
| MetricFlow | `datus-semantic-metricflow` | MetricFlow YAML | MetricFlow | Ready |
| OSI | `datus-semantic-osi` | strict OSI core YAML + DATUS custom extensions | MetricFlow by default | Ready |

MetricFlow and OSI are peer semantic adapters. The difference is what users and generation agents author:

- MetricFlow mode authors MetricFlow YAML directly.
- OSI mode authors OSI core YAML and lets `datus-semantic-osi` compile it to Datus Semantic IR before lowering to MetricFlow.

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
    └── datus-semantic-osi
        └── DatusOSIAdapter
```

Adapters are discovered through Python entry points under `datus.semantic_adapters`.

## Configuration

Configure semantic adapters under `agent.services.semantic_layer` in `agent.yml`.

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        default: true

      osi:
        execution_backend: metricflow
```

The key under `services.semantic_layer` must equal the adapter type, for example `metricflow` or `osi`. If a `type:` field is present, it must match the key.
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
- you want generated files to be MetricFlow-native
- your team expects to inspect or maintain MetricFlow assets directly

Use OSI when:

- you want the authored source to follow OSI core schema
- you want Datus-specific execution hints isolated in `custom_extensions`
- you want LLM generation to avoid backend YAML fields such as `measure_proxy`, `type_params`, and `data_source`
- you still want to execute through MetricFlow today

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
