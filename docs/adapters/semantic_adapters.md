# Semantic Adapters

Datus Agent supports connecting to various semantic layer services through a plugin-based adapter system. This document explains the available adapters, how to install them, and how to configure semantic layer connections.

## Overview

Datus uses a modular semantic adapter architecture that allows you to connect to different semantic layer backends:

- **MetricFlow**: dbt's semantic layer for metrics and dimensions

This design provides a unified interface for metric discovery, querying, and validation across different semantic layer implementations.

## Architecture

```text
datus-agent (Core)
├── Semantic Tools Layer
│   ├── BaseSemanticAdapter (Abstract)
│   ├── SemanticAdapterRegistry (Factory)
│   └── Data Models (MetricDefinition, QueryResult, etc.)
│
└── Plugin System (Entry Points)
    └── datus-semantic-metricflow
        └── MetricFlowAdapter
```

The adapter system uses Python's entry points mechanism for automatic discovery. When you install an adapter package, it registers itself with Datus Agent and becomes available for use.

## Supported Semantic Layers

| Semantic Layer | Package | Installation | Status |
|----------------|---------|-------------|--------|
| MetricFlow | datus-semantic-metricflow | `pip install datus-semantic-metricflow` | Ready |

## Installation

### MetricFlow Adapter

```bash
# Install MetricFlow adapter
pip install datus-semantic-metricflow

# Or install from source
pip install -e ../datus-semantic-adapter/datus_semantic_metricflow
```

Once installed, Datus Agent will automatically detect and load the adapter.

## Configuration

Configure your semantic layer in the `config.yaml` file under the `semantic` section:

### MetricFlow

```yaml
semantic:
  type: metricflow
  namespace: my_project
  timeout: 300  # optional, default is 300 seconds
  config_path: /path/to/agent.yml  # optional, uses default lookup if not specified
```

**Semantic Model File Location**:
MetricFlow automatically locates semantic model files at:
```
{agent.home}/semantic_models/{namespace}/
```
- `agent.home` is read from `agent.yml` (defaults to `~/.datus`)

### Configuration Lookup Priority

When initializing MetricFlow adapter:

1. `config_path` parameter (if explicitly provided)
2. `./conf/agent.yml` (current directory)
3. `~/.datus/conf/agent.yml` (home directory)

## Core Interfaces

### Metrics Interface

All semantic adapters implement these core async methods:

| Method | Description | Return Type |
|--------|-------------|-------------|
| `list_metrics(path, limit, offset)` | List available metrics with filtering | `List[MetricDefinition]` |
| `get_dimensions(metric_name, path)` | Get dimensions for a metric | `List[DimensionInfo]` |
| `query_metrics(metrics, dimensions, ...)` | Query metrics with filters, time range, where clause | `QueryResult` |
| `validate_semantic()` | Validate semantic layer configuration | `ValidationResult` |

### Semantic Model Interface (Optional)

| Method | Description | Return Type |
|--------|-------------|-------------|
| `get_semantic_model(table_name, ...)` | Get semantic model for a table | `Optional[Dict]` |
| `list_semantic_models(...)` | List available semantic models | `List[str]` |

## Data Models

| Model | Key Fields |
|-------|------------|
| `MetricDefinition` | `name`, `description`, `type`, `dimensions`, `measures`, `unit`, `format`, `path` |
| `QueryResult` | `columns`, `data`, `metadata` |
| `ValidationResult` | `valid`, `issues` |
| `ValidationIssue` | `severity`, `message`, `location` |
| `DimensionInfo` | `name`, `description` |

## Usage Examples

### Direct Adapter Usage

```python
from datus.tools.semantic_tools import semantic_adapter_registry
from datus_semantic_metricflow.config import MetricFlowConfig

config = MetricFlowConfig(namespace="my_project")
adapter = semantic_adapter_registry.create_adapter("metricflow", config)

metrics = await adapter.list_metrics(limit=10)
dimensions = await adapter.get_dimensions(metric_name="revenue")
result = await adapter.query_metrics(
    metrics=["revenue"], dimensions=["date"], time_start="2024-01-01"
)
```

### Dry Run (SQL Preview)

```python
result = await adapter.query_metrics(metrics=["revenue"], dry_run=True)
print(result.data[0]["sql"])
```

### Bootstrap from Adapter

```bash
datus-agent bootstrap-kb --namespace my_project --components metrics \
  --from_adapter metricflow --kb-update-strategy overwrite
```

## Features by Adapter

### Common Features

All semantic adapters support:

- Metric discovery and listing
- Dimension retrieval per metric
- Metric querying with filters
- Configuration validation
- Storage sync for caching

### MetricFlow Adapter

- Full MetricFlow API integration
- YAML-based semantic model files
- Three-stage validation (lint, parse, semantic)
- SQL generation and explain
- Time range filtering with granularity

## Implementing a Custom Adapter

You can implement your own semantic adapter by extending `BaseSemanticAdapter` and registering it via Python entry points.

### Required Methods

Your adapter must implement these abstract methods:

| Method | Description | Return Type |
|--------|-------------|-------------|
| `list_metrics()` | List available metrics with optional filtering | `List[MetricDefinition]` |
| `get_dimensions()` | Get queryable dimensions for a metric | `List[DimensionInfo]` |
| `query_metrics()` | Execute metric queries with filters | `QueryResult` |
| `validate_semantic()` | Validate semantic layer configuration | `ValidationResult` |

### Optional Methods

| Method | Description | Default |
|--------|-------------|---------|
| `get_semantic_model()` | Get semantic model for a table | Returns `None` |
| `list_semantic_models()` | List available semantic models | Returns `[]` |

### Package Structure

```
datus_semantic_myservice/
├── pyproject.toml
└── datus_semantic_myservice/
    ├── __init__.py    # register() function
    ├── adapter.py     # MyServiceAdapter
    └── config.py      # MyServiceConfig
```

### Entry Point Configuration

```toml
# pyproject.toml
[project.entry-points."datus.semantic_adapters"]
myservice = "datus_semantic_myservice:register"
```

### Reference Implementation

See the MetricFlow adapter implementation for a complete example:
- [datus-semantic-metricflow](https://github.com/Datus-ai/datus-semantic-adapter)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Adapter not found | Install the adapter: `pip install datus-semantic-metricflow` |
| Connection issues | Verify `agent.yml` config, check namespace matches semantic model directory |
| Validation errors | Run `adapter.validate_semantic()` to check configuration |

## Next Steps

- [MetricFlow Configuration](../metricflow/introduction.md) - Detailed MetricFlow setup
- [Configuration Reference](../configuration/introduction.md) - General configuration options
