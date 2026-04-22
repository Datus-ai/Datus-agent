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
pip install -e ../datus-semantic-adapter/datus-semantic-metricflow
```

Once installed, Datus Agent will automatically detect and load the adapter.

## Configuration

Configure semantic adapters under `agent.services.semantic_layer` in `agent.yml`:

The entire `semantic_layer` block is optional when you use MetricFlow with default settings. In that case Datus defaults to `metricflow` automatically.

### MetricFlow

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300  # optional, default is 300 seconds
        config_path: /path/to/agent.yml  # optional advanced override

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow
    gen_metrics:
      semantic_adapter: metricflow
```

**Semantic Model File Location**:
By default, Datus points MetricFlow at the current project's semantic model directory:
```text
{project_root}/subject/semantic_models/
```
- `project_root` is the active Datus project root.

### Selection Rules

- The key under `services.semantic_layer` **must equal the adapter type** (for example `metricflow`). If a `type:` field is present, it must match the key; otherwise Datus raises a configuration error at startup. Comparison is case-insensitive and trims surrounding whitespace, so `MetricFlow` and ` metricflow ` also match.
- Semantic nodes choose the adapter with `semantic_adapter`.
- If both `services.semantic_layer` and `semantic_adapter` are omitted, Datus defaults to `metricflow`.
- If `semantic_adapter` is omitted and only one semantic layer is configured, Datus uses that adapter automatically.
- If multiple semantic layers are configured, set `semantic_adapter` explicitly.

### About `config_path`

`config_path` is optional. The normal runtime path builds MetricFlow config from:

1. the selected datasource in `services.datasources`
2. the current project's semantic model directory
3. the active `agent.home`

Use `config_path` only when you explicitly need MetricFlow to initialize from a different agent config file.

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
import asyncio
from datus.tools.semantic_tools import semantic_adapter_registry
from datus_semantic_metricflow.config import MetricFlowConfig

async def main():
    config = MetricFlowConfig(datasource="my_project")
    adapter = semantic_adapter_registry.create_adapter("metricflow", config)

    metrics = await adapter.list_metrics(limit=10)
    dimensions = await adapter.get_dimensions(metric_name="revenue")
    result = await adapter.query_metrics(
        metrics=["revenue"], dimensions=["date"], time_start="2024-01-01"
    )

asyncio.run(main())
```

### Dry Run (SQL Preview)

```python
async def dry_run_example():
    result = await adapter.query_metrics(metrics=["revenue"], dry_run=True)
    print(result.data[0]["sql"])

asyncio.run(dry_run_example())
```

### Bootstrap from Adapter

```bash
datus-agent bootstrap-kb --database my_project --components metrics \
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

```text
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
| Connection issues | Verify `agent.yml` config, check the selected database and semantic model directory |
| Validation errors | Run `adapter.validate_semantic()` to check configuration |

## Next Steps

- [MetricFlow Configuration](../metricflow/introduction.md) - Detailed MetricFlow setup
- [Configuration Reference](../configuration/introduction.md) - General configuration options
