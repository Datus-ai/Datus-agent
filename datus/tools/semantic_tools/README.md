# Semantic Tools - Unified Semantic Layer Abstraction

This package provides a unified abstraction layer for semantic layer services (MetricFlow, dbt Semantic Layer, Cube, etc.), following the adapter pattern similar to `datus/tools/db_tools/`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Agent/Tools Layer                      │
│  ┌────────────────┐        ┌──────────────────┐         │
│  │ DBFuncTool     │        │ SemanticTools    │         │
│  │ describe_table │───────▶│                  │         │
│  └────────────────┘        └──────────────────┘         │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
                   │                  ▼
                   │         ┌─────────────────────┐
                   │         │ BaseSemanticAdapter │
                   │         │  (abstract class)   │
                   │         └──────────┬──────────┘
                   │                    │
                   │         ┌──────────┴──────────┐
                   │         │                     │
                   │    ┌────▼──────┐      ┌──────▼─────┐
                   │    │MetricFlow │      │ dbt/cube   │
                   │    │ Adapter   │      │ (future)   │
                   │    └───────────┘      └────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │   Unified Storage Layer   │
        │  ┌────────────────────┐  │
        │  │SemanticModelStorage│  │
        │  │   MetricStorage    │  │
        │  └────────────────────┘  │
        └──────────────────────────┘
```

## Core Components

### 1. Data Models (`models.py`)

Standard data models for semantic layer operations:

```python
from datus.tools.semantic_tools.models import (
    MetricDefinition,   # BaseModel: name, description, type, dimensions, etc.
    QueryResult,        # BaseModel: columns, data, metadata
    ValidationResult,   # BaseModel: valid, issues
    AnomalyContext,     # BaseModel: rule, observed_change_pct
)
```

### 2. Base Adapter (`base.py`)

Abstract base class that all semantic adapters must implement:

```python
from datus.tools.semantic_tools import BaseSemanticAdapter

class BaseSemanticAdapter(ABC):
    # Semantic Model Interface (Simplified)
    def get_semantic_model(table_name, catalog, database, schema) -> Dict
    def list_semantic_models(catalog, database, schema) -> List[str]

    # Metrics Interface (Complete - Async)
    async def list_metrics(path, limit, offset) -> List[MetricDefinition]
    async def get_dimensions(metric_name, path) -> List[str]
    async def query_metrics(metrics, dimensions, ...) -> QueryResult
    async def validate_semantic() -> ValidationResult

    # Storage Sync Interface
    def sync_to_storage(storage_manager) -> Dict[str, int]
```

### 3. Adapter Registry (`registry.py`)

Factory pattern for adapter registration and discovery:

```python
from datus.tools.semantic_tools import semantic_adapter_registry

# Register adapter
semantic_adapter_registry.register(
    service_type="metricflow",
    adapter_class=MetricFlowAdapter,
    config_class=MetricFlowConfig,
    display_name="MetricFlow"
)

# Create adapter instance
adapter = semantic_adapter_registry.create_adapter("metricflow", config)
```

### 4. Configuration (`config.py`)

Base configuration class for semantic adapters. Specific adapter configurations
(MetricFlowConfig, DbtConfig, CubeConfig, etc.) are defined in their respective adapter packages.

```python
from datus.tools.semantic_tools import SemanticAdapterConfig  # Base config
```

### 5. Storage Sync (`storage_sync.py`)

Syncs data from adapters to unified storage:

```python
from datus.tools.semantic_tools import SemanticStorageManager

manager = SemanticStorageManager(agent_config)
manager.store_semantic_model(model_data)
manager.store_metric(metric_data, subject_path=["Finance"])
```

### 6. Function Tools (`semantic_tools.py`)

High-level tool interface for LLM agents:

```python
from datus.tools.func_tool.semantic_tools import SemanticTools

tool = SemanticTools(agent_config, sub_agent_name="gen_metrics")
tools = tool.available_tools()  # Returns list of Tools for LLM
```

## Implementing a New Adapter

### Step 1: Create Adapter Package

Create a new package in the `datus-semantic-adapter` repository.

**Repository Structure** (following `datus-db-adapters` pattern):

```
datus-semantic-adapter/              # Repository root
├── datus_semantic_metricflow/       # Each adapter is a subdirectory
│   ├── pyproject.toml               # Package config for this adapter
│   └── datus_semantic_metricflow/   # Python package
│       ├── __init__.py              # Contains register() function
│       ├── adapter.py               # MetricFlowAdapter implementation
│       ├── config.py                # MetricFlowConfig
│       ├── models.py                # Data models
│       ├── tests/
│       │   └── test_adapter.py
│       └── examples/
└── datus_semantic_cube/             # Future: Cube adapter
    ├── pyproject.toml
    └── datus_semantic_cube/
        └── ...
```

**Installation**:
```bash
# Install specific adapter from local directory
pip install -e ../datus-semantic-adapter/datus_semantic_metricflow
```

### Step 2: Implement Adapter Class

```python
# datus_metricflow/adapter.py
import asyncio
from typing import List, Optional, Dict, Any

from datus.tools.semantic_tools import BaseSemanticAdapter
from datus.tools.semantic_tools.models import (
    MetricDefinition,
    QueryResult,
    ValidationResult,
    TimeRange,
)

class MetricFlowAdapter(BaseSemanticAdapter):
    def __init__(self, config):
        super().__init__(config, service_type="metricflow")
        self.cli_path = config.cli_path
        self.namespace = config.namespace
        self.timeout = config.timeout

    # Semantic Model Interface (Optional)
    def get_semantic_model(self, table_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        MetricFlow doesn't directly expose semantic models.
        Return None or infer from metrics.
        """
        return None

    def list_semantic_models(self, **kwargs) -> List[str]:
        """Return empty list - MetricFlow uses semantic models internally."""
        return []

    # Metrics Interface (Required)
    async def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MetricDefinition]:
        """
        Call: mf --namespace {namespace} list-metrics
        Parse output and convert to MetricDefinition objects.
        """
        cmd = [self.cli_path]

        # Add namespace parameter
        if self.namespace:
            cmd.extend(["--namespace", self.namespace])

        cmd.append("list-metrics")

        # Execute command asynchronously
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=self.timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"MetricFlow CLI error: {stderr.decode()}")

        # Parse output and convert to MetricDefinition
        metrics = self._parse_metrics_output(stdout.decode())

        # Apply filtering by path if needed
        if path:
            metrics = [m for m in metrics if m.path and m.path[:len(path)] == path]

        # Apply pagination
        return metrics[offset:offset + limit]

    async def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Call: mf --namespace {namespace} list-dimensions --metric-names {metric_name}
        Parse and return dimension names.
        """
        cmd = [self.cli_path]
        if self.namespace:
            cmd.extend(["--namespace", self.namespace])
        cmd.extend(["list-dimensions", "--metric-names", metric_name])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"MetricFlow CLI error: {stderr.decode()}")

        return self._parse_dimensions_output(stdout.decode())

    async def query_metrics(
        self,
        metrics: List[str],
        dimensions: List[str] = [],
        path: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> QueryResult:
        """
        Build and execute: mf --namespace {namespace} query --metrics m1,m2 --group-by d1,d2 ...
        For dry_run=True, use --explain flag.
        """
        cmd = [self.cli_path]

        if self.namespace:
            cmd.extend(["--namespace", self.namespace])

        cmd.append("query")

        # Add metrics
        cmd.extend(["--metrics", ",".join(metrics)])

        # Add dimensions (use --group-by for datus-metricflow)
        if dimensions:
            cmd.extend(["--group-by", ",".join(dimensions)])

        # Add time range
        if time_range:
            if time_range.start:
                cmd.extend(["--start-time", time_range.start])
            if time_range.end:
                cmd.extend(["--end-time", time_range.end])
            if time_range.granularity:
                cmd.extend(["--granularity", time_range.granularity.value])

        # Add WHERE clause
        if where:
            cmd.extend(["--where", where])

        # Add LIMIT
        if limit:
            cmd.extend(["--limit", str(limit)])

        # Add ORDER BY
        if order_by:
            cmd.extend(["--order", ",".join(order_by)])

        # Dry run mode
        if dry_run:
            cmd.append("--explain")

        # Execute query
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"MetricFlow CLI error: {stderr.decode()}")

        return self._parse_query_result(stdout.decode(), dry_run)

    async def validate_semantic(self) -> ValidationResult:
        """
        Call: mf --namespace {namespace} validate-configs
        Check configuration validity.

        Note: datus-metricflow automatically finds semantic model files at:
        {agent.home}/semantic_models/{namespace}/
        """
        cmd = [self.cli_path]

        if self.namespace:
            cmd.extend(["--namespace", self.namespace])

        cmd.append("validate-configs")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        valid = proc.returncode == 0
        issues = []

        if not valid:
            issues = self._parse_validation_errors(stderr.decode())

        return ValidationResult(valid=valid, issues=issues)

    # Helper methods for parsing CLI output
    def _parse_metrics_output(self, output: str) -> List[MetricDefinition]:
        """Parse 'mf list-metrics' output into MetricDefinition objects."""
        # Implementation depends on MetricFlow CLI output format
        # Example: parse JSON/YAML/table format
        pass

    def _parse_dimensions_output(self, output: str) -> List[str]:
        """Parse 'mf list-dimensions' output into dimension names."""
        pass

    def _parse_query_result(self, output: str, dry_run: bool) -> QueryResult:
        """Parse query result or explain plan."""
        pass

    def _parse_validation_errors(self, error_output: str) -> List[str]:
        """Parse validation error messages."""
        pass
```

### Step 3: Create Config Class

```python
# datus_semantic_metricflow/config.py
from typing import Optional
from pydantic import BaseModel, Field

class SemanticAdapterConfig(BaseModel):
    """Base configuration for semantic adapters."""
    namespace: str = Field(..., description="Namespace for this semantic layer instance")
    service_type: str = Field(default="metricflow", description="Type of semantic service")

class MetricFlowConfig(SemanticAdapterConfig):
    """Configuration for MetricFlow adapter."""
    service_type: str = Field(default="metricflow", description="Service type")
    cli_path: str = Field(default="mf", description="Path to MetricFlow CLI executable")
    timeout: int = Field(default=300, description="Command timeout in seconds")

    class Config:
        extra = "allow"
```

**Note**: For datus-metricflow, the semantic model files are automatically located at:
- `{agent.home}/semantic_models/{namespace}/`
- `agent.home` is read from `agent.yml` config (defaults to `~/.datus`)

### Step 4: Create Registration Function

```python
# datus_semantic_metricflow/__init__.py
from datus_semantic_metricflow.adapter import MetricFlowAdapter
from datus_semantic_metricflow.config import MetricFlowConfig

def register():
    """Register MetricFlow adapter with Datus semantic adapter registry."""
    # Import at runtime to avoid circular dependencies
    from datus.tools.semantic_tools.registry import semantic_adapter_registry

    semantic_adapter_registry.register(
        service_type="metricflow",
        adapter_class=MetricFlowAdapter,
        config_class=MetricFlowConfig,
        display_name="MetricFlow",
    )

__all__ = ["MetricFlowAdapter", "MetricFlowConfig", "register"]
```

### Step 5: Configure Entry Point

```toml
# pyproject.toml (in datus_semantic_metricflow directory)
[project]
name = "datus-semantic-metricflow"
version = "0.1.0"
description = "MetricFlow adapter for Datus semantic layer"
dependencies = [
    "pydantic>=2.0.0",
]

[project.entry-points."datus.semantic_adapters"]
metricflow = "datus_semantic_metricflow:register"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["datus_semantic_metricflow*"]
```

### Step 6: Install and Use

```bash
# Install specific adapter from datus-semantic-adapter repository
pip install -e ../datus-semantic-adapter/datus_semantic_metricflow

# Verify installation
pip show datus-semantic-metricflow
```

The adapter will be automatically discovered via entry points!

## Usage Examples

### 1. Bootstrap from Adapter

```bash
# Pull semantic models from MetricFlow
datus-agent bootstrap-kb \
  --namespace my_project \
  --components semantic_model \
  --from_adapter metricflow \
  --kb-update-strategy overwrite

# Pull metrics from MetricFlow with subject tree categorization
datus-agent bootstrap-kb \
  --namespace my_project \
  --components metrics \
  --from_adapter metricflow \
  --subject-path "Finance,Sales,Operations" \
  --kb-update-strategy overwrite
```

### 2. Use in Agent Node

```python
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.configuration.agent_config import AgentConfig

# Initialize tool (uses storage by default)
semantic_tool = SemanticTools(
    agent_config=agent_config,
    sub_agent_name="gen_metrics",
    adapter_type=None,  # Storage-only mode
)

# Or with adapter for direct queries
semantic_tool = SemanticTools(
    agent_config=agent_config,
    sub_agent_name="gen_metrics",
    adapter_type="metricflow",  # Enable MetricFlow adapter
)

# Get available tools for LLM
tools = semantic_tool.available_tools()
# Returns: [search_metrics, list_metrics, get_dimensions, query_metrics, validate_semantic]
```

### 3. Direct Adapter Usage

```python
from datus.tools.semantic_tools import semantic_adapter_registry

# Create adapter (config class is loaded from adapter's metadata)
from datus_semantic_metricflow.config import MetricFlowConfig

config = MetricFlowConfig(
    namespace="my_project",
    cli_path="mf",
)

adapter = semantic_adapter_registry.create_adapter("metricflow", config)

# Or let Datus auto-create config from namespace
metadata = semantic_adapter_registry.get_metadata("metricflow")
config = metadata.config_class(namespace="my_project")
adapter = semantic_adapter_registry.create_adapter("metricflow", config)

# List metrics
metrics = await adapter.list_metrics(limit=10)

# Get dimensions
dimensions = await adapter.get_dimensions(metric_name="revenue")

# Query metrics
result = await adapter.query_metrics(
    metrics=["revenue", "orders"],
    dimensions=["date", "region"],
    time_range={"start": "2024-01-01", "end": "2024-12-31"},
)
```

### 4. Sync to Storage

```python
from datus.tools.semantic_tools import SemanticStorageManager

# Create storage manager
storage_manager = SemanticStorageManager(agent_config)

# Sync from adapter
stats = await storage_manager.sync_from_adapter(
    adapter=adapter,
    sync_semantic_models=True,
    sync_metrics=True,
    subject_path=["Finance", "Revenue"],
)

print(f"Synced {stats['semantic_models_synced']} models, {stats['metrics_synced']} metrics")
```

## Design Decisions

### 1. Storage-First Strategy
- Tools query unified storage (SemanticModelStorage, MetricStorage) first
- Adapter calls only when storage is empty or for live operations
- Provides fast vector search and consistent interface

### 2. Async Metrics Interface
- All metrics methods use `async/await`
- Supports HTTP/RPC calls to remote services (dbt Cloud, Cube API)
- MetricFlow CLI calls use `asyncio.subprocess`

### 3. Sync Semantic Model Interface
- `get_semantic_model()` remains synchronous
- Used by `describe_table()` for fast enrichment
- Most adapters won't implement full semantic model support

### 4. Entry Points Discovery
- Adapters auto-register via `datus.semantic_adapters` entry point
- No manual registration needed
- Supports plugin ecosystem

## Testing

```python
# tests/test_adapter.py
import pytest
from datus.tools.semantic_tools import semantic_adapter_registry
from datus_semantic_metricflow.adapter import MetricFlowAdapter
from datus_semantic_metricflow.config import MetricFlowConfig

@pytest.mark.asyncio
async def test_list_metrics():
    config = MetricFlowConfig(
        namespace="test",
        cli_path="mf",
    )
    adapter = MetricFlowAdapter(config)

    metrics = await adapter.list_metrics(limit=5)
    assert len(metrics) <= 5
    assert all(isinstance(m.name, str) for m in metrics)

@pytest.mark.asyncio
async def test_query_metrics():
    config = MetricFlowConfig(namespace="test")
    adapter = MetricFlowAdapter(config)

    result = await adapter.query_metrics(
        metrics=["test_metric"],
        dimensions=["date"],
        dry_run=True,  # Don't execute, just validate
    )

    assert result.metadata.get("sql") is not None
```

## Configuration Details

### MetricFlow Adapter Configuration

The MetricFlow adapter uses datus-metricflow CLI which automatically finds semantic model files:

1. **Namespace parameter**: Required, used to locate files
2. **Model path resolution**: `{agent.home}/semantic_models/{namespace}/`
   - `agent.home` is read from `agent.yml` (defaults to `~/.datus`)
   - No need to specify `project_root` manually

3. **Agent config lookup priority**:
   - `--config` parameter (if provided to `mf` CLI)
   - `./conf/agent.yml` (current directory)
   - `~/.datus/conf/agent.yml` (home directory)

### Example agent.yml

```yaml
agent:
  home: /Users/myuser/.datus  # Optional, defaults to ~/.datus

namespaces:
  starrocks:
    - name: starrocks_db
      type: starrocks
      host: localhost
      port: 9030
      # ... database config
```

## Error Handling

Use the predefined error codes from `datus.utils.exceptions`:

```python
from datus.utils.exceptions import DatusException, ErrorCode

# Adapter not found
raise DatusException(
    ErrorCode.SEMANTIC_ADAPTER_NOT_FOUND,
    message_args={"adapter_type": "unknown_service"}
)

# Adapter operation failed
raise DatusException(
    ErrorCode.SEMANTIC_ADAPTER_ERROR,
    message_args={"error_message": "Connection timeout"}
)

# Configuration error
raise DatusException(
    ErrorCode.SEMANTIC_ADAPTER_CONFIG_ERROR,
    message_args={"error_message": "Missing project_root"}
)

# Sync failed
raise DatusException(
    ErrorCode.SEMANTIC_ADAPTER_SYNC_FAILED,
    message_args={"error_message": "Invalid metric definition"}
)
```

## References

- **Architecture Pattern**: [datus/tools/db_tools/](../db_tools/) - Database adapter pattern
- **Storage Integration**: [datus/storage/semantic_model/](../../storage/semantic_model/), [datus/storage/metric/](../../storage/metric/)
- **Example Usage**: [datus/agent/node/gen_metrics_agentic_node.py](../../agent/node/gen_metrics_agentic_node.py)
