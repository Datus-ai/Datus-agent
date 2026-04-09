# Semantic Adapter Interface Specification

This document defines the contract for implementing a Datus semantic adapter.
It is designed for LLM consumption during adapter code generation.

## Base Class

Inherit from `BaseSemanticAdapter` in `datus/tools/semantic_tools/base.py`.

```python
from datus.tools.semantic_tools.base import BaseSemanticAdapter
```

## Required Abstract Methods

### 1. `list_metrics`

```python
async def list_metrics(
    self,
    path: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[MetricDefinition]
```

**Purpose**: List available metrics from the semantic layer.

**Parameters**:
- `path`: Filter by subject hierarchy, e.g. `["domain", "subdomain"]`. `None` or `[]` = all.
- `limit`: Max results (pagination).
- `offset`: Skip count (pagination).

**Returns**: `List[MetricDefinition]` where each item has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Metric name (use platform's full name, e.g. `orders.count`) |
| `description` | `Optional[str]` | No | Human-readable description |
| `type` | `Optional[Any]` | No | Metric type: `simple`, `ratio`, `derived`, `cumulative`, `count`, `sum`, etc. |
| `dimensions` | `List[str]` | No | Dimension names available for grouping |
| `measures` | `List[str]` | No | Underlying measure names |
| `unit` | `Optional[str]` | No | Unit: `USD`, `count`, `percent` |
| `format` | `Optional[str]` | No | Display format: `currency`, `percent`, `,.2f` |
| `path` | `Optional[List[str]]` | No | Subject tree path for categorization |
| `metadata` | `Dict[str, Any]` | No | Platform-specific data (e.g. `cube_name`, `aggType`) |

**Example return**:
```python
[
    MetricDefinition(
        name="orders.count",
        description="Total number of orders",
        type="count",
        dimensions=["orders.status", "orders.created_at"],
        measures=[],
        metadata={"cube_name": "orders", "aggType": "count"},
    )
]
```

---

### 2. `get_dimensions`

```python
async def get_dimensions(
    self,
    metric_name: str,
    path: Optional[List[str]] = None,
) -> List[DimensionInfo]
```

**Purpose**: Get queryable dimensions for a specific metric.

**Returns**: `List[DimensionInfo]` where each item has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Dimension name |
| `description` | `Optional[str]` | No | Description |
| `type` | `Optional[str]` | No | Platform-native type: `string`, `number`, `time`, `boolean`, `geo`, `categorical` |
| `is_primary_key` | `Optional[bool]` | No | Whether this is the primary key dimension |

**Example return**:
```python
[
    DimensionInfo(name="orders.status", type="string", description="Order status"),
    DimensionInfo(name="orders.created_at", type="time", description="Creation date"),
]
```

---

### 3. `query_metrics`

```python
async def query_metrics(
    self,
    metrics: List[str],
    dimensions: Optional[List[str]] = None,
    path: Optional[List[str]] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    time_granularity: Optional[str] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    order_by: Optional[List[str]] = None,
    dry_run: bool = False,
) -> QueryResult
```

**Purpose**: Execute a metric query or explain the execution plan.

**Parameters**:
- `metrics`: Metric names to query, e.g. `["orders.count", "orders.total_revenue"]`
- `dimensions`: Group-by dimensions, e.g. `["orders.status"]`
- `time_start` / `time_end`: ISO date strings, e.g. `"2024-01-01"`
- `time_granularity`: `day`, `week`, `month`, `quarter`, `year`
- `where`: SQL-style filter string. **Each adapter converts internally** to its native format.
- `dry_run`: If `True`, return the generated SQL/query plan without executing.

**Returns**: `QueryResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `columns` | `List[str]` | Column names |
| `data` | `List[Dict[str, Any]]` | Result rows as dicts |
| `metadata` | `Dict[str, Any]` | Execution info: `execution_time`, `sql`, `row_count` |

**dry_run=True example**:
```python
QueryResult(
    columns=["sql"],
    data=[{"sql": "SELECT status, COUNT(*) FROM orders GROUP BY 1", "valid": True}],
    metadata={"dry_run": True},
)
```

---

### 4. `validate_semantic`

```python
async def validate_semantic(self) -> ValidationResult
```

**Purpose**: Validate the semantic layer configuration / connectivity.

**Returns**: `ValidationResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `bool` | Whether the configuration is valid |
| `issues` | `List[ValidationIssue]` | List of issues found |

Each `ValidationIssue` has: `severity` (`error`/`warning`/`info`), `message`, `location`.

---

## Optional Methods (with defaults)

### `list_semantic_models`

```python
def list_semantic_models(
    self, catalog_name="", database_name="", schema_name=""
) -> List[SemanticModelInfo]
```

Default: returns `[]`. Override to enable semantic model discovery.

**Returns**: `List[SemanticModelInfo]` where each item has:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Model name (cube name, explore name, etc.) |
| `description` | `Optional[str]` | Description |
| `platform_type` | `Optional[str]` | Platform type: `cube`, `view`, `explore`, `semantic_model` |
| `dimensions` | `List[DimensionInfo]` | Dimensions with type info |
| `measures` | `List[str]` | Measure names |
| `extra` | `Dict[str, Any]` | Platform-specific: joins, segments, connectedComponent, etc. |

### `get_semantic_model`

```python
def get_semantic_model(
    self, table_name, catalog_name="", database_name="", schema_name=""
) -> Optional[SemanticModelInfo]
```

Default: returns `None`. Override to get a single model by name.

---

## Configuration

Inherit from `SemanticAdapterConfig` in `datus/tools/semantic_tools/config.py`:

```python
from datus.tools.semantic_tools.config import SemanticAdapterConfig

class MyPlatformConfig(SemanticAdapterConfig):
    service_type: str = "my_platform"
    # SemanticAdapterConfig already provides:
    #   namespace, timeout_seconds, api_base_url, auth_token, username, password
    # Add platform-specific fields:
    custom_field: str = Field(default="", description="...")
```

---

## Registration

In `__init__.py`:

```python
from .adapter import MyPlatformAdapter
from .config import MyPlatformConfig

def register():
    from datus.tools.semantic_tools.registry import semantic_adapter_registry
    semantic_adapter_registry.register(
        service_type="my_platform",
        adapter_class=MyPlatformAdapter,
        config_class=MyPlatformConfig,
        display_name="My Platform",
    )
```

In `pyproject.toml`:

```toml
[project.entry-points."datus.semantic_adapters"]
my_platform = "datus_semantic_my_platform:register"
```

---

## Constructor Pattern

```python
class MyPlatformAdapter(BaseSemanticAdapter):
    def __init__(self, config: MyPlatformConfig):
        super().__init__(config=config, service_type="my_platform")
        self.api_base_url = config.api_base_url
        self.auth_token = config.auth_token
        # Initialize HTTP client, etc.
```

---

## Reference Implementation

The MetricFlow adapter at `datus-semantic-adapter/datus_semantic_metricflow/` serves as reference:

| Concept | MetricFlow | Pattern |
|---------|-----------|---------|
| `list_metrics()` | `client.semantic_model.metric_semantics` | Iterate native metrics, convert to MetricDefinition |
| `get_dimensions()` | `client.list_dimensions(metric_names=[...])` | Convert native dims to DimensionInfo |
| `query_metrics()` | `client.query()` / `client.explain()` | Build native query, convert DataFrame to QueryResult |
| `validate_semantic()` | ConfigLinter + ModelValidator | Multi-step validation, collect issues |
| `__init__()` | Build MetricFlowClient from config | Connect to backend, initialize client |

---

## Platform Concept Mapping

| Datus Concept | Cube | Looker | dbt Semantic Layer |
|--------------|------|--------|--------------------|
| MetricDefinition | Measure (in `/v1/meta`) | Measure (in explore fields) | Metric (via GraphQL `metricsPaginated`) |
| DimensionInfo | Dimension (in `/v1/meta`) | Dimension (in explore fields) | Dimension (via `dimensionsPaginated`) |
| SemanticModelInfo | Cube / View | Explore / View | Semantic Model |
| `list_metrics()` | `GET /v1/meta` → cubes[].measures | `GET /lookml_models/{m}/explores/{e}` → fields.measures | GraphQL `metricsPaginated` |
| `get_dimensions()` | `GET /v1/meta` → cubes[].dimensions | Same explore endpoint → fields.dimensions | GraphQL `dimensionsPaginated` |
| `query_metrics()` | `POST /v1/load` (JSON query) | Looker SDK `run_inline_query` | GraphQL `createQuery` → poll `query` |
| `query_metrics(dry_run)` | `POST /v1/sql` | N/A | GraphQL `compileSql` |
| `validate_semantic()` | `GET /readyz` | `GET /lookml_models` (parse errors) | N/A (always valid if deployed) |
| Auth | JWT in `Authorization` header (no Bearer) | Client ID/Secret → token exchange | Service token as Bearer |
| Filter (where) | JSON: `{"member","operator","values"}` | LookML filter expression | Jinja: `{{ Dimension('x') }} = 'y'` |
| Time dimension | Separate `timeDimensions` param | `dimension_group` with timeframes | `metric_time` canonical dim |
