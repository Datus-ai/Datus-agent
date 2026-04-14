# Adapter Generation Guide

## Overview

The adapter generation feature helps you create Datus adapter project scaffolding for integrating external platforms. Using an AI assistant, you can generate a complete adapter project skeleton, get guidance on implementing platform-specific logic, and validate the result with contract tests — all within a single interactive workflow.

## What is an Adapter?

An **adapter** is a Python package that connects Datus to an external platform. Datus supports four adapter types:

| Type | Base Class | Example Platforms | Package Naming |
|------|-----------|-------------------|----------------|
| **semantic** | `BaseSemanticAdapter` | Cube, Looker, dbt Semantic Layer | `datus_semantic_{platform}` |
| **bi** | `BIAdapterBase` | Superset, Metabase, Grafana | `datus_bi_{platform}` |
| **db** | `BaseSqlConnector` | ClickHouse, Snowflake, BigQuery | `datus_{platform}` |
| **scheduler** | `BaseSchedulerAdapter` | Airflow, Dagster, Prefect | `datus_scheduler_{platform}` |

Each adapter type has a defined interface contract specifying the methods to implement, return types, and registration mechanism.

## Quick Start

Start Datus CLI with `datus --database <namespace>`, then invoke the gen_adapter subagent:

```bash
/gen_adapter Generate a semantic adapter for the Cube platform
```

Or describe the integration need naturally — the chat agent will delegate to gen_adapter automatically:

```bash
I need to integrate Cube as a semantic layer platform
```

## How It Works

### Generation Workflow

```
User describes platform → Agent confirms intent → Scaffolds project → Assists implementation →
Validates static checks → Runs contract tests → Iterates until passing → Summary
```

The workflow has five phases:

### Phase 1: Understand Intent

The agent identifies the adapter type, platform name, and output directory from your request, then asks for confirmation before generating any files.

**Example interaction:**

```
User: Generate a semantic adapter for Cube

Agent: I'll create a semantic adapter for the Cube platform:
  - Adapter type: semantic
  - Platform: cube
  - Output directory: ./datus-semantic-cube
  - Package name: datus_semantic_cube

Shall I proceed? [confirm via ask_user]
```

### Phase 2: Generate Skeleton

After confirmation, the agent calls `scaffold_adapter` to generate a complete project:

```
datus-semantic-cube/
├── datus_semantic_cube/
│   ├── __init__.py          # register() function + exports
│   ├── adapter.py           # CubeAdapter class with method stubs
│   ├── config.py            # CubeConfig (Pydantic model)
│   └── py.typed             # PEP 561 marker
├── tests/
│   ├── conftest.py          # Shared test fixtures
│   └── unit/
│       ├── test_adapter.py  # Basic instantiation tests
│       └── test_contract.py # Contract test suite (semantic only)
├── pyproject.toml           # Package metadata + entry point
└── README.md
```

The generated `adapter.py` contains method stubs that raise `NotImplementedError`:

```python
from datus_semantic_core import BaseSemanticAdapter, DimensionInfo, MetricDefinition, QueryResult, ValidationResult

class CubeAdapter(BaseSemanticAdapter):
    """Cube adapter for Datus."""

    def __init__(self, config: "CubeConfig"):
        super().__init__(config)

    async def list_metrics(self, path=None, limit=100, offset=0) -> List[MetricDefinition]:
        raise NotImplementedError("TODO: Implement list_metrics")

    async def get_dimensions(self, metric_name, path=None) -> List[DimensionInfo]:
        raise NotImplementedError("TODO: Implement get_dimensions")

    async def query_metrics(self, metrics, dimensions=None, ...) -> QueryResult:
        raise NotImplementedError("TODO: Implement query_metrics")

    async def validate_semantic(self) -> ValidationResult:
        raise NotImplementedError("TODO: Implement validate_semantic")
```

### Phase 3: Assist with Implementation

The agent helps implement each method stub:

1. **Gathers platform knowledge** — Searches official API documentation using `web_search_document`
2. **Proposes implementation** — Maps platform APIs to Datus interface requirements
3. **Writes code** — After user confirmation, updates `adapter.py` and `config.py` via `write_file` / `edit_file`
4. **Fills contract test factory** (semantic only) — Updates `tests/unit/test_contract.py` with mocked fixtures

### Phase 4: Validate

Validation is a two-step process:

**4a. Static validation** via `validate_adapter`:

- Module imports successfully
- `register()` function is exposed
- Adapter class exists
- No methods still raise `NotImplementedError`

**4b. Contract test execution** (semantic adapters) via `run_adapter_pytest`:

```bash
# The agent runs this internally:
run_adapter_pytest(
    project_dir="/absolute/path/to/datus-semantic-cube",
    test_subpath="tests/unit/test_contract.py"
)
```

The contract test suite (from `datus_semantic_core.testing`) enforces:

| Contract | Assertion |
|----------|-----------|
| `list_metrics` return type | `list[MetricDefinition]` |
| `MetricDefinition.dimensions` | `list[str]` (not DimensionInfo) |
| `get_dimensions` return type | `list[DimensionInfo]` |
| `query_metrics` return type | `QueryResult` with `.data` as `list[dict]` |
| `query_metrics(dry_run=True)` | Sets `metadata['dry_run']` or includes `'sql'` column |
| `validate_semantic` return type | `ValidationResult` |

If tests fail, the agent reads the failure output, fixes the code, and re-runs — looping until all tests pass.

### Phase 5: Summary

The agent presents a summary of the completed adapter: files created/modified, contract test results, and any remaining TODOs.

## Configuration

Most configurations are built-in. In `agent.yml`, minimal setup is needed:

```yaml
agentic_nodes:
  gen_adapter:
    model: claude        # Optional: defaults to configured model
    max_turns: 30        # Optional: defaults to 30
```

**Built-in configurations** (automatically enabled):

- **Scaffold tools**: `scaffold_adapter`, `validate_adapter`, `list_adapter_types`
- **Filesystem tools**: `read_file`, `read_multiple_files`, `write_file`, `edit_file`, `list_directory`
- **Documentation tools**: `list_document_nav`, `get_document`, `search_document`, `web_search_document`
- **Test runner**: `run_adapter_pytest`
- **Interactive mode**: `ask_user` (only in interactive mode)

### Configuration Options

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `model` | No | LLM model to use | Uses default configured model |
| `max_turns` | No | Maximum conversation turns | 30 |

## Available Tools

| Tool | Description |
|------|-------------|
| `list_adapter_types` | List all supported adapter types with metadata |
| `scaffold_adapter` | Generate a complete adapter project skeleton |
| `validate_adapter` | Static validation: imports, register(), adapter class, stub check |
| `run_adapter_pytest` | Run pytest inside a scaffolded adapter project (restricted scope) |
| `read_file` / `write_file` / `edit_file` | Read and modify generated adapter code |
| `list_directory` | Browse project file structure |
| `web_search_document` | Search external platform API documentation |
| `ask_user` | Confirm decisions with the user (interactive mode only) |

## Usage Examples

### Example 1: Semantic Adapter (Cube)

**User Input:**
```bash
/gen_adapter Generate a semantic adapter for the Cube platform
```

**Agent Actions:**
1. Confirms: semantic adapter, platform "cube", output directory
2. Scaffolds `datus-semantic-cube/` with 11 files
3. Searches Cube REST API documentation
4. Implements `list_metrics` using Cube's `/v1/meta` endpoint
5. Implements `get_dimensions` using `/v1/meta` dimension metadata
6. Implements `query_metrics` using Cube's `/v1/load` endpoint
7. Implements `validate_semantic` using `/v1/meta` health check
8. Fills `test_contract.py` factory with mocked HTTP responses
9. Runs contract tests — all pass
10. Reports summary

### Example 2: BI Adapter (Metabase)

**User Input:**
```bash
/gen_adapter Create a BI adapter for Metabase
```

**Agent Actions:**
1. Confirms: bi adapter, platform "metabase"
2. Scaffolds `datus-bi-metabase/` with 10 files
3. Implements `platform_name`, `list_dashboards`, `get_dashboard_info`, `list_charts`, `list_datasets`
4. Validates — all methods implemented
5. Reports summary

### Example 3: Database Connector

**User Input:**
```bash
/gen_adapter Generate a database adapter for ClickHouse
```

**Agent Actions:**
1. Confirms: db adapter, platform "clickhouse"
2. Scaffolds `datus-clickhouse/` with 10 files
3. Implements `execute`, `test_connection`, `get_databases`
4. Validates — all methods implemented

## Adapter Interface Contracts

### Semantic Adapter (4 methods)

| Method | Signature | Return Type |
|--------|-----------|-------------|
| `list_metrics` | `(self, path=None, limit=100, offset=0)` | `List[MetricDefinition]` |
| `get_dimensions` | `(self, metric_name, path=None)` | `List[DimensionInfo]` |
| `query_metrics` | `(self, metrics, dimensions=None, path=None, time_start=None, time_end=None, time_granularity=None, where=None, limit=None, order_by=None, dry_run=False)` | `QueryResult` |
| `validate_semantic` | `(self)` | `ValidationResult` |

### BI Adapter (5 methods)

| Method | Signature | Return Type |
|--------|-----------|-------------|
| `platform_name` | `(self)` | `str` |
| `list_dashboards` | `(self, search='', page_size=20)` | `list` |
| `get_dashboard_info` | `(self, dashboard_id)` | `object` |
| `list_charts` | `(self, dashboard_id)` | `list` |
| `list_datasets` | `(self, dashboard_id)` | `list` |

### DB Adapter (3 methods)

| Method | Signature | Return Type |
|--------|-----------|-------------|
| `execute` | `(self, input_params, result_format=None)` | `object` |
| `test_connection` | `(self)` | `bool` |
| `get_databases` | `(self, catalog_name='', include_sys=False)` | `list` |

### Scheduler Adapter (5 methods)

| Method | Signature | Return Type |
|--------|-----------|-------------|
| `platform_name` | `(self)` | `str` |
| `test_connection` | `(self)` | `bool` |
| `submit_job` | `(self, payload)` | `object` |
| `get_job` | `(self, job_id)` | `object` |
| `list_jobs` | `(self, project=None, status=None, limit=50, offset=0)` | `list` |

## Contract Tests (Semantic Only)

Semantic adapters get auto-generated contract tests wired to `datus_semantic_core.testing.make_semantic_contract_suite`. This shared test suite enforces the interface spec without duplicating assertion logic across adapters.

To make the contract tests pass after implementing the adapter:

1. Fill in the `factory()` function in `tests/unit/test_contract.py` — construct a test-ready adapter with mocked HTTP/SDK layer
2. Set `SAMPLE_METRIC_NAME` and `SAMPLE_DIMENSION_NAME` to values your mock fixtures expose
3. Run: `pytest tests/unit/test_contract.py -v`

### Example factory:

```python
from unittest.mock import AsyncMock

async def factory() -> CubeAdapter:
    config = CubeConfig(api_url="http://localhost:4000", api_token="test-token")
    adapter = CubeAdapter(config)
    # Mock the HTTP layer with fixture responses
    adapter._http_get = AsyncMock(return_value={
        "cubes": [{"name": "orders", "measures": [...], "dimensions": [...]}]
    })
    return adapter
```

## Security Constraints

The `run_adapter_pytest` tool is intentionally restricted:

- `project_dir` must be absolute and contain `pyproject.toml`
- `test_subpath` must start with `tests/` and cannot contain `..`
- No free-form pytest flags — arguments are fixed to `-q --tb=short --no-header`
- Hard 120-second timeout per invocation
- Output truncated to 8KB tail to prevent context overflow

## Summary

The adapter generation feature provides:

- Automated project scaffolding for 4 adapter types (semantic, bi, db, scheduler)
- AI-assisted implementation with platform API documentation search
- Static validation (import, register, stub detection)
- Contract test generation and execution (semantic adapters)
- Iterative fix loop: scaffold → implement → test → fix → re-test
- Security-scoped pytest runner for generated projects
- Interactive workflow with user confirmation at key decision points
