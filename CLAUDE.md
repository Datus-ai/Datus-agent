# Datus-Agent Project Instructions

## Overview

Datus-Agent is an AI-powered data analysis agent: natural language → SQL, multi-database, RAG knowledge base, MCP protocol.

- **Stack**: Python 3.12+, OpenAI Agents SDK + LiteLLM, LanceDB, FastAPI, FastMCP, Streamlit
- **Package manager**: uv
- **Version**: 0.2.6 | License: Apache-2.0

## Build & Run

```bash
uv sync                                    # Install dependencies
uv run pytest tests/unit_tests/ -q                # CI tests (zero external deps)
uv run pytest -m nightly tests/             # Nightly tests (needs API keys)
uv run pytest -m "nightly or regression" tests/  # Full regression
uv run ruff format . && uv run ruff check --fix .      # Lint & format
bash build_scripts/build_test_data.sh       # Build test knowledge base
```

## Coding Conventions

### Style

- **Formatter**: ruff format, line-length = 120, extend-exclude = `mcp/`
- **Linter**: ruff check (E/W/F/B/I/C90 rules), line-length = 120
- **Imports**: ruff isort rules, group order: stdlib → third-party → `datus.*`
- **Type hints**: use throughout; Pydantic models for data structures

### Logging

Always use structured logging — never `print()`:
```python
from datus.utils.loggings import get_logger
logger = get_logger(__name__)
```

### Error Handling

Use `DatusException` with `ErrorCode` enum — never raise raw exceptions for expected failures:
```python
from datus.utils.exceptions import DatusException, ErrorCode
raise DatusException(ErrorCode.DB_CONNECTION_FAILED, message_args={"error_message": str(e)}) from e
```

Error code ranges: 100000–199999 (common), 200000–299999 (node), 300000–399999 (model/LLM), 400000–499999 (tool/storage), 500000–599999 (database), 600000–699999 (semantic adapter).

### Async

- Mark async tests with `@pytest.mark.asyncio`
- Use `pytest_asyncio.fixture` for async fixtures
- See `datus/utils/async_utils.py` for event loop handling (especially Windows)

## Architecture Patterns

### Adding a New Node

1. Create `datus/agent/node/{name}_node.py`
2. Inherit from `Node(ABC)` (standard) or `AgenticNode(Node)` (OpenAI Agents SDK-based)
3. Register the type constant in `datus/configuration/node_type.py`
4. Add the mapping in `Node.new_instance()` factory in `datus/agent/node/node.py`
5. Add nightly-level test in `tests/test_node.py`

### Adding a New LLM Model

1. Create `datus/models/{provider}_model.py`
2. Inherit from `LLMBaseModel(ABC)` in `datus/models/base.py`
3. Register in `LLMBaseModel.MODEL_TYPE_MAP`
4. Add to `PROVIDER_MODELS` in `tests/regression/test_regression_llm.py`

### Adding a New Database Connector

1. Create `datus/tools/db_tools/{db}_connector.py`
2. Inherit from `BaseSqlConnector(ABC)` in `datus/tools/db_tools/base.py`
3. Register via `ConnectorRegistry.register(db_type, connector_class)` in `registry.py`
4. Write DDL/schema/query tests following `tests/test_connector_duckdb.py`

### Adding a New MCP Tool

1. Add tool function in `datus/tools/func_tool/`
2. Register in the MCP server tool list
3. Add registration + invocation tests in `tests/test_mcp_server.py`

## PR Title Convention

PR titles **must** include a type prefix: `[BugFix]`, `[Enhancement]`, `[Feature]`, `[Refactor]`, `[UT]`, `[Doc]`, `[Tool]`, `[Others]`. CI will reject PRs without the prefix.

Examples:
- `[Feature] Add metric definition skill`
- `[BugFix] Fix tools display in LangSmith traces`
- `[Enhancement] Optimize schema linking performance`

## Commit Workflow

1. **Coverage gate**: Before committing, run `uv run pytest tests/unit_tests/ --cov=datus --cov-report=term-missing --cov-fail-under=80`. If coverage < 80%, add tests for uncovered lines in modified files until it passes. Do NOT commit until coverage passes.
2. **Pre-commit hook failures**: Never stop or use `--no-verify`. Auto-fix all issues (run `uv run ruff format . && uv run ruff check --fix .`), re-stage, and retry the commit until it succeeds.
3. **Push target**: Always push to `origin` only. Never push directly to `upstream`.

## Guardrails

- **No direct DB imports**: Use `ConnectorRegistry` / `db_manager_instance` — never import connector classes directly in business logic
- **No hardcoded LLM calls in Nodes**: Always go through `LLMBaseModel` and model config
- **No external deps in CI tests**: CI tests must run with zero API keys, zero pre-built data, zero network access
- **No print()**: Use `get_logger(__name__)` for all output
- **No raw exceptions for expected failures**: Use `DatusException(ErrorCode.XXX)`
- **No secrets in code**: API keys go in env vars or `agent.yml` with `${ENV_VAR}` substitution
- **Config via YAML**: New configurable parameters belong in `agent.yml` sections, not hardcoded constants

## Testing Rules

### Three-Tier Classification

| Tier | Marker | Criteria |
|------|--------|----------|
| CI | No marker (run by directory) | Zero external deps, zero pre-built data, deterministic, < 5s per test |
| Nightly | `@pytest.mark.nightly` | Requires real LLM API or pre-built LanceDB indexes |
| Regression | `@pytest.mark.regression` | Requires external DB instances or multiple LLM providers |

### Mock Strategy by Tier

| Tier | Strategy |
|------|----------|
| CI | **Must** mock all external calls (LLM, remote databases, network requests) |
| Nightly | Real LLM APIs allowed; still mock unstable external services |
| Regression | Use real services; handle missing API keys with `@pytest.mark.skipif` |

### Test File Naming

| Location | Pattern | Example |
|----------|---------|---------|
| `tests/unit_tests/` | `test_{module_name}.py` | `test_sql_utils.py` |
| `tests/` root | `test_{feature_name}.py` | `test_connector_duckdb.py` |
| `tests/integration/` | `test_{scenario}.py` | `test_mcp_server.py` |
| `tests/regression/` | `test_regression_{dimension}.py` | `test_regression_llm.py` |

### Source → Test File Mapping Rule

Unit test files **strictly mirror** the source path:

`datus/a/b/c.py` → `tests/unit_tests/a/b/test_c.py`

| Source File | Test File |
|-------------|-----------|
| `datus/utils/json_utils.py` | `tests/unit_tests/utils/test_json_utils.py` |
| `datus/agent/node/gen_sql_agentic_node.py` | `tests/unit_tests/agent/node/test_gen_sql_agentic_node.py` |
| `datus/tools/func_tool/db_func_tools.py` | `tests/unit_tests/tools/func_tool/test_db_func_tools.py` |

Create intermediate `__init__.py` files when adding tests to new subdirectories.

### Common Test Patterns

```python
# Skip when API key is missing
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DeepSeek API key not set")

# Parameterize across database types
@pytest.mark.parametrize("db_type", [DBType.SQLITE, DBType.DUCKDB])
```

### Test Checklist When Modifying Code

| Modified Module | Required Test Files |
|----------------|-------------------|
| `datus/configuration/` | test_configuration_load.py, test_openai_headers.py |
| `datus/models/{provider}_model.py` | integration/models/test_*_model.py, regression/test_regression_llm.py |
| `datus/agent/node/` | test_node.py, test_schema_linking.py, test_date_parser_*.py |
| `datus/agent/workflow.py` | test_workflow.py, test_planning.py |
| `datus/cli/repl.py` | test_cli_commands.py, regression/test_regression_web*.py |
| `datus/cli/tutorial.py` | test_tutorial.py |
| `datus/cli/bi_dashboard.py` | test_bi_dashboard.py, unit_tests/test_bi_superset_adaptor.py |
| `datus/tools/func_tool/` | unit_tests/test_db_func_tools.py, unit_tests/test_context_search_tools.py, test_func_tools_db.py |
| `datus/tools/skill_tools/` | test_skill_config.py, test_skill_registry.py, test_skill_manager.py, test_skill_bash_tool.py, test_skill_func_tool.py |
| `datus/tools/permission/` | test_permission_config.py, test_permission_hooks.py, test_permission_manager.py |
| `datus/tools/bi_tools/` | unit_tests/test_bi_superset_adaptor.py, test_bi_dashboard.py |
| `datus/mcp_server.py` | test_mcp_server.py, integration/tools/test_mcp_server.py |
| `datus/storage/` | unit_tests/test_storage_*.py, unit_tests/test_subject_tree_store.py, test_storage.py |
| `datus/storage/schema_metadata/` | test_schema_recall_*.py, test_llm_recall.py |
| `datus/storage/reference_template/` | unit_tests/storage/reference_template/test_*.py, integration/tools/test_reference_template.py |
| `datus/storage/document/` | test_doc_search.py, integration/test_integration_platform_doc.py |
| `datus/utils/sql_utils.py` | unit_tests/test_sql_utils.py |
| `datus/utils/json_utils.py` | unit_tests/test_json_utils.py |
| `datus/utils/pyarrow_utils.py` | unit_tests/test_pyarrow_utils.py |
| `datus/schemas/` | test_input_result.py, test_nav_resolver.py, unit_tests/test_sub_agent_manager.py |

### New Code Requirements

- **New public functions/classes** → CI-level tests in `tests/unit_tests/`
- **New agent nodes** → nightly test in `test_node.py`
- **New LLM adaptors** → register in `regression/test_regression_llm.py`
- **New CLI commands** → test in `test_cli_commands.py`
- **New MCP tools** → test in `test_mcp_server.py`
- **New reference template files** → integration test in `integration/tools/test_reference_template.py`
- **New connectors** → tests following `test_connector_duckdb.py`
