# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Datus is an open-source data engineering agent that builds evolvable context for data systems. It provides three main interfaces:
- **Datus-CLI**: AI-powered command-line interface for data engineers
- **Datus-Chat**: Web chatbot for data analysts
- **Datus-API**: REST APIs for programmatic access

## Common Commands

### Development Setup
```bash
# Install in development mode
make setup-dev
# or
pip install -e ".[dev]"

# Install from source
make build
make install-dist
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schema_recall_bird.py -v

# Run acceptance tests (quick validation)
pytest tests/test_*_model.py -m acceptance -q

# Run tests with output shown
pytest tests/ -s -vv
```

### Code Quality
```bash
# Format code (Black with 120 char line length)
black datus/ --line-length=120

# Lint code
flake8 datus/ --max-line-length=120

# Sort imports
isort datus/ --profile=black --line-length=120

# Run pre-commit hooks
pre-commit run --all-files
```

### Building and Publishing
```bash
# Clean build artifacts
make clean

# Build package
make build

# Quick build (clean + build)
make quick-build

# Publish to PyPI (full workflow)
make publish
```

### Running the Application
```bash
# Interactive initialization wizard
datus-agent init

# Run CLI with specific namespace
datus-cli --namespace demo

# Start API server
python datus/api/server.py

# Start API server in daemon mode
python datus/api/server.py --daemon

# Run benchmarks
python datus/main.py bootstrap-kb --benchmark bird_dev --namespace bird_sqlite
```

## Architecture

### Core Components

**Agent System (`datus/agent/`)**
- `agent.py`: Main Agent orchestrator that coordinates workflow execution
- `workflow.py`: Workflow management with node execution orchestration
- `workflow_runner.py`: Executes workflows with plan selection and reflection rounds
- `plan.py`: Handles workflow planning and dynamic plan generation
- `node/`: Individual workflow nodes (schema_linking, generate_sql, execute_sql, etc.)

**Node Types**
Nodes are the building blocks of workflows. Key node categories:
- **Agentic Nodes**: Multi-turn AI-powered nodes (chat_agentic_node, gen_sql_agentic_node, semantic_agentic_node)
- **Processing Nodes**: Single-step operations (schema_linking_node, generate_sql_node, execute_sql_node)
- **Control Flow**: Orchestration nodes (parallel_node, selection_node, subworkflow_node)

**CLI Interface (`datus/cli/`)**
- `repl.py`: Interactive REPL with command processing
- `agent_commands.py`: Tool commands (!, @, /) for workflow interaction
- `chat_commands.py`: Natural language chat interface
- `autocomplete.py`: SQL and command auto-completion
- `sub_agent_wizard.py`: Interactive subagent creation wizard
- `bi_dashboard.py`: Business intelligence dashboard integration

**Storage Layer (`datus/storage/`)**
- Built on LanceDB for vector storage and hybrid search
- `base.py`: StorageBase and BaseEmbeddingStore abstract classes
- `schema_metadata/`: Database schema vector storage
- `metric/`: Metrics definition storage with semantic search
- `reference_sql/`: SQL history and example storage
- `document/`: External knowledge document storage

**Models (`datus/models/`)**
- Unified multi-LLM interface supporting OpenAI, Claude, DeepSeek, Qwen, Gemini
- `base.py`: Abstract LLMBaseModel with factory pattern
- `openai_compatible.py`: Shared base for OpenAI-compatible APIs
- `claude_model.py`: Anthropic Claude with native API and prompt caching
- MCP (Model Context Protocol) integration via OpenAI Agents

**Configuration (`datus/configuration/`)**
- `agent_config.py`: Main configuration dataclasses (AgentConfig, ModelConfig, NodeConfig)
- `agent_config_loader.py`: YAML config loading with environment variable resolution
- `node_type.py`: Node type registry and definitions

**API Server (`datus/api/`)**
- FastAPI-based REST API with streaming support
- `server.py`: Server lifecycle and daemon management
- `service.py`: Workflow execution service with async/sync modes
- `auth.py`: OAuth2 client credentials authentication

### Workflow Execution Model

Workflows execute as directed acyclic graphs (DAGs) of nodes:

1. **Sequential Execution**: Nodes execute in order defined in workflow YAML
2. **Parallel Execution**: Multiple nodes run concurrently using `parallel:` block
3. **Selection**: Choose best output from parallel branches using `selection` node
4. **Reflection**: Re-execute node sequences on failure via `reflection_nodes` config
5. **Subworkflows**: Nested workflows with isolated configuration

Example workflow from `conf/agent.yml.example`:
```yaml
workflow:
  plan: bird_para

  bird_para:
    - schema_linking
    - parallel:
      - generate_sql
      - reasoning
    - selection
    - execute_sql
    - output
```

### Configuration System

**Primary Config**: `conf/agent.yml`
- LLM models with API keys and base URLs
- Database namespaces (Snowflake, SQLite, DuckDB, StarRocks, etc.)
- Node configurations with model assignments
- Workflow definitions and reflection strategies
- Storage/embedding model settings
- Benchmark configurations

**Environment Variables**: Resolved in config using `${VAR_NAME}` syntax

**Storage Paths**: Fixed at `{agent.home}/data/` where `agent.home` defaults to `~/.datus`
- Sessions: `{agent.home}/sessions/`
- Embeddings: `{agent.home}/data/datus_db_{namespace}/`
- Subagents: `{agent.home}/data/sub_agents/{agent_name}/`

### Key Data Flow

1. **User Input** → CLI/API receives natural language query
2. **Schema Linking** → Vector search finds relevant tables/columns
3. **SQL Generation** → LLM generates SQL (possibly with reasoning/metrics)
4. **Execution** → SQL runs against database via namespace connection
5. **Reflection** (optional) → On error, retry with error context
6. **Output** → Results formatted and saved to CSV

### Subagent Architecture

Subagents are domain-specific chatbots with scoped context:
- Each subagent has its own knowledge base (vector store)
- Configurable table/metric scope limits
- Custom rules and instructions
- Isolated tool configurations
- Stored in `{agent.home}/data/sub_agents/{name}/`

Created via interactive wizard (`datus/cli/sub_agent_wizard.py`) or CLI commands.

## Important Implementation Details

### MCP Integration
- Uses OpenAI Agents SDK for Model Context Protocol
- MCP servers provide tools (database queries, file operations)
- Connection managed in `models/mcp_utils.py`
- Tool results extracted in `models/mcp_result_extractors.py`

### Streaming Responses
- Both CLI and API support streaming output
- Action history tracked via `schemas/action_history.py`
- CLI hooks in `cli/generation_hooks.py` for real-time display
- API uses Server-Sent Events (SSE) for streaming

### Database Adapters
- Database connectors moved to separate packages: https://github.com/Datus-ai/Datus-adapters
- Core supports: Snowflake, SQLite, DuckDB, StarRocks (via SQLAlchemy)
- Multi-database support for SQLite/DuckDB via `dbs:` array in config

### Prompt Templates
- Jinja2 templates in `datus/prompts/prompt_templates/`
- Version-controlled prompts via `prompt_version` in node config
- Templates receive context (schemas, metrics, examples) for rendering

### Error Handling
- Custom exceptions in `utils/exceptions.py`
- DatusException with ErrorCode enum for categorized errors
- LLM retry logic with exponential backoff

### Testing Strategy
- Unit tests for core components
- Integration tests for workflows (`test_integration_benchmark.py`)
- Acceptance tests marked with `@pytest.mark.acceptance`
- Benchmark tests for BIRD/Spider2 datasets

## Code Style Conventions

- **Line Length**: 120 characters (enforced by Black and Flake8)
- **Import Order**: Use isort with Black profile
- **Type Hints**: Preferred but not required everywhere
- **Async/Await**: Used for LLM calls, MCP integration, and streaming
- **Logging**: Use `datus.utils.loggings.get_logger(__name__)`
- **Configuration**: Dataclasses for type safety, YAML for user config
- **Naming**: Snake_case for functions/variables, PascalCase for classes
- **Exclude MCP**: MCP server code (`mcp/`) excluded from linting

## Development Workflow

1. **Make changes** in relevant module
2. **Run formatters**: `black` and `isort`
3. **Run linter**: `flake8`
4. **Run tests**: `pytest` with appropriate markers
5. **Update version**: Bump in `datus/__init__.py` and `pyproject.toml`
6. **Build**: `make build` to verify package builds
7. **Commit**: Follow conventional commit style for PR titles

## Common Pitfalls

- **Path Issues**: Storage paths are fixed at `{agent.home}/data/`, not configurable separately
- **Environment Variables**: Must be set before loading config, resolved at load time
- **MCP Servers**: Require async context managers, handle cleanup properly
- **Session Management**: Sessions stored in SQLite, need proper lifecycle management
- **Embedding Models**: Dimension size must match between config and storage schema
- **Node Dependencies**: Nodes access prior node outputs via workflow context
- **Streaming**: Action history must be yielded incrementally for streaming to work

## Testing Individual Components

```bash
# Test specific model
pytest tests/test_claude_model.py -v

# Test schema linking
pytest tests/test_schema_recall_bird.py -v

# Test document search
pytest tests/test_doc_search.py -v

# Test full workflow
pytest tests/test_integration_benchmark.py -v

# Test with specific benchmark
python datus/main.py benchmark-test --benchmark bird_dev --limit 5
```

## When Adding New Features

**New Node Type:**
1. Create node class in `datus/agent/node/`
2. Define input/output schemas in `datus/schemas/`
3. Register in `configuration/node_type.py`
4. Add to workflow YAML and test

**New LLM Provider:**
1. Implement in `datus/models/{provider}_model.py`
2. Add to `MODEL_TYPE_MAP` in `models/base.py`
3. Add provider constant in `utils/constants.py`
4. Update config schema in `conf/agent.yml.example`
5. Add tests in `tests/test_{provider}_model.py`

**New Storage Module:**
1. Inherit from `BaseEmbeddingStore` in `storage/base.py`
2. Define PyArrow schema with vector field
3. Implement search methods
4. Create indices for performance
5. Follow patterns in `storage/README.md`

## Resources

- Documentation: https://docs.datus.ai/
- Quick Start: https://docs.datus.ai/getting_started/Quickstart/
- CLI Guide: `datus/cli/README.md`
- Storage Guide: `datus/storage/README.md`
- Models Guide: `datus/models/README.md`
- API Guide: `datus/api/README.md`