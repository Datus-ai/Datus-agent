# Configuration Introduction

The Agent is the central configuration unit in Datus. It defines how your agent behaves, which models and databases it connects to, and how workflows are executed. By customizing the agent, you can adapt Datus to fit different environments and business scenarios.

## Key Highlights of Agent Configuration

- **Flexibility at scale**: Configure multiple models, connect to multiple databases, and define node strategies or workflow rules
- **Agent configuration file**: `agent.yml` is the primary configuration file for both datus-agent and datus-cli
- **Startup priority**:

    1. File specified by `-f`
    2. `./conf/agent.yml`
    3. `~/.datus/conf/agent.yml`

- **Separation of concerns**: MCP (Model Context Protocol) configuration is stored in `.mcp.json`, not in `agent.yml`, ensuring a clear boundary between agent settings and MCP server management

With this structure, agents in Datus remain modular, portable, and easy to maintain, giving you full control over how they run across different environments.

## Configuration Overview

The Datus Agent configuration is the heart of how your system behaves—it governs which models to use, how different components (nodes, workflows, storage, namespaces, benchmarks) connect, and how queries get processed end-to-end.

Here's a high-level summary of each module and how they relate:

| Module | Purpose | Key Concepts / Responsibilities |
|--------|---------|--------------------------------|
| **[Agent](agent.md)** | Global settings & model providers | Defines the default target LLM, and the set of supported LLM providers (with types, base URLs, API keys, model names) |
| **[Nodes](nodes.md)** | Task-level processing units | Each "node" handles a specific step (schema linking, SQL generation, reasoning, reflection, output formatting, chat, utilities) in the data-to-SQL pipeline |
| **[Workflow](workflow.md)** | Orchestration of nodes | Defines execution plans (sequential, parallel, sub-workflows, reflection paths) that specify how nodes are chained to answer a user's query |
| **[Storage](storage.md)** | Embeddings & vector store configuration | Manages embedding models, device settings, embedding storage paths, and how metadata / documents / metrics are embedded and retrieved |
| **[Namespace](namespace.md)** | Database connection abstraction | Encapsulates configurations for different databases (Snowflake, StarRocks, SQLite, DuckDB, etc.), allowing multi-database support under logical "namespaces" |
| **[Benchmark](benchmark.md)** | Evaluation & testing setup | Defines benchmark datasets (e.g. BIRD-DEV, Spider2, semantic layer) and paths to evaluate the SQL-generation performance of the agent |

## Configuration Structure

The main configuration file follows a hierarchical structure:

```yaml
# Global model configurations
models:
  default: "openai:gpt-4"
  providers:
    openai:
      type: "openai"
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"

# Database connections
namespace:
  production:
    type: snowflake
    account: "${SNOWFLAKE_ACCOUNT}"
    username: "${SNOWFLAKE_USER}"
    password: "${SNOWFLAKE_PASSWORD}"

# Workflow execution plans
workflow:
  default: "reflection"
  plans:
    reflection:
      - schema_linking
      - generate_sql
      - execute_sql
      - reflect
      - output

# Node-specific configurations
nodes:
  schema_linking:
    model: "openai:gpt-3.5-turbo"
    similarity_threshold: 0.7

# Storage and embeddings
storage:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store_path: "./data/vector_store"

# Benchmark datasets
benchmark:
  bird_dev:
    path: "./benchmark/bird/dev_20240627"
    database_pattern: "**/*.sqlite"
```

## Environment Variable Support

All configuration values support environment variable expansion with default values:

```yaml
# Direct environment variable reference
api_key: ${OPENAI_API_KEY}

# Environment variable with default value
timeout: ${API_TIMEOUT:-30}

# Complex string interpolation
connection_string: "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME}"
```

## Multi-Environment Configuration

Configure different environments using separate configuration files:

```text
conf/
├── agent.yml              # Main configuration
├── agent.yml.dev          # Development overrides
├── agent.yml.staging      # Staging environment
├── agent.yml.production   # Production settings
└── .mcp.json              # MCP server configuration
```

## Next Steps

Explore the detailed configuration for each component:

- **[Agent Settings](agent.md)**: Configure models, providers, and global settings
- **[Database Namespaces](namespace.md)**: Set up multi-database connections
- **[Database Adapters](../adapters/db_adapters.md)**: Install additional database connectors
- **[Workflow Definitions](workflow.md)**: Define custom execution patterns
- **[Node Configuration](nodes.md)**: Customize individual node behavior
- **[Storage Settings](storage.md)**: Configure knowledge base and vector storage
- **[Benchmark Datasets](benchmark.md)**: Set up evaluation and testing