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

The Datus Agent configuration is the heart of how your system behaves—it governs which models to use, how different components (nodes, workflows, storage, databases, benchmarks) connect, and how queries get processed end-to-end.

Here's a high-level summary of each module and how they relate:

| Module | Purpose | Key Concepts / Responsibilities |
|--------|---------|--------------------------------|
| **[Agent](agent.md)** | Global settings & model providers | Defines the default target LLM, and the set of supported LLM providers (with types, base URLs, API keys, model names) |
| **[Nodes](nodes.md)** | Task-level processing units | Each "node" handles a specific step (schema linking, SQL generation, reasoning, reflection, output formatting, chat, utilities) in the data-to-SQL pipeline |
| **[Workflow](workflow.md)** | Orchestration of nodes | Defines execution plans (sequential, parallel, sub-workflows, reflection paths) that specify how nodes are chained to answer a user's query |
| **[Storage](storage.md)** | Embeddings & vector store configuration | Manages embedding models, device settings, embedding storage paths, and how metadata / documents / metrics are embedded and retrieved |
| **[Datasources](datasources.md)** | Datasource configuration | Defines database connections under `agent.services.datasources`; semantic adapters, BI platforms, and schedulers are documented on sibling pages |
| **[Benchmark](benchmark.md)** | Evaluation & testing setup | Defines benchmark datasets (e.g. BIRD-DEV, Spider2, semantic layer) and paths to evaluate the SQL-generation performance of the agent |

## Configuration Structure

The main configuration file follows a hierarchical structure:

```yaml
agent:
  # Provider-level credentials (preferred)
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
    deepseek:
      api_key: "${DEEPSEEK_API_KEY}"

  # Custom / self-hosted models (optional)
  models:
    my-internal:
      type: openai
      base_url: "https://internal.example.com/v1"
      api_key: "${MY_KEY}"
      model: "internal-gpt-4"

  services:
    databases:
      production:
        type: snowflake
        account: "${SNOWFLAKE_ACCOUNT}"
        username: "${SNOWFLAKE_USER}"
        password: "${SNOWFLAKE_PASSWORD}"
        default: true

    semantic_layer:
      metricflow: {}

    bi_platforms:
      superset:
        type: superset
        api_base_url: "http://localhost:8088"
        username: "${SUPERSET_USER}"
        password: "${SUPERSET_PASSWORD}"

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: "${AIRFLOW_URL}"
        username: "${AIRFLOW_USER}"
        password: "${AIRFLOW_PASSWORD}"
        dags_folder: "${AIRFLOW_DAGS_DIR}"

  agentic_nodes:
    gen_metrics:
      semantic_adapter: metricflow
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod

storage:
  database:
    registry_name: sentence-transformers
    model_name: all-MiniLM-L6-v2
    dim_size: 384
  document:
    model_name: intfloat/multilingual-e5-large-instruct
    dim_size: 1024
  metric:
    model_name: all-MiniLM-L6-v2
    dim_size: 384

# Benchmark datasets
benchmark:
  my_custom_benchmark:
    benchmark_path: benchmark/my_custom
```

Additionally, each project directory can have a `.datus/config.yml` override for project-specific model and database selection:

```yaml title=".datus/config.yml"
target:
  provider: openai
  model: gpt-4.1
default_database: production
```

## Environment Variable Support

All configuration values support environment variable expansion with default values:

```yaml
# Direct environment variable reference
api_key: ${OPENAI_API_KEY}

# Environment variable with fallback
timeout: ${API_TIMEOUT}

# Complex string interpolation
connection_string: "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
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
- **[Datasources](datasources.md)**: Set up database connections under `agent.services.datasources`
- **[Semantic Layer](semantic_layer.md)**: Configure semantic adapters such as MetricFlow
- **[BI Platforms](bi_platforms.md)**: Configure Superset or Grafana access
- **[Schedulers](schedulers.md)**: Configure Airflow scheduler services
- **[Database Adapters](../adapters/db_adapters.md)**: Install additional database connectors
- **[Workflow Definitions](workflow.md)**: Define custom execution patterns
- **[Node Configuration](nodes.md)**: Customize individual node behavior
- **[Storage Settings](storage.md)**: Configure knowledge base and vector storage
- **[Benchmark Datasets](benchmark.md)**: Set up evaluation and testing
