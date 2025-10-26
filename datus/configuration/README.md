# Datus Configuration Guide

This directory contains the configuration management system for Datus, handling all aspects of agent behavior, LLM
models, database connections, and workflow orchestration.

## Overview

The configuration system is built on a YAML-based structure that allows flexible configuration of:

- **LLM Models**: Multiple providers (OpenAI, Anthropic, Google, DeepSeek)
- **Database Connections**: Support for SQLite, MySQL, DuckDB, Snowflake, StarRocks
- **Workflow Plans**: Define execution pipelines with parallel processing
- **Storage**: Vector database configuration for RAG capabilities
- **Benchmarks**: Support for bird_dev, spider2, and semantic layer benchmarks

## Files Structure

- **`agent_config.py`** - Core configuration class with dataclasses for type-safe configuration
- **`agent_config_loader.py`** - YAML configuration loading and environment variable resolution
- **`node_type.py`** - Enumeration of available node types for workflow orchestration
- **`README.md`** - This documentation file

## Configuration File Structure

Datus uses a hierarchical YAML configuration file (`agent.yml`) with the following main sections:

### 1. Agent Configuration

```yaml
agent:
  target: anthropic  # Default LLM provider
  models: # Define available models
    openai:
      type: openai
      vendor: openai
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
      model: gpt-4-turbo
```

### 2. Node Configuration

Configure specific behaviors for each workflow node:

```yaml
  nodes:
    schema_linking:
      model: openai
      matching_rate: fast
      prompt_version: "1.0"
    generate_sql:
      model: deepseek_v3
      max_context_length: 8000
      max_value_length: 500
```

### 3. Database Configuration

Configure database connections by namespace:

```yaml
  namespace:
    snowflake:
      type: snowflake
      account: ${SNOWFLAKE_ACCOUNT}
      username: ${SNOWFLAKE_USER}
      password: ${SNOWFLAKE_PASSWORD}

    local_sqlite:
      type: sqlite
      dbs: # Multi-database
        - name: mydb1
          uri: sqlite:////path/to/mybd1.db # absolute path
        - name: mydb2
          uri: sqlite:///path/to/mybd2.db # relative path
    bird_sqlite:
      path_pattern: ~/benchmark/bird/dev_20240627/dev_databases/**/*.sqlite # fuzzy matching, just support glob pattern
```

### 4. Storage Configuration

Configure vector storage for RAG functionality:

```yaml
  storage:
    base_path: data # Path to store vector data
    embedding_device_type: cpu # embedding use CPU or GPU. It will be specified as cpu, otherwise, it will be automatically selected based on the current machine
    database:
      registry_name: openai # the library name of embedding function，support for openai, sentence-transformers.
      model_name: text-embedding-v3 # the model name 
      dim_size: 1024 # the dimension size of embedding function
      batch_size: 10 # Number of batches of embedding
      target_model: openai # the openai config
```

### 5. Workflow Plans

Define execution workflows:

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

## Configuration Classes

### ModelConfig

Configuration for LLM models with support for:

- **OpenAI**: GPT-3.5, GPT-4, GPT-4-turbo ...
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 4 Sonnet ...
- **Google**: Gemini 2.0 Flash ...
- **DeepSeek**: DeepSeek-chat, deepseek-reasoner (AliCloud Support deepseek-r1 deepseek-v3 ...)...

### DbConfig

Database connection configuration supporting:

- **SQLite**: Single file or multiple databases
- **DuckDB**: Local or remote DuckDB instances
- **MySQL**: MySQL and MariaDB
- **Snowflake**: Cloud data warehouse
- **StarRocks**: Real-time analytics database
- **PostgreSQL**: Standard PostgreSQL connections(Not supported yet)

### Workflow Configuration

Flexible workflow orchestration with:

- **Fixed**: Performed according to standard processes
- **Reflection**: After generating SQL and execute, reflected by LLMs
- **Metrics**: Use of metrics in workflow
- **Reasoning**: Generate SQL via MCP, execute it, and return the execution result.
- **Customizable**: Develop your own nodes and customize your own processes based on requirements

## Environment Variables

Use environment variables for sensitive configuration:

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# Database Credentials
export SNOWFLAKE_ACCOUNT="your-account"
export SNOWFLAKE_USER="your-user"
export SNOWFLAKE_PASSWORD="your-password"
export STARROCKS_HOST="localhost"
export STARROCKS_PORT="9030"
export STARROCKS_USER="root"
export STARROCKS_PASSWORD=""
```

## Adding New Configuration

### 1. Adding New LLM Provider

Add to the `models` section in agent.yml:

```yaml
models:
  my_provider:
    type: openai  # Use openai for OpenAI-compatible APIs
    vendor: custom
    base_url: https://api.my-provider.com/v1
    api_key: ${MY_PROVIDER_API_KEY}
    model: my-model-name
```

### 2. Adding New Database Configuration

Add to the `namespace` section:

```yaml
namespace:
  my_database:
    type: postgres
    host: ${POSTGRES_HOST}
    port: ${POSTGRES_PORT}
    username: ${POSTGRES_USER}
    password: ${POSTGRES_PASSWORD}
    database: ${POSTGRES_DB}
```

### 3. Creating Custom Workflow

Define new workflow plans in the `workflow` section:

```yaml
workflow:
  my_custom_plan:
    - schema_linking
    - generate_sql
    - execute_sql
    - custom_node  # Add your custom node type
    - output
```

### 4. Adding New Node Configuration

Extend node-specific configurations:

```yaml
nodes:
  my_custom_node:
    model: deepseek_v3
    prompt_version: "1.0"
    max_table_schemas_length: 10000
    max_data_details_length: 10000
    max_context_length: 80000
```

## Configuration Examples

### Basic SQLite Setup

```yaml
agent:
  target: openai
  models:
    openai:
      type: openai
      api_key: ${OPENAI_API_KEY}
      model: gpt-4-turbo

  namespace:
    my_data:
      type: sqlite
      uri: sqlite:///my_database.db

  storage:
    base_path: ./data
    # embedding_device_type: cpu
```

### Multi-Database Setup

```yaml
agent:
  target: anthropic
  models:
    anthropic:
      type: claude
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-3-sonnet-20241022

  namespace:
    production:
      type: postgresql
      host: ${POSTGRES_HOST}
      port: 5432
      username: ${POSTGRES_USER}
      password: ${POSTGRES_PASSWORD}
      database: production

    analytics:
      type: duckdb
      dbs:
        - name: sales
          uri: duckdb:///sales.duckdb
        - name: inventory
          uri: duckdb:///inventory.duckdb
```

### Cloud Database Setup

```yaml
agent:
  target: openai
  models:
    openai:
      type: openai
      api_key: ${OPENAI_API_KEY}
      model: gpt-4-turbo

  namespace:
    snowflake_prod:
      type: snowflake
      account: ${SNOWFLAKE_ACCOUNT}
      username: ${SNOWFLAKE_USER}
      password: ${SNOWFLAKE_PASSWORD}
      database: PRODUCTION
      schema: PUBLIC
      warehouse: COMPUTE_WH
```

## Benchmark Configuration

### BIRD-dev Setup

```yaml
agent:
  benchmark:
    bird_dev:
      benchmark_path: benchmark/bird/dev_20240627 
```

### Spider2 Setup

```yaml
agent:
  benchmark:
    spider2:
      benchmark_path: benchmark/spider2/spider2-snow
```

## Storage Configuration

### Local CPU-based Embedding

```yaml
storage:
  base_path: ./data
  embedding_device_type: cpu
  database:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
  document:
    model_name: all-MiniLM-L6-v2
    dim_size: 384
```

### Cloud-based Embedding

```yaml
storage:
  base_path: ./data
  embedding_device_type: cuda
  database:
    registry_name: openai # the library name of embedding function，support for openai, sentence-transformers.
    model_name: text-embedding-v3 # the model name 
    dim_size: 1024 # the dimension size of embedding function
    batch_size: 10 # Number of batches of embedding
    target_model: openai # the openai config
```

## Validation

The configuration system validates:

- Required fields are present
- Environment variables are resolved
- Database connections are successful
- Model configurations are valid
- Workflow plans are properly defined

## Contributing

When adding new configuration options:

1. Update the corresponding dataclass in `agent_config.py`
2. Add validation logic if needed
3. Update this documentation
4. Provide examples in the examples section(agent.yml.example)
5. Ensure backward compatibility

## Support

For configuration-related issues:

1. Check environment variables are properly set
2. Validate YAML syntax using online validators
3. Test database connections independently
4. Review error messages in logs
5. Refer to the troubleshooting section above