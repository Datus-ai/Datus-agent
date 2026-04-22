# Storage

Configure storage settings for Datus Agent's embedding models and vector databases. The storage configuration manages how metadata, documents, and metrics are embedded and stored for efficient retrieval during schema linking and knowledge search.

## Storage Configuration Structure

The storage configuration defines the base path for vector databases and embedding models for different data types:

```yaml
storage:
  embedding_device_type: cpu         # Device type for embedding models

  # Database metadata and sample data embedding
  database:
    registry_name: openai            # Embedding provider
    model_name: text-embedding-3-small
    dim_size: 1536
    batch_size: 10
    target_model: openai

  # Document embedding configuration
  document:
    model_name: all-MiniLM-L6-v2     # Local embedding model
    dim_size: 384

  # Metrics embedding configuration
  metric:
    model_name: all-MiniLM-L6-v2     # Local embedding model
    dim_size: 384
```

## Base Configuration

### Device Type
```yaml
storage:
  embedding_device_type: cpu  # Or cude/mps
```

The final data paths will be:

- `data/datus_db_<datasource_name>` for each configured datasource
- Example: `data/datus_db_snowflake`, `data/datus_db_local_sqlite`

### Device Configuration
```yaml
storage:
  embedding_device_type: cpu  # cpu, cuda, mps, auto
```

**Device Options:**

- **`cpu`**: Force CPU usage for embedding models
- **`cuda`**: Use NVIDIA GPU (if available)
- **`mps`**: Use Apple Metal Performance Shaders (Apple Silicon)
- **`auto`**: Automatically select best available device

## Embedding Model Configuration

### Database Embeddings

For table metadata, schema information, and sample data:

```yaml
database:
  registry_name: openai              # openai or sentence-transformers
  model_name: text-embedding-3-small
  dim_size: 1536
  batch_size: 10
  target_model: openai               # Reference to agent.models
```

**Configuration Parameters:**

- **`registry_name`**: Embedding provider type (`openai` or `sentence-transformers`)
- **`model_name`**: Specific embedding model to use
- **`dim_size`**: Output embedding dimension size
- **`batch_size`**: Number of texts to process in each batch
- **`target_model`**: LLM model key from [`models`](agent.md#models-configuration) (for OpenAI embeddings)

### Document Embeddings

For knowledge base documents and extended documentation:

```yaml
document:
  model_name: all-MiniLM-L6-v2       # Lightweight model (~100MB)
  dim_size: 384                      # Smaller dimension for efficiency
```

### Metric Embeddings

For business metrics and KPI definitions:

```yaml
metric:
  model_name: all-MiniLM-L6-v2       # Consistent with document embeddings
  dim_size: 384                      # Matching dimension size
```

## Embedding Provider Options

### OpenAI Embeddings (Cloud)

For high-quality embeddings with cloud API:

```yaml
database:
  registry_name: openai
  model_name: text-embedding-3-small    # or text-embedding-3-large
  dim_size: 1536                         # 1536 for 3-small, 3072 for 3-large
  batch_size: 10                         # Adjust based on rate limits
  target_model: openai                   # Must reference valid model in models configuration
```

!!! tip "Environment Variables"
    Ensure your OpenAI API key is configured:

    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

### Sentence Transformers (Local)

For local embedding models without external API calls:

```yaml
database:
  registry_name: sentence-transformers   # Default local provider
  model_name: all-MiniLM-L6-v2          # Lightweight option
  dim_size: 384
```

!!! info "Alternative Local Models"
    Consider these high-quality alternatives:

    - **`intfloat/multilingual-e5-large-instruct`**: 1.2GB, 1024 dimensions, multilingual
    - **`BAAI/bge-large-en-v1.5`**: 1.2GB, 1024 dimensions (English optimized)
    - **`BAAI/bge-large-zh-v1.5`**: 1.2GB, 1024 dimensions (Chinese optimized)

## Model Selection Guidelines

=== "Performance-Focused (Small Models)"

    Optimized for speed and minimal resource usage:

    ```yaml
    document:
      model_name: all-MiniLM-L6-v2         # ~100MB, 384 dimensions
      dim_size: 384

    metric:
      model_name: intfloat/multilingual-e5-small  # ~460MB, 384 dimensions
      dim_size: 384
    ```

=== "Balanced Performance and Quality"

    Good balance of speed and retrieval quality:

    ```yaml
    database:
      model_name: intfloat/multilingual-e5-large-instruct  # ~1.2GB, 1024 dimensions
      dim_size: 1024
    ```

=== "Quality-Focused (Large Models)"

    Maximum retrieval quality with cloud-based embeddings:

    ```yaml
    database:
      registry_name: openai
      model_name: text-embedding-3-large   # Highest quality
      dim_size: 3072
      target_model: openai
    ```

## Complete Configuration Examples

=== "High-Performance Local Setup"

    Fast local embeddings optimized for development:

    ```yaml
    storage:
      base_path: data
      embedding_device_type: auto          # Use best available device

      # Fast local embeddings for all data types
      database:
        registry_name: sentence-transformers
        model_name: all-MiniLM-L6-v2
        dim_size: 384

      document:
        model_name: all-MiniLM-L6-v2
        dim_size: 384

      metric:
        model_name: all-MiniLM-L6-v2
        dim_size: 384
    ```

    !!! success "Best For"
        - Development environments
        - Resource-constrained systems
        - Offline deployment requirements

=== "Hybrid Cloud-Local Setup"

    Combines cloud quality for critical data with local efficiency:

    ```yaml
    storage:
      base_path: data
      embedding_device_type: cpu

      # High-quality cloud embeddings for database metadata
      database:
        registry_name: openai
        model_name: text-embedding-3-small
        dim_size: 1536
        batch_size: 10
        target_model: openai

      # Local embeddings for documents and metrics
      document:
        model_name: intfloat/multilingual-e5-large-instruct
        dim_size: 1024

      metric:
        model_name: intfloat/multilingual-e5-large-instruct
        dim_size: 1024
    ```

    !!! success "Best For"
        - Production environments with mixed requirements
        - Cost-conscious deployments
        - Balancing quality and performance

=== "Enterprise Quality Setup"

    Maximum quality embeddings for production systems:

    ```yaml
    storage:
      base_path: /opt/datus/embeddings
      embedding_device_type: cuda          # Use GPU acceleration

      # High-quality embeddings across all data types
      database:
        registry_name: openai
        model_name: text-embedding-3-large
        dim_size: 3072
        batch_size: 5                      # Smaller batches for large model
        target_model: openai

      document:
        model_name: BAAI/bge-large-en-v1.5
        dim_size: 1024

      metric:
        model_name: BAAI/bge-large-en-v1.5
        dim_size: 1024
    ```

    !!! success "Best For"
        - Enterprise production environments
        - High-accuracy requirements
        - Systems with GPU acceleration

## Integration with Other Components

### Metrics Configuration

The storage configuration works with the metrics section to embed business metrics:

```yaml
metrics:
  duckdb:                              # Datasource reference
    domain: sale                       # Business domain
    layer1: layer1                     # Metric layer classification
    layer2: layer2                     # Sub-layer classification
    ext_knowledge: ""                  # Extended knowledge base path

storage:
  metric:
    model_name: all-MiniLM-L6-v2       # Model for embedding metrics
    dim_size: 384
```

## Storage Backends

Datus Agent uses a dual-track storage architecture — **Vector DB** for embeddings and **RDB** for structured metadata. Both support pluggable backends via a registry + entry-point mechanism.

> For an architectural overview, see [Knowledge Base Introduction](../knowledge_base/introduction.md#storage-backends).

### Default Backend

By default (no `rdb` / `vector` section in YAML), Datus Agent uses:

- **Vector**: LanceDB (file-based, zero-config)
- **RDB**: SQLite (file-based, zero-config)

Data is stored under `data/datus_db_<datasource>/`.

### PostgreSQL Backend

#### Prerequisites
- PostgreSQL 15+ with the `pgvector` extension enabled
- Install the adapter package:
  ```bash
  pip install datus-storage-postgresql
  # or
  uv add datus-storage-postgresql
  ```
  The entry-point registers automatically — no code changes needed.

#### Configuration
```yaml
storage:
  rdb:
    type: postgresql
    host: ${PG_HOST:-localhost}
    port: 5432
    user: ${PG_USER:-postgres}
    password: ${PG_PASSWORD}
    dbname: datus
    pool_min_size: 1
    pool_max_size: 10

  vector:
    type: postgresql
    host: ${PG_HOST:-localhost}
    port: 5432
    user: ${PG_USER:-postgres}
    password: ${PG_PASSWORD}
    dbname: datus
    pool_min_size: 1
    pool_max_size: 10
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `type` | Yes | — | `postgresql` |
| `host` | Yes | — | PostgreSQL server hostname |
| `port` | Yes | — | PostgreSQL server port |
| `user` | Yes | — | Database user |
| `password` | Yes | — | User password |
| `dbname` | Yes | — | Database name |
| `pool_min_size` | No | 1 | Minimum connection pool size |
| `pool_max_size` | No | 10 | Maximum connection pool size |

#### Mixed Backend

You can mix backends — for example, use PostgreSQL for RDB and LanceDB for vectors:

```yaml
storage:
  rdb:
    type: postgresql
    host: localhost
    port: 5432
    user: postgres
    password: ${PG_PASSWORD}
    dbname: datus

  vector:
    type: lance    # default, no extra install needed
```
