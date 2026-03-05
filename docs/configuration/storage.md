# Storage

Configure storage settings for Datus Agent's embedding models and vector databases. The storage configuration manages how metadata, documents, and metrics are embedded and stored for efficient retrieval during schema linking and knowledge search.

## Storage Configuration Structure

The storage configuration defines the base path for vector databases and embedding models for different data types:

```yaml
storage:
  base_path: data                    # RAG storage base path
  embedding_device_type: cpu         # Device type for embedding models
  
  # Database metadata and sample data embedding
  database:
    registry_name: openai            # Embedding provider
    model_name: text-embedding-v3-small
    dim_size: 1024
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

### Storage Path
```yaml
storage:
  base_path: data  # Base directory for vector storage
```

The final data paths will be:

- `data/datus_db_<namespace_name>` for each configured namespace
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
  model_name: text-embedding-v3-small
  dim_size: 1024
  batch_size: 10
  target_model: openai               # Reference to agent.models
```

**Configuration Parameters:**

- **`registry_name`**: Embedding provider type (`openai` or `sentence-transformers`)
- **`model_name`**: Specific embedding model to use
- **`dim_size`**: Output embedding dimension size
- **`batch_size`**: Number of texts to process in each batch
- **`target_model`**: LLM model key from [`agent.models`](agent.md#models-configuration) (for OpenAI embeddings)

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
  model_name: text-embedding-v3-small    # or text-embedding-v3-large
  dim_size: 1536                         # 1536 for v3-small, 3072 for v3-large
  batch_size: 10                         # Adjust based on rate limits
  target_model: openai                   # Must reference valid model in agent.models
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
      model_name: text-embedding-v3-large   # Highest quality
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
        model_name: text-embedding-v3-small
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
        model_name: text-embedding-v3-large
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
  duckdb:                              # Namespace reference
    domain: sale                       # Business domain
    layer1: layer1                     # Metric layer classification
    layer2: layer2                     # Sub-layer classification
    ext_knowledge: ""                  # Extended knowledge base path

storage:
  metric:
    model_name: all-MiniLM-L6-v2       # Model for embedding metrics
    dim_size: 384
```
