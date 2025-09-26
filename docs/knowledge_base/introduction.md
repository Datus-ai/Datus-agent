# Knowledge Base Introduction

The Datus Agent Knowledge Base is a multi-modal intelligence system that transforms scattered data assets into a unified, searchable repository. Think of it as "Google for your data" with deep understanding of SQL, business metrics, and data relationships.

## Core Purpose

- **Data Discovery**: Find relevant tables, columns, and patterns
- **Query Intelligence**: Understand business intent and generate SQL
- **Knowledge Preservation**: Capture and organize SQL expertise
- **Semantic Search**: Find information by meaning, not keywords

## Core Components

### 1. [Schema Metadata](metadata.md)
**Purpose**: Understand database structure and provide intelligent table recommendations.

- **Stores**: Table definitions, column info, sample data, statistics
- **Capabilities**: Find tables by business meaning, get table structures, semantic search
- **Use**: Automatic table selection, data discovery, schema understanding

### 2. [Business Metrics](metrics.md)
**Purpose**: Manage and discover business metrics and KPIs.

- **Stores**: Semantic models, business metrics, hierarchical categorization
- **Capabilities**: Find metrics by concept, get calculations, discover related metrics
- **Use**: Standardized definitions, quick lookup, consistent reporting

### 3. [SQL History](sql_history.md)
**Purpose**: Capture, analyze, and make searchable SQL expertise.

- **Stores**: Historical queries, LLM summaries, query patterns, best practices
- **Capabilities**: Find queries by intent, get similar queries, learn patterns
- **Use**: Knowledge sharing, optimization through examples, team onboarding

## How It Works

1. **Data Ingestion**: Initialize components via `datus bootstrap-kb` command with various data sources
2. **Processing Pipeline**: Raw data → Parsing → LLM Analysis → Vector Embedding → Indexing
3. **Search**: Multi-modal search combining vector similarity, full-text search, and filtering for optimal results
4. **Storage**: Built on LanceDB vector database with optimized indexing and scalable architecture

## Key Features

- **Unified Search**: Single interface across all knowledge domains
- **Semantic Search**: Find by meaning using vector embeddings
- **Intelligent Classification**: Automatic categorization and organization
- **Scalable**: Lazy loading, batch processing, incremental updates
