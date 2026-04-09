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

### 2. [Semantic Models](semantic_model.md)

**Purpose**: Enrich database schemas with semantic information for better SQL generation.

- **Stores**: Table structures, dimensions, measures, entity relationships
- **Capabilities**: Schema linking, column usage patterns, foreign key discovery
- **Use**: Accurate ad-hoc SQL generation, smart filtering, proper JOIN construction

### 3. [Business Metrics](metrics.md)

**Purpose**: Manage and query standardized business KPIs.

- **Stores**: Metric definitions, subject tree categorization
- **Capabilities**: Direct metric queries via MetricFlow, metrics-first strategy
- **Use**: Consistent reporting, eliminate duplicate SQL, standardized definitions

### 4. [Reference SQL](reference_sql.md)

**Purpose**: Capture, analyze, and make searchable SQL expertise.

- **Stores**: Historical queries, LLM summaries, query patterns, best practices
- **Capabilities**: Find queries by intent, get similar queries, learn patterns
- **Use**: Knowledge sharing, optimization through examples, team onboarding

### 5. [Reference Template](reference_template.md)

**Purpose**: Manage parameterized SQL templates for stable, repeatable query generation.

- **Stores**: Jinja2 templates, parameter definitions, LLM summaries, subject tree classification
- **Capabilities**: Search templates by intent, retrieve with parameter metadata, server-side rendering
- **Use**: Stable SQL output for production scenarios, parameterized report queries, template-based SQL generation

### 6. [External Knowledge](ext_knowledge.md)

**Purpose**: Process and index domain-specific business knowledge for intelligent search.

- **Stores**: Business terminology, rules, concepts, hierarchical categorization
- **Capabilities**: Semantic search for business terms, context enrichment, term resolution
- **Use**: Agent context enhancement, terminology standardization, knowledge onboarding

### 7. [Platform Documentation](platform_doc.md)

**Purpose**: Provide authoritative platform documentation for SQL generation and validation.

- **Stores**: Official documentation chunks per platform and version
- **Capabilities**: Navigation browsing, document retrieval, semantic search
- **Use**: Verify platform-specific syntax and features before writing SQL


## Storage Backends

All knowledge base components rely on a dual-track storage architecture:

- **Vector Database**: Stores embedding vectors, powering semantic search (schema linking, document search, etc.)
- **Relational Database (RDB)**: Stores structured metadata (task, feedback, success story, etc.)

Datus Agent supports pluggable storage backends via a Registry + entry-point mechanism — switch backends without modifying business code.

### Default: LanceDB + SQLite
- Zero configuration, works out of the box
- Data stored under `data/datus_db_<namespace>/`
- Ideal for development and single-machine deployment

### PostgreSQL (pgvector)
- Production-grade backend, automatically registered after installing `datus-storage-postgresql`
- Vector: pgvector extension provides vector search
- RDB: Native PostgreSQL relational storage
- Database isolation via PostgreSQL schemas

### Database Isolation
- Each namespace is stored independently with no cross-contamination
  - LanceDB: One directory per namespace
  - PostgreSQL: One schema per namespace

> For detailed configuration, see [Storage Configuration](../configuration/storage.md#storage-backends).

## Key Features

- **Unified Search**: Single interface across all knowledge domains
- **Semantic Search**: Find by meaning using vector embeddings
- **Intelligent Classification**: Automatic categorization and organization
- **Scalable**: Lazy loading, batch processing, incremental updates
