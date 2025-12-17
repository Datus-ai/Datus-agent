# Reference SQL Storage Module

This module handles the storage, processing, and analysis of SQL query history files. It provides functionality to extract SQL statements from files, analyze them with LLM, and store them in a searchable knowledge base.


## Data Flow

```
SQL Files → File Processor → LLM Analysis → Storage
     ↓              ↓             ↓           ↓
Comment-SQL     Validation    Metadata    Vector DB
  Pairs         + Cleaning   Extraction   + Search
```

### Processing Pipeline

1. **File Processing**: Extract comment-SQL pairs from `.sql` files
2. **Validation**: Validate SQL syntax using multiple dialects
3. **LLM Analysis**: Extract business metadata using language models
4. **Storage**: Store enriched data in LanceDB for vector search
5. **Indexing**: Create search indices for efficient retrieval

## Configuration

### Build Modes

- **`overwrite`**: Replace all existing data
- **`incremental`**: Only process new items (based on SQL+comment hash)

### Performance Tuning

- **`pool_size`**: Number of parallel threads for LLM analysis (default: 4)
- **Batch processing**: Items are processed in batches to optimize LLM API usage

## Data Schema

### Input Format (from SQL files)
```python
{
    "sql": "SELECT * FROM users WHERE active = 1",
    "comment": "Query active users for analysis",
    "filepath": "/path/to/source.sql"
}
```

### Output Format (after LLM analysis)
```python
{
    "id": "md5_hash_of_sql_and_comment",
    "name": "active_users",
    "sql": "SELECT * FROM users WHERE active = 1",
    "comment": "Query active users for analysis",
    "filepath": "/path/to/source.sql",
    "summary": "Retrieve all active user records for business analysis",
    "subject_path": ["user_analysis", "user_behavior", "activity_tracking"],
    "tags": "users,active,analysis"
}
```

## Usage

```bash
# Basic usage - initialize reference SQL with overwrite mode
python -m datus.main bootstrap-kb \
  --namespace your_namespace \
  --components reference_sql \
  --sql_dir /path/to/sql/directory \
  --kb_update_strategy overwrite

# Incremental update - only process new SQL files
python -m datus.main bootstrap-kb \
  --namespace your_namespace \
  --components reference_sql \
  --sql_dir /path/to/sql/directory \
  --kb_update_strategy incremental

# High performance - use 8 parallel threads for LLM analysis
python -m datus.main bootstrap-kb \
  --namespace your_namespace \
  --components reference_sql \
  --sql_dir /path/to/sql/directory \
  --kb_update_strategy overwrite \
  --pool_size 8

# Debug mode with detailed logging
python -m datus.main bootstrap-kb \
  --namespace your_namespace \
  --components reference_sql \
  --sql_dir /path/to/sql/directory \
  --kb_update_strategy overwrite \
  --debug

# Validate-only mode - process and validate SQL files without LLM analysis or storage
python -m datus.main bootstrap-kb \
  --namespace your_namespace \
  --components reference_sql \
  --sql_dir /path/to/sql/directory \
  --validate-only
```
