# Database Tools Module

## Overview

The `db_tools` module provides a unified interface for interacting with SQL databases in the Datus agent system. It abstracts database-specific operations through a common connector pattern, with built-in support for SQLite and DuckDB. Additional database adapters (MySQL, PostgreSQL, Snowflake, StarRocks, etc.) are available as separate adapter packages.

**Key Problems Solved:**
- Provides a consistent API for different SQL database types
- Handles connection management, query execution, and result formatting
- Supports multiple output formats (CSV, Arrow, JSON)
- Manages database metadata retrieval (schemas, tables, DDL)
- Offers context switching between catalogs, databases, and schemas

**Technologies and Patterns Used:**
- **Abstract Base Classes**: `BaseSqlConnector` defines the common interface
- **Factory Pattern**: `DBManager` creates appropriate connectors based on configuration
- **SQLAlchemy**: Provides ORM capabilities and database abstraction
- **Apache Arrow**: Efficient columnar data format for large result sets
- **Connection Pooling**: Automatic connection lifecycle management(Both alchemy and snowflake_connector have connection pooling capabilities)
- **Context Managers**: Safe resource management with Single Instance or `with` statements

## File Structure & Capabilities

### Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `base.py` | Abstract base class defining the connector interface | `execute()`, `get_schema()`, `get_tables()` |
| `db_manager.py` | Central manager for database connections | `get_conn()`, `get_connections()`, connection lifecycle |
| `db_tool.py` | Tool wrapper for database operations | `execute()` method for SQL execution |

### Built-in Database Adapters

| File | Database Type | Features |
|------|---------------|----------|
| `sqlite_connector.py` | SQLite | Lightweight file database, DDL extraction, table metadata |
| `duckdb_connector.py` | DuckDB | Analytical database, read-only mode, schema introspection |

**Note:** Other database adapters (MySQL, Snowflake, StarRocks, etc.) are available as separate adapter packages. Install them separately as needed, e.g., `pip install datus-mysql`.

### Entry Points and Public Interfaces

**Primary Entry Points:**
- `DBManager`: Main interface for database connection management
- `DBTool`: Tool wrapper for SQL execution in agent workflows
- Individual connector classes for direct database access

**Public Interfaces:**
- `BaseSqlConnector`: All database operations through consistent API
- `ExecuteSQLInput/ExecuteSQLResult`: Standardized input/output schemas
- `DbConfig`: Configuration structure for database connections

## How to Use This Module

### Basic Usage Example

```python
from datus.tools.db_tools import DBManager, SQLiteConnector
from datus.configuration.agent_config import DbConfig

# Configuration-based usage
configs = {
    "analytics": {
        "main_db": DbConfig(
            type="sqlite",
            uri="sqlite:///data/analytics.db"
        )
    }
}

# Initialize DBManager
with DBManager(configs) as manager:
    # Get specific connection
    conn = manager.get_conn("analytics", "sqlite", "main_db")

    # Execute query
    result = conn.execute({"sql_query": "SELECT * FROM users LIMIT 10"})
    print(f"Rows returned: {result.row_count}")
    print(f"Data: {result.sql_return}")
```

### Direct Connector Usage

```python
from datus.tools.db_tools import SQLiteConnector
from datus.tools.db_tools.config import SQLiteConfig
from datus.schemas.node_models import ExecuteSQLInput

# Create configuration
config = SQLiteConfig(
    db_path="/path/to/database.db"
)

# Direct connector instantiation
connector = SQLiteConnector(config=config)

# Execute with different formats
result = connector.execute(
    ExecuteSQLInput(sql_query="SELECT * FROM sales WHERE date > '2024-01-01'"),
    result_format="csv"  # Options: "csv", "arrow", "list"
)
```

### Metadata Operations

```python
from datus.tools.db_tools import SQLiteConnector
from datus.tools.db_tools.config import SQLiteConfig

# Create configuration
config = SQLiteConfig(
    db_path="/path/to/ecommerce.db"
)

# Get database schema information
conn = SQLiteConnector(config=config)

# List all tables
tables = conn.get_tables()
print("Available tables:", tables)

# Get table DDL
tables_with_ddl = conn.get_tables_with_ddl()
for table_info in tables_with_ddl:
    print(f"Table: {table_info['table_name']}")
    print(f"DDL: {table_info['definition'][:100]}...")

# Get sample data
samples = conn.get_sample_rows(tables=["users", "orders"], top_n=3)
for sample in samples:
    print(f"Sample from {sample['table_name']}:")
    print(sample['sample_rows'])
```

## Environment Variables and Configuration

### Database Configuration Structure

The module uses `DbConfig` objects with the following fields:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `type` | string | Database type (sqlite, mysql, postgresql, snowflake, etc.) | Yes |
| `uri` | string | Full connection URI (alternative to individual fields) | No |
| `host` | string | Database host | For network databases |
| `port` | int | Database port | For network databases |
| `username` | string | Database username | For network databases |
| `password` | string | Database password | For network databases |
| `database` | string | Default database/schema | Yes |
| `warehouse` | string | Snowflake warehouse | For Snowflake |
| `catalog` | string | StarRocks catalog | For StarRocks |
| `path_pattern` | string | Glob pattern for SQLite/DuckDB files | For file databases |

### Example Configuration

```yaml
# agent.yml
namespace:
  local_data:
    sqlite_db:
      type: "sqlite"
      uri: "sqlite:///data/local.db"

    duckdb_files:
      type: "duckdb"
      path_pattern: "data/*.duckdb"
```

## How to Contribute to This Module

### Adding a New Database Adapter

New database adapters should be created as separate adapter packages (e.g., `datus-newdb`). This keeps the core package lightweight while allowing users to install only the database adapters they need.

See the [Datus-adapters](https://github.com/Datus-ai/Datus-adapters) repository for existing adapter implementations and contribution guidelines.

1. **Create a new adapter package** following the structure in the Datus-adapters repository:

```
datus-newdb/
├── datus_newdb/
│   ├── __init__.py
│   ├── connector.py
│   └── README.md
├── tests/
│   └── test_connector.py
└── pyproject.toml
```

2. **Implement the connector class** in `connector.py`:

```python
from datus.tools.db_tools.base import BaseSqlConnector
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult

class NewDatabaseConnector(BaseSqlConnector):
    def __init__(self, connection_params):
        super().__init__(dialect="newdb")
        # Initialize connection

    def do_execute(self, input_params, result_format="csv"):
        # Implement query execution
        pass

    def get_tables_with_ddl(self, ...):
        # Implement DDL extraction
        pass
```

3. **Register the adapter** in the adapter's `__init__.py`:

```python
from datus.tools.db_tools import connector_registry
from .connector import NewDatabaseConnector

def register():
    """Register this adapter with Datus"""
    connector_registry.register("newdb", NewDatabaseConnector)

__all__ = ["NewDatabaseConnector", "register"]
```

4. **Add entry point** in `pyproject.toml`:

```toml
[project.entry-points."datus.adapters"]
newdb = "datus_newdb:register"
```

### Extending Built-in Adapters

To add features to the built-in SQLite and DuckDB adapters:

1. **Add method to base class** in `base.py`:

```python
@abstractmethod
def get_table_stats(self, table_name: str) -> Dict[str, Any]:
    """Get table statistics like row count, size, etc."""
    raise NotImplementedError
```

2. **Implement in specific adapter connector classes**:

```python
@override
def get_table_stats(self, table_name: str) -> Dict[str, Any]:
    query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    result = self.execute_query(query)
    return {"row_count": result.iloc[0, 0]}
```

## Error Handling and Best Practices

- **Connection Errors**: All adapters implement proper connection lifecycle management
- **Query Errors**: Standardized error codes and messages through `DatusException`
- **Resource Management**: Use context managers (`with` statements) for automatic cleanup
- **Thread Safety**: Each connection should be used in a single thread
- **Configuration Validation**: Validate database configurations before connection

## Performance Considerations

- **Connection Pooling**: SQLAlchemy-based adapters use connection pooling
- **Streaming**: Use `execute_arrow_iterator()` for large result sets
- **Batch Processing**: Configure `batch_size` parameter for optimal memory usage
- **Read-Only Mode**: DuckDB adapter uses read-only mode to prevent lock conflicts