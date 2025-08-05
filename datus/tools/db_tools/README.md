# Database Tools Module

## Overview

The `db_tools` module provides a unified interface for interacting with various SQL databases in the Datus agent system. It abstracts database-specific operations through a common connector pattern, supporting multiple database types including SQLite, MySQL, PostgreSQL, DuckDB, Snowflake, and StarRocks.

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

### Database Connectors

| File | Database Type | Features |
|------|---------------|----------|
| `sqlite_connector.py` | SQLite | File-based database, DDL extraction, table metadata |
| `mysql_connector.py` | MySQL | MySQL protocol support, schema introspection |
| `snowflake_connector.py` | Snowflake | Cloud data warehouse, Arrow format, warehouse management |
| `sqlalchemy_connector.py` | Generic SQLAlchemy | Base for SQL databases, connection pooling, error handling |
| `sqlite_connector.py` | SQLite | Lightweight file database, DDL extraction |
| `starrocks_connector.py` | StarRocks | OLAP database, materialized views, catalog support |
| `duckdb_connector.py` | DuckDB | Analytical database, read-only mode, schema introspection |

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
from datus.tools.db_tools import SnowflakeConnector
from datus.schemas.node_models import ExecuteSQLInput

# Direct connector instantiation
connector = SnowflakeConnector(
    account="<your-account>",
    user="<your_username>",
    password="<your_password>",
    warehouse="COMPUTE_WH_PARTICIPANT",
    database="ANALYTICS"
)

# Execute with different formats
result = connector.execute(
    ExecuteSQLInput(sql_query="SELECT * FROM sales WHERE date > '2024-01-01'"),
    result_format="arrow"  # Options: "csv", "arrow", "list"
)
```

### Metadata Operations

```python
from datus.tools.db_tools import MySQLConnector

# Get database schema information
conn = MySQLConnector(
    host="localhost",
    port=3306,
    user="<your_username>",
    password="<your_password>",
    database="ecommerce"
)

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

### Context Switching

```python
from datus.tools.db_tools import SnowflakeConnector

conn = SnowflakeConnector(...)

# Switch between databases and schemas. It is also possible to switch only one layer.
conn.switch_context(
    catalog_name="PRODUCTION",
    database_name="ANALYTICS", 
    schema_name="PUBLIC"
)

# Execute in new context
result = conn.execute({"sql_query": "SELECT * FROM events"})
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
  analytics:
    main_db:
      type: "snowflake"
      account: "<your-account>"
      username: "${SNOWFLAKE_USER}"
      password: "${SNOWFLAKE_PASSWORD}"
      warehouse: "COMPUTE_WH"
      database: "ANALYTICS"
    
  local_data:
    sqlite_db:
      type: "sqlite"
      uri: "sqlite:///data/local.db"
    
    duckdb_files:
      type: "duckdb"
      path_pattern: "data/*.duckdb"
```

### Environment Variables

For secure configuration, use environment variables:

```bash
# Snowflake
export SNOWFLAKE_USER=your_username
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_ACCOUNT=your_account

# MySQL
export MYSQL_USER=root
export MYSQL_PASSWORD=secret

# StarRocks
export STARROCKS_USER=admin
export STARROCKS_PASSWORD=secret
```

## How to Contribute to This Module

### Adding a New Database Connector

1. **Create the connector file** in `datus/tools/db_tools/`:

```python
from .base import BaseSqlConnector
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

2. **Register the connector** in `__init__.py`:

```python
from .newdb_connector import NewDatabaseConnector

__all__ = [
    # ... existing exports ...
    "NewDatabaseConnector",
]
```

3. **Add to DBManager** in `db_manager.py`:

```python
elif db_config.type == DBType.NEWDB:
    conn = NewDatabaseConnector(
        host=db_config.host,
        port=int(db_config.port),
        # ... other parameters
    )
```

### Example: Adding PostgreSQL Support

```python
# postgresql_connector.py
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.constants import DBType

class PostgreSQLConnector(SQLAlchemyConnector):
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        super().__init__(connection_string, dialect=DBType.POSTGRESQL)
    
    def full_name(self, catalog_name="", database_name="", schema_name="", table_name=""):
        # PostgreSQL-specific full name formatting
        if schema_name:
            return f'"{schema_name}"."{table_name}"'
        return f'"{table_name}"'
```

### Adding New Features

To extend existing connectors with new capabilities:

1. **Add method to base class** in `base.py`:

```python
@abstractmethod
def get_table_stats(self, table_name: str) -> Dict[str, Any]:
    """Get table statistics like row count, size, etc."""
    raise NotImplementedError
```

2. **Implement in specific connectors**:

```python
@override
def get_table_stats(self, table_name: str) -> Dict[str, Any]:
    query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    result = self.execute_query(query)
    return {"row_count": result.iloc[0, 0]}
```

3. **Update configuration handling** if new parameters are needed:

```python
# In db_manager.py _init_conn method
elif db_config.type == DBType.NEWDB:
    conn = NewDatabaseConnector(
        host=db_config.host,
        port=int(db_config.port),
        user=db_config.username,
        password=db_config.password,
        database=db_config.database,
        # Add new parameters here
        ssl_mode=getattr(db_config, 'ssl_mode', 'require')
    )
```

### Testing New Connectors

```python
# test_new_connector.py
from datus.tools.db_tools import NewDatabaseConnector

def test_connection():
    conn = NewDatabaseConnector(**test_config)
    assert conn.test_connection()
    
def test_query_execution():
    conn = NewDatabaseConnector(**test_config)
    result = conn.execute({"sql_query": "SELECT 1 as test"})
    assert result.success
    assert result.row_count == 1
```

### Configuration Schema Updates

When adding a new database type, update the configuration schema in `datus/configuration/agent_config.py` if necessary to include the new database-specific fields.

## Error Handling and Best Practices

- **Connection Errors**: All connectors implement proper connection lifecycle management
- **Query Errors**: Standardized error codes and messages through `DatusException`
- **Resource Management**: Use context managers (`with` statements) for automatic cleanup
- **Thread Safety**: Each connection should be used in a single thread
- **Configuration Validation**: Validate database configurations before connection

## Performance Considerations

- **Connection Pooling**: SQLAlchemy-based connectors use connection pooling
- **Streaming**: Use `execute_arrow_iterator()` for large result sets
- **Batch Processing**: Configure `batch_size` parameter for optimal memory usage
- **Read-Only Mode**: DuckDB connector uses read-only mode to prevent lock conflicts