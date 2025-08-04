# Database Tools - README

This directory contains the database connection and management infrastructure for the Datus agent. It provides a unified interface for connecting to and querying various database types.

## Architecture Overview

The database tools follow a plugin-style architecture with these key components:

- **Base Classes**: Abstract base classes defining the interface
- **Connectors**: Concrete implementations for specific database types
- **Manager**: Central orchestrator for database connections
- **Utilities**: Helper functions and utilities

## File Structure & Capabilities

### Core Files

#### `base.py` - Abstract Base Classes
**Purpose**: Defines the contract that all database connectors must implement

**Key Classes**:
- `BaseSqlConnector`: Abstract base class for all database connectors

**Core Methods**:
- `execute()`: Execute SQL queries with configurable output formats
- `execute_csv()`: Execute queries returning CSV format
- `execute_arrow()`: Execute queries returning Apache Arrow format
- `get_tables()`: List all tables in a database/schema
- `get_schemas()`: Get table schema information
- `get_sample_rows()`: Get sample data from tables
- `get_tables_with_ddl()`: Get table definitions with DDL
- `test_connection()`: Validate database connectivity
- `switch_context()`: Switch between catalogs/databases/schemas

#### `db_manager.py` - Connection Manager
**Purpose**: Central management of database connections across namespaces

**Key Classes**:
- `DBManager`: Manages all database connections
- `db_manager_instance()`: Singleton pattern for global access

**Features**:
- Multi-namespace support (isolated connection pools)
- Database type auto-detection
- Connection pooling and lifecycle management
- Configuration-based initialization
- Support for multiple databases per namespace

#### `db_tool.py` - Database Tool Integration
**Purpose**: Integration layer for database tools in Datus workflows

### Database Connectors

#### `sqlite_connector.py` - SQLite Support
**Capabilities**:
- Local SQLite database connections
- File-based database access
- Full SQL support
- Schema introspection

#### `duckdb_connector.py` - DuckDB Support
**Capabilities**:
- In-process analytical database
- CSV/Parquet file querying
- High-performance analytics
- MotherDuck cloud integration

#### `mysql_connector.py` - MySQL Support
**Capabilities**:
- MySQL/MariaDB connections
- TCP/IP socket connections
- Authentication support
- SSL/TLS encryption

#### `snowflake_connector.py` - Snowflake Support
**Capabilities**:
- Cloud data warehouse connections
- Account-based authentication
- Warehouse selection
- Schema navigation

#### `starrocks_connector.py` - StarRocks Support
**Capabilities**:
- Real-time analytics database
- Catalog/database/schema support
- MySQL protocol compatibility

#### `sqlalchemy_connector.py` - Generic SQLAlchemy Support
**Capabilities**:
- Fallback for any SQLAlchemy-supported database
- PostgreSQL, Oracle, SQL Server support
- Connection string flexibility
- Generic SQL execution

## How to Add New Database Types

### Step 1: Create a New Connector

Create a new file `{database_type}_connector.py` in the `db_tools` directory:

```python
from datus.tools.db_tools.base import BaseSqlConnector
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult

class NewDatabaseConnector(BaseSqlConnector):
    def __init__(self, connection_params: dict):
        super().__init__(dialect="new_database_type")
        # Initialize your connection
        
    def do_execute(self, input_params: ExecuteSQLInput, result_format="csv") -> ExecuteSQLResult:
        """Implement query execution logic"""
        pass
        
    def test_connection(self):
        """Implement connection testing"""
        pass
        
    # Implement other required abstract methods...
```

### Step 2: Register in DBManager

Update the `DBManager._init_conn()` method in `db_manager.py`:

```python
elif db_config.type == DBType.NEW_DATABASE:
    conn = NewDatabaseConnector(
        host=db_config.host,
        port=db_config.port,
        user=db_config.username,
        password=db_config.password,
        database=db_config.database,
        # Add any custom parameters
    )
```

### Step 3: Add Database Type Definition

Add the new database type to `datus.utils.constants.DBType`:

```python
class DBType:
    NEW_DATABASE = "new_database"
```

### Step 4: Update Configuration Schema

Ensure the configuration schema in `datus.configuration.agent_config.DbConfig` supports the new database type's parameters.

### Step 5: Add Connection URI Generation (if needed)

Update the `gen_uri()` function in `db_manager.py` if your database uses custom connection strings:

```python
elif db_config.type == DBType.NEW_DATABASE:
    return f"new_database://{username}:{password}@{host}:{port}/{database}"
```

## Configuration Examples

### SQLite Configuration
```yaml
databases:
  sqlite_namespace:
    sqlite_db:
      type: sqlite
      uri: sqlite:///path/to/database.db
```

### MySQL Configuration
```yaml
databases:
  mysql_namespace:
    production:
      type: mysql
      host: localhost
      port: 3306
      username: user
      password: pass
      database: mydb
```

### Snowflake Configuration
```yaml
databases:
  snowflake_namespace:
    analytics:
      type: snowflake
      account: myaccount
      username: user
      password: pass
      warehouse: compute_wh
      database: analytics
```

### DuckDB Configuration with Pattern
```yaml
databases:
  duckdb_namespace:
    analytics:
      type: duckdb
      path_pattern: "/data/**/*.duckdb"
```

## Usage Patterns

### Basic Query Execution
```python
from datus.tools.db_tools.db_manager import db_manager_instance

# Get database connection
manager = db_manager_instance()
conn = manager.get_conn("namespace", "mysql", "database_name")

# Execute query
result = conn.execute({"sql_query": "SELECT * FROM users LIMIT 10"})
```

### Schema Exploration
```python
# Get all tables
tables = conn.get_tables()

# Get table schema
schema = conn.get_schema(table_name="users")

# Get sample data
samples = conn.get_sample_rows(tables=["users"], top_n=5)
```

### Context Switching
```python
# Switch database/schema context
conn.switch_context(database_name="analytics", schema_name="public")
```

## Best Practices

1. **Connection Management**: Always use the `DBManager` singleton for connection management
2. **Error Handling**: Implement proper error handling in new connectors
3. **Testing**: Always implement `test_connection()` for validation
4. **Configuration**: Use the standard `DbConfig` schema for configuration
5. **Documentation**: Document any database-specific requirements or limitations

## Security Considerations

- Credentials are handled through configuration files
- Connection strings use URL encoding for special characters
- SSL/TLS support varies by database type
- Connection pooling helps prevent resource leaks

## Troubleshooting

### Common Issues

1. **Connection Timeouts**: Check network connectivity and firewall settings
2. **Authentication Errors**: Verify credentials and permissions
3. **Schema Issues**: Ensure proper catalog/database/schema context
4. **Driver Dependencies**: Install required database drivers

### Debug Tips

- Use `test_connection()` to validate configurations
- Check logs for connection errors
- Verify database-specific configuration parameters
- Use the singleton pattern for consistent connection management