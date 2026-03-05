# Namespace

Configure database namespaces and connections for different data sources

## Overview

Namespaces in Datus Agent provide a comprehensive data connectivity framework that abstracts and organizes database connections across diverse data ecosystems. Each namespace serves as a logical container that encapsulates database connection configurations, enabling seamless multi-database operations within a unified interface.

The namespace configuration system is built on a **polymorphic architecture** that supports heterogeneous resource types through a unified abstraction layer. This design pattern enables:

- **Universal Connectivity**: Support for cloud data warehouses (Snowflake, StarRocks), local databases (SQLite, DuckDB), and specialized benchmark datasets
- **Environment Isolation**: Logical separation of development, staging, and production environments
- **Credential Security**: Environment variable-based credential management with secure connection protocols
- **Dynamic Discovery**: Pattern-based database discovery for automated inclusion of multiple database files
- **Scalable Organization**: Hierarchical namespace structure that grows with organizational complexity

The namespace system operates as a **configuration-driven abstraction layer** that translates high-level business requirements into specific database connection parameters, providing developers and analysts with a consistent interface regardless of the underlying database technology.

## Namespace Structure

Namespaces are configured under the `namespace` section and contain resource connection details for different service types:

```yaml
namespace:
  # service configuration
  service:
    type: cloud_provider
    endpoint: ${SERVICE_ENDPOINT}
    access_key: ${ACCESS_KEY}
    secret_key: ${SECRET_KEY}
    region: ${SERVICE_REGION}
    
  # Local resource configuration
  local_resource:
    type: local_service
    resources:
      - name: primary
        uri: protocol://path/to/resource
      - name: secondary
        uri: protocol://path/to/backup
```

## Supported Database Types

### Snowflake
```yaml
snowflake:
  type: snowflake
  # Option 1: Using individual parameters
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  password: ${SNOWFLAKE_PASSWORD}
  database: ${SNOWFLAKE_DATABASE}    # Optional
  schema: ${SNOWFLAKE_SCHEMA}        # Optional
  warehouse: ${SNOWFLAKE_WAREHOUSE}  # Optional
```

### StarRocks
```yaml
starrocks:
  type: starrocks
  host: ${STARROCKS_HOST}
  port: ${STARROCKS_PORT}
  username: ${STARROCKS_USER}
  password: ${STARROCKS_PASSWORD}
  database: ${STARROCKS_DATABASE}
  catalog: ${STARROCKS_CATALOG}      # Optional
```

### SQLite
```yaml
# Single database configuration
local_sqlite:
  type: sqlite
  name: ssb                          # Required for SQLite
  uri: sqlite:////Users/xxx/benchmark/SSB.db

# Multiple databases configuration
local_sqlite_multi:
  type: sqlite
  dbs:
    - name: ssb
      uri: sqlite:////Users/xxx/benchmark/SSB.db
    - name: northwind
      uri: sqlite:////Users/xxx/data/northwind.db
```

### DuckDB
```yaml
# Single database configuration
local_duckdb:
  type: duckdb
  name: analytics
  uri: duckdb:////absolute/path/to/analytics.db

# Multiple databases configuration
local_duckdb_multi:
  type: duckdb
  dbs:
    - name: ssb
      uri: duckdb:////absolute/path/to/ssb.db
    - name: tpch
      uri: duckdb:///relative/path/to/tpch.duckdb  # Relative path
```


## Configuration Parameters

### Common Parameters

- **type**: Database dialect/type (required)
- **name**: Database identifier (required for SQLite and DuckDB)
- **uri**: Connection URI for local databases
- **host**: Database server hostname
- **port**: Database server port
- **username**: Database username
- **password**: Database password
- **database**: Database name

### Database-Specific Parameters

#### Snowflake Parameters
- **account**: Snowflake account identifier (top-level container)
- **warehouse**: Compute warehouse to use
- **role**: User role for permissions
- **schema**: Default schema

#### StarRocks Parameters
- **catalog**: Catalog name for multi-catalog setups
- **ssl**: Enable SSL connection

#### SQLite/DuckDB Parameters
- **path_pattern**: Glob pattern for multiple database files
- **dbs**: Array of database configurations for multi-database setup


## Complete Namespace Configuration

```yaml
namespace:
  # Production Snowflake
  production_snowflake:
    type: snowflake
    account: ${SNOWFLAKE_ACCOUNT}
    username: ${SNOWFLAKE_USER}
    password: ${SNOWFLAKE_PASSWORD}
    database: ANALYTICS
    schema: PUBLIC
    warehouse: COMPUTE_WH
    
  # Development StarRocks
  dev_starrocks:
    type: starrocks
    host: ${STARROCKS_HOST}
    port: ${STARROCKS_PORT}
    username: ${STARROCKS_USER}
    password: ${STARROCKS_PASSWORD}
    database: dev_analytics
    
  # Local SQLite for testing
  test_sqlite:
    type: sqlite
    dbs:
      - name: orders
        uri: sqlite:////Users/data/orders.db
      - name: customers
        uri: sqlite:////Users/data/customers.db
      - name: products
        uri: sqlite:////Users/data/products.db
        
  # Local DuckDB for analytics
  analytics_duckdb:
    type: duckdb
    dbs:
      - name: sales
        uri: duckdb:////opt/data/sales.db
      - name: marketing
        uri: duckdb:///data/marketing.duckdb
        
  # BIRD benchmark databases
  bird_benchmark:
    type: sqlite
    path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

## Multi-Database Configuration

### SQLite Multi-Database Setup

For SQLite and DuckDB, you can configure multiple databases within a single namespace:

```yaml
multi_sqlite:
  type: sqlite
  dbs:
    - name: sales_2023        # Each database must have a unique name
      uri: sqlite:////data/sales_2023.db
    - name: sales_2024
      uri: sqlite:////data/sales_2024.db
    - name: customer_master
      uri: sqlite:////data/customers.db
```

### Path Pattern Configuration

Use glob patterns to automatically include multiple database files:

```yaml
benchmark_dbs:
  type: sqlite
  path_pattern: benchmarks/**/*.sqlite  # Includes all .sqlite files recursively
```

**Supported patterns:**
- `*.sqlite` - All SQLite files in current directory
- `**/*.sqlite` - All SQLite files recursively
- `data/2024/*.db` - All .db files in data/2024 directory
- `benchmark/bird/**/*.sqlite` - All SQLite files under benchmark/bird

## URI Formats

### SQLite URI Format
```text
sqlite:////absolute/path/to/database.db      # Absolute path
sqlite:///relative/path/to/database.db       # Relative path
```

### DuckDB URI Format
```text
duckdb:////absolute/path/to/database.db      # Absolute path
duckdb:///relative/path/to/database.db       # Relative path
```

## Namespace Manager CLI

Datus Agent provides an interactive CLI tool for managing namespace configurations without manually editing YAML files.

### Commands

#### List Namespaces

View all configured namespaces and their connection details:

```bash
datus-agent namespace list
```

Output example:
```
Configured namespaces:

Namespace: production_snowflake
  Database: ANALYTICS
    Type: snowflake
    Account: my_account
    Warehouse: COMPUTE_WH
    Database: ANALYTICS
    Schema: PUBLIC
    Username: admin

Namespace: local_duckdb
  Database: analytics
    Type: duckdb
    URI: duckdb:////data/analytics.db
```

#### Add Namespace

Interactively add a new namespace configuration:

```bash
datus-agent namespace add
```

The command will prompt you for:

1. **Namespace name**: Unique identifier for the namespace
2. **Database type**: Choose from sqlite, duckdb, snowflake, mysql, starrocks
3. **Connection parameters**: Varies by database type

**For file-based databases (SQLite, DuckDB):**
- Connection string (file path)

**For host-based databases (MySQL, StarRocks):**
- Host
- Port
- Username
- Password
- Database name

**For Snowflake:**
- Username
- Account
- Warehouse
- Password
- Database (optional)
- Schema (optional)

After entering the configuration, the tool will:
- Test database connectivity
- Save the configuration to `conf/agent.yml` if successful

Example session:
```text
Add New Namespace
- Namespace name: my_analytics
- Database type [sqlite/duckdb/snowflake/mysql/starrocks] (duckdb): snowflake
- Username: admin
- Account: my_account
- Warehouse: COMPUTE_WH
- Password: ********
- Database (optional): ANALYTICS
- Schema (optional): PUBLIC
→ Testing database connectivity...
✔ Database connection test successful

Configuration saved to conf/agent.yml
✔ Namespace 'my_analytics' added successfully
```

#### Delete Namespace

Interactively delete an existing namespace:

```bash
datus-agent namespace delete
```

The command will:
1. Display available namespaces
2. Prompt for the namespace name to delete
3. Ask for confirmation before deletion

Example session:
```
Delete Namespace
Available namespaces:
  - production_snowflake
  - local_duckdb
  - test_sqlite
- Namespace name to delete: test_sqlite
Are you sure you want to delete namespace 'test_sqlite'? This action cannot be undone. [y/N]: y
Configuration saved to conf/agent.yml
✔ Namespace 'test_sqlite' deleted successfully
```

### Usage with Custom Config

Specify a custom configuration file:

```bash
datus-agent namespace list --config /path/to/agent.yml
datus-agent namespace add --config /path/to/agent.yml
datus-agent namespace delete --config /path/to/agent.yml
```

## Security Considerations

### Credential Management
```yaml
# Good: Using environment variables
username: ${DB_USERNAME}
password: ${DB_PASSWORD}

# Avoid: Hardcoded credentials
username: "actual_username"
password: "actual_password"
```

## See Also

- [Database Adapters](../adapters/db_adapters.md) - Install plugin adapters for MySQL, Snowflake, StarRocks, and more
