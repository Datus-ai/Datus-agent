# Service Configuration

Configure database connections and services for your data sources.

## Overview

The service configuration in Datus Agent organizes database connections under `service.databases` in `agent.yml`. Each database is an independent entry with its own connection parameters.

Key features:

- **Universal Connectivity**: Support for cloud data warehouses (Snowflake, StarRocks), local databases (SQLite, DuckDB), and more
- **Credential Security**: Environment variable-based credential management (`${ENV_VAR}` syntax)
- **Default Database**: Mark one database as `default: true` for auto-selection
- **Plugin Adapters**: Install additional database adapters via `datus-agent configure`
- **Dynamic Discovery**: Glob pattern-based database discovery for multiple database files

> **Migration note**: The old `namespace:` config format is auto-migrated to `service.databases` at runtime. You can also run `python -m datus.configuration.config_migrator --config conf/agent.yml` to migrate offline.

## Configuration Structure

Databases are configured under `agent.service.databases`. Each entry is an independent database connection:

```yaml
agent:
  service:
    databases:
      my_snowflake:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        default: true

      my_duckdb:
        type: duckdb
        uri: ./data/analytics.duckdb

    bi_tools: {}       # Future: BI tool connections
    schedulers: {}     # Future: Scheduler connections
```

## Supported Database Types

### Snowflake
```yaml
my_snowflake:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  password: ${SNOWFLAKE_PASSWORD}
  database: ${SNOWFLAKE_DATABASE}    # Optional
  schema: ${SNOWFLAKE_SCHEMA}        # Optional
  warehouse: ${SNOWFLAKE_WAREHOUSE}  # Optional
  default: true                      # Optional: mark as default
```

### StarRocks
```yaml
my_starrocks:
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
my_sqlite:
  type: sqlite
  uri: sqlite:////Users/xxx/data/orders.db
```

### DuckDB
```yaml
my_duckdb:
  type: duckdb
  uri: ./data/analytics.duckdb
```

### Path Pattern (Multiple Files)

Use glob patterns to auto-discover database files:

```yaml
bird_benchmark:
  type: sqlite
  path_pattern: benchmark/bird/dev_20240627/dev_databases/**/*.sqlite
```

Supported patterns: `*.sqlite`, `**/*.sqlite`, `data/2024/*.db`

## Configuration Parameters

### Common Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `type` | Yes | Database type: `sqlite`, `duckdb`, `snowflake`, `starrocks`, `mysql`, `postgresql` |
| `default` | No | Set to `true` to mark as default database |
| `uri` | For file DBs | Connection URI for SQLite/DuckDB |
| `host` | For server DBs | Database server hostname |
| `port` | For server DBs | Database server port |
| `username` | For server DBs | Database username |
| `password` | For server DBs | Database password |
| `database` | No | Database/schema name |

### Database-Specific Parameters

- **Snowflake**: `account`, `warehouse`, `role`, `schema`
- **StarRocks**: `catalog`
- **SQLite/DuckDB**: `path_pattern` for glob-based discovery

## Managing Databases

### Interactive Configuration

Use `datus-agent configure` to add, delete, or manage databases interactively:

```bash
datus-agent configure
```

This shows your current models and databases, then offers a menu:

```
Current Databases:
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Name         ┃ Type      ┃ Connection              ┃ Default ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ demo         │ duckdb    │ ./demo.duckdb           │ *       │
│ prod_sf      │ snowflake │ account=my_account      │         │
└──────────────┴───────────┴─────────────────────────┴─────────┘

What would you like to do?
  → [add_database] Add a database
    [delete_database] Delete a database
    [done] Done
```

Database adapter plugins are auto-installed when you select an uninstalled type (e.g., snowflake, mysql).

### CLI Commands

```bash
# List all databases
datus-agent service list

# Add a database interactively
datus-agent service add

# Delete a database interactively
datus-agent service delete
```

### Specify a Custom Config

```bash
datus-agent service list --config /path/to/agent.yml
datus-agent configure --config /path/to/agent.yml
```

## Default Database Selection

When running CLI commands, specify which database to use:

```bash
datus-cli --database my_duckdb
datus-agent run --database my_snowflake --task "..." --task_db_name ANALYTICS
```

If `--database` is not specified:
1. If a database has `default: true` → auto-selected
2. If only one database configured → auto-selected
3. If multiple without default → shows available list

## Security Considerations

### Credential Management
```yaml
# Recommended: Using environment variables
username: ${DB_USERNAME}
password: ${DB_PASSWORD}

# Avoid: Hardcoded credentials
username: "actual_username"
password: "actual_password"
```

## See Also

- [Database Adapters](../adapters/db_adapters.md) - Install plugin adapters for MySQL, Snowflake, StarRocks, and more
- [CLI Commands](../cli-commands.md) - Full CLI reference including configure, init, and service commands
