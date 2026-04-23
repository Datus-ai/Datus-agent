# Datasource Configuration

Configure database connections under `agent.services.datasources`.

## Overview

The runtime services in Datus Agent live under `agent.services` in `agent.yml`. This page focuses on database connections in `services.datasources`. Semantic adapters, BI platforms, and schedulers are documented on their sibling pages.

Key features:

- **Universal Connectivity**: Support for cloud data warehouses (Snowflake, StarRocks), local databases (SQLite, DuckDB), and more
- **Credential Security**: Environment variable-based credential management (`${ENV_VAR}` syntax)
- **Default Database**: Mark one database as `default: true` for auto-selection
- **Plugin Adapters**: Install additional database adapters via `datus-agent configure`
- **Dynamic Discovery**: Glob pattern-based database discovery for multiple database files

> **Migration note**: The old `namespace:` config format is auto-migrated to `services.datasources` at runtime. The earlier `services.databases` key has been renamed to `services.datasources`; run `python -m datus.configuration.config_migrator --config conf/agent.yml` to rewrite the YAML automatically, or rename the key manually — the runtime rejects the old name.

## Configuration Structure

Datasources are configured under `agent.services.datasources`. Each entry is an independent database connection:

```yaml
agent:
  services:
    datasources:
      my_snowflake:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        default: true

      my_duckdb:
        type: duckdb
        uri: ./data/analytics.duckdb

    semantic_layer:
      metricflow: {}

    bi_platforms:
      superset:
        type: superset
        api_base_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: ${AIRFLOW_URL}
        username: ${AIRFLOW_USER}
        password: ${AIRFLOW_PASSWORD}
        dags_folder: ${AIRFLOW_DAGS_DIR}
```

## Service Sections

| Section | Purpose | Selector |
|---------|---------|----------|
| `services.datasources` | Database connections used by SQL and KB operations | `--database` / current database / default database |
| `services.semantic_layer` | Semantic adapter configuration such as MetricFlow | `semantic_adapter` |
| `services.bi_platforms` | BI platform credentials and dataset materialization config | `bi_platform` |
| `services.schedulers` | Scheduler service instances such as Airflow | `scheduler_service` |

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
  uri: duckdb:////Users/xxx/data/analytics.duckdb
```

### MySQL
```yaml
my_mysql:
  type: mysql
  host: localhost
  port: 3306
  username: ${MYSQL_USER}
  password: ${MYSQL_PASSWORD}
  database: analytics
```

### PostgreSQL
```yaml
my_postgresql:
  type: postgresql
  host: localhost
  port: 5432
  username: ${POSTGRES_USER}
  password: ${POSTGRES_PASSWORD}
  database: analytics
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
- **MySQL/PostgreSQL**: `host`, `port`, `username`, `password`, `database`

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
- [Semantic Layer Configuration](semantic_layer.md) - Configure semantic adapters
- [BI Platforms Configuration](bi_platforms.md) - Configure Superset or Grafana
- [Scheduler Configuration](schedulers.md) - Configure Airflow services
- [CLI Commands](../cli-commands.md) - Full CLI reference including configure, init, and service commands
