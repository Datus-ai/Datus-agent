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

> **Note**: The earlier `services.databases` key has been renamed to `services.datasources`. Rename the key manually in your `agent.yml` — the runtime rejects the old name.

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
        password: ${SNOWFLAKE_PASSWORD}  # Use private_key, or exactly one of password/private_key_file
        # private_key: ${SNOWFLAKE_PRIVATE_KEY}
        # private_key_file: ${SNOWFLAKE_PRIVATE_KEY_FILE}
        # private_key_file_pwd: ${SNOWFLAKE_PRIVATE_KEY_FILE_PWD}
        database: ${SNOWFLAKE_DATABASE}  # Optional
        schema: ${SNOWFLAKE_SCHEMA}      # Optional
        warehouse: ${SNOWFLAKE_WAREHOUSE}
        role: ${SNOWFLAKE_ROLE}          # Optional
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
| `services.datasources` | Database connections used by SQL and KB operations | `--datasource` / current database / default database |
| `services.semantic_layer` | Semantic adapter configuration such as MetricFlow or OSI | active/default semantic layer |
| `services.bi_platforms` | BI platform credentials and dataset materialization config | `bi_platform` |
| `services.schedulers` | Scheduler service instances such as Airflow | `scheduler_service` |

## Supported Database Types

### Snowflake
```yaml
my_snowflake:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  password: ${SNOWFLAKE_PASSWORD}      # Use private_key, or exactly one of password/private_key_file
  # private_key: ${SNOWFLAKE_PRIVATE_KEY}
  # private_key_file: ${SNOWFLAKE_PRIVATE_KEY_FILE}
  # private_key_file_pwd: ${SNOWFLAKE_PRIVATE_KEY_FILE_PWD}  # Optional
  database: ${SNOWFLAKE_DATABASE}    # Optional
  schema: ${SNOWFLAKE_SCHEMA}        # Optional
  warehouse: ${SNOWFLAKE_WAREHOUSE}
  role: ${SNOWFLAKE_ROLE}            # Optional
  default: true                      # Optional: mark as default
```

Snowflake supports password authentication and key-pair authentication. For hosted/SaaS deployments, prefer
`private_key` to store the PEM private key as a secret; for local or CI setups with an existing key file, use
`private_key_file`. When `private_key` is provided, it takes precedence over `private_key_file` and `password`;
without `private_key`, configure exactly one of `password` or `private_key_file`. Set `private_key_file_pwd`
when the private key is encrypted. The adapter uses Snowflake JWT authentication internally.

Snowflake uses a `database` + `schema` namespace. Leave `catalog` unset for Snowflake; catalog filters are for
catalog-aware engines such as StarRocks.

### Google BigQuery

```yaml
my_bigquery:
  type: bigquery
  catalog: ${BIGQUERY_PROJECT}
  database: ${BIGQUERY_DATASET}
  location: ${BIGQUERY_LOCATION:-US}
  credentials_path: ${GOOGLE_APPLICATION_CREDENTIALS}
  # credentials_base64: ${BIGQUERY_CREDENTIALS_BASE64}
  # billing_project_id: ${BIGQUERY_BILLING_PROJECT}
```

`catalog` maps to the Google Cloud project and `database` maps to the BigQuery dataset. Leave `schema` unset. Configure
only one of `credentials_path`, `credentials_info`, or `credentials_base64`; omit all three to use Application Default
Credentials. In YAML, `credentials_info` must be a mapping, not a quoted JSON string. See
[Database Adapters](../adapters/db_adapters.md#google-bigquery) for correct and incorrect examples, the GitHub Secret
flow, and detailed credential and namespace guidance.

### MaxCompute

```yaml
my_maxcompute:
  type: maxcompute
  database: ${MAXCOMPUTE_PROJECT}
  endpoint: ${MAXCOMPUTE_ENDPOINT}
  access_key_id: ${MAXCOMPUTE_ACCESS_KEY_ID}
  access_key_secret: ${MAXCOMPUTE_ACCESS_KEY_SECRET}
  # schema: default               # Optional; schema-enabled projects only
  namespace_mode: auto            # auto, two_level, or three_level
  # quota_name: ${MAXCOMPUTE_QUOTA_NAME}
  # tunnel_endpoint: ${MAXCOMPUTE_TUNNEL_ENDPOINT}
  # query_timeout_seconds: 600
```

`database` names the MaxCompute project. Leave `schema` unset for two-level projects; schema-enabled projects default
to the `default` schema. A datasource has no catalog and stays within its configured project. See
[Database Adapters](../adapters/db_adapters.md#maxcompute) for namespace and endpoint details.

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

### Apache Doris

```yaml
my_doris:
  type: doris
  host: ${DORIS_HOST}
  port: 9030                    # FE query port (MySQL protocol)
  username: ${DORIS_USER}
  password: ${DORIS_PASSWORD}
  database: ${DORIS_DATABASE}
  catalog: internal             # Optional, default is internal
```

### TiDB

```yaml
my_tidb:
  type: tidb
  host: ${TIDB_HOST}
  port: 4000                    # TiDB's own default, not MySQL's 3306
  username: ${TIDB_USER}
  password: ${TIDB_PASSWORD}
  database: ${TIDB_DATABASE}
```

### Hologres

```yaml
my_hologres:
  type: hologres
  host: ${HOLOGRES_ENDPOINT}    # Console endpoint, hostname or hostname:port
  port: 80
  username: ${HOLOGRES_ACCESS_KEY_ID}
  password: ${HOLOGRES_ACCESS_KEY_SECRET}
  database: ${HOLOGRES_DATABASE}
  schema: public                # Optional
  sslmode: prefer               # Optional
```

### GaussDB / openGauss

```yaml
my_gaussdb:
  type: gaussdb
  host: ${GAUSSDB_HOST}
  port: 5432
  username: ${GAUSSDB_USER}
  password: ${GAUSSDB_PASSWORD}
  database: postgres
  schema: public
  # driver: pg8000              # Optional; omit to use the platform default
  sslmode: verify-ca             # Recommended production baseline
  sslrootcert: /etc/datus/certs/gaussdb-ca.pem
```

Supported drivers are `gaussdb` (Linux; sha256/md5/sm3 authentication), `pg8000`
(Linux/macOS; sha256/md5), and the md5-only `psycopg2` escape hatch. Supported TLS modes are
`disable`, `allow`, `prefer` (default), `require`, `verify-ca`, and `verify-full`. Use
`verify-ca` with the server CA in `sslrootcert` as the production baseline. When the configured
hostname is guaranteed to match the certificate, use `verify-full` for stricter hostname
validation; otherwise keep `verify-ca`. The adapter supports one-way TLS, not client certificates
for mutual TLS. See [Database Adapters](../adapters/db_adapters.md#gaussdb) for mode, platform,
authentication, and A/B/PG compatibility details.

### Huawei Cloud GaussDB(DWS)

```yaml
my_dws:
  type: dws
  host: ${DWS_HOST}              # console endpoint, may embed ":8000"
  port: 8000
  username: ${DWS_USER}
  password: ${DWS_PASSWORD}
  database: gaussdb              # the cluster default
  schema: public
  sslmode: verify-ca             # recommended production baseline
  sslrootcert: /etc/datus/certs/dws-cacert.pem
```

DWS answers standard MD5 authentication over the PostgreSQL wire protocol, so no driver
selection is needed. Use `verify-ca` with the server CA in `sslrootcert` as the production
baseline — take that CA from `v2/sslcert/cacert.pem` in the console's `dws_ssl_cert` bundle,
since the v1 CA does not match the server certificate issuer. `verify-full` cannot succeed
against the default server certificate, whose CN is `server` with no `subjectAltName`.
`sslrootcert` accepts a file path or inline PEM content.

Note what `verify-ca` does not cover: it proves the certificate chains to the configured CA,
not that you reached the intended cluster. The DWS default certificate is not issued per
cluster, so any endpoint presenting one from the same CA passes — and with `verify-full`
unavailable, no `sslmode` closes that gap. Reach the cluster over a VPC or a verified fixed
EIP; a substituted endpoint would still receive the configured password.

New clusters default to ORA compatibility mode, which stores empty strings as NULL and
evaluates `7/2` as `3.5` rather than integer `3`. See
[Database Adapters](../adapters/db_adapters.md#gaussdbdws) for compatibility modes, TLS
details, and DDL portability.

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
| `type` | Yes | Database type: `sqlite`, `duckdb`, `snowflake`, `starrocks`, `mysql`, `postgresql`, `doris`, `hologres`, ... |
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
- **Apache Doris**: `catalog` (default `internal`)
- **Hologres**: `schema`, `sslmode`; `access_key_id`/`access_key_secret` are accepted as aliases for `username`/`password`
- **GaussDB/openGauss**: `schema`, `driver`, `sslmode`, `sslrootcert`

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
datus-cli --datasource my_duckdb
datus-agent run --datasource my_snowflake --task "..." --task_db_name ANALYTICS
```

If `--datasource` is not specified:
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
- [CLI Commands](../cli/other_commands.md) - Full CLI reference including configure, init, and service commands
