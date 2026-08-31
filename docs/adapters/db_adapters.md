# Database Adapters

Datus Agent supports connecting to various databases through a plugin-based adapter system. This document explains the available adapters, how to install them, and how to configure your database connections.

## Overview

Datus uses a modular adapter architecture that allows you to connect to different databases:

- **Built-in Adapters**: SQLite and DuckDB are included with the core package
- **Plugin Adapters**: Additional databases (MySQL, Snowflake, StarRocks, etc.) can be installed as separate packages

This design keeps the core package lightweight while allowing you to add support for specific databases as needed.

## Supported Databases

| Database | Package | Installation | Status |
|----------|---------|-------------|--------|
| SQLite | Built-in | Included | Ready |
| DuckDB | Built-in | Included | Ready |
| MySQL | datus-mysql | `pip install datus-mysql` | Ready |
| PostgreSQL | datus-postgresql | `pip install datus-postgresql` | Ready |
| StarRocks | datus-starrocks | `pip install datus-starrocks` | Ready |
| Snowflake | datus-snowflake | `pip install datus-snowflake` | Ready |
| ClickZetta | datus-clickzetta | `pip install datus-clickzetta` | Ready |
| Hive | datus-hive | `pip install datus-hive` | Ready |
| Spark | datus-spark | `pip install datus-spark` | Ready |
| ClickHouse | datus-clickhouse | `pip install datus-clickhouse` | Ready |
| Trino | datus-trino | `pip install datus-trino` | Ready |
| Apache Doris | datus-doris | `pip install datus-doris` | Ready |
| TiDB | datus-tidb | `pip install datus-tidb` | Ready |
| Hologres | datus-hologres | `pip install datus-hologres` | Ready |
| MaxCompute | datus-maxcompute | `pip install datus-maxcompute` | Ready |
| Google BigQuery | datus-bigquery | `pip install datus-bigquery` | Ready |
| Oracle | datus-oracle | `pip install datus-oracle` | Ready |
| GaussDB / openGauss | datus-gaussdb | `pip install datus-gaussdb` | Ready (Linux and macOS) |
| Huawei Cloud GaussDB(DWS) | datus-dws | `pip install datus-dws` | Ready |

## Installation

### Built-in Databases

SQLite and DuckDB are included with Datus Agent and require no additional installation.

### Plugin Adapters

Install the adapter package for your database:

```bash
# MySQL
pip install datus-mysql

# PostgreSQL
pip install datus-postgresql

# Snowflake
pip install datus-snowflake

# StarRocks
pip install datus-starrocks

# ClickZetta
pip install datus-clickzetta

# Hive
pip install datus-hive

# Spark
pip install datus-spark

# ClickHouse
pip install datus-clickhouse

# Trino
pip install datus-trino

# Apache Doris
pip install datus-doris

# Hologres
pip install datus-hologres

# MaxCompute
pip install datus-maxcompute

# Google BigQuery
pip install datus-bigquery

# Oracle
pip install datus-oracle

# GaussDB / openGauss
pip install datus-gaussdb

# Huawei Cloud GaussDB(DWS)
pip install datus-dws
```

Once installed, Datus Agent will automatically detect and load the adapter.

## Configuration

Configure database connections under `agent.services.datasources` in `agent.yml`:

```yaml
agent:
  services:
    datasources:
      mydata:
        type: sqlite
        uri: sqlite:///path/to/database.db
```

Each entry under `services.datasources` is one logical database connection.

### SQLite

```yaml
mydata:
  type: sqlite
  uri: sqlite:///path/to/database.db
```

### DuckDB

```yaml
analytics:
  type: duckdb
  uri: duckdb:///path/to/database.duckdb
```

### MySQL

```yaml
production:
  type: mysql
  host: localhost
  port: 3306
  username: your_username
  password: your_password
  database: your_database
```

### PostgreSQL

```yaml
production_pg:
  type: postgresql
  host: localhost
  port: 5432
  username: your_username
  password: your_password
  database: your_database
  schema: public  # optional, default is public
  sslmode: prefer  # optional, default is prefer
```

### Snowflake

```yaml
warehouse:
  type: snowflake
  account: your_account
  username: your_username
  password: your_password  # Use private_key, or exactly one of password/private_key_file
  # private_key: |
  #   -----BEGIN PRIVATE KEY-----
  #   ...
  #   -----END PRIVATE KEY-----
  # private_key_file: /path/to/rsa_key.p8
  # private_key_file_pwd: optional_key_passphrase
  warehouse: your_warehouse
  database: your_database
  schema: your_schema
  role: your_role  # optional
```

Snowflake supports password authentication and key-pair authentication. Configure `private_key`, or exactly one of
`password` or `private_key_file` when `private_key` is absent. `private_key` takes precedence over
`private_key_file` and `password`; set `private_key_file_pwd` only when the private key is encrypted. Snowflake uses
`database` and `schema`; do not set `catalog` for Snowflake.

### Google BigQuery

```yaml
bigquery_data:
  type: bigquery
  catalog: ${BIGQUERY_PROJECT}
  database: ${BIGQUERY_DATASET}
  location: ${BIGQUERY_LOCATION:-US}
  credentials_path: ${GOOGLE_APPLICATION_CREDENTIALS}
  # credentials_base64: ${BIGQUERY_CREDENTIALS_BASE64}
  # billing_project_id: ${BIGQUERY_BILLING_PROJECT}
  # timeout_seconds: 60
```

`catalog` is the Google Cloud project and `database` is the BigQuery dataset; BigQuery has no schema level below a
dataset. The adapter also accepts `project` and `dataset` as aliases. Use exactly one of `credentials_path`,
`credentials_info`, or `credentials_base64`. Local environments normally use `credentials_path`, while hosted secret
stores can provide `credentials_base64`. The datasource form accepts a service-account JSON object in
`credentials_info`; in YAML, provide that field as a mapping rather than a JSON string. When none is configured, the
Google client uses Application Default Credentials.

For example, this is a YAML mapping. Preserve all fields from the downloaded service-account JSON file; the values
below are placeholders:

```yaml
bigquery_data:
  type: bigquery
  catalog: your-gcp-project-id
  database: your_dataset
  credentials_info:
    type: service_account
    project_id: your-gcp-project-id
    private_key_id: your-private-key-id
    private_key: |
      -----BEGIN PRIVATE KEY-----
      REPLACE_WITH_THE_PRIVATE_KEY_BODY
      -----END PRIVATE KEY-----
    client_email: datus-ci@your-gcp-project-id.iam.gserviceaccount.com
    client_id: "123456789012345678901"
    auth_uri: https://accounts.google.com/o/oauth2/auth
    token_uri: https://oauth2.googleapis.com/token
    auth_provider_x509_cert_url: https://www.googleapis.com/oauth2/v1/certs
    client_x509_cert_url: https://www.googleapis.com/robot/v1/metadata/x509/...
```

Do not quote the entire JSON object:

```yaml
# Wrong: this is one YAML string, not a mapping.
credentials_info: '{"type":"service_account","project_id":"your-gcp-project-id"}'
```

The GitHub cloud test is a separate environment-variable flow: store the original JSON file contents in the
`BIGQUERY_CREDENTIALS_INFO` repository secret. The test fixture parses that string with `json.loads()` before creating
the adapter config. For normal Agent YAML backed by a string-only secret store, prefer `credentials_base64` instead.
Base64 is an encoding, not encryption.

### MaxCompute

```yaml
maxcompute_data:
  type: maxcompute
  database: ${MAXCOMPUTE_PROJECT}
  endpoint: ${MAXCOMPUTE_ENDPOINT}
  access_key_id: ${MAXCOMPUTE_ACCESS_KEY_ID}
  access_key_secret: ${MAXCOMPUTE_ACCESS_KEY_SECRET}
  # schema: default               # optional; schema-enabled projects only
  namespace_mode: auto            # auto, two_level, or three_level
  # quota_name: ${MAXCOMPUTE_QUOTA_NAME}
  # tunnel_endpoint: ${MAXCOMPUTE_TUNNEL_ENDPOINT}
  # timeout_seconds: 30
  # query_timeout_seconds: 600
```

`database` is the MaxCompute project. Each datasource is bound to one project, has no catalog, and does not generate
cross-project SQL. Leave `schema` unset for a two-level `project.table` project. Schema-enabled projects use
`project.schema.table`; when `schema` is omitted, the adapter uses `default`. Keep `namespace_mode: auto` unless the
configured identity cannot probe schema support. `endpoint` is the MaxCompute service endpoint; configure
`tunnel_endpoint` separately only when Instance Tunnel uses a different endpoint. Advanced execution settings also
include `timeout_seconds`, `query_timeout_seconds`, and `default_hints`.

### StarRocks

```yaml
analytics:
  type: starrocks
  host: localhost
  port: 9030
  username: root
  password: your_password
  database: your_database
```

### ClickZetta

```yaml
lakehouse:
  type: clickzetta
  service: CLICKZETTA_SERVICE
  username: CLICKZETTA_USERNAME
  password: CLICKZETTA_PASSWORD
  instance: CLICKZETTA_INSTANCE
  workspace: CLICKZETTA_WORKSPACE
  schema: CLICKZETTA_SCHEMA
  vcluster: CLICKZETTA_VCLUSTER
```

### Hive

```yaml
hive_data:
  type: hive
  host: 127.0.0.1
  port: 10000
  username: hive
  database: default
  auth: NONE  # optional: NONE, LDAP, CUSTOM, KERBEROS
  configuration:  # optional Hive session config
    hive.execution.engine: spark
```

### Spark

```yaml
spark_data:
  type: spark
  host: localhost
  port: 10000
  username: spark
  database: default
  auth_mechanism: NONE  # optional: NONE, PLAIN, KERBEROS
```

### ClickHouse

```yaml
analytics:
  type: clickhouse
  host: localhost
  port: 8123
  username: default
  password: your_password
  database: your_database
```

### Trino

```yaml
trino_data:
  type: trino
  host: localhost
  port: 8080
  username: trino
  catalog: hive
  schema: default
  http_scheme: http  # optional: http or https
```

### Apache Doris

```yaml
doris_data:
  type: doris
  host: localhost
  port: 9030
  username: root
  password: your_password
  catalog: internal      # optional, default is internal
  database: your_database
  charset: utf8mb4       # optional, default is utf8mb4
  autocommit: true       # optional, default is true
  timeout_seconds: 30    # optional, default is 30
```

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `127.0.0.1` | Frontend (FE) host |
| `port` | `9030` | FE MySQL-protocol query port, not the FE HTTP port (8030) |
| `username` | required | Doris user |
| `password` | empty | Doris password |
| `catalog` | `internal` | Catalog the connection starts in |
| `database` | none | Database the connection starts in |
| `charset` | `utf8mb4` | Connection character set |
| `autocommit` | `true` | Autocommit mode |
| `timeout_seconds` | `30` | Connection timeout |

Doris speaks the MySQL protocol, so `port` is the FE query port (default 9030). Objects are addressed as
`catalog.database.table`: Doris has no schema level between database and table, so leave `schema` unset
and let `catalog` carry the extra level.

`internal` is the built-in catalog holding Doris-managed tables. To start a connection on an external
catalog — a Hive Metastore catalog, for example — name it in `catalog`:

```yaml
doris_hive:
  type: doris
  host: localhost
  port: 9030
  username: root
  password: your_password
  catalog: hive_catalog
  database: warehouse
```

The catalog has to exist in Doris already — Datus selects a catalog, it never creates one. Create it on
the Doris side first (`CREATE CATALOG hive_catalog PROPERTIES (...)`, with the metastore URI and storage
credentials the catalog type needs) and confirm it is listed by `SHOW CATALOGS`.

Within a session, `SWITCH <catalog>` changes the catalog and `USE [<catalog>.]<database>` changes the
database. Switching catalogs clears the current database, because a database of that name usually does
not exist under the new catalog; issue a `USE` afterwards to select one.

### TiDB

```yaml
tidb_data:
  type: tidb
  host: 127.0.0.1
  port: 4000
  username: root
  password: your_password
  database: your_database
  charset: utf8mb4       # optional, default is utf8mb4
  autocommit: true       # optional, default is true
  timeout_seconds: 30    # optional, default is 30
```

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `127.0.0.1` | TiDB server host |
| `port` | `4000` | TiDB's own default SQL port — not MySQL's 3306 |
| `username` | required | TiDB user |
| `password` | empty | TiDB password |
| `database` | none | Database the connection starts in |
| `charset` | `utf8mb4` | Connection character set |
| `autocommit` | `true` | Autocommit mode |
| `timeout_seconds` | `30` | Connection timeout |

TiDB speaks the MySQL wire protocol and is addressed as `database.table`: like MySQL it has no schema
level, so leave both `catalog` and `schema` unset. Set `port` explicitly — TiDB listens on 4000, and
3306 on the same host is usually a different server entirely.

Being MySQL-compatible does not make TiDB a MySQL superset. It has no `FULL OUTER JOIN`, `JSON_TABLE`,
`LATERAL`, `CREATE TABLE ... AS SELECT`, `CORR`/`COVAR_*`, materialized views, or updatable views, and
its default collation is the case-sensitive `utf8mb4_bin`. Two clauses are accepted and then ignored:
`CHECK` constraints are not enforced unless `tidb_enable_check_constraint` is `ON`, and `FULLTEXT`
indexes are silently dropped. The adapter ships a SQL skill that keeps generated SQL clear of all of
these.

**TiFlash.** TiDB's columnar replica engine is per table: `ALTER TABLE t SET TIFLASH REPLICA 1` and the
optimizer starts choosing between row store and columnar on its own — no query change, and nothing to
configure in Datus. `information_schema.TIFLASH_REPLICA` reports which tables have a synced replica.
Aggregate window functions are the one thing that tends not to gain from a replica: on TiDB v8.5 they
fall back to single-node computation on the TiDB layer — correct, but not parallel — so prefer
`GROUP BY` aggregation where a query can be written either way. Push-down coverage varies by version
and by the operators around the call, so confirm with `EXPLAIN` and look for `mpp[tiflash]` on the
window operator before assuming either behaviour.

The `datus-tidb` adapter does not support TLS configuration yet, so it cannot reach TiDB Cloud endpoints that require TLS. This is an adapter limitation, not a TiDB one.

### Hologres

```yaml
hologres_data:
  type: hologres
  host: your-instance-cn-hangzhou.hologres.aliyuncs.com  # console endpoint, may embed ":80"
  port: 80
  username: ${HOLOGRES_ACCESS_KEY_ID}
  password: ${HOLOGRES_ACCESS_KEY_SECRET}
  database: your_database
  schema: public   # optional, default is public
  sslmode: prefer  # optional, default is prefer
```

Hologres (Alibaba Cloud) uses the PostgreSQL wire protocol with AccessKey credentials.
`access_key_id`/`access_key_secret` are also accepted as aliases for `username`/`password`. The `host`
accepts either a plain hostname or a `hostname:port` console endpoint; an explicit `port` must match an
embedded one.

### Oracle

```yaml
oracle_data:
  type: oracle
  host: localhost
  port: 1521
  username: datus_user
  password: your_password
  service_name: FREEPDB1
  schema: DATUS_USER
```

Configure exactly one connection target: `service_name` (recommended), `sid`, or `dsn`. The service/PDB
selects the connection target but is not part of an SQL object name; qualify objects as `SCHEMA.TABLE`.

### GaussDB

```yaml
gaussdb_data:
  type: gaussdb
  host: localhost
  port: 5432
  username: datus
  password: ${GAUSSDB_PASSWORD}
  database: postgres
  schema: public   # optional, default is public
  # driver: pg8000  # optional; omit to use gaussdb on Linux or pg8000 on macOS
  sslmode: verify-ca
  sslrootcert: /etc/datus/certs/gaussdb-ca.pem
```

GaussDB and openGauss speak the PostgreSQL wire protocol. Choose the driver to match the platform and the
server account's password authentication:

| Driver | Platform | Authentication |
|--------|----------|----------------|
| `gaussdb` (Linux default) | Linux | sha256, md5, sm3 |
| `pg8000` (macOS default) | Linux and macOS | sha256, md5 |
| `psycopg2` | Linux and macOS | md5 only; escape hatch |

All drivers accept `disable`, `allow`, `prefer` (default), `require`, `verify-ca`, and `verify-full`, but
their retry behavior is not identical. For production, use `verify-ca` with `sslrootcert` as the baseline:
the connection is encrypted and the server certificate must chain to that CA. When the configured `host`
is guaranteed to match the server certificate, use `verify-full` for stricter hostname validation;
otherwise keep `verify-ca`. `require` encrypts but does not authenticate the server; `prefer` permits an
unencrypted fallback. If the server merely enables TLS, the client does not need a certificate. If it
requires TLS, `prefer` normally negotiates TLS, but explicitly use `require` or either `verify-*` mode to
prevent a plaintext fallback against a differently configured endpoint. Only `verify-ca` and `verify-full`
need the server CA configured on the client. The `pg8000` path treats `allow` like `prefer` (TLS first),
because its API cannot express libpq's plaintext-first retry order. The adapter currently supports one-way
TLS, not mutual TLS (`sslcert`/`sslkey`).

Both centralized and distributed deployments are supported. The connector auto-detects the database's
A (Oracle), B (MySQL), or PG compatibility mode so generated SQL follows the server semantics.

### GaussDB(DWS)

```yaml
dws_data:
  type: dws
  host: example.dws.myhuaweicloud.com   # console endpoint, may embed ":8000"
  port: 8000
  username: dbadmin
  password: ${DWS_PASSWORD}
  database: gaussdb   # the cluster default
  schema: public      # optional, default is public
  sslmode: verify-ca
  sslrootcert: /etc/datus/certs/dws-cacert.pem
```

DWS is a shared-nothing MPP warehouse that speaks the PostgreSQL wire protocol and answers standard MD5
authentication, so the adapter uses psycopg2 and needs no driver selection.

**TLS.** Use `verify-ca` with `sslrootcert` for production. Two DWS-specific points:

- `verify-full` **cannot succeed** against the default server certificate, whose CN is `server` and which
  carries no `subjectAltName`; hostname validation can never match a real endpoint. This is a property of
  the certificate, not a misconfiguration.
- The console's `dws_ssl_cert` bundle contains two CAs. Use `v2/sslcert/cacert.pem` — the v1 CA is
  `Huawei Equipment CA` and does not match the server certificate issuer.

`sslrootcert` accepts a file path or inline PEM content. If the cluster has SSL enforcement enabled under
**Security Settings**, `disable` fails outright while the default `prefer` upgrades automatically.

**Compatibility modes.** The connector detects `ORA`, `TD`, or `MySQL` mode from the catalog. ORA is the
default for new clusters and changes expression semantics — notably `7/2` yields `3.5` rather than integer
`3`, and an empty string is stored as NULL so `col = ''` never matches. These are surfaced in the packaged
SQL skill and in migration notes. TD and MySQL modes are not verified.

**DDL.** Table definitions come from DWS's `pg_get_tabledef()`, preserving storage orientation,
compression, distribution and partitioning. The `TO GROUP` and `TABLESPACE` clauses it emits name objects
of the source cluster and must be removed before replaying the DDL elsewhere.

### Multiple Database Entries

```yaml
agent:
  services:
    datasources:
      source_db:
        type: mysql
        host: source-server
        username: reader
        password: password
        database: source

      target_db:
        type: snowflake
        account: your_account
        username: writer
        password: password
        warehouse: compute_wh
        database: target
```

## Features by Adapter

### Common Features

All adapters support:

- SQL query execution (SELECT, INSERT, UPDATE, DELETE)
- DDL operations (CREATE, ALTER, DROP)
- Metadata retrieval (tables, views, schemas)
- Sample data retrieval
- Connection pooling and timeout management

### Adapter-Specific Features

#### MySQL
- INFORMATION_SCHEMA queries
- SHOW CREATE TABLE/VIEW support
- Full CRUD operations

#### PostgreSQL
- INFORMATION_SCHEMA queries
- Tables, views, and materialized views support
- Multi-schema datasource support
- SSL connection modes (disable, allow, prefer, require, verify-ca, verify-full)
- SQLAlchemy-based (psycopg2 driver)

#### Snowflake
- Multi-database and schema support
- Tables, views, and materialized views
- Arrow format for efficient data transfer
- Native SDK integration

#### Google BigQuery
- GoogleSQL guidance through the adapter-provided `db-bigquery-sql` skill
- Project and dataset navigation with fully qualified `project.dataset.table` identifiers
- Tables, views, materialized views, and Arrow/Pandas result formats
- Application Default Credentials, service-account file, JSON object, and base64 credential flows
- Migration target hints, source-type mapping, BigQuery DDL validation, partitioning, and clustering suggestions

#### StarRocks
- Multi-Catalog support
- Materialized view support
- MySQL protocol compatibility

#### ClickZetta
- Workspace and schema management
- Volume/Stage file operations
- Native SDK integration

#### Hive
- HiveServer2/Thrift protocol connection
- Hive session configuration support
- Multiple auth mechanisms (NONE, LDAP, CUSTOM, KERBEROS)
- Database context switching (USE statement)

#### Spark
- Spark Thrift Server connection via HiveServer2 protocol
- Multiple auth mechanisms (NONE, PLAIN, KERBEROS)
- Spark SQL dialect support

#### ClickHouse
- HTTP protocol connection
- No schema layer (databases serve as schemas)
- ClickHouse-specific DML syntax (ALTER TABLE UPDATE)
- Lightweight deletes support

#### Trino
- Three-level hierarchy: catalog → schema → table
- Cross-catalog query support
- Built-in TPC-H connector for benchmarking
- HTTP/HTTPS connection with SSL support

#### Apache Doris
- MySQL protocol compatibility on the FE query port
- Multi-catalog support: `SHOW CATALOGS` discovery, plus `SWITCH <catalog>` and `USE [catalog.]database` context switching
- Catalog-qualified `information_schema` reads, so metadata queries need no session-level catalog switch and stay thread-safe
- Three-part identifiers (`catalog.database.table`) with backtick quoting; no schema level
- Asynchronous materialized views discovered through `mv_infos()`, with DDL retrieval
- Key-model-aware column metadata (Duplicate, Unique, and Aggregate key columns)
- Catalog-aware sample-row retrieval, and list, CSV, Pandas, and Arrow result formats
- A packaged `db-doris-sql` skill covering table models, distribution, materialized views, and loading through Stream Load, Routine Load, or `INSERT INTO SELECT` over a TVF or catalog
- Migration target support: table-layout suggestions, DDL validation, source-type mapping, and a dry-run `CREATE TABLE` against the cluster

#### Hologres
- PostgreSQL wire protocol (PostgreSQL-compatible SQL dialect)
- Alibaba Cloud AccessKey authentication
- Multi-schema datasource support
- Console endpoint normalization (`hostname` or `hostname:port`)
- SSL connection modes (disable, allow, prefer, require, verify-ca, verify-full)

#### Oracle
- Oracle Database 19c SQL and PL/SQL syntax guidance through the adapter-provided skill
- Service name, SID, or DSN connection targets
- Schema-scoped metadata discovery through `ALL_*` dictionary views
- Oracle-compatible profiling and bound-parameter data transfers

#### GaussDB
- PostgreSQL wire protocol (PostgreSQL-compatible SQL dialect)
- sha256, md5, and sm3 authentication through the official `gaussdb` driver
- Pure-Python `pg8000` path with sha256/md5 authentication on Linux and macOS
- TLS modes through `verify-full`; `verify-ca` is the production baseline, and `verify-full` adds hostname validation
- A / B / PG compatibility-mode auto-detection, so SQL generation follows the server's semantics
- Centralized and distributed deployments, with distribution-aware table DDL
- Multi-schema datasource support

## Troubleshooting

### Adapter Not Found

If you see an error like `Connector 'mysql' not found`, make sure you have installed the corresponding adapter package:

```bash
pip install datus-mysql
```

### Connection Issues

Check the following:

1. **Network connectivity**: Ensure you can reach the database server
2. **Credentials**: Verify username and password are correct
3. **Port**: Confirm the correct port is specified
4. **Database name**: Ensure the database exists

### Driver Dependencies

Some adapters require additional system dependencies:

- **MySQL**: Requires `pymysql` (installed automatically)
- **PostgreSQL**: Requires `psycopg2-binary` (installed automatically)
- **Snowflake**: Requires `snowflake-connector-python` (installed automatically)
- **Google BigQuery**: Requires `sqlalchemy-bigquery` and the Google Cloud BigQuery client (installed automatically)
- **Hive**: Requires `pyhive`, `thrift`, `thrift-sasl`, `pure-sasl` (installed automatically)
- **Spark**: Requires `pyhive`, `thrift`, `thrift-sasl`, `pure-sasl` (installed automatically)
- **ClickHouse**: Requires `clickhouse-sqlalchemy` (installed automatically)
- **Trino**: Requires `trino` (installed automatically)
- **Apache Doris**: Requires `datus-mysql` and `pymysql` (installed automatically)
- **Hologres**: Requires `datus-postgresql` and `psycopg2-binary` (installed automatically)
- **Oracle**: Requires `oracledb` (installed automatically; Thin mode needs no Oracle Client)
- **GaussDB**: Requires `datus-postgresql`; dependencies are installed automatically. Linux defaults to
  the official `gaussdb` driver and the package wheel bundles its GaussDB-family libpq. macOS defaults to
  the pure-Python `pg8000` driver because no compatible native libpq is published for Darwin.

## Architecture

```text
datus-agent (Core)
├── Built-in Adapters
│   ├── SQLite Connector
│   └── DuckDB Connector
│
└── Plugin System (Entry Points)
    ├── datus-sqlalchemy (Base layer)
    │   ├── datus-mysql
    │   │   ├── datus-starrocks
    │   │   └── datus-doris
    │   ├── datus-postgresql
    │   │   ├── datus-hologres
    │   │   └── datus-gaussdb
    │   ├── datus-hive
    │   ├── datus-spark
    │   ├── datus-clickhouse
    │   ├── datus-trino
    │   ├── datus-oracle
    │   └── datus-bigquery
    │
    └── Native SDK Adapters
        ├── datus-snowflake
        └── datus-clickzetta
```

The adapter system uses Python's entry points mechanism for automatic discovery. When you install an adapter package, it registers itself with Datus Agent and becomes available for use.

## Next Steps

- [Quick Start Guide](../getting_started/Quickstart.md) - Get started with Datus Agent
- [Configuration Reference](../configuration/introduction.md) - Detailed configuration options
