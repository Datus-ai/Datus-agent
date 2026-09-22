# Datasources

A datasource is a named database connection used by Datus for SQL execution, metadata discovery, and knowledge-base indexing. Configure datasources under `agent.services.datasources` in `agent.yml`.

```yaml
agent:
  services:
    datasources:
      analytics:
        type: postgresql
        host: ${POSTGRES_HOST}
        port: 5432
        username: ${POSTGRES_USER}
        password: ${POSTGRES_PASSWORD}
        database: analytics
        schema: public
        default: true
```

Each key under `datasources` is the datasource name. Names may contain letters, numbers, underscores, and hyphens. Set `default: true` on at most one entry; when only one datasource exists, Datus selects it automatically.

!!! note
    The configuration path remains `agent.services.datasources`. Configure semantic adapters under [Adapters](../adapters/semantic_adapters.md), and prefer [plugins](../plugin/introduction.md) for integrations such as Airflow.

## Choose a datasource

SQLite and DuckDB are built into Datus. Every other datasource is provided by an independently installable `datus-<type>` adapter. Datus can install a missing adapter when you add the datasource with `/datasource`; you can also install the package explicitly.

| Datasource | `type` | Package | Namespace |
|---|---|---|---|
| [SQLite](datasources/sqlite.md) | `sqlite` | built in | database file → table |
| [DuckDB](datasources/duckdb.md) | `duckdb` | built in | database → schema → table |
| [MySQL](datasources/mysql.md) | `mysql` | `datus-mysql` | database → table |
| [PostgreSQL](datasources/postgresql.md) | `postgresql` | `datus-postgresql` | database → schema → table |
| [Greenplum](datasources/greenplum.md) | `greenplum` | `datus-greenplum` | database → schema → table |
| [Amazon Redshift](datasources/redshift.md) | `redshift` | `datus-redshift` | database → schema → table |
| [Snowflake](datasources/snowflake.md) | `snowflake` | `datus-snowflake` | database → schema → table |
| [Google BigQuery](datasources/bigquery.md) | `bigquery` | `datus-bigquery` | project → dataset → table |
| [StarRocks](datasources/starrocks.md) | `starrocks` | `datus-starrocks` | catalog → database → table |
| [Apache Doris](datasources/doris.md) | `doris` | `datus-doris` | catalog → database → table |
| [TiDB](datasources/tidb.md) | `tidb` | `datus-tidb` | database → table |
| [ClickHouse](datasources/clickhouse.md) | `clickhouse` | `datus-clickhouse` | database → table |
| [Trino](datasources/trino.md) | `trino` | `datus-trino` | catalog → schema → table |
| [Hive](datasources/hive.md) | `hive` | `datus-hive` | database → table |
| [Spark SQL](datasources/spark.md) | `spark` | `datus-spark` | database → table |
| [ClickZetta](datasources/clickzetta.md) | `clickzetta` | `datus-clickzetta` | instance/workspace → schema → table |
| [Hologres](datasources/hologres.md) | `hologres` | `datus-hologres` | database → schema → table |
| [MaxCompute](datasources/maxcompute.md) | `maxcompute` | `datus-maxcompute` | project → optional schema → table |
| [Oracle](datasources/oracle.md) | `oracle` | `datus-oracle` | schema → table |
| [GaussDB / openGauss](datasources/gaussdb.md) | `gaussdb` | `datus-gaussdb` | database → schema → table |
| [GaussDB(DWS)](datasources/dws.md) | `dws` | `datus-dws` | database → schema → table |

## Common profile keys

Each datasource page documents only its type-specific connection keys. The following profile keys are handled by Datus and apply to every datasource:

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `type` | string | yes | — | Adapter type from the table above. |
| `default` | boolean | no | `false` | Marks this entry as the default datasource. |

Adapter models reject unknown keys, so copy the field names exactly. Use environment variables for credentials:

```yaml
username: ${DB_USERNAME}
password: ${DB_PASSWORD}
```

`${NAME:-fallback}` supplies a fallback. A missing `${NAME}` expands to a visible missing-value marker and the connection will fail rather than silently using an empty credential.

## File datasource patterns

SQLite and DuckDB can bind one datasource to several files with `path_pattern` instead of `uri`:

```yaml
agent:
  services:
    datasources:
      benchmark:
        type: sqlite
        path_pattern: benchmark/databases/**/*.sqlite
        database: california_schools  # optional initial file stem
```

The matched files become databases within one datasource. Datus skips the entry at startup when the pattern matches no files.

## Add, test, and select a datasource

The recommended path is the interactive datasource manager:

```bash
datus
```

Run `/datasource` in the CLI. Adding or editing an entry validates its fields, installs a missing adapter, tests connectivity, and writes the result to the active `agent.yml` selected by `--config` or the normal configuration search order. Use `/datasource <name>` to switch the current session.

For a manually edited file, start Datus with the target entry. Connector construction and the initial metadata load surface configuration, network, and authentication failures:

```bash
datus --config conf/agent.yml --datasource analytics
```

Project-specific selection belongs in `.datus/config.yml`:

```yaml
default_datasource: analytics
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Unsupported value ... for field datasource` | The selected name is not a key under `agent.services.datasources`. Check spelling and the active config file. |
| Adapter import or installation error | Install the package shown in the datasource's page into the same Python environment as `datus`. |
| Unknown or extra field | The installed adapter validates its own schema. Remove the field or use the exact key documented on that datasource page. |
| Connection succeeds but objects are missing | Check the adapter's namespace fields (`catalog`, `database`, `schema`, or `schema_name`) and the database user's grants. |
| Credential appears literally as `<MISSING:...>` | Export the referenced environment variable before starting Datus. |

## Related documentation

- [Database adapter architecture](../adapters/db_adapters.md)
- [SQL policy and deployment-wide read-only mode](sql_policy.md)
- [Plugin system](../plugin/introduction.md)
- [CLI reference](../cli/reference.md)
