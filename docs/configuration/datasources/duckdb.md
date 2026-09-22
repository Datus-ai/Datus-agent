# DuckDB datasource

DuckDB is built into Datus. It can use a persistent `.duckdb` file, an in-memory database, or a DuckDB Iceberg REST catalog attached to the local engine.

## Connection profile

```yaml
agent:
  services:
    datasources:
      local_duckdb:
        type: duckdb
        uri: duckdb:///data/analytics.duckdb
        read_only: false
        enable_external_access: true
        memory_limit: 2GB
        default: true
```

Use `duckdb:///:memory:` for a process-local in-memory database. Multiple files can be discovered with `path_pattern`, using the same rules as [SQLite](sqlite.md).

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `uri` | string | yes* | — | DuckDB URI or file path. Required unless `path_pattern` is used. |
| `path_pattern` | string | yes* | — | Glob for multiple `.duckdb` files. Required unless `uri` is used. |
| `database` | string | no | first match | With `path_pattern`, selects the initial database by file stem. |
| `read_only` | boolean | no | `false` | Opens the DuckDB file read-only. |
| `enable_external_access` | boolean | no | `true` | Controls DuckDB access to external files and extensions. |
| `memory_limit` | string | no | DuckDB default | Passed to `SET memory_limit`, for example `2GB`. |
| `iceberg` | mapping | no | — | Attaches an Iceberg REST catalog; see below. |

## Iceberg REST catalog

```yaml
lakehouse:
  type: duckdb
  uri: duckdb:///:memory:
  iceberg:
    catalog_uri: ${ICEBERG_REST_URI}
    warehouse: s3://analytics-warehouse/
    catalog_alias: lake
    client_id: ${ICEBERG_CLIENT_ID}
    client_secret: ${ICEBERG_CLIENT_SECRET}
    oauth2_server_uri: ${ICEBERG_OAUTH_TOKEN_URI}
    s3_region: us-east-1
    s3_endpoint: ${S3_ENDPOINT}
    s3_access_key_id: ${S3_ACCESS_KEY_ID}
    s3_secret_access_key: ${S3_SECRET_ACCESS_KEY}
    read_only: true
```

`catalog_uri` and `warehouse` are required inside `iceberg`. `catalog_alias` defaults to `lake`. OAuth fields create a DuckDB Iceberg secret; `token` can be supplied instead. S3 credentials accept the `s3_*` names shown above or their `aws_*` aliases. Advanced attach options include `endpoint_type`, `authorization_type`, `access_delegation_mode`, `support_nested_namespaces`, `support_stage_create`, `max_table_staleness`, and `purge_requested`.

!!! note
    The top-level `read_only` controls the local DuckDB connection. `iceberg.read_only` independently controls the attached catalog.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource local_duckdb
```

Run `/databases`, `/tables`, or `SELECT 1` in SQL mode. For Iceberg, `/databases` should include the configured `catalog_alias` after the `httpfs` and `iceberg` extensions load.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| External file or extension access is refused | Set `enable_external_access: true`; keep it disabled when the datasource must be isolated from the filesystem/network. |
| `DuckDB iceberg config requires catalog_uri` | Add `iceberg.catalog_uri` (the aliases `iceberg_catalog_uri` and `endpoint` are also accepted). |
| `DuckDB iceberg config requires warehouse` | Add the logical warehouse identifier, such as an S3 URI. |
| File is locked or writes fail | Do not open the same file with incompatible read/write modes; verify `read_only` and filesystem permissions. |

## Reference

- [DuckDB documentation](https://duckdb.org/docs/)
