# SQLite datasource

SQLite is built into Datus and connects directly to a local database file. It has no server, credentials, catalog, or schema layer.

## Connection profile

```yaml
agent:
  services:
    datasources:
      local_sqlite:
        type: sqlite
        uri: sqlite:////absolute/path/to/analytics.sqlite
        read_only: true
        default: true
```

A relative path uses three slashes:

```yaml
uri: sqlite:///data/analytics.sqlite
```

To expose several files as databases within one datasource, replace `uri` with a glob:

```yaml
benchmark:
  type: sqlite
  path_pattern: benchmark/databases/**/*.sqlite
  database: california_schools  # optional initial file stem
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `uri` | string | yes* | — | SQLite URI or file path. Required unless `path_pattern` is used. |
| `path_pattern` | string | yes* | — | Glob for multiple files. Required unless `uri` is used. |
| `database` | string | no | first match | With `path_pattern`, selects the initial database by file stem. |
| `read_only` | boolean | no | `false` | Opens the file through SQLite's read-only URI mode. |

## Paths and namespaces

- `sqlite:////tmp/orders.sqlite` resolves to the absolute path `/tmp/orders.sqlite`.
- `sqlite:///data/orders.sqlite` resolves relative to the process working directory.
- A single-file datasource uses the file stem as its database name and addresses tables directly.
- A `path_pattern` datasource lists each matched file as a separate database.

!!! warning
    `read_only: true` protects the SQLite file at the connection layer. The deployment-wide `agent.sql_read_only` setting is a separate SQL execution policy and can be enabled in addition.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource local_sqlite
```

In the CLI, run `/tables` or execute `SELECT 1` in SQL mode. The `/datasource` editor also tests the file before saving it.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Datasource is skipped at startup | `path_pattern` matched no files. Check the working directory and glob. |
| `unable to open database file` | The path does not exist or the Datus process lacks directory/file permissions. |
| Write statement fails | The profile has `read_only: true`, the file is not writable, or `agent.sql_read_only` is enabled. |

## Reference

- [SQLite documentation](https://www.sqlite.org/docs.html)
