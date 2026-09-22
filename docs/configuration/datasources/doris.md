# Apache Doris datasource

The Doris adapter uses the MySQL protocol and supports native and external catalogs with `catalog.database.table` identifiers.

## Install

```bash
pip install datus-doris
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      doris_prod:
        type: doris
        host: ${DORIS_HOST}
        port: 9030
        username: ${DORIS_USER}
        password: ${DORIS_PASSWORD}
        catalog: internal
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | FE query host. |
| `port` | integer | no | `9030` | MySQL-protocol query port. |
| `username` | string | yes | — | Doris user. |
| `password` | string | no | empty | Login password. |
| `catalog` | string | no | `internal` | Initial catalog. |
| `database` | string | no | — | Initial database. |
| `charset` | string | no | `utf8mb4` | Connection character set. |
| `autocommit` | boolean | no | `true` | Enables autocommit. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Catalogs and objects

Use `catalog: internal` for native Doris tables. External catalogs, including Hive Metastore catalogs, can be selected by name. Metadata calls and fully qualified SQL preserve all three namespace levels.

Connect to the FE MySQL-protocol query port, not the FE HTTP port (`8030` in many deployments).

## Verify the connection

```bash
datus --config conf/agent.yml --datasource doris_prod
```

Run `/databases` and `/tables`; for an external catalog, verify that its backing metastore is healthy and reachable from Doris.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Connection reaches HTTP instead of MySQL | Use FE port `9030`, not `8030`. |
| External catalog objects are missing | Check `catalog`, Doris catalog grants, and the external metastore. |
| Native objects are missing | Set `catalog: internal` and the intended `database`. |

## Reference

- [Apache Doris connection documentation](https://doris.apache.org/docs/dev/db-connect/database-connect/)
