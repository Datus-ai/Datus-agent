# StarRocks datasource

The StarRocks adapter uses the MySQL protocol and adds StarRocks catalog, materialized-view, and three-part namespace support.

## Install

```bash
pip install datus-starrocks
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      starrocks_prod:
        type: starrocks
        host: ${STARROCKS_HOST}
        port: 9030
        username: ${STARROCKS_USER}
        password: ${STARROCKS_PASSWORD}
        catalog: default_catalog
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | Frontend query host. |
| `port` | integer | no | `9030` | MySQL-protocol query port. |
| `username` | string | yes | — | StarRocks user. |
| `password` | string | no | empty | Login password. |
| `catalog` | string | no | `default_catalog` | Initial StarRocks catalog. |
| `database` | string | no | — | Initial database. |
| `charset` | string | no | `utf8mb4` | Connection character set. |
| `autocommit` | boolean | no | `true` | Enables autocommit. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Catalogs and objects

StarRocks objects use `catalog.database.table`. Keep `default_catalog` for native tables or select an external catalog such as a Hive catalog. The login must be able to list the catalog and database as well as query the target objects.

Connect to the FE query port (`9030` by default), not the FE HTTP port or a BE port.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource starrocks_prod
```

Run `/databases` and `/tables`; query a materialized view if your workflow relies on MV discovery.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| MySQL handshake or connection fails | Use the FE MySQL-protocol endpoint and port `9030`. |
| External catalog is missing | Set `catalog` to its exact name and grant catalog/database privileges. |
| Native tables are missing | Use `catalog: default_catalog` and the intended `database`. |

## Reference

- [StarRocks client connection](https://docs.starrocks.io/docs/sql-reference/System_variable/)
