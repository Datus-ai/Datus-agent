# TiDB datasource

The TiDB adapter uses TiDB's MySQL-compatible protocol while preserving TiDB-specific metadata and SQL behavior.

## Install

```bash
pip install datus-tidb
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      tidb_cluster:
        type: tidb
        host: ${TIDB_HOST:-127.0.0.1}
        port: 4000
        username: ${TIDB_USER}
        password: ${TIDB_PASSWORD}
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | TiDB server host. |
| `port` | integer | no | `4000` | TiDB server default, not MySQL's `3306`. |
| `username` | string | yes | — | TiDB user. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | — | Initial database. |
| `charset` | string | no | `utf8mb4` | Connection character set. |
| `autocommit` | boolean | no | `true` | Enables autocommit. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## TLS and TiDB behavior

The current adapter profile does not support TLS fields. TiDB Cloud endpoints that require TLS are therefore not supported by this adapter version; adding MySQL SSL keys will fail profile validation rather than enabling TLS.

Objects use `database.table`. The adapter understands TiDB/TiFlash metadata, but generated or hand-written SQL must still follow TiDB limitations—for example, TiDB does not support `FULL OUTER JOIN`.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource tidb_cluster
```

Run `/databases` and `/tables`. If TiFlash is relevant, query `information_schema.TIFLASH_REPLICA` with an account that can read it.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Connection is refused on `3306` | TiDB defaults to port `4000`; use the actual TiDB server/listener port. |
| TLS-required cloud endpoint fails | This adapter version has no TLS profile fields; use an endpoint that permits the supported connection mode. |
| MySQL-compatible SQL still fails | Check TiDB's own unsupported syntax and version-specific behavior. |

## Reference

- [TiDB client connection](https://docs.pingcap.com/tidb/stable/connect-to-tidb/)
