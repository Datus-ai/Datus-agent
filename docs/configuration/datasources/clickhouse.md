# ClickHouse datasource

The ClickHouse adapter connects through ClickHouse's HTTP interface and exposes databases directly; ClickHouse has no separate schema layer.

## Install

```bash
pip install datus-clickhouse
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      clickhouse_prod:
        type: clickhouse
        host: ${CLICKHOUSE_HOST:-localhost}
        port: 8123
        username: ${CLICKHOUSE_USER}
        password: ${CLICKHOUSE_PASSWORD}
        database: analytics
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `localhost` | ClickHouse server host. |
| `port` | integer | no | `8123` | HTTP interface port. |
| `username` | string | yes | — | ClickHouse user. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | — | Initial database. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Protocol and namespace

Connect to the HTTP endpoint (`8123` by default), not the native TCP port (`9000` by default). Objects use `database.table`; do not add a `schema` field.

The adapter profile currently has no TLS/HTTPS flag. Place a compatible HTTP endpoint or trusted proxy in front of ClickHouse when transport requirements differ.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource clickhouse_prod
```

Run `/databases`, `/tables`, and a small `SELECT 1` query.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Protocol/handshake error on port `9000` | Use the HTTP interface, normally port `8123`. |
| Unknown field `schema` | ClickHouse databases serve as the namespace; remove `schema`. |
| Authentication fails | Check the ClickHouse user, password, allowed networks, and database grants. |

## Reference

- [ClickHouse HTTP interface](https://clickhouse.com/docs/interfaces/http)
