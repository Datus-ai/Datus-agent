# ClickHouse datasource

The ClickHouse adapter connects through ClickHouse's HTTP(S) interface and exposes databases directly; ClickHouse has no separate schema layer.

## Install

```bash
pip install datus-clickhouse
```

This installs the published adapter for the HTTP profile below. The HTTPS settings on this page target the upcoming `0.1.4` release; see the release requirements below.

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
| `port` | integer | no | `8123` / `8443` | Default follows `protocol`; explicit custom ports are preserved. |
| `username` | string | yes | — | ClickHouse user. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | — | Initial database. |
| `timeout_seconds` | integer | no | `30` | SQLAlchemy pool wait timeout, not a query deadline. |
| `protocol` | string | no | `http` | `http` or `https`; a port does not select the protocol. |
| `verify` | boolean | no | `true` | Verify the HTTPS certificate chain and hostname. |
| `ca_cert` | string | no | — | Readable PEM CA bundle file for verified HTTPS. |

## Protocol and namespace

Connect to the HTTP endpoint (`8123` by default), not the native TCP port (`9000` by default). Objects use `database.table`; do not add a `schema` field.

TLS requires `datus-clickhouse>=0.1.4` and its shared dependencies (`datus-sqlalchemy>=0.1.10`, `datus-db-core>=0.1.8`). These versions are not yet published on PyPI; use the following HTTPS configuration after they are released. Existing profiles keep HTTP.

```yaml
agent:
  services:
    datasources:
      clickhouse_prod:
        type: clickhouse
        host: clickhouse.example.com
        protocol: https
        port: 8443
        username: ${CLICKHOUSE_USER}
        password: ${CLICKHOUSE_PASSWORD}
        database: analytics
        verify: true
        ca_cert: /absolute/path/clickhouse-ca.pem
```

Omit `ca_cert` to use Requests' default trust configuration. For a private CA, provide a readable PEM bundle file; `~` expands to the user's home and relative paths resolve from the process working directory. Inline PEM and client certificates are not supported by these settings.

HTTPS verifies both certificate trust and hostname. `ca_cert` requires HTTPS with verification enabled; conflicting settings and invalid files fail configuration validation. Wrong CA/hostname errors fail the connection without a fallback retry over HTTP. `verify: false` explicitly disables verification for development diagnosis. Connect to the server's HTTPS endpoint (commonly 8443), not the native TLS port 9440.

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
| Certificate verification fails | Check the CA bundle and use the hostname listed in the certificate SAN. |
| HTTPS handshake fails on an HTTP port | Set `protocol` and the HTTP(S) endpoint port together. |
| Authentication fails | Check the ClickHouse user, password, allowed networks, and database grants. |

## Reference

- [ClickHouse HTTP interface](https://clickhouse.com/docs/interfaces/http)
