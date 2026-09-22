# Hologres datasource

The Hologres adapter uses the PostgreSQL wire protocol while preserving Hologres-specific namespaces, foreign tables, table properties, and SQL rules.

## Install

```bash
pip install datus-hologres
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      hologres_prod:
        type: hologres
        host: ${HOLOGRES_ENDPOINT}
        port: 80
        username: ${HOLOGRES_ACCESS_KEY_ID}
        password: ${HOLOGRES_ACCESS_KEY_SECRET}
        database: ${HOLOGRES_DATABASE}
        schema: public
        sslmode: require
        timeout_seconds: 30
```

`host` may be a hostname or the console's `hostname:port` value. Omit `port` when it is already embedded.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | yes | — | Endpoint hostname, optionally including its port. |
| `port` | integer | no | `80` | Endpoint port, 1–65535. Must match an embedded port. |
| `username` | string | yes | — | AccessKey ID; `access_key_id` is also accepted. |
| `password` | string | yes | — | AccessKey secret; `access_key_secret` is also accepted. |
| `database` | string | yes | — | Hologres database. |
| `schema` | string | no | `public` | Initial schema. |
| `sslmode` | string | no | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full`. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout; must be positive. |

## Endpoint, TLS, and namespaces

Do not include a URI scheme, credentials, path, query, or fragment in `host`. When both `host` and `port` carry a port, they must agree. Hologres public endpoints commonly use port `80`; use the value shown for your instance.

The adapter default is `sslmode: prefer`; the credentialed baseline above uses `require` so it cannot silently fall back to plaintext. The adapter exposes `sslmode` but not a custom `sslrootcert` profile field. Objects can be addressed as `table`, `schema.table`, or `database.schema.table`; Hologres internal schemas are filtered from normal discovery.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource hologres_prod
```

Run `/schemas` and `/tables`; include a known foreign table if your project uses external storage.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Endpoint port conflicts with explicit port | Remove `port` or make it equal to the port embedded in `host`. |
| Host validation rejects the endpoint | Remove `http://`, `https://`, paths, and user information. |
| Login is rejected | Check the AccessKey pair, database, endpoint type, and instance network allowlist. |

## Reference

- [Hologres client connections](https://www.alibabacloud.com/help/en/hologres/user-guide/connect-to-hologres)
