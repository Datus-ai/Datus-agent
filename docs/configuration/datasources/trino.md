# Trino datasource

The Trino adapter connects over Trino's HTTP protocol and preserves the three-level `catalog → schema → table` hierarchy.

## Install

```bash
pip install datus-trino
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      trino_prod:
        type: trino
        host: ${TRINO_HOST}
        port: 8443
        username: ${TRINO_USER}
        password: ${TRINO_PASSWORD}
        catalog: hive
        schema_name: analytics
        http_scheme: https
        verify: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | Trino coordinator host. |
| `port` | integer | no | `8080` | HTTP/HTTPS port, 1–65535. |
| `username` | string | yes | — | Trino user; blank values are rejected. |
| `password` | string | no | empty | Password used by the Trino client. |
| `catalog` | string | no | `hive` | Initial catalog. |
| `schema_name` | string | no | `default` | Initial schema. Use this exact key, not `schema`. |
| `http_scheme` | string | no | `http` | `http` or `https`. |
| `verify` | boolean | no | `true` | Verifies HTTPS certificates. |
| `timeout_seconds` | integer | no | `30` | Connection timeout; must be positive. |

## TLS and namespaces

For TLS, set `http_scheme: https` and keep `verify: true` in production. The adapter currently exposes a boolean verification switch, not a custom CA bundle path.

Trino profiles must use `schema_name`; unlike several SQL adapters, this model does not define `schema` as an alias. Catalogs can be switched at runtime, subject to Trino access-control rules.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource trino_prod
```

Run `/databases` and `/tables`; Datus maps Trino schemas into its database listing while retaining the catalog context.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Unknown field `schema` | Rename it to `schema_name`. |
| HTTPS certificate fails | Use a publicly/system-trusted certificate or terminate TLS at a trusted proxy; avoid `verify: false` in production. |
| Catalog or schema is missing | Check the exact names, connector configuration on the coordinator, and Trino access-control grants. |

## Reference

- [Trino client protocol](https://trino.io/docs/current/develop/client-protocol.html)
