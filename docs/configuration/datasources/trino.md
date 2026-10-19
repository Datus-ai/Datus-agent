# Trino datasource

The Trino adapter connects over Trino's HTTP protocol and preserves the three-level `catalog → schema → table` hierarchy.

## Install

```bash
pip install datus-trino
```

The CA bundle setting below targets the upcoming `datus-trino>=0.1.4`, with `datus-sqlalchemy>=0.1.10` and `datus-db-core>=0.1.8`. These versions are not yet published; use a source checkout containing CMP-TLS-03 until release. Existing HTTPS profiles without `ca_cert` remain supported.

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
| `verify` | boolean | no | `true` | Verify the HTTPS certificate chain and hostname. |
| `ca_cert` | string | no | — | Readable PEM CA bundle file; requires HTTPS and `verify: true` (from 0.1.4). |
| `timeout_seconds` | integer | no | `30` | Connection timeout; must be positive. |

## TLS and namespaces

For TLS, set `http_scheme: https` and keep `verify: true` in production. For a private CA, add `ca_cert: /absolute/path/trino-ca.pem` to the profile above. Omit it to use the Python client's Requests trust configuration. `verify` stays a boolean; do not put a file path in that field.

`ca_cert` accepts a readable PEM bundle file. `~` expands to the user's home and relative paths resolve from the process working directory. Invalid files, HTTP plus CA, and `verify: false` plus CA are rejected before connecting. Inline PEM and client certificates are outside this setting's scope. `verify: false` disables chain and hostname checks and is intended only for temporary development. HTTPS failures do not retry over HTTP. Ports do not select the protocol; set both `port` and `http_scheme` explicitly for TLS.

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
| HTTPS certificate fails | Check the endpoint hostname and issuing CA; on 0.1.4+, supply a private CA via `ca_cert`. Keep `verify: true` in production. |
| Catalog or schema is missing | Check the exact names, connector configuration on the coordinator, and Trino access-control grants. |

## Reference

- [Trino TLS and HTTPS](https://trino.io/docs/current/security/tls.html)
- [Trino Python client TLS configuration](https://github.com/trinodb/trino-python-client#ssl)
- [Trino client protocol](https://trino.io/docs/current/develop/client-protocol.html)
