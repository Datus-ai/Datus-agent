# Huawei Cloud GaussDB(DWS) datasource

The DWS adapter connects to Huawei Cloud GaussDB(DWS) over its PostgreSQL-compatible protocol and handles DWS compatibility modes, native table DDL, and certificate constraints.

## Install

```bash
pip install datus-dws
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      dws_analytics:
        type: dws
        host: ${DWS_HOST}
        port: 8000
        username: ${DWS_USER}
        password: ${DWS_PASSWORD}
        database: gaussdb
        schema: public
        sslmode: verify-ca
        sslrootcert: /etc/datus/certs/dws-cacert.pem
        timeout_seconds: 30
```

The console's `hostname:port` value may be used directly as `host`; omit `port` in that form.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | yes | — | Coordinator endpoint, optionally including the port. |
| `port` | integer | no | `8000` | Coordinator port, 1–65535; must match an embedded port. |
| `username` | string | yes | — | DWS database user. |
| `password` | string | no | empty | Login password. |
| `database` | string | yes | — | DWS database; clusters commonly default to `gaussdb`. |
| `schema` | string | no | `public` | Initial schema. |
| `sslmode` | string | no | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full`. |
| `sslrootcert` | string | no* | — | CA path or inline PEM. Supply it for `verify-ca` unless libpq's standard CA location is configured. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout; must be positive. |

## TLS

Use `verify-ca` with `v2/sslcert/cacert.pem` from the console's `dws_ssl_cert` bundle. The v1 CA does not match the server certificate issuer. `sslrootcert` accepts a file path or inline PEM content.

!!! warning
    `verify-full` cannot succeed against the default DWS server certificate: it uses `CN=server` and has no `subjectAltName`, so it cannot match a real cluster endpoint. `verify-ca` validates the issuing CA but not the cluster hostname. Reach the cluster through a trusted VPC path or verified fixed endpoint.

`require` encrypts without authenticating the server; `prefer` upgrades when the cluster offers/enforces TLS but does not verify its identity; `disable` fails when the cluster enforces SSL.

## Compatibility modes

DWS databases can use `ORA`, `TD`, or `MySQL` compatibility mode. New clusters commonly default to `ORA`, where `7/2` evaluates to `3.5`, empty strings become `NULL`, concatenation absorbs `NULL`, and `DATE` maps to `timestamp(0)`. Confirm the mode before reusing PostgreSQL assumptions.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource dws_analytics
```

Run `/schemas` and `/tables`, then confirm the database compatibility mode and inspect a table DDL if migrations depend on distribution, partitioning, tablespace, or resource-group clauses.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Embedded and explicit ports conflict | Remove `port` or make it match the port in `host`. |
| `verify-ca` reports an issuer error | Use the v2 CA from the DWS certificate bundle, not v1. |
| `verify-full` reports hostname mismatch | This is expected with the default DWS certificate; use `verify-ca` plus a trusted network path. |
| SQL semantics differ from PostgreSQL | Check whether the database is in `ORA`, `TD`, or `MySQL` mode. |

## Reference

- [Huawei Cloud DWS SSL connection settings](https://support.huaweicloud.com/intl/en-us/mgtg-dws/dws_01_0038.html)
