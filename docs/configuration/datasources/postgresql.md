# PostgreSQL datasource

The PostgreSQL adapter connects through the PostgreSQL wire protocol with psycopg2. Datus preserves the native `database → schema → table` hierarchy and can discover tables, views, and materialized views.

## Install

```bash
pip install datus-postgresql
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      warehouse_pg:
        type: postgresql
        host: ${POSTGRES_HOST}
        port: 5432
        username: ${POSTGRES_USER}
        password: ${POSTGRES_PASSWORD}
        database: analytics
        schema: public
        sslmode: verify-full
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | PostgreSQL server host. |
| `port` | integer | no | `5432` | PostgreSQL wire-protocol port. |
| `username` | string | yes | — | Login role. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | `postgres` at connection time | Initial database. |
| `schema` | string | no | `public` | Initial schema; accepted as the alias for the adapter's `schema_name`. |
| `sslmode` | string | no | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full`. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout. |

## TLS and namespaces

`sslmode` is passed to psycopg2/libpq. The adapter default is `prefer`; the credentialed baseline above uses `verify-full` to encrypt the connection, validate the CA chain, and verify that `host` matches the server certificate. The profile does not expose an `sslrootcert` field, so configure a trusted CA through libpq's standard certificate locations or environment and use a matching hostname. Do not add unsupported certificate keys to the datasource profile.

The configured `database` is the initial PostgreSQL database and `schema` defaults to `public`. Datus filters system databases/schemas during normal discovery. Access to additional databases still depends on network reachability and role permissions.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource warehouse_pg
```

Run `/databases`, `/schemas`, and `/tables`. This distinguishes a working login from missing catalog grants.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `password authentication failed` | Check the role/password and matching `pg_hba.conf` rule. |
| `no pg_hba.conf entry` | Permit the Datus client address and choose the intended TLS/auth rule. |
| Certificate verification fails | Use a valid `sslmode` and configure libpq's trusted CA through its standard environment/files; the adapter profile has no `sslrootcert` key. |
| Relation not found | Check `database`, `schema`, `search_path`, identifier case, and grants. |

## Reference

- [PostgreSQL libpq SSL support](https://www.postgresql.org/docs/current/libpq-ssl.html)
