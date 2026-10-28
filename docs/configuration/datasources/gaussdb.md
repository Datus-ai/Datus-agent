# GaussDB / openGauss datasource

The GaussDB adapter connects to GaussDB and openGauss over the PostgreSQL wire protocol while handling GaussDB authentication, compatibility modes, distributed-table metadata, and TLS behavior.

## Install

```bash
pip install datus-gaussdb
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      gaussdb_prod:
        type: gaussdb
        host: ${GAUSSDB_HOST}
        port: 5432
        username: ${GAUSSDB_USER}
        password: ${GAUSSDB_PASSWORD}
        database: postgres
        schema: public
        # driver: pg8000
        sslmode: verify-ca
        sslrootcert: /etc/datus/certs/gaussdb-ca.pem
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | GaussDB/openGauss host. |
| `port` | integer | no | `5432` | PostgreSQL-compatible port. |
| `username` | string | yes | — | Database user. |
| `password` | string | no | empty | Password; supported authentication depends on `driver`. |
| `database` | string | no | `postgres` at connection time | Initial database. |
| `schema` | string | no | `public` | Initial schema. |
| `driver` | string | no | platform-specific | `gaussdb`, `pg8000`, or `psycopg2`. |
| `sslmode` | string | no | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full`. |
| `sslrootcert` | string | no* | — | CA file path or inline PEM. Required by `pg8000` verification modes; libpq drivers may use their standard CA locations. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout. |

## Driver and authentication

| Driver | Default platform | Authentication | Use case |
|---|---|---|---|
| `gaussdb` | Linux | SHA-256, MD5, SM3 | Official driver; works with a stock server. |
| `pg8000` | macOS | SHA-256, MD5 | Pure Python; may be selected on any platform. |
| `psycopg2` | none | MD5 only | Compatibility escape hatch. |

The official driver has no macOS build, so macOS selects `pg8000`. The `psycopg2` path requires both an MD5 `pg_hba.conf` rule and a role password stored with GaussDB's MD5-compatible setting; changing the server setting does not re-encode an existing password.

## TLS

| `sslmode` | Encryption | Certificate verification |
|---|---|---|
| `disable` | off | none |
| `allow` | after a plaintext failure | none |
| `prefer` | preferred, plaintext fallback allowed | none |
| `require` | required | none; with `pg8000`, a supplied `sslrootcert` enables CA verification |
| `verify-ca` | required | CA chain against `sslrootcert` |
| `verify-full` | required | CA chain and hostname |

Use `verify-ca` as the production baseline; use `verify-full` when the configured hostname matches the certificate. `sslrootcert` accepts a path or inline PEM. The adapter supports server verification only and does not expose client certificate fields for mutual TLS. The `pg8000` driver treats `allow` like `prefer` because it cannot express libpq's plaintext-first retry order.

## Compatibility modes

GaussDB databases may use `A` (Oracle), `B` (MySQL), or `PG` compatibility mode. The adapter probes the mode and distributed/centralized shape at runtime. In `A` mode, empty strings become `NULL`; in distributed deployments, table DDL includes reconstructed `DISTRIBUTE BY` clauses.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource gaussdb_prod
```

Run `/schemas` and `/tables`, then confirm the detected compatibility mode before relying on empty-string, boolean, or arithmetic semantics.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| SHA-256 authentication fails with `psycopg2` | Use `gaussdb`/`pg8000`, or reconfigure the role for MD5 correctly. |
| Official driver is unavailable on macOS | Omit `driver` to use the macOS default, or set `driver: pg8000`. |
| `verify-ca`/`verify-full` fails | Supply the issuing CA through `sslrootcert`; for `verify-full`, also use the certificate's hostname. |
| Queries differ from PostgreSQL | Check the database's `A`/`B`/`PG` compatibility mode. |

## Reference

- [openGauss client connection security](https://docs.opengauss.org/en/docs/latest/docs/DatabaseAdministrationGuide/configuring-client-connection-security.html)
