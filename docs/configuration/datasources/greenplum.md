# Greenplum datasource

Greenplum uses the PostgreSQL wire protocol and the same connection fields as PostgreSQL. Its adapter adds Greenplum-aware system-schema filtering, distribution-policy metadata, and storage details.

## Install

```bash
pip install datus-greenplum
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      greenplum_warehouse:
        type: greenplum
        host: ${GREENPLUM_HOST}
        port: 5432
        username: ${GREENPLUM_USER}
        password: ${GREENPLUM_PASSWORD}
        database: analytics
        schema: public
        sslmode: prefer
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | Greenplum coordinator host. |
| `port` | integer | no | `5432` | PostgreSQL-compatible port. |
| `username` | string | yes | — | Login role. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | `postgres` at connection time | Initial database. |
| `schema` | string | no | `public` | Initial schema. |
| `sslmode` | string | no | `prefer` | PostgreSQL/libpq SSL mode. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout. |

## Greenplum behavior

Objects use `database.schema.table`. Datus reads Greenplum distribution policy and storage metadata when available and removes Greenplum internal schemas from normal discovery. The connection profile intentionally matches [PostgreSQL](postgresql.md); the adapter does not add coordinator/segment management fields.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource greenplum_warehouse
```

Run `/schemas` and `/tables`, then inspect a distributed table to confirm the role can read Greenplum catalogs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Login reaches the wrong service | Confirm the coordinator host/port rather than a segment endpoint. |
| Distribution metadata is missing | Grant the login role access to the relevant Greenplum catalog views. |
| TLS or authentication fails | Apply the same libpq/`pg_hba.conf` checks as PostgreSQL. |

## Reference

- [Greenplum documentation](https://greenplum.org/)
- [PostgreSQL datasource](postgresql.md)
