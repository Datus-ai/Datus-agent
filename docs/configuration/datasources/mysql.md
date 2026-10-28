# MySQL datasource

The MySQL adapter connects through PyMySQL and exposes MySQL databases as Datus databases. There is no separate schema level.

## Install

```bash
pip install --upgrade 'datus-mysql>=0.1.10' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'
```

These published versions include the TLS and AWS authentication fields below. For Aurora password, IAM, and Secrets Manager profiles, follow the [Aurora guide](aurora.md); IAM and Secrets Manager need the adapter's optional `aws` extra. Install into the Agent environment and restart Datus after upgrading shared packages.

## Connection profile

```yaml
agent:
  services:
    datasources:
      mysql_prod:
        type: mysql
        host: ${MYSQL_HOST:-127.0.0.1}
        port: 3306
        username: ${MYSQL_USER}
        password: ${MYSQL_PASSWORD}
        database: analytics
        charset: utf8mb4
        autocommit: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
| --- | --- | --: | --- | --- |
| `host` | string | no | `127.0.0.1` | MySQL server host. |
| `port` | integer | no | `3306` | MySQL protocol port. |
| `username` | string | yes | — | Login user. |
| `password` | string | no | empty | Login password; use an environment variable. |
| `database` | string | no | — | Initial/default database. |
| `charset` | string | no | `utf8mb4` | Connection character set. |
| `autocommit` | boolean | no | `true` | Enables autocommit on the connection. |
| `timeout_seconds` | integer | no | `30` | Pool checkout and default driver connect/read/write timeout. |
| `connect_timeout_seconds` | integer | no | — | Positive connection-timeout override. |
| `read_timeout_seconds` / `write_timeout_seconds` | integer | no | — | Positive socket read/write timeout overrides. |
| `query_timeout_seconds` | integer | no | — | Optional positive server-side SELECT deadline; otherwise preserve the server setting. |
| `ssl_ca` | string | no | — | Database CA file; required for Aurora/IAM. |
| `ssl_cert` / `ssl_key` | string | no | — | Client certificate and key; supply both together. |
| `ssl_verify_cert` / `ssl_verify_identity` | boolean | no | unset | Certificate and hostname checks; require `ssl_ca` when enabled. Aurora/IAM enables both and rejects explicit `false`. |
| `aurora` | boolean | no | `false` | Require verified TLS and check recognized Aurora cluster writer endpoints. |
| `auth_mode` | string | no | `password` | `password` or AWS RDS `iam`. |
| `aws` | object | no | SDK defaults | `region`, `profile`, and AWS HTTPS `ca_bundle`. |
| `secret_ref` | string | no | — | Secrets Manager name/ARN in password mode; cannot include an inline password or IAM. |
| `secret_cache_ttl_seconds` | integer | no | `300` | Secret cache lifetime, 1–86400 seconds. |

## Namespace and authentication

Objects are addressed as `database.table`. The adapter requires a username; an empty password is valid for ordinary password-authenticated servers that permit it. For verified TLS, add these settings to the datasource:

```yaml
ssl_ca: /absolute/path/mysql-ca.pem
ssl_verify_cert: true
ssl_verify_identity: true
```

For mutual TLS, also supply both `ssl_cert` and `ssl_key`. MySQL uses file paths; `aws.ca_bundle` configures AWS API trust separately. [Aurora](aurora.md) uses the same `mysql` type with `aurora: true` and supports password, IAM, or Secret-backed password authentication. IAM cannot include `password` or `secret_ref`.

For metadata discovery, grant the configured user access to every database Datus should inspect. Restrict the account to read-only SQL privileges when Datus must not modify data.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource mysql_prod
```

Run `/databases` and `/tables` to verify both connectivity and metadata permissions.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Access denied for user` | Check `username`, `password`, allowed source host, and MySQL grants. |
| Connection refused or timed out | Verify `host`, port `3306`, bind address, firewall, and container port mapping. |
| Tables from another database are missing | Grant access to that database or set `database` to the intended initial namespace. |

## Reference

- [MySQL documentation](https://dev.mysql.com/doc/)
