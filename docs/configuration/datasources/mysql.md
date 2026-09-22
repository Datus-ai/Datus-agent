# MySQL datasource

The MySQL adapter connects through PyMySQL and exposes MySQL databases as Datus databases. There is no separate schema level.

## Install

```bash
pip install datus-mysql
```

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
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | MySQL server host. |
| `port` | integer | no | `3306` | MySQL protocol port. |
| `username` | string | yes | — | Login user. |
| `password` | string | no | empty | Login password; use an environment variable. |
| `database` | string | no | — | Initial/default database. |
| `charset` | string | no | `utf8mb4` | Connection character set. |
| `autocommit` | boolean | no | `true` | Enables autocommit on the connection. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Namespace and authentication

Objects are addressed as `database.table`. The adapter requires a username; an empty password is valid for servers that permit it. The current profile schema does not expose MySQL TLS certificate options, so unknown keys such as `ssl_ca` are rejected.

For metadata discovery, grant the configured user access to every database Datus should inspect. Restrict the account to read-only SQL privileges when Datus must not modify data.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource mysql_prod
```

Run `/databases` and `/tables` to verify both connectivity and metadata permissions.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Access denied for user` | Check `username`, `password`, allowed source host, and MySQL grants. |
| Connection refused or timed out | Verify `host`, port `3306`, bind address, firewall, and container port mapping. |
| Tables from another database are missing | Grant access to that database or set `database` to the intended initial namespace. |

## Reference

- [MySQL documentation](https://dev.mysql.com/doc/)
