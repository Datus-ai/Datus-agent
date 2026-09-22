# Spark SQL datasource

The Spark adapter connects to Spark Thrift Server through HiveServer2/PyHive. Spark uses a two-level `database → table` hierarchy.

## Install

```bash
pip install datus-spark
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      spark_warehouse:
        type: spark
        host: ${SPARK_HOST:-127.0.0.1}
        port: 10000
        username: ${SPARK_USER}
        password: ${SPARK_PASSWORD}
        database: default
        auth_mechanism: LDAP
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | Spark Thrift Server host. |
| `port` | integer | no | `10000` | Thrift Server port. |
| `username` | string | yes | — | Spark/Thrift user. |
| `password` | string | no | empty | Password for the selected auth mode. |
| `database` | string | no | `default` at connection time | Initial Spark database. |
| `auth_mechanism` | string | no | `NONE` | `NONE`, `NOSASL`, `LDAP`, `KERBEROS`, or `CUSTOM`. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication and namespaces

`auth_mechanism` is passed to PyHive. Use `LDAP` or `CUSTOM` when the server requires password-based SASL. `PLAIN` is the underlying SASL mechanism, not a valid value for this profile.

Spark databases are namespaces; `schema` and `catalog` are not accepted profile fields. The adapter connects to Spark Thrift Server, not directly to a Spark driver process.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource spark_warehouse
```

Run `/databases` and `/tables`, then execute `SELECT 1` in SQL mode.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `auth_mechanism` validation fails | Use one of the five documented values; do not use `PLAIN`. |
| Connection hangs during startup | Check Spark Thrift Server health and whether port `10000` is reachable. |
| Database is empty or missing | Verify the Thrift Server catalog implementation and the user's permissions. |

## Reference

- [Spark Thrift JDBC/ODBC server](https://spark.apache.org/docs/latest/sql-distributed-sql-engine.html)
