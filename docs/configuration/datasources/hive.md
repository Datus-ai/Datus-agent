# Hive datasource

The Hive adapter connects to HiveServer2 through PyHive/Thrift. Hive databases are the table namespace, and optional session properties are passed through `configuration`.

## Install

```bash
pip install datus-hive
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      hive_warehouse:
        type: hive
        host: ${HIVE_HOST:-127.0.0.1}
        port: 10000
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        database: default
        auth: LDAP
        configuration:
          hive.execution.engine: tez
          hive.vectorized.execution.enabled: true
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | HiveServer2 host. |
| `port` | integer | no | `10000` | HiveServer2 Thrift port. |
| `username` | string | yes | — | Hive user. |
| `password` | string | no | empty | Used by password-based auth. |
| `database` | string | no | `default` at connection time | Initial Hive database. |
| `auth` | string | no | driver default | Supported modes include `NONE`, `LDAP`, `CUSTOM`, and `KERBEROS`. |
| `configuration` | mapping | no | `{}` | Hive session properties; values are normalized to strings. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication and sessions

Use `LDAP` or `CUSTOM` with a password as required by the HiveServer2 deployment. Kerberos still requires the surrounding Kerberos client/ticket configuration; those settings are not represented by extra datasource fields.

`configuration` is sent as Hive session configuration. YAML booleans become `true`/`false` strings, other scalar values are stringified, and `null` becomes an empty string.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource hive_warehouse
```

Run `/databases` and `/tables`, then query a small table to verify HiveServer2 and backing metastore access.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Thrift connection fails | Check HiveServer2 host/port, service health, and network routing. |
| SASL/authentication error | Match `auth`, username/password, and the server's HiveServer2 auth mode. |
| Session option has no effect | Use the exact Hive property key under `configuration` and confirm the server permits overrides. |

## Reference

- [HiveServer2 clients](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients)
