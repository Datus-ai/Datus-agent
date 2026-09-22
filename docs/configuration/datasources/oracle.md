# Oracle datasource

The Oracle adapter uses `python-oracledb` Thin mode, so no Oracle Client installation is required. It generates Oracle Database 19c-compatible SQL and connects to Oracle Database 12.1 or later.

## Install

```bash
pip install datus-oracle
```

## Connection profile

Use a service/PDB name for modern deployments:

```yaml
agent:
  services:
    datasources:
      oracle_prod:
        type: oracle
        host: ${ORACLE_HOST:-127.0.0.1}
        port: 1521
        username: ${ORACLE_USER}
        password: ${ORACLE_PASSWORD}
        service_name: FREEPDB1
        schema: ANALYTICS
        timeout_seconds: 30
```

For legacy databases, replace `service_name` with `sid`. For a TNS alias or full connect descriptor, use `dsn` instead.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | Listener host; used with `service_name` or `sid`. |
| `port` | integer | no | `1521` | Listener port. |
| `username` | string | yes | — | Oracle user. |
| `password` | string | no | empty | Login password. |
| `service_name` | string | exactly one | — | Recommended service/PDB target. |
| `sid` | string | exactly one | — | Legacy SID target. |
| `dsn` | string | exactly one | — | TNS alias or full connect descriptor. |
| `database` | string | no | — | Compatibility alias for `service_name`; prefer the explicit field. |
| `schema` | string | no | connecting user | Default object namespace. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Connection target and namespace

Exactly one of `service_name`, `sid`, and `dsn` must be set. The service/PDB selects the connection target but is not part of SQL identifiers; Oracle objects are addressed as `SCHEMA.TABLE`. Datus defaults the schema to the connecting user's uppercase name.

The adapter runs in Thin mode. Do not add Thick-mode client-library fields to the profile.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource oracle_prod
```

The connection test executes Oracle's `SELECT 1 FROM DUAL`; then run `/schemas` and `/tables` to verify data-dictionary grants.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Connection-target validation fails | Keep exactly one of `service_name`, `sid`, and `dsn`. |
| `ORA-12514` or `DPY-6001` | The listener does not know the configured service; check service registration and spelling. |
| Tables are missing | Set `schema` to the owning schema and grant access to the user; unquoted Oracle names normally appear uppercase. |

## Reference

- [python-oracledb connection handling](https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html)
