# Snowflake datasource

The Snowflake adapter uses the native Snowflake Python connector. It supports password authentication and RSA key-pair authentication, with the native `database → schema → table` hierarchy.

## Install

```bash
pip install datus-snowflake
```

## Connection profile

Password authentication:

```yaml
agent:
  services:
    datasources:
      snowflake_prod:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        username: ${SNOWFLAKE_USER}
        password: ${SNOWFLAKE_PASSWORD}
        warehouse: ${SNOWFLAKE_WAREHOUSE}
        database: ANALYTICS
        schema: PUBLIC
        role: ANALYST
```

Key-pair authentication with an in-memory secret:

```yaml
snowflake_ci:
  type: snowflake
  account: ${SNOWFLAKE_ACCOUNT}
  username: ${SNOWFLAKE_USER}
  private_key: ${SNOWFLAKE_PRIVATE_KEY}
  warehouse: ${SNOWFLAKE_WAREHOUSE}
  database: ANALYTICS
  schema: PUBLIC
```

For a local key file, replace `private_key` with `private_key_file` and add `private_key_file_pwd` only when the PEM is encrypted.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `account` | string | yes | — | Snowflake account identifier. |
| `username` | string | yes | — | Snowflake user. |
| `password` | secret string | conditional | — | Password auth; mutually exclusive with `private_key_file` when `private_key` is absent. |
| `private_key` | secret string | conditional | — | Inline PEM; takes precedence over the other credentials. |
| `private_key_file` | string | conditional | — | Path to a PEM-encoded RSA private key. |
| `private_key_file_pwd` | secret string | no | — | Passphrase for an encrypted private key. |
| `warehouse` | string | yes | — | Compute warehouse. |
| `database` | string | no | — | Initial database. |
| `schema` | string | no | — | Initial schema. |
| `role` | string | no | — | Role activated for the session. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication, TLS, and namespaces

Configure `private_key`, or—when it is absent—exactly one of `password` and `private_key_file`. `private_key` wins if other credential fields are also present. Key-pair auth uses Snowflake JWT and is the practical choice for MFA-enabled users and CI.

`private_key` and `private_key_file` are RSA keys for Snowflake JWT user authentication; they are not TLS certificates. The current adapter profile does not expose a custom CA bundle or mutual-TLS client certificate. The native Snowflake connector manages HTTPS server-certificate validation.

Snowflake uses `database` and `schema`; do not add `catalog`. The login role must have `USAGE` on the warehouse, database, and schema plus privileges on the objects Datus should inspect or query.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource snowflake_prod
```

Run `/databases`, `/schemas`, and `/tables` to verify role activation and grants.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Credential validation says exactly one is required | Use `private_key`, or exactly one of `password`/`private_key_file`. |
| Private key cannot be loaded | Use PEM encoding and provide `private_key_file_pwd` only for an encrypted key. |
| Warehouse or objects are not visible | Check `role` and `USAGE`/object grants; object names may be case-sensitive when quoted. |

## Reference

- [Snowflake key-pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth)
