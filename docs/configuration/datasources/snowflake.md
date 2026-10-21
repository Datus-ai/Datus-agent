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
agent:
  services:
    datasources:
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
| `username` | string | conditional | — | Snowflake user; may instead come from a Secret mapping. |
| `auth_mode` | string | with `secret_ref` | — | `key_pair` or `password`; static profiles keep their existing credential selection. |
| `secret_ref` | string | no | — | AWS Secrets Manager Secret name or full ARN. |
| `secret_key_mapping` | object | no | — | Top-level JSON keys for `username`, `private_key`, `private_key_file_pwd`, or `password`. |
| `secret_cache_ttl_seconds` | integer | no | `300` | Credential cache lifetime, from 1 to 300 seconds. |
| `aws` | object | no | SDK defaults | Optional `region`, `profile`, and AWS HTTPS `ca_bundle`. |
| `password` | secret string | conditional | — | Password auth; mutually exclusive with `private_key_file` when `private_key` is absent. |
| `private_key` | secret string | conditional | — | Inline PEM; takes precedence over the other credentials. |
| `private_key_file` | string | conditional | — | Path to a PEM-encoded RSA private key. |
| `private_key_file_pwd` | secret string | no | — | Passphrase for an encrypted private key. |
| `warehouse` | string | yes | — | Compute warehouse. |
| `database` | string | no | — | Initial database. |
| `schema` | string | no | — | Initial schema. |
| `role` | string | no | — | Role activated for the session. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## AWS Secrets Manager credentials

This requires `datus-snowflake` 0.1.10 or later and its optional AWS dependency:

```bash
pip install 'datus-snowflake[aws]>=0.1.10'
```

Store a JSON `SecretString`, such as `{"sf_user":"READER","sf_key":"<PEM private key>","sf_passphrase":"<key passphrase>"}`, then map its top-level keys:

```yaml
agent:
  services:
    datasources:
      snowflake_prod:
        type: snowflake
        account: ${SNOWFLAKE_ACCOUNT}
        warehouse: ${SNOWFLAKE_WAREHOUSE}
        database: ANALYTICS
        schema: PUBLIC
        role: READER
        auth_mode: key_pair
        secret_ref: ${SNOWFLAKE_SECRET_ARN}
        secret_cache_ttl_seconds: 300
        secret_key_mapping:
          username: sf_user
          private_key: sf_key
          private_key_file_pwd: sf_passphrase
        aws:
          region: us-east-1
          profile: datus-prod
```

Omit the passphrase mapping for an unencrypted PEM. Unrelated Secret fields are ignored. Mapped values must be nonempty strings; mappings select literal top-level JSON keys. Account, warehouse, role and namespace always come from the connection profile. A configured username must match a mapped username. Changing the effective username requires replacing the connector and refreshing metadata cached under the old identity.

Without a mapping (including an all-blank mapping), the entire `SecretString` is the selected credential. For example, use `auth_mode: key_pair`, `username: READER`, and `secret_ref` for a raw unencrypted PEM. For a raw password use `auth_mode: password`; for JSON passwords map `password` to its key. Password authentication remains subject to the Snowflake user's MFA policy. Do not combine `secret_ref` with inline passwords, private keys, key files or passphrases. `SecretBinary` is unsupported.

Credentials are read when a physical connection is created. Open Snowflake sessions continue normally after the cache expires; there is no background polling or periodic session termination. A subsequent login reads `AWSCURRENT` after expiry. Keep the old key valid for at least the cache window while publishing and validating its replacement. Revoking it earlier can make a new login fail while the old credential is cached. SDK callers can use `connector.refresh_credentials()` to update the cache for the next login; this leaves the current connection open. `close()` followed by `connect()` establishes a new session with the current cache. Reconnection restores the managed database/schema and configured role/warehouse; it does not restore temporary objects or arbitrary session settings. Business SQL is never replayed automatically.

[AWS Secrets Manager changes can take time to propagate](https://docs.aws.amazon.com/secretsmanager/latest/userguide/troubleshoot.html#troubleshoot_general_eventual-consistency). Validate that the replacement is visible as `AWSCURRENT` before retiring the old key; the 300-second limit applies to the adapter cache, rather than total AWS propagation time.

An unexpectedly closed SDK session returns a connection error. Call `connect()` explicitly to start a fresh session before submitting more SQL; the adapter does not silently continue a lost transaction on a new session.

AWS authentication uses the SDK credential chain. For cross-account access, configure the role and External ID in the shared AWS config, then select the profile in the datasource:

```ini
[profile datus-prod]
role_arn = arn:aws:iam::123456789012:role/DatusSecretReader
source_profile = default
external_id = customer-external-id
region = us-east-1
```

The caller needs permission to assume the role; the role trust policy must accept that caller and External ID. The reading identity needs `secretsmanager:GetSecretValue` for the Secret, plus `kms:Decrypt` when it uses a customer-managed KMS key. The SDK handles temporary AWS credential renewal. `aws.ca_bundle` applies to AWS HTTPS APIs. No OIDC issuer or Snowflake OAuth flow is added by this credential source.

## Authentication, TLS, and namespaces

For static credentials, configure `private_key`, or—when it is absent—exactly one of `password` and `private_key_file`. `private_key` wins if other credential fields are also present. Key-pair auth uses Snowflake JWT and is the practical choice for MFA-enabled users and CI.

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
- [AWS SDK credential profiles](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)
- [AWS GetSecretValue permissions and version stages](https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html)
