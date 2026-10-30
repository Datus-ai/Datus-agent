# Amazon Aurora datasources

Aurora MySQL uses the [MySQL adapter](mysql.md) (`type: mysql`); Aurora PostgreSQL uses the [PostgreSQL adapter](postgresql.md) (`type: postgresql`). Configure `aurora: true` to enable verified TLS and writer-endpoint checks. Connections use the normal MySQL/PostgreSQL database protocol, with the corresponding SQL dialect and metadata hierarchy.

## Install

Install the adapter for your engine into the same Python environment as `datus`. The following published versions support the documented Aurora profiles and compatible shared credential handling:

```bash
# Aurora MySQL
pip install --upgrade 'datus-mysql[aws]>=0.1.10' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'

# Aurora PostgreSQL
pip install --upgrade 'datus-postgresql[aws]>=0.1.11' \
  'datus-sqlalchemy>=0.1.12' 'datus-db-core>=0.1.10' 'SQLAlchemy==2.0.23'
```

The `aws` extra installs the SDK needed for IAM and Secrets Manager; omit it for password-only access. Restart Datus after upgrading. If launching through uv from a checkout, use `uv run --no-sync` after installing the adapters so synchronization does not restore older shared packages.

## Connection profiles

### Aurora MySQL

```yaml
agent:
  services:
    datasources:
      aurora_mysql:
        type: mysql
        host: ${AURORA_MYSQL_ENDPOINT}
        port: 3306
        username: ${AURORA_DB_USER}
        database: analytics
        aurora: true
        auth_mode: iam
        aws:
          region: us-east-1
          # profile: analytics
        ssl_ca: /absolute/path/rds-ca-bundle.pem
        ssl_verify_cert: true
        ssl_verify_identity: true
        connect_timeout_seconds: 10
        read_timeout_seconds: 60
        write_timeout_seconds: 60
```

### Aurora PostgreSQL

```yaml
agent:
  services:
    datasources:
      aurora_postgresql:
        type: postgresql
        host: ${AURORA_POSTGRESQL_ENDPOINT}
        port: 5432
        username: ${AURORA_DB_USER}
        database: analytics
        schema: public
        aurora: true
        auth_mode: iam
        aws:
          region: us-east-1
          # profile: analytics
        sslrootcert: /absolute/path/rds-ca-bundle.pem
        sslmode: verify-full
        connect_timeout_seconds: 10
```

Use the actual Aurora endpoint as `host` and allow the Datus runtime to reach its database TCP port. AWS API access alone does not establish database connectivity. Omit `aws.profile` when using the SDK default credential chain. The MySQL namespace is `database.table`; PostgreSQL uses `database → schema → table`.

### Password authentication

In either complete profile above, replace `auth_mode: iam` with the following settings. Keep `aurora: true` and the engine's database CA settings:

```yaml
auth_mode: password
password: ${AURORA_PASSWORD}
```

### Secrets Manager authentication

Use password mode with `secret_ref` instead of an inline password:

```yaml
auth_mode: password
secret_ref: ${AURORA_SECRET_ARN}
secret_cache_ttl_seconds: 300
```

The SecretString must be a JSON object with a string `password`. If it includes `username`, that value must match the configured database user. The runtime AWS identity needs `secretsmanager:GetSecretValue` and any required KMS access. The adapter reads the current Secret version after cache expiry and disposes old pools when the version changes; already checked-out work continues. Changing the database user or AWS identity requires replacing the connector, rather than rotating only the password.

## Parameters and TLS

Connection fields such as `host`, `database`, and `schema` follow the selected [MySQL](mysql.md) or [PostgreSQL](postgresql.md) adapter. Aurora authentication uses these additional fields:

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `aurora` | boolean | `false` | Require verified TLS for Aurora and enable checks on recognized cluster writer endpoints. |
| `auth_mode` | string | `password` | `password` or `iam`. IAM cannot include `password` or `secret_ref`. |
| `secret_ref` | string | — | Secrets Manager name/ARN in password mode; cannot include an inline password. |
| `secret_cache_ttl_seconds` | integer | `300` | Secret cache lifetime, from 1 to 86400 seconds. |
| `aws` | object | SDK defaults | `region`, `profile`, and `ca_bundle`; SDK providers own credentials and refresh. |
| `connect_timeout_seconds` | integer | — | Positive database connection timeout override. |

For Aurora MySQL, `ssl_ca` is required. `ssl_verify_cert` and `ssl_verify_identity` become `true`; explicitly disabling either is rejected. For Aurora PostgreSQL, `sslrootcert` is required and `sslmode` must be `verify-full` (selected automatically if omitted). Download the appropriate [RDS CA bundle](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.SSL.html), keep it readable by Datus, and use a certificate-matching endpoint. Database CA settings are separate from `aws.ca_bundle`, which controls AWS HTTPS API trust.

## IAM prerequisites

1. Enable IAM database authentication on the Aurora cluster.
2. Configure the database user with MySQL `AWSAuthenticationPlugin` or the PostgreSQL `rds_iam` role, and grant its required database/schema/table privileges.
3. Grant the runtime AWS identity `rds-db:connect` on the database-user resource. Its ARN uses the **cluster resource ID**, not the cluster name: `arn:aws:rds-db:<region>:<account-id>:dbuser:<DbClusterResourceId>/<db-user-name>`.
4. Use the original AWS endpoint for IAM signing, including its actual port and exact database username. An arbitrary DNS alias is not a substitute for the signing endpoint.

For cross-account access, select an SDK AssumeRole profile for a role in the database's account. The SDK manages temporary AWS credential refresh. IAM database tokens are generated for each new physical database connection; established sessions do not need to reconnect just because their login token expires.

## Timeouts and failover

MySQL `timeout_seconds` defaults to 30 seconds for pool and driver waits; `connect_timeout_seconds`, `read_timeout_seconds`, and `write_timeout_seconds` override their respective bounds. PostgreSQL `timeout_seconds` supplies pool, socket-operation, and query deadlines; `connect_timeout_seconds`, `read_timeout_seconds`, and `query_timeout_seconds` override their respective settings. These are distinct controls; a client/socket timeout is not proof of server-side query cancellation.

For Serverless instances that auto-pause, allow sufficient connection time for resume. For canonical Aurora cluster writer endpoints, the adapter checks the server role when a connection is checked out, preventing a demoted writer from serving application SQL through the writer pool. Reader, instance, and custom endpoints retain their own role semantics. Recover from DNS or connection failures with a new explicit request; SQL and transactions are not automatically replayed.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource aurora_mysql
# Or select the PostgreSQL profile:
datus --config conf/agent.yml --datasource aurora_postgresql
```

Run `/databases` and `/tables`; for PostgreSQL also check `/schemas`. Execute `SELECT 1 AS connection_ok;` to verify SQL access. Successful AWS login, database login, and table privileges are separate checks.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `aurora`, TLS, or AWS fields are rejected | Upgrade the adapter and shared packages listed above in the Agent environment, then restart Datus. |
| IAM login fails | Check cluster IAM support, the database user's IAM setup, `rds-db:connect` resource ID/username, region, endpoint, and runtime AWS identity. |
| Authentication options conflict | Choose IAM, inline password, or Secret-backed password; do not combine them. |
| Certificate verification fails | Use the engine-specific CA field and the original certificate-matching endpoint. |
| Secret login fails after rotation | Check Secret JSON fields, username, AWS/KMS permissions, and the configured cache lifetime. |
| AWS API works but database connection times out | Check VPC routing, security groups, the database TCP port, and any VPN/tunnel separately. |
| Writer connection is rejected after failover | Let endpoint DNS converge and retry explicitly; do not replay an uncertain transaction automatically. |

## Reference

- [Adapter Aurora guide](https://github.com/Datus-ai/datus-db-adapters/blob/main/docs/aurora.md)
- [IAM database user setup](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.IAMDBAuth.DBAccounts.html)
- [IAM database access policy](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.IAMDBAuth.IAMPolicy.html)
- [Aurora TLS and CA certificates](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.SSL.html)
