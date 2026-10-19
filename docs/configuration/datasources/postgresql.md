# PostgreSQL datasource

The PostgreSQL adapter connects through the PostgreSQL wire protocol with psycopg2. Datus preserves the native `database → schema → table` hierarchy and can discover tables, views, and materialized views.

## Install

```bash
pip install datus-postgresql
```

Certificate and AWS authentication fields below target the upcoming `datus-postgresql>=0.1.10`, `datus-sqlalchemy>=0.1.10`, and `datus-db-core>=0.1.8`. These versions are not yet published. Until release, install the corresponding source packages; older published adapters only expose `sslmode` and rely on libpq certificate files/environment.

Before release, install all three compatible source packages into the Agent environment. The following pinned adapter revision includes the documented TLS/AWS fields. Replace the Python path with the environment used to run Datus:

```bash
git clone https://github.com/Datus-ai/datus-db-adapters.git
git -C datus-db-adapters checkout 66fadf1bffc1710c6b85e4149d6bb08d3458d464
uv pip install --python /path/to/Datus-agent/.venv/bin/python \
  -e ./datus-db-adapters/datus-db-core \
  -e ./datus-db-adapters/datus-sqlalchemy \
  -e './datus-db-adapters/datus-postgresql[aws]' \
  'SQLAlchemy==2.0.23'
```

The `aws` extra installs the SDK used by IAM/Secrets Manager; it is optional for ordinary password connections. SQLAlchemy remains pinned to the Agent's `2.0.23`. After source installation, use `uv run --no-sync` when launching through uv; a normal sync can restore the older core from the Agent lockfile.

## Connection profile

```yaml
agent:
  services:
    datasources:
      warehouse_pg:
        type: postgresql
        host: ${POSTGRES_HOST}
        port: 5432
        username: ${POSTGRES_USER}
        password: ${POSTGRES_PASSWORD}
        database: analytics
        schema: public
        sslmode: verify-full
        timeout_seconds: 30
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | PostgreSQL server host. |
| `port` | integer | no | `5432` | PostgreSQL wire-protocol port. |
| `username` | string | yes | — | Login role. |
| `password` | string | no | empty | Login password. |
| `database` | string | no | `postgres` at connection time | Initial database. |
| `schema` | string | no | `public` | Initial schema; accepted as the alias for the adapter's `schema_name`. |
| `sslmode` | string | no | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full`. |
| `sslrootcert` | string | no | — | CA bundle file path passed to libpq. Required for Aurora/IAM. |
| `sslcert` / `sslkey` | string | no | — | Client certificate and private key file paths; supply both together for mutual TLS. |
| `aurora` | boolean | no | `false` | Require a CA and `verify-full` for an Aurora endpoint. |
| `auth_mode` | string | no | `password` | `password` or AWS RDS `iam`. |
| `aws` | object | no | — | SDK identity options: `region`, `profile`, `ca_bundle` (AWS API trust only). |
| `secret_ref` | string | no | — | Secrets Manager name/ARN; mutually exclusive with inline password and IAM. |
| `secret_cache_ttl_seconds` | integer | no | `300` | Secret refresh interval, 1–86400 seconds. |
| `connect_timeout_seconds` | integer | no | — | Optional positive DBAPI connection timeout. |
| `timeout_seconds` | integer | no | `30` | Connection and pool timeout. |

## TLS and namespaces

`sslmode` is passed to psycopg2/libpq. The adapter default remains `prefer` for existing profiles; this can fall back to plaintext and does not provide verified server identity. New production profiles should explicitly select `verify-full`, use a certificate-matching hostname, and configure the issuing CA. On 0.1.10+, add:

```yaml
sslrootcert: /absolute/path/postgresql-ca.pem
# For mutual TLS, supply both client files:
# sslcert: /absolute/path/client.crt
# sslkey: /absolute/path/client.key
```

These fields accept file paths, not inline PEM. Without `sslrootcert`, ordinary PostgreSQL can use libpq's standard CA location/environment. Moving an old `prefer` profile to `verify-full` requires a trusted CA and matching hostname first; changing the adapter version alone does not migrate its policy.

### AWS authentication

Install `datus-postgresql[aws]>=0.1.10` after release (or the source package with its `aws` extra before release). Keep database TLS trust in `sslrootcert`; `aws.ca_bundle` only affects AWS HTTPS APIs. Use the SDK credential chain or an AWS shared profile, including SDK-managed role refresh; do not paste temporary AWS credentials into the datasource.

For RDS/Aurora IAM, omit `password` and `secret_ref` and add:

```yaml
aurora: true  # Only for Aurora endpoints
auth_mode: iam
aws:
  region: us-east-1
  profile: analytics
sslmode: verify-full
sslrootcert: /absolute/path/rds-ca-bundle.pem
connect_timeout_seconds: 10
```

Both `aurora: true` and `auth_mode: iam` require an explicit CA path. When omitted, `sslmode` becomes `verify-full`; explicitly selecting another mode is rejected. IAM tokens are generated for new physical connections. Password authentication can instead use `secret_ref` with `auth_mode: password`: the Secret must contain a JSON `password` string, and any `username` must match the configured role. Secret refresh follows the TTL and changed versions dispose old pools before new checkouts. Enable the matching server-side IAM login and AWS permissions separately.

[GaussDB](gaussdb.md) and [DWS](dws.md) use separate adapters and certificate constraints. Follow their datasource pages; the PostgreSQL `verify-full` template does not establish hostname verification for those products.

The configured `database` is the initial PostgreSQL database and `schema` defaults to `public`. Datus filters system databases/schemas during normal discovery. Access to additional databases still depends on network reachability and role permissions.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource warehouse_pg
```

Run `/databases`, `/schemas`, and `/tables`. This distinguishes a working login from missing catalog grants.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `password authentication failed` | Check the role/password and matching `pg_hba.conf` rule. |
| `no pg_hba.conf entry` | Permit the Datus client address and choose the intended TLS/auth rule. |
| Certificate verification fails | On 0.1.10+, set `sslrootcert` to the issuing CA file and match `host` to the certificate. Older packages use libpq's standard files/environment. |
| Relation not found | Check `database`, `schema`, `search_path`, identifier case, and grants. |

## Reference

- [PostgreSQL libpq SSL support](https://www.postgresql.org/docs/current/libpq-ssl.html)
