# SAP HANA datasource

The HANA adapter uses SAP's official Python `hdbcli` client. SQL objects use `schema.table`; an optional tenant database selects the connection target. The client machine needs the Python adapter, while the HANA server runs separately.

## Install

Use Python 3.12 or later and a Datus build with HANA SQL-analysis support. Until `datus-hana` is published on PyPI, install this source revision into the same Python environment as `datus`:

```bash
pip install --upgrade 'datus-db-core>=0.1.10' \
  'datus-hana @ git+https://github.com/Datus-ai/datus-db-adapters.git@6f5fb395cee2fa0217f281aa9a0af27f33dac68d#subdirectory=datus-hana'
```

The source installation requires Git and installs the `hdbcli` dependency. Restart Datus after upgrading the adapter or shared core. Installing the SDK into an older Agent alone does not add HANA query analysis. If launching through uv from a checkout, use `uv run --no-sync` after installing these packages.

## Connection profile

Password authentication with verified TLS:

```yaml
agent:
  services:
    datasources:
      hana_prod:
        type: hana
        host: ${HANA_HOST}
        port: 39041
        username: ${HANA_USER}
        password: ${HANA_PASSWORD}
        database: HXE
        schema: ANALYTICS
        encrypt: true
        ssl_validate_certificate: true
        ssl_trust_store: /absolute/path/hana-ca.pem
        timeout_seconds: 30
        communication_timeout_seconds: 60
```

Use the actual tenant SQL endpoint and port supplied by the administrator. HANA Cloud commonly uses port `443`; `39041` is only the adapter's default. Omit `database` if the endpoint already selects the intended tenant and no explicit database name is needed.

### Secure user store

For an existing HANA secure user-store key, replace the username/password credentials with `userkey`:

```yaml
agent:
  services:
    datasources:
      hana_userstore:
        type: hana
        userkey: DATUS_READER
        schema: ANALYTICS
        encrypt: true
        ssl_validate_certificate: true
        ssl_trust_store: /absolute/path/hana-ca.pem
```

Provision the key with SAP's `hdbuserstore` utility for the OS user running Datus. Installing the Python wheel does not provision that utility or a key. Omit `host` to use the endpoint stored in the key. `schema` is required with `userkey`; do not also supply `username` or `password`. TLS settings continue to apply.

## Parameters

| Key | Type | Required | Default | Notes |
| --- | --- | --: | --- | --- |
| `host` | string | password auth | — | Tenant SQL endpoint; optional with `userkey`. |
| `port` | integer | no | `39041` | SQL port, from 1 to 65535; used when `host` is supplied. |
| `username` | string | password auth | — | HANA database user. |
| `password` | string | password auth | empty | Nonempty password; use an environment variable. |
| `userkey` | string | no | — | Existing secure user-store key; mutually exclusive with username/password. |
| `database` | string | no | — | Tenant connection target; fixed for this datasource. |
| `schema` | string | with `userkey` | configured username | Object namespace; accepted as the alias for `schema_name`. |
| `encrypt` | boolean | no | `true` | Enables encrypted database connections. |
| `ssl_validate_certificate` | boolean | no | `true` | Validates the server certificate. |
| `ssl_trust_store` | string | no | SDK trust configuration | Trust-store file path for the issuing CA. |
| `ssl_host_name_in_certificate` | string | no | SDK default | Expected certificate hostname when an explicit name is needed. |
| `timeout_seconds` | integer | no | `30` | Positive connection timeout in seconds. |
| `communication_timeout_seconds` | integer | no | `60` | Positive network-request timeout in seconds. |
| `max_result_rows` | integer | no | `10000` | Positive result row limit. |
| `max_result_bytes` | integer | no | `16777216` | Positive result byte limit. |

## TLS and namespaces

Keep encryption and certificate validation enabled for production. Provide the CA trust store used by the deployment and a certificate-matching hostname. The adapter converts timeout seconds to the milliseconds expected by the SAP client; a communication timeout is a network bound, not confirmation of server-side query cancellation.

`database` selects the tenant when connecting. Write SQL object names as `schema.table`, for example `"ANALYTICS"."ORDERS"`. Unquoted HANA identifiers fold to uppercase; quote mixed-case names exactly as metadata returns them. Changing the tenant requires another datasource; per-query context changes only select a schema.

## Query support and result limits

Datus checks common SELECT, JOIN, CTE, aggregation, window, set-operation, and LIMIT queries before execution. Unsupported SQLScript, Calculation View parameters, sequence access, table functions, locking queries, and multi-statement input are rejected by the local query checks. Full HANA SQL formatting, policy rewriting, transpilation, and static lineage are not supported by this adapter. Queries that pass analysis are sent unchanged to the SAP client.

Results support list, CSV, pandas, and Arrow formats. Exceeding the row or byte budget fails explicitly rather than silently truncating data. DML batches can share a transaction, but HANA DDL may commit implicitly. Writes are not automatically replayed after connection failures. Metadata DDL describes columns; it is not a lossless schema-migration export.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource hana_prod
```

Run `/schemas` and `/tables` to check metadata grants, then execute:

```sql
SELECT 1 AS CONNECTION_OK FROM DUMMY;
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Adapter cannot be installed from PyPI | Use the pinned source installation above until a package release is available. |
| Authentication or connection failure | Check the actual tenant endpoint, port, user/password or secure user-store key, and network access. |
| `userkey` validation fails | Supply `schema`, omit username/password, and provision the key for the OS user running Datus. |
| TLS verification fails | Check the issuing CA, trust-store path, and expected certificate hostname. |
| Table not found | Use `schema.table`, match quoted identifier case, and check schema/object privileges. |
| Query rejected before execution | Restrict the query to the supported analysis subset; installing `hdbcli` alone does not expand Agent SQL analysis. |
| Result exceeds its budget | Select fewer rows/columns or set appropriate `max_result_rows` and `max_result_bytes`. |

## Reference

- [Adapter configuration and support boundaries](https://github.com/Datus-ai/datus-db-adapters/blob/main/datus-hana/README.md)
- [SAP Python client](https://pypi.org/project/hdbcli/)
- [SAP client connection properties](https://help.sap.com/docs/hana-cloud-database/f1b440ded6144a54ada97ff95dac7adf/ee592e89dcce4480a99571a4ae7a702f.html)
