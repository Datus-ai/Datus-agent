# Hive datasource

The Hive adapter connects to the binary HiveServer2 endpoint through PyHive/Thrift, including compatible Transwarp Inceptor endpoints. Hive databases are the table namespace, and optional session properties are passed through `configuration`.

!!! note "Adapter availability"
    The Kerberos configuration and Inceptor metadata improvements described below require a `datus-hive` build containing those changes. The published `datus-hive` 0.1.3 package does not include `kerberos_service_name`, the `kerberos` extra, or the metadata fixes. Install a build containing them before using these features; installing GSSAPI dependencies alone does not add the missing adapter configuration.

The commands below pin a source revision containing the changes in [adapter PR #158](https://github.com/Datus-ai/datus-db-adapters/pull/158). They require Git and Python 3.12 or newer. Both installation variants use the same revision. `--force-reinstall` replaces an existing PyPI 0.1.3 installation even though the source build currently has the same package version.

## Install

```bash
pip install --force-reinstall 'datus-hive @ git+https://github.com/Datus-ai/datus-db-adapters.git@5dbac1871985f349b917c6876ca6487b709a65b0#subdirectory=datus-hive'
```

Install the adapter into the same Python environment as `datus`. For Kerberos, install its optional GSSAPI dependencies after choosing an adapter build with Kerberos support:

```bash
pip install --force-reinstall 'datus-hive[kerberos] @ git+https://github.com/Datus-ai/datus-db-adapters.git@5dbac1871985f349b917c6876ca6487b709a65b0#subdirectory=datus-hive'
```

The Kerberos extension needs system Kerberos libraries and development headers to build, plus a C compiler. For example, Debian/Ubuntu builds use `libkrb5-dev`; install the Kerberos client tools such as `kinit` and `klist` as well.

## Connection profile

```yaml
agent:
  services:
    datasources:
      hive_warehouse:
        type: hive
        host: ${HIVE_HOST:-127.0.0.1}
        port: 10000
        username: hive
        database: default
        auth: NONE
        timeout_seconds: 30
```

This profile is for a server configured with `NONE` authentication. Choose the matching authentication mode below for a secured server.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | no | `127.0.0.1` | HiveServer2 host. |
| `port` | integer | no | `10000` | HiveServer2 Thrift port. |
| `username` | string | yes | — | Hive user. For Kerberos, the authenticated identity comes from the ticket cache. |
| `password` | string | no | empty | Password for `LDAP` or `CUSTOM`; omit for `NONE` and `KERBEROS`. |
| `database` | string | no | `default` at connection time | Initial Hive database. |
| `auth` | string | no | `None` → PyHive `NONE` | Set explicitly to `NONE`, `LDAP`, `CUSTOM`, or `KERBEROS`; no automatic authentication detection. |
| `kerberos_service_name` | string | for `KERBEROS` | `None` | Required nonblank service component of the server principal, usually `hive`. Omit for other modes. |
| `configuration` | mapping | no | `{}` | Hive session properties; values are normalized to strings. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication

### NONE, LDAP, and CUSTOM

Omitting `auth` uses PyHive's `NONE` mode, which still uses SASL PLAIN. It does not select LDAP or Kerberos automatically. Supplying a password requires an explicit `LDAP` or `CUSTOM` mode.

```yaml
agent:
  services:
    datasources:
      hive_ldap:
        type: hive
        host: ${HIVE_HOST}
        port: 10000
        database: default
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        auth: LDAP
      hive_custom:
        type: hive
        host: ${HIVE_HOST}
        port: 10000
        database: default
        username: ${HIVE_USER}
        password: ${HIVE_PASSWORD}
        auth: CUSTOM
```

Both modes send a username and password. `LDAP` asks the server to validate them against its configured directory; `CUSTOM` invokes the server's configured authentication provider. The server administrator must configure that provider or directory. These datasource settings select the client mode and do not enable authentication on the server.

### Kerberos

Acquire a ticket before starting Datus. For example, use a keytab and a credential cache dedicated to the Datus process:

```bash
export KRB5_CONFIG=/path/to/krb5.conf
export KRB5CCNAME=FILE:/path/to/datus.ccache
kinit -kt /path/to/alice.keytab alice@EXAMPLE.COM
klist
```

Start Datus from the same environment with this profile:

```yaml
agent:
  services:
    datasources:
      hive_kerberos:
        type: hive
        host: hs2.example.com
        port: 10000
        database: default
        username: alice
        auth: KERBEROS
        kerberos_service_name: hive
```

This example expects the server principal `hive/hs2.example.com@EXAMPLE.COM`. Set `kerberos_service_name` to its service component, `hive`. Use a hostname matching the service principal, with working DNS, realm configuration, KDC connectivity, and synchronized clocks.

PyHive uses SASL GSSAPI and the current ticket cache to authenticate. The profile's `username` remains required, but changing it does not change the authenticated ticket identity. Omit `password`; the adapter rejects it for Kerberos. The application or deployment environment owns ticket acquisition and renewal. The adapter does not run `kinit`, renew expired tickets, or set process-wide Kerberos configuration. New connections require valid credentials.

## Session configuration

`configuration` is sent as Hive session configuration. YAML booleans become `true`/`false` strings, other scalar values are stringified, and `null` becomes an empty string.

Add engine-specific options only when supported by the target server. For example, a Hive deployment using Tez might accept:

```yaml
configuration:
  hive.execution.engine: tez
  hive.vectorized.execution.enabled: true
```

## Transwarp Inceptor

Use `type: hive` with Inceptor's HiveServer2-compatible binary Thrift endpoint:

```yaml
agent:
  services:
    datasources:
      inceptor_warehouse:
        type: hive
        host: ${INCEPTOR_HOST}
        port: 10000
        database: default
        username: ${INCEPTOR_USER}
        password: ${INCEPTOR_PASSWORD}
        auth: LDAP
```

Replace the endpoint, port, and authentication mode with those supplied by the Inceptor administrator. The same `NONE`, `LDAP`, `CUSTOM`, and Kerberos client configurations apply when the server enables the corresponding mode. Use the Kerberos profile above for an Inceptor Kerberos endpoint. A vendor JDBC driver is not required for this connection path.

With the metadata fixes, ordinary views are excluded from table listings and retrieved through view discovery. Column comments are read by field name so Inceptor's additional `DESCRIBE` fields do not shift the comment value. Database, table, view, column/comment, DDL, and sample-row retrieval were verified with the updated adapter against Apache Hive 4.0.1 and TDH Community Development Edition 2024.5 (Inceptor 8.37.3).

Table/view discovery requires server support for `SHOW VIEWS` and metadata-read permissions. A failed inventory is reported to the caller.

Inceptor validation covered `NONE`, `LDAP`, `CUSTOM`, and client-to-HiveServer2 Kerberos authentication. It did not validate a fully Kerberized TDH/Guardian cluster or all enterprise features. Guardian Token/CAS authentication, HTTP/HTTPS transport, and ZooKeeper service discovery are not exposed by this adapter's profile. Session engine options and SQL features still depend on the target server.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource hive_warehouse
```

Run `/databases` and `/tables`, then query a small table to verify HiveServer2 and backing metastore access.

For a secured profile, replace `hive_warehouse` with its configured name. Check ordinary views, column comments, DDL, and sample rows on known objects as well. Authentication success and SQL authorization are separate: the authenticated user still needs access to the target databases and objects.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Thrift connection fails | Check HiveServer2 host/port, service health, and network routing. |
| SASL/authentication error | Match `auth`, username/password, and the server's HiveServer2 auth mode. |
| `kerberos_service_name` is rejected, or installation reports a missing `kerberos` extra | The installed adapter build lacks the Kerberos changes. Install a build containing both the field and the optional dependency group. |
| No Kerberos credentials or an expired ticket | Run `klist` in the Datus process environment, check `KRB5CCNAME`, and acquire/renew the ticket before opening a new connection. |
| Kerberos service principal is not found | Check `host`, realm/DNS configuration, and the service component in `kerberos_service_name`; do not put the complete principal in that field. |
| Login succeeds but queries or metadata access are denied | Check the user's database grants and active roles with the server administrator. |
| Session option has no effect | Use the exact Hive property key under `configuration` and confirm the server permits overrides. |

## Reference

- [HiveServer2 clients](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients)
- [HiveServer2 authentication configuration](https://hive.apache.org/docs/latest/admin/setting-up-hiveserver2/)
- [Transwarp Inceptor connection guide](https://www.transwarp.cn/doc/inceptor/9.5/quick-start--connect-database)
