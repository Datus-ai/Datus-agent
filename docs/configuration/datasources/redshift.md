# Amazon Redshift datasource

The Redshift adapter uses the native `redshift_connector` driver and supports password or IAM authentication. Objects follow the `database → schema → table` hierarchy.

## Install

```bash
pip install datus-redshift
```

## Connection profile

Password authentication:

```yaml
agent:
  services:
    datasources:
      redshift_prod:
        type: redshift
        host: ${REDSHIFT_HOST}
        port: 5439
        username: ${REDSHIFT_USER}
        password: ${REDSHIFT_PASSWORD}
        database: analytics
        schema: public
        ssl: true
        timeout_seconds: 30
```

IAM authentication:

```yaml
agent:
  services:
    datasources:
      redshift_iam:
        type: redshift
        host: ${REDSHIFT_HOST}
        port: 5439
        username: ${REDSHIFT_USER}
        database: analytics
        schema: public
        iam: true
        ssl: true
        cluster_identifier: analytics-cluster
        region: us-east-1
        # access_key_id: ${AWS_ACCESS_KEY_ID}
        # secret_access_key: ${AWS_SECRET_ACCESS_KEY}
```

When the runtime already has an AWS role or standard AWS credentials, omit the static access-key fields.

For both authentication methods, `username` is an existing Redshift **database user**. With IAM authentication, the adapter forwards it to the driver's `db_user` parameter when obtaining temporary database credentials. Configure that database username, rather than an IAM user or role ARN. `db_user` is an internal driver parameter and is not a supported field in `agent.yml`.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | yes | — | Redshift cluster endpoint. |
| `port` | integer | no | `5439` | Redshift server port. |
| `username` | string | yes | — | Database user; also forwarded as `db_user` for IAM authentication. |
| `password` | string | yes* | — | Required when `iam` is `false`. |
| `database` | string | no | `dev` at connection time | Initial database. |
| `schema` | string | no | — | Initial schema; the adapter uses `public` when omitted. |
| `ssl` | boolean | no | `true` | Enables TLS; required for IAM authentication. |
| `iam` | boolean | no | `false` | Enables IAM authentication. |
| `cluster_identifier` | string | IAM | — | Cluster identifier used to obtain temporary credentials. |
| `region` | string | IAM | — | AWS region. |
| `access_key_id` | string | no | AWS credential chain | Static access key for IAM auth. |
| `secret_access_key` | string | no | AWS credential chain | Static secret key for IAM auth. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication and TLS

Set `iam: true` to make `password` optional, and keep `ssl: true` for IAM connections. In AWS-hosted environments, prefer an attached role over keys in `agent.yml`. The adapter exposes a boolean `ssl`; it does not expose CA-bundle or hostname-verification fields in the profile.

The adapter does not pass `sslmode`; hostname checks follow the installed driver's default policy. `ssl: true` does not guarantee strict hostname verification, and the current profile cannot explicitly request `sslmode: verify-full`.

## Connecting from a local machine

The AWS CLI calls AWS HTTPS APIs, while the adapter opens a separate database TCP connection. A successful CLI call or an HTTP proxy does not establish database connectivity. Check the cluster endpoint's actual database port, VPC routes, network ACLs, and security groups. For public endpoints, allow the database client's public egress IP; an HTTP proxy may use a different address.

For a private endpoint, use a VPN or an existing Systems Manager managed host that can reach and resolve the database endpoint. Allow the database port from that host's security group or private IP, and install the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) locally.

For an SSM tunnel, set `REDSHIFT_HOST` to the original cluster endpoint, `REDSHIFT_REGION` to its AWS region, and `REDSHIFT_SSM_TARGET` to the managed host's instance ID. Set `REDSHIFT_REMOTE_PORT` to the cluster's actual database port (`Endpoint.Port`, usually `5439`). Run in a separate terminal:

```bash
aws ssm start-session \
  --region "$REDSHIFT_REGION" \
  --target "$REDSHIFT_SSM_TARGET" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "{\"host\":[\"$REDSHIFT_HOST\"],\"portNumber\":[\"$REDSHIFT_REMOTE_PORT\"],\"localPortNumber\":[\"15439\"]}"
```

For the tunnel, keep the datasource profile's `host` set to the original Redshift endpoint. While the tunnel is open, temporarily map that endpoint to `127.0.0.1` in local DNS or the hosts file, and set the profile's `port` to `15439`. This changes local routing; TLS hostname checks still follow the driver's policy described above. The managed host resolves the remote endpoint independently of this local override. Remove the override and close the SSM session after use.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource redshift_prod
```

Run `/schemas` and `/tables`. For IAM profiles, also confirm the runtime identity can call the Redshift credential APIs for the configured cluster.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Password required validation error | Supply `password`, or set `iam: true`. |
| IAM authentication fails | Check the database `username`, `cluster_identifier`, `region`, runtime AWS credentials/role, and IAM policy. |
| IAM error reports a missing `db_user` | Use an adapter release containing the [IAM username-forwarding fix](https://github.com/Datus-ai/datus-db-adapters/pull/157). Keep the database user in `username`; adding `db_user` to the profile fails validation. |
| CLI calls succeed but the database connection times out | Check database TCP connectivity separately: endpoint, actual database port, VPC routing, security groups, and any VPN or tunnel. |
| `SSLContext` / `PROTOCOL_TLS` deprecation warnings on Python 3.12 | Driver versions including `redshift_connector` 2.2.0 use deprecated SSL-context initialization. Keep TLS enabled and use an official driver release containing the [upstream fix](https://github.com/aws/amazon-redshift-python-driver/pull/288) when available. Treating these warnings as errors can prevent a connection. |

## Reference

- [Amazon Redshift connection guidance](https://docs.aws.amazon.com/redshift/latest/mgmt/configure-jdbc-connection.html)
- [Session Manager port forwarding to a remote host](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html#sessions-remote-port-forwarding)
- [Python SSLContext deprecation notes](https://docs.python.org/3.12/library/ssl.html#ssl.SSLContext)
