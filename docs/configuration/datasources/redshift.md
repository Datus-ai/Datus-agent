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
        cluster_identifier: analytics-cluster
        region: us-east-1
        # access_key_id: ${AWS_ACCESS_KEY_ID}
        # secret_access_key: ${AWS_SECRET_ACCESS_KEY}
```

When the runtime already has an AWS role or standard AWS credentials, omit the static access-key fields.

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `host` | string | yes | — | Redshift cluster endpoint. |
| `port` | integer | no | `5439` | Redshift server port. |
| `username` | string | yes | — | Database user. |
| `password` | string | yes* | — | Required when `iam` is `false`. |
| `database` | string | no | `dev` at connection time | Initial database. |
| `schema` | string | no | — | Initial schema; the driver uses `public` when omitted. |
| `ssl` | boolean | no | `true` | Enables TLS. |
| `iam` | boolean | no | `false` | Enables IAM authentication. |
| `cluster_identifier` | string | IAM | — | Cluster identifier used to obtain temporary credentials. |
| `region` | string | IAM | — | AWS region. |
| `access_key_id` | string | no | AWS credential chain | Static access key for IAM auth. |
| `secret_access_key` | string | no | AWS credential chain | Static secret key for IAM auth. |
| `timeout_seconds` | integer | no | `30` | Connection timeout. |

## Authentication and TLS

Set `iam: true` to make `password` optional. In AWS-hosted environments, prefer an attached role over keys in `agent.yml`. The adapter exposes a boolean `ssl`; it does not expose CA-bundle or hostname-verification fields in the profile.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource redshift_prod
```

Run `/schemas` and `/tables`. For IAM profiles, also confirm the runtime identity can call the Redshift credential APIs for the configured cluster.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Password required validation error | Supply `password`, or set `iam: true`. |
| IAM authentication fails | Check `cluster_identifier`, `region`, runtime AWS credentials/role, and IAM policy. |
| Connection times out | Confirm VPC routing, security groups, endpoint, and port `5439`. |

## Reference

- [Amazon Redshift connection guidance](https://docs.aws.amazon.com/redshift/latest/mgmt/configure-jdbc-connection.html)
