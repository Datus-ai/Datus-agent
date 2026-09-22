# MaxCompute datasource

The MaxCompute adapter supports both legacy `project.table` projects and schema-enabled `project.schema.table` projects through PyODPS.

## Install

```bash
pip install datus-maxcompute
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      maxcompute_prod:
        type: maxcompute
        database: ${MAXCOMPUTE_PROJECT}
        endpoint: ${MAXCOMPUTE_ENDPOINT}
        access_key_id: ${MAXCOMPUTE_ACCESS_KEY_ID}
        access_key_secret: ${MAXCOMPUTE_ACCESS_KEY_SECRET}
        namespace_mode: auto
        # schema: default
        # quota_name: ${MAXCOMPUTE_QUOTA_NAME}
        # tunnel_endpoint: ${MAXCOMPUTE_TUNNEL_ENDPOINT}
        timeout_seconds: 30
        query_timeout_seconds: 600
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `database` | string | yes | — | MaxCompute project; `project` is also accepted. |
| `endpoint` | string | yes | — | MaxCompute service endpoint. |
| `access_key_id` | secret string | yes | — | Alibaba Cloud AccessKey ID. |
| `access_key_secret` | secret string | yes | — | Alibaba Cloud AccessKey secret. |
| `schema` | string | no | `default` in three-level mode | Default schema for a schema-enabled project. |
| `namespace_mode` | string | no | `auto` | `auto`, `two_level`, or `three_level`. |
| `quota_name` | string | no | — | MaxCompute quota name. |
| `tunnel_endpoint` | string | no | service endpoint behavior | Separate Instance Tunnel endpoint. |
| `timeout_seconds` | integer | no | `30` | Connection timeout; must be positive. |
| `query_timeout_seconds` | integer | no | `600` | SQL job timeout; must be positive. |
| `default_hints` | mapping | no | `{}` | Default MaxCompute SQL hints. |

## Namespace modes

`database` names the MaxCompute project; the adapter also accepts the native name `project`. Keep `namespace_mode: auto` unless the configured identity cannot probe schema support.

- `two_level`: objects are `project.table`; leave `schema` unset.
- `three_level`: objects are `project.schema.table`; omitted `schema` means `default`.

A datasource stays within its configured project and does not generate cross-project SQL. Set `tunnel_endpoint` only when Instance Tunnel uses a different endpoint from the SQL service.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource maxcompute_prod
```

Run `/databases` and `/tables`; in three-level mode, also verify schema discovery with the configured identity.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Namespace probing is denied | Set `namespace_mode` explicitly to `two_level` or `three_level`. |
| Table names resolve in the wrong schema | Set `schema`, or leave it unset for two-level projects. |
| Query succeeds but downloads fail | Configure the correct `tunnel_endpoint` and network route. |
| Job exceeds the operation timeout | Increase `query_timeout_seconds`; `timeout_seconds` is the connection timeout. |

## Reference

- [MaxCompute endpoints](https://www.alibabacloud.com/help/en/maxcompute/user-guide/endpoints)
