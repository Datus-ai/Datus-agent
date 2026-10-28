# ClickZetta datasource

The ClickZetta adapter connects to a service endpoint and selects an instance, workspace, schema, and virtual cluster for SQL execution.

## Install

```bash
pip install datus-clickzetta
```

## Connection profile

```yaml
agent:
  services:
    datasources:
      clickzetta_lakehouse:
        type: clickzetta
        service: ${CLICKZETTA_SERVICE}
        username: ${CLICKZETTA_USERNAME}
        password: ${CLICKZETTA_PASSWORD}
        instance: ${CLICKZETTA_INSTANCE}
        workspace: ${CLICKZETTA_WORKSPACE}
        schema: ${CLICKZETTA_SCHEMA:-PUBLIC}
        vcluster: ${CLICKZETTA_VCLUSTER:-DEFAULT_AP}
        secure: true
        hints:
          key: value
```

## Parameters

| Key | Type | Required | Default | Notes |
|---|---|---:|---|---|
| `service` | string | yes | — | ClickZetta service endpoint. |
| `username` | string | yes | — | Login user; blank values are rejected. |
| `password` | string | yes | — | Login password; blank values are rejected. |
| `instance` | string | yes | — | Instance identifier. |
| `workspace` | string | yes | — | Workspace name. |
| `schema` | string | no | `PUBLIC` | Default schema; alias for `schema_name`. |
| `vcluster` | string | no | `DEFAULT_AP` | Virtual cluster. |
| `secure` | boolean | no | connector default | Enables a secure connection. |
| `hints` | mapping | no | — | Additional ClickZetta connection hints. |

## Profile boundary

The adapter model also defines a generic field named `extra`, but `extra` is reserved internally by Datus Agent for carrying adapter-specific top-level fields. Do not add `extra:` to `agent.yml`; use the explicit fields above, including `hints`, instead.

The service, user, password, instance, and workspace must all be non-empty. Schema and virtual-cluster names are normalized by the ClickZetta service rather than by Datus.

## Verify the connection

```bash
datus --config conf/agent.yml --datasource clickzetta_lakehouse
```

Run `/schemas` and `/tables` to verify the selected workspace, schema, and virtual cluster.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Required field cannot be empty | Export all five required environment variables before starting Datus. |
| Workspace or schema is not visible | Check `instance`, `workspace`, `schema`, `vcluster`, and the user's grants. |
| Secure connection fails | Confirm the endpoint expects TLS and set `secure` accordingly. |

## Reference

- [ClickZetta documentation](https://www.clickzetta.com/)
