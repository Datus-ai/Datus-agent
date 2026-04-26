# BI Platforms Configuration

BI platform connections are configured under `agent.services.bi_platforms`.

## Structure

The serving DB (the database the BI platform reads from and Datus writes to)
is registered as a **regular Datus datasource**. The BI platform entry then
references it by name via `dataset_db.datasource_ref`. This keeps connector
pooling, schema metadata, and credentials shared with the rest of Datus.

```yaml
agent:
  services:
    datasources:
      # Existing source warehouse (read-only from the BI side)
      src_warehouse:
        type: starrocks
        host: ${SRC_WAREHOUSE_HOST}
        port: 9030
        username: ${SRC_WAREHOUSE_USER}
        password: ${SRC_WAREHOUSE_PASSWORD}
        database: warehouse

      # Serving DB — Datus writes here, the BI platform reads from here
      serving_pg:
        type: postgresql
        host: 127.0.0.1
        port: 5433
        database: superset_examples
        schema: bi_public
        username: ${SERVING_WRITE_USER}
        password: ${SERVING_WRITE_PASSWORD}

    bi_platforms:
      superset:
        type: superset
        api_base_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}
        dataset_db:
          datasource_ref: serving_pg          # ← references services.datasources.serving_pg
          bi_database_name: examples          # Superset Database connection name shown in Settings > Database Connections

      grafana:
        type: grafana
        api_base_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          datasource_ref: serving_pg          # ← can share the same serving DB
          bi_database_name: PostgreSQL        # ← Grafana datasource name

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
```

## `dataset_db` fields

`dataset_db` is the BI-platform-specific layer on top of a Datus datasource.
It carries only what's BI-specific:

| Field | Required | Description |
|-------|----------|-------------|
| `datasource_ref` | Yes | Name of a `services.datasources` entry. Datus uses that datasource's connector for both schema introspection and writes. |
| `bi_database_name` | Recommended | Alias under which the BI platform itself has registered the same DB. `gen_dashboard` matches it against `list_bi_databases()` to resolve `database_id` for `create_dataset`. |

The legacy inline form (`dataset_db: {uri: "..."}` or
`dataset_db: {type: ..., host: ..., ...}`) is no longer accepted —
move the connection fields under `services.datasources` and reference them
by name.

## Selection rules

- `bi_platform` selects one entry from `services.bi_platforms`.
- The config key should match the platform name (`superset`, `grafana`, ...).
- If `bi_platform` is omitted and only one BI platform is configured, Datus
  auto-selects it.
- If multiple BI platforms are configured, set `bi_platform` explicitly.

## Ownership

Dashboard creation is split into three explicit steps:

1. `gen_job` or `scheduler` prepares / refreshes data in the serving DB
   referenced by `dataset_db.datasource_ref`.
2. `gen_dashboard` builds the dataset / chart / dashboard on the BI side
   from tables or SQL datasets that already exist in that BI-registered DB.
3. `bi-validation` runs post-creation checks automatically through `ValidationHook.on_end`.

Source DB credentials never leave Datus — Superset / Grafana see only the
serving DB registered under `bi_database_name`.

## Notes

- `services.bi_platforms` is the only runtime source for BI credentials.
- Top-level `dashboard:` is no longer read at runtime.
- `services.datasources.<datasource_ref>` must exist before
  `services.bi_platforms.<x>.dataset_db.datasource_ref` is resolved —
  Datus validates this at startup and fails loudly if the ref is dangling.
