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

`BIFuncTool._resolved_platform` resolves the active BI service in this
order — identical to Scheduler and Semantic resolution:

1. Explicit `bi_service` argument at the call site (or `bi_platform` on
   the agentic node).
2. Project-level pin in `./.datus/config.yml`'s `dashboard:` field.
3. Global `default: true` flag — at most one entry under
   `services.bi_platforms` may carry it; multiple defaults are rejected
   at config load time so the user fixes the YAML rather than us silently
   picking one.
4. Single-entry shortcut when only one BI service is configured.
5. Otherwise raise `Multiple BI platforms configured` so the operator
   sets a default explicitly.

Set the global default in YAML:

```yaml
agent:
  services:
    bi_platforms:
      superset:
        type: superset
        default: true     # global default — picked when no project pin set
        ...
```

## Configuring through the CLI (`/services`)

Run `/services` inside the Datus REPL to enter the configuration TUI
directly (Dashboard tab by default; pass `/services scheduler` to land on
the Scheduler tab; `/services list` keeps the legacy read-only listing).
The two-tab TUI lets you:

- Add a new dashboard with `Enter` on the trailing `+ Add new dashboard` row.
  When you pick a `type` whose adapter package isn't installed yet
  (`datus-bi-superset`, `datus-bi-grafana`, …), Datus runs
  `pip install` for you and hot-reloads the registry — no restart needed.
- Edit credentials with `e`, delete an entry with `x`, run a connectivity
  probe with `t`.
- Set the **global** `default: true` flag with `d`. Pressing `d` on a row
  marks it as the workspace-wide default and clears the flag from every
  other entry, so you cannot end up with two defaults.
- Pin a **project-level** default with `p`. The pin is written to
  `./.datus/config.yml` as `dashboard: <name>` and outranks the global
  flag for the current project only. Press `p` again on the pinned row to
  clear it.

On the first interactive launch, if a section is configured but has no
project pin, Datus auto-pins the only entry (or the one flagged
`default: true`) to `./.datus/config.yml` so subsequent runs are
explicit. When multiple entries are configured without a default, the
launch prompts for a quick choice.

Service definitions are written to `~/.datus/conf/agent.yml`, so the same
credentials are shared across every project. Only the active selection is
project-local.

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
