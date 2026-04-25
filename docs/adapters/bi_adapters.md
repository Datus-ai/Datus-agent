# BI Adapters

Datus Agent supports BI platform integrations through a plugin-based adapter system. These adapters power dashboard creation workflows such as `gen_dashboard` and BI bootstrapping.

## Overview

BI adapters let Datus connect to external dashboard platforms and expose a common tool layer for:

- listing dashboards, charts, and datasets
- creating dashboards and charts
- registering datasets when the platform supports them

Data movement into the BI serving DB is **not** handled by BI adapters or
`gen_dashboard`. Prepare or refresh serving tables separately with `gen_job`
or `scheduler`, then invoke `gen_dashboard` to create BI datasets, charts, and
dashboards on top of those existing tables.

BI runtime configuration lives under `agent.services.bi_platforms` in `agent.yml`.

## Supported BI Platforms

| Platform | Package | Installation | Status |
|----------|---------|--------------|--------|
| Apache Superset | `datus-bi-superset` | `pip install datus-bi-superset` | Ready |
| Grafana | `datus-bi-grafana` | `pip install datus-bi-grafana` | Ready |

## Installation

Install the adapter package for the BI platform you want to use:

```bash
# Superset
pip install datus-bi-superset

# Grafana
pip install datus-bi-grafana
```

Once installed, Datus discovers the adapter automatically through Python entry points.

## Configuration

Configure BI platforms under `agent.services.bi_platforms`. The serving DB
itself is a regular `services.datasources` entry; the BI platform references
it by name via `dataset_db.datasource_ref`.

```yaml
agent:
  services:
    datasources:
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
          datasource_ref: serving_pg
          bi_database_name: analytics_pg

      grafana:
        type: grafana
        api_base_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          datasource_ref: serving_pg          # can share the same DB
          bi_database_name: PostgreSQL        # Grafana datasource name

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
```

## Selection Rules

- `bi_platform` selects one entry from `services.bi_platforms`.
- The config key should match the platform name, such as `superset` or `grafana`.
- If `bi_platform` is omitted and only one BI platform is configured, Datus selects it automatically.
- If multiple BI platforms are configured, set `bi_platform` explicitly.

## Platform Notes

### Superset

- Authentication uses `username` and `password`.
- `dataset_db.datasource_ref` points at the serving DB. Datus uses the
  referenced datasource's adapter (e.g. `datus-postgresql`) for both schema
  introspection and writes.
- `dataset_db.bi_database_name` matches the alias Superset shows in
  `list_bi_databases()` — `gen_dashboard` uses it to resolve `database_id`
  for `create_dataset`.

### Grafana

- Authentication uses `api_key`.
- `dataset_db.bi_database_name` should match an existing Grafana datasource
  name (Grafana's equivalent of a connection alias).
- Grafana panels embed SQL directly, so there is no separate dataset
  registration step.

## Workflow Differences

| Dimension | Superset | Grafana |
|-----------|----------|---------|
| Authentication | Username + password | API key |
| Dataset registration | Yes | No |
| Chart prerequisite | `dataset_id` | `dashboard_id` + SQL |
| Serving DB alias | `dataset_db.bi_database_name` | `dataset_db.bi_database_name` |

## Runtime Notes

- `services.bi_platforms` is the only runtime source for BI credentials.
- Sensitive values support `${ENV_VAR}` substitution.
- The legacy inline forms (`dataset_db: {uri: "..."}` or
  `dataset_db: {type: ..., host: ..., ...}`) are no longer supported — move
  the connection to `services.datasources.<name>` and reference it via
  `dataset_db.datasource_ref`.

## Related Docs

- [BI Platforms Configuration](../configuration/bi_platforms.md)
- [Generate Dashboard](../subagent/gen_dashboard.md)
- [Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md)
