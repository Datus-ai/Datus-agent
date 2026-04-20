# BI Adapters

Datus Agent supports BI platform integrations through a plugin-based adapter system. These adapters power dashboard creation workflows such as `gen_dashboard` and BI bootstrapping.

## Overview

BI adapters let Datus connect to external dashboard platforms and expose a common tool layer for:

- listing dashboards, charts, and datasets
- creating dashboards and charts
- materializing SQL results into a BI-facing database
- registering datasets when the platform supports them

BI runtime configuration lives under `agent.services.bi_tools` in `agent.yml`.

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

Configure BI platforms under `agent.services.bi_tools`:

```yaml
agent:
  services:
    bi_tools:
      superset:
        type: superset
        api_url: http://localhost:8088
        username: ${SUPERSET_USER}
        password: ${SUPERSET_PASSWORD}
        dataset_db:
          uri: ${SUPERSET_DB_URI}
          schema: public

      grafana:
        type: grafana
        api_url: http://localhost:3000
        api_key: ${GRAFANA_API_KEY}
        dataset_db:
          uri: ${GRAFANA_DB_URI}
          datasource_name: PostgreSQL

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
```

## Selection Rules

- `bi_platform` selects one entry from `services.bi_tools`.
- The config key should match the platform name, such as `superset` or `grafana`.
- If `bi_platform` is omitted and only one BI platform is configured, Datus selects it automatically.
- If multiple BI platforms are configured, set `bi_platform` explicitly.

## Platform Notes

### Superset

- Authentication uses `username` and `password`.
- `dataset_db.uri` is the SQLAlchemy target where `write_query` materializes data.
- The corresponding database connection should already exist in Superset so `create_dataset` can register the table.

### Grafana

- Authentication uses `api_key`.
- `dataset_db.uri` is the SQLAlchemy target where `write_query` materializes data.
- `dataset_db.datasource_name` should match an existing Grafana datasource.
- Grafana panels embed SQL directly, so there is no separate dataset registration step.

## Workflow Differences

| Dimension | Superset | Grafana |
|-----------|----------|---------|
| Authentication | Username + password | API key |
| Dataset registration | Yes | No |
| Chart prerequisite | `dataset_id` | `dashboard_id` + SQL |
| Materialization target | `dataset_db.uri` | `dataset_db.uri` |
| Extra selector | none | `dataset_db.datasource_name` |

## Runtime Notes

- `services.bi_tools` is the only runtime source for BI credentials.
- Sensitive values support `${ENV_VAR}` substitution.

## Related Docs

- [BI Tools Configuration](../configuration/bi_tools.md)
- [Generate Dashboard](../subagent/gen_dashboard.md)
- [Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md)
