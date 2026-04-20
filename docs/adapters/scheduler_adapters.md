# Scheduler Adapters

Datus Agent supports external schedulers through a plugin-based adapter system. These adapters power the `scheduler` subagent and scheduled job lifecycle operations.

## Overview

Scheduler adapters let Datus submit, inspect, and manage scheduled jobs on external orchestration platforms. The current runtime focuses on Airflow-backed SQL and SparkSQL jobs.

Scheduler runtime configuration lives under `agent.services.schedulers` in `agent.yml`.

## Supported Scheduler Platforms

| Platform | Package | Installation | Status |
|----------|---------|--------------|--------|
| Apache Airflow | `datus-scheduler-airflow` | `pip install datus-scheduler-airflow` | Ready |

## Installation

Install the concrete scheduler adapter package:

```bash
pip install datus-scheduler-airflow
```

`datus-scheduler-airflow` pulls in `datus-scheduler-core` transitively, so you do not need to install the core package separately.

## Configuration

Configure scheduler services under `agent.services.schedulers`:

```yaml
agent:
  services:
    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: ${AIRFLOW_URL}
        username: ${AIRFLOW_USER}
        password: ${AIRFLOW_PASSWORD}
        dags_folder: ${AIRFLOW_DAGS_DIR}
        default: true
        connections:
          postgres_prod: PostgreSQL production

      airflow_dev:
        type: airflow
        api_base_url: ${AIRFLOW_DEV_URL}
        username: ${AIRFLOW_DEV_USER}
        password: ${AIRFLOW_DEV_PASSWORD}
        dags_folder: /tmp/airflow-dags

  agentic_nodes:
    scheduler:
      scheduler_service: airflow_prod
```

## Selection Rules

- `scheduler_service` selects one scheduler instance from `services.schedulers`.
- If only one scheduler is configured, Datus can select it automatically.
- If multiple schedulers are configured, either set `scheduler_service` explicitly or mark exactly one service with `default: true`.
- Do not configure more than one `default: true`.

## Airflow Notes

- `api_base_url` should point to the Airflow REST API, for example `http://localhost:8080/api/v1`.
- `dags_folder` is the host-side directory where Datus writes generated DAG files.
- If Airflow runs in Docker or Kubernetes, mount that directory into Airflow's DAG directory such as `/opt/airflow/dags`.
- `connections` is an optional curated map of available target connections for scheduled jobs.
- SQL files referenced by `submit_sql_job` or `update_job` must already exist on the agent host.

## Runtime Notes

- `services.schedulers` is the only runtime source for scheduler configuration.
- Sensitive values support `${ENV_VAR}` substitution.

## Supported Job Types

Current Airflow adapter capabilities include:

- scheduled SQL jobs from `.sql` files
- scheduled SparkSQL jobs from `.sql` files
- trigger, pause, resume, update, delete, and inspect job runs

## Related Docs

- [Scheduler Configuration](../configuration/schedulers.md)
- [Scheduler Subagent Guide](../subagent/scheduler.md)
- [Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md)
