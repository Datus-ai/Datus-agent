# Scheduler Configuration

Scheduler services are configured under `agent.services.schedulers`.

## Structure

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
          starrocks_default: StarRocks production

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
- If only one scheduler is configured, Datus can use it automatically.
- If multiple schedulers are configured, either:
  - set `scheduler_service`, or
  - mark exactly one scheduler with `default: true`.
- Do not configure more than one `default: true`.

## Notes

- `services.schedulers` is now the only runtime source for scheduler config.
- Top-level `scheduler:` is no longer read at runtime.
