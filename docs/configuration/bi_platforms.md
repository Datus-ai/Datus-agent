# BI Platforms Configuration

BI platform connections are configured under `agent.services.bi_platforms`.

## Structure

```yaml
agent:
  services:
    bi_platforms:
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

- `bi_platform` selects one entry from `services.bi_platforms`.
- The config key should match the platform name, such as `superset` or `grafana`.
- If `bi_platform` is omitted and only one BI platform is configured, Datus auto-selects it.
- If multiple BI platforms are configured, set `bi_platform` explicitly.

## Notes

- `services.bi_platforms` is now the only runtime source for BI credentials.
- Top-level `dashboard:` is no longer read at runtime.
