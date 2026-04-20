# Semantic Layer Configuration

Semantic adapters are configured under `agent.services.semantic_layer`.

## Structure

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # optional advanced override

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow
```

## Selection Rules

- The key under `services.semantic_layer` is the adapter type, for example `metricflow`.
- Semantic nodes choose the adapter with `semantic_adapter`.
- There is no `default: true` for semantic adapters.
- If `semantic_adapter` is omitted and only one semantic layer is configured, Datus uses that adapter automatically.
- If multiple semantic layers are configured, set `semantic_adapter` explicitly.

## MetricFlow Notes

- `config_path` is optional.
- Datus prefers the current `services.databases` entry and the project semantic model directory to build runtime config automatically.
- `config_path` is only needed when you want MetricFlow to read a specific `agent.yml` file directly.
