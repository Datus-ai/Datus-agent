# Semantic Layer Configuration

Semantic adapters are configured under `agent.services.semantic_layer`.

When you use MetricFlow with default settings, the entire `semantic_layer` block can be omitted. Datus defaults to `metricflow`.

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

- The key under `services.semantic_layer` **must equal the adapter type** (for example `metricflow`). If a `type:` field is present, it must match the key; otherwise Datus raises a configuration error at startup. Comparison is case-insensitive and trims surrounding whitespace, so `MetricFlow` and ` metricflow ` also match.
- Semantic nodes choose the adapter with `semantic_adapter`.
- There is no `default: true` for semantic adapters.
- If both `services.semantic_layer` and `semantic_adapter` are omitted, Datus defaults to `metricflow`.
- If `semantic_adapter` is omitted and only one semantic layer is configured, Datus uses that adapter automatically.
- If multiple semantic layers are configured, set `semantic_adapter` explicitly.

## MetricFlow Notes

- `config_path` is optional.
- Datus prefers the current `services.databases` entry and the project semantic model directory to build runtime config automatically.
- `config_path` is only needed when you want MetricFlow to read a specific `agent.yml` file directly.
