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
- Datus prefers the current `services.datasources` entry and the project semantic model directory to build runtime config automatically.
- MetricFlow validation reads YAML files from the configured project semantic model directory directly, including generated files under gitignored project paths.
- `config_path` is only needed when you want MetricFlow to read a specific `agent.yml` file directly.

## Configuring through the CLI (`/services`)

Run `/services semantic` inside the Datus REPL (or press `Tab` from any
other tab) to enter the configuration TUI on the **Semantic** tab. The
tab lets you:

- Add a new semantic layer by pressing `Enter` on the trailing `+ Add
  new semantic` row. Only `metricflow` (`datus-semantic-metricflow`)
  ships today and **takes no parameters** — picking it from the type
  picker is enough. If the adapter package isn't installed, Datus runs
  `pip install datus-semantic-metricflow` for you and hot-reloads the
  registry — no restart needed.
- Delete an entry with `x` and run a registration probe with `t`.
- `e edit` and `p project default` are intentionally hidden on this tab:
  metricflow has no editable fields, and semantic layers don't yet expose
  a project-level default API.

Service definitions are written to `~/.datus/conf/agent.yml` as
`services.semantic_layer.<type>: {type: <type>}`.
