# Semantic Layer Configuration

Semantic adapters are configured under `agent.services.semantic_layer`.
At least one entry must be configured — running a semantic node with an
empty `services.semantic_layer` block now raises
`No semantic layer configured` instead of silently falling back to a
hard-coded `metricflow` default.

> **Migration note**: previous Datus releases inserted an implicit
> `metricflow` default when the section was missing. The default now
> requires an explicit YAML entry. The bundled `conf/agent.yml.example`
> already ships with `metricflow: {}`, so fresh installs continue to
> Just Work.

## Structure

```yaml
agent:
  services:
    semantic_layer:
      metricflow:
        timeout: 300
        config_path: ./conf/agent.yml   # optional advanced override
        default: true                   # global default — picked when no project pin set

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow

    gen_metrics:
      semantic_adapter: metricflow
```

## Selection Rules

`AgentConfig.resolve_semantic_adapter` resolves the active semantic
adapter in this order — identical to BI Dashboard and Scheduler
resolution:

1. Explicit `adapter_type` argument at the call site (or
   `semantic_adapter` on the agentic node).
2. Project-level pin in `./.datus/config.yml`'s `semantic:` field.
3. Global `default: true` flag — at most one entry under
   `services.semantic_layer` may carry it; multiple defaults are
   rejected at config load time.
4. Single-entry shortcut when only one semantic adapter is configured.
5. Otherwise raise:
   - `No semantic layer configured ...` when the section is empty.
   - `Multiple semantic layers are configured ...` when there are
     multiple without a default.

The key under `services.semantic_layer` **must equal the adapter type**
(for example `metricflow`). If a `type:` field is present, it must match
the key; otherwise Datus raises a configuration error at startup.
Comparison is case-insensitive and trims surrounding whitespace.

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
- Toggle the **global** `default: true` flag with `d`. Pressing `d`
  marks the current row as default and clears the flag from every other
  entry.
- Pin a **project-level** default with `p` — the value lands in
  `./.datus/config.yml` as `semantic: <name>` and outranks the global
  flag for the current project only. Press `p` again on the pinned row
  to clear it.
- `e edit` is hidden on this tab: metricflow has no editable fields.

Service definitions are written to `~/.datus/conf/agent.yml` as
`services.semantic_layer.<type>: {type: <type>}`.

On the first interactive launch, if no project pin exists, Datus
auto-pins the only entry (or the one flagged `default: true`) to
`./.datus/config.yml` so subsequent runs are explicit. When multiple
entries are configured without a default, the launch prompts for a quick
choice. Set `DATUS_DISABLE_SERVICE_BOOTSTRAP=1` to opt out (CI / Docker).
