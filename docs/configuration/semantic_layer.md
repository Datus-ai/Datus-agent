# Semantic Layer Configuration

Semantic adapters are configured under `agent.services.semantic_layer`.
If the section is omitted or empty, Datus uses its built-in semantic
adapter default: `dosi`. Dosi supports both semantic authoring and queries.
MetricFlow and OSI remain available when explicitly configured, but only for
query compatibility with existing assets.

Semantic layer selection is global for the project. Node-level semantic
format fields from older configs are ignored; use this section to pin a
format explicitly.

## Structure

```yaml
agent:
  services:
    semantic_layer:
      dosi:
        default: true                   # optional; Dosi is also the built-in default
        # Native Dosi execution; semantic_model_path is only needed when the
        # datasource directory contains more than one OSI model file.

      metricflow:                       # query-only compatibility
        timeout: 300
        config_path: ./conf/agent.yml   # optional advanced override

      osi:                              # query-only compatibility
        # execution_backend defaults to metricflow and normally does not need
        # to be configured.
```

## Selection Rules

`AgentConfig.resolve_semantic_adapter` resolves the active semantic
adapter in this order — identical to BI Dashboard and Scheduler
resolution:

1. Explicit `adapter_type` argument at service-management call sites.
2. Project-level pin in `./.datus/config.yml`'s `semantic:` field.
3. Global `default: true` flag — at most one entry under
   `services.semantic_layer` may carry it; multiple defaults are
   rejected at config load time.
4. Single-entry shortcut when only one semantic adapter is configured.
5. Built-in default when the section is empty.

Multiple configured semantic adapters without a `default: true` entry are
rejected as ambiguous.

The key under `services.semantic_layer` **must equal the adapter type**
(for example `metricflow`). If a `type:` field is present, it must match
the key; otherwise Datus raises a configuration error at startup.
Comparison is case-insensitive and trims surrounding whitespace.

## MetricFlow Notes

- MetricFlow is not selected or installed by default. Configure `metricflow`
  explicitly to query an existing MetricFlow project.
- MetricFlow is query-only in Datus; use Dosi for new semantic authoring.
- `config_path` is optional.
- Datus prefers the current `services.datasources` entry and the project semantic model directory to build runtime config automatically.
- MetricFlow validation reads YAML files from the configured project semantic model directory directly, including generated files under gitignored project paths.
- `config_path` is only needed when you want MetricFlow to read a specific `agent.yml` file directly.

## OSI Notes

- OSI is an explicitly configured query-only compatibility adapter for existing strict OSI core YAML.
- The current OSI execution backend is MetricFlow by default. You normally do not need to set `execution_backend`.
- Install it with `pip install "datus-semantic-osi[metricflow]"`; the CLI uses the same extra so the query backend is present.
- Configure `services.semantic_layer.osi` and mark it `default: true` to select this path globally when other adapters are also configured. An empty `osi: {}` entry is selected automatically only when it is the sole semantic adapter, or when the current project pins `semantic: osi`.

## Dosi Notes

- Dosi uses the same OSI authoring format and DATUS `custom_extensions` workflow.
- `datus-semantic-dosi` executes directly through the native engine; it does not set `execution_backend: metricflow`.
- The adapter loads one OSI document per instance. If the datasource model directory contains several files, configure `semantic_model_path` explicitly.
- `pip install datus-semantic-dosi` installs the adapter and `dosi-engine` together.
- On interactive launch with no semantic-layer configuration, Datus installs
  and selects Dosi automatically. Non-interactive environments must install
  their selected adapter explicitly.

## Configuring through the CLI (`/services`)

Run `/services semantic` inside the Datus REPL (or press `Tab` from any
other tab) to enter the configuration TUI on the **Semantic** tab. The
tab lets you:

- Add a new semantic layer by pressing `Enter` on the trailing `+ Add
  new semantic` row. Choose the adapter type, such as `metricflow`, `osi`, or
  `dosi`. If the adapter package isn't installed, install the matching
  package first, for example `datus-semantic-metricflow`,
  `datus-semantic-osi[metricflow]`, or `datus-semantic-dosi`.
- Delete an entry with `x` and run a registration probe with `t`.
- Toggle the **global** `default: true` flag with `d`. Pressing `d`
  marks the current row as default and clears the flag from every other
  entry.
- Pin a **project-level** default with `p` — the value lands in
  `./.datus/config.yml` as `semantic: <name>` and outranks the global
  flag for the current project only. Press `p` again on the pinned row
  to clear it.
- `e edit` is hidden for adapters that have no editable fields.

Service definitions are written to `~/.datus/conf/agent.yml` as
`services.semantic_layer.<type>: {type: <type>}`.

On the first interactive launch, if no project pin exists, Datus
auto-pins the only entry (or the one flagged `default: true`) to
`./.datus/config.yml` so subsequent runs are explicit. When multiple
entries are configured without a default, the launch prompts for a quick
choice. Set `DATUS_DISABLE_SERVICE_BOOTSTRAP=1` to opt out (CI / Docker).
