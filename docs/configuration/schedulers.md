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

`get_scheduler_config(service_name)` resolves the active scheduler in this
order:

1. Explicit `service_name` argument at the call site.
2. Project-level pin in `./.datus/config.yml`'s top-level `scheduler:` field.
3. Global `default: true` flag (must be unique across `services.schedulers`).
4. Single-entry shortcut when only one scheduler is configured.

A stale project pin (the named service no longer exists in agent.yml) is
ignored with a warning so the lookup falls through to the global default.

## Configuring through the CLI (`/services`)

Run `/services scheduler` inside the Datus REPL to enter the
configuration TUI on the Scheduler tab (bare `/services` lands on the
Dashboard tab; `/services list` keeps the legacy read-only listing). The
two-tab TUI lets you:

- Add a new scheduler with `Enter` on the trailing `+ Add new scheduler`
  row. Only `airflow` (`datus-scheduler-airflow`) ships today; if the
  adapter package isn't installed yet, Datus runs `pip install` for you
  and hot-reloads the registry — no restart needed.
- Edit credentials with `e`, delete with `x`, run a connectivity probe
  with `t`.
- Toggle the **global** `default: true` flag with `d`. Pressing `d` on a
  row sets it as the workspace-wide default and clears the flag from every
  other entry, so you cannot end up with two defaults.
- Pin a **project-level** default with `p` — the value lands in
  `./.datus/config.yml` as `scheduler: <name>` and outranks the global
  flag for the current project only. Press `p` again on the pinned row to
  clear it.

Service definitions are written to `~/.datus/conf/agent.yml` (shared across
projects); only the active selection is project-local.

On the first interactive launch, if no project pin exists, Datus
auto-pins the only entry (or the one flagged `default: true`) to
`./.datus/config.yml` so subsequent runs are explicit. When multiple
entries are configured without a default, the launch prompts for a quick
choice. Set `DATUS_DISABLE_SERVICE_BOOTSTRAP=1` to opt out (CI / Docker).

## Notes

- `services.schedulers` is now the only runtime source for scheduler config.
- Top-level `scheduler:` is no longer read at runtime.
