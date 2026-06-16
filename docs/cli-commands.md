# Datus CLI Commands

## Setup Commands

Datus is configured entirely from inside the REPL. After installation, launch
`datus` and use slash commands:

| Command | Purpose |
|---------|---------|
| [`/model`](cli/model_command.md) | Pick an LLM provider, capture credentials, persist to `~/.datus/conf/agent.yml` |
| [`/datasource`](cli/reference.md) | Add / edit / delete / switch datasources (DuckDB, SQLite, Snowflake, MySQL, PostgreSQL, StarRocks, …); writes to `~/.datus/conf/agent.yml` under `services.datasources` |
| [`/init`](cli/init_command.md) | Lightweight init: scans files + database metadata, writes an `AGENTS.md` inventory plus file-based knowledge/memory (no vector KB, no confirmation gate); accepts optional free-text goal/scope hints |
| [`/build-kb`](cli/build_kb_command.md) | Build the vector-indexed KB — semantic models / metrics / reference SQL — after you confirm a generation manifest, then refresh the `AGENTS.md` index; accepts optional free-text file/table/domain scope |

The resulting `~/.datus/conf/agent.yml` looks like:

```yaml
agent:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
  services:
    datasources:
      my_duckdb:
        type: duckdb
        uri: ./data.duckdb
        default: true
    semantic_layer: {}
    bi_platforms: {}
    schedulers: {}
  project_root: ~/.datus/workspace
```

`datus-agent service add | list | delete` provide a non-interactive surface
for the same datasource CRUD operations (handy in scripts or CI).

---

## Service Management

### `datus-agent service list`

Show all configured datasources, semantic adapters, BI platforms, and schedulers.

```bash
datus-agent service list
```

### `datus-agent service add`

Interactively add a new datasource. Equivalent to running `/datasource` inside the REPL and choosing **Add**.

```bash
datus-agent service add
```

### `datus-agent service delete`

Interactively remove a datasource.

```bash
datus-agent service delete
```

---

## Database Selection

### `--datasource` flag

`datus-agent` subcommands require specifying which datasource to use. Interactive `datus-cli` can auto-select a datasource when the configuration is unambiguous.

```bash
datus-cli --datasource my_duckdb
datus-agent run --datasource my_duckdb --task "show tables" --task_db_name demo
datus-agent check-db --datasource my_duckdb
datus-agent bootstrap-kb --datasource my_duckdb --components metadata
```

**Interactive CLI auto-selection:** If `--datasource` is not specified for `datus-cli`:
- If a database has `default: true` in config, it's auto-selected
- If only one database is configured, it's auto-selected
- Otherwise, a list of available databases is shown

The old database-selection flag is no longer accepted by the current CLI; use `--datasource`.

---

## Self-Upgrade

### `datus upgrade`

Upgrade `datus-agent` and every installed `datus-*` adapter package to the
latest release in one `uv` / `pip` run. Third-party dependencies are bumped only
when a new datus release requires it; editable / source (`git`) checkouts are
skipped. Add `--check` to report the latest version without installing.

```bash
datus upgrade
datus upgrade --check
```

On an interactive launch, `datus` also prints a one-line hint when a newer
release is available. Set `DATUS_DISABLE_VERSION_CHECK=1` to silence it.

---
