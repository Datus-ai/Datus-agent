# Other Commands

This page collects the remaining CLI commands — runtime configuration and datasource/service setup — that don't have a dedicated page of their own.

## Configuration Commands

These slash commands adjust runtime behavior from inside the REPL. Each opens an interactive selector when run without arguments, or accepts a shortcut argument.

### `/model`

Switch the active LLM provider and model without editing YAML.

```text
/model                       # open the interactive selector
/model openai/gpt-4.1        # switch directly to a provider/model
/model openai                # open the selector at a provider
```

The selector groups choices into **Providers**, **Plans**, and **Custom** (self-hosted models from `agent.models`). Provider credentials live in `agent.yml`; `/model` only switches the active selection, which persists to `./.datus/config.yml` under `target`:

```yaml
target:
  provider: openai
  model: gpt-4.1
```

The change takes effect on the next query — no restart needed.

### `/effort`

Control the reasoning effort level for the LLM.

```text
/effort                      # open the selector
/effort high                 # set the level (asks for scope)
/effort high --project       # persist to ./.datus/config.yml
/effort high --global        # persist to agent.yml
/effort off                  # disable reasoning
```

| Level | Behavior |
|-------|----------|
| `off` | Disable reasoning |
| `minimal` | Least reasoning, fastest |
| `low` | Low effort |
| `medium` | Balanced (default) |
| `high` | Most thorough, slowest |

Higher effort uses more tokens and takes longer. The level maps to each provider's native dialect, so one knob covers every provider; models without reasoning support ignore it. The selection is persisted at project scope (`./.datus/config.yml`) or global scope (`agent.yml`) — use `/effort --clear` to remove a project override, and `/effort status` to show the effective level.

### `/language`

Set the language the assistant replies in.

```text
/language                    # open the selector
/language zh                 # set the language (asks for scope)
/language zh --project       # persist to ./.datus/config.yml
/language zh --global        # persist to agent.yml
```

Accepts language codes such as `en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`, `pt`, `ru`, `it`, plus `auto` to let the model decide. It affects only the assistant's natural-language responses, not SQL or code. The setting is persisted at project or global scope; use `/language --clear` to remove a project override.

## Setup & Service Commands

Datus is configured from inside the REPL with slash commands; a few datasource operations also have a non-interactive `datus-agent` surface for scripts and CI.

### `/datasource`

Add, edit, delete, or switch datasources (DuckDB, SQLite, Snowflake, MySQL, PostgreSQL, StarRocks, …). Changes are written to `agent.yml` under `services.datasources`.

### `/init` and `/build-kb`

`/init` and `/build-kb` bootstrap a project workspace — they delegate to built-in skills. `/init` does a fast, lightweight scan; `/build-kb` builds the vector-indexed knowledge base. See [Init](../skills/init.md) and [Build KB](../skills/build_kb.md) for details.

### `datus-agent service`

A non-interactive surface for the same datasource CRUD operations, handy in scripts or CI:

```bash
datus-agent service list      # show configured datasources, adapters, BI platforms, schedulers
datus-agent service add       # add a datasource interactively
datus-agent service delete    # remove a datasource interactively
```

### `--datasource` flag

`datus-agent` subcommands require `--datasource` to pick which datasource to use:

```bash
datus-cli --datasource my_duckdb
datus-agent run --datasource my_duckdb --task "show tables" --task_db_name demo
```

Interactive `datus-cli` can auto-select when the configuration is unambiguous: a datasource marked `default: true`, or the only one configured, is chosen automatically; otherwise you pick from a list.

### `datus upgrade`

Upgrade `datus-agent` and every installed `datus-*` adapter package to the latest release in one run. Editable / source checkouts are skipped. Add `--check` to report the latest version without installing.

```bash
datus upgrade
datus upgrade --check
```

On an interactive launch, Datus also prints a one-line hint when a newer release is available. Set `DATUS_DISABLE_VERSION_CHECK=1` to silence it.
