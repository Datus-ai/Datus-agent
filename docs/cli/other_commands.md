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
/effort high                 # set directly
```

| Level | Behavior |
|-------|----------|
| `minimal` | Least reasoning, fastest |
| `low` | Less reasoning |
| `medium` | Balanced (default) |
| `high` | Most thorough, slowest |

Higher effort uses more tokens and takes longer. Not all providers/models support effort levels; unsupported models ignore the setting. The selection persists for the session.

### `/language`

Set the language the assistant replies in.

```text
/language                    # open the selector
/language zh                 # set directly
```

It affects only the assistant's natural-language responses, not SQL or code. The setting persists for the session.

## Setup & Service Commands

Datus is configured from inside the REPL with slash commands.

### `/datasource`

Add, edit, delete, or switch datasources (DuckDB, SQLite, Snowflake, MySQL, PostgreSQL, StarRocks, …). Changes are written to `agent.yml` under `services.datasources`.

### `/init` and `/build-kb`

`/init` and `/build-kb` bootstrap a project workspace — they delegate to built-in skills. `/init` does a fast, lightweight scan; `/build-kb` builds the vector-indexed knowledge base. See [Init](../skills/init.md) and [Build KB](../skills/build_kb.md) for details.

### `datus upgrade`

Upgrade `datus-agent` and every installed `datus-*` adapter package to the latest release in one run. Editable / source checkouts are skipped. Add `--check` to report the latest version without installing.

```bash
datus upgrade
datus upgrade --check
```

On an interactive launch, Datus also prints a one-line hint when a newer release is available. Set `DATUS_DISABLE_VERSION_CHECK=1` to silence it.
