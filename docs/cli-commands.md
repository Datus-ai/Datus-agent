# Datus CLI Commands

## Setup Commands

### `datus configure`

Interactive wizard to configure LLM provider, database connections, and workspace settings.

Writes configuration to `~/.datus/conf/agent.yml` using the `service.databases` format.

```bash
datus configure
```

**Steps:**
1. **[1/3] Configure LLM** — Select provider (OpenAI, DeepSeek, Claude, etc.), enter API key, test connectivity
2. **[2/3] Configure Database** — Name the database, select type (duckdb, sqlite, snowflake, etc.), enter connection details, test connectivity
3. **[3/3] Configure Workspace** — Set workspace directory path

**Repeatable:** Yes. If `~/.datus/conf/agent.yml` already exists, you'll be asked to confirm overwrite. The existing configuration is fully replaced.

**Output:** `~/.datus/conf/agent.yml` with structure:
```yaml
agent:
  target: deepseek
  models:
    deepseek:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: sk-xxx
      model: deepseek-chat
  service:
    databases:
      my_duckdb:
        type: duckdb
        uri: ./data.duckdb
        default: true
    bi_tools: {}
    schedulers: {}
  project_root: ~/.datus/workspace
```

**To add more databases later:** Use `datus service add`.

---

### `datus init`

Initialize a project workspace by generating `AGENTS.md` in the current directory.

```bash
cd /path/to/your/project
datus init
```

**Requires:** A configured LLM (`datus configure` must be run first).

**What it does:**
1. Loads LLM and service configuration from `~/.datus/conf/agent.yml`
2. Scans current directory structure (up to 3 levels deep)
3. Detects project type (Python, Node.js, Docker, dbt, etc.)
4. Reads README.md if present
5. Uses LLM to generate `AGENTS.md` with project-specific content
6. Falls back to a template skeleton if LLM generation fails

**If AGENTS.md already exists:** Prompts to overwrite or cancel.

**Generated sections:**
- `## Architecture` — Project architecture description with ASCII diagram
- `## Directory Map` — Table mapping directories to purposes
- `## Services` — Configured databases and services from agent.yml
- `## Artifacts` — Data catalogs, semantic models, SQL files, etc.

---

## Service Management

### `datus service list`

Show all configured databases, BI tools, and schedulers.

```bash
datus service list
```

### `datus service add`

Interactively add a new database connection.

```bash
datus service add
```

### `datus service delete`

Interactively remove a database connection.

```bash
datus service delete
```

---

## Database Selection

### `--database` flag

Most commands require specifying which database to use:

```bash
datus-cli --database my_duckdb
datus run --database my_duckdb --task "show tables" --task_db_name demo
datus check-db --database my_duckdb
datus bootstrap-kb --database my_duckdb --components metadata
```

**Auto-selection:** If `--database` is not specified:
- If a database has `default: true` in config, it's auto-selected
- If only one database is configured, it's auto-selected
- Otherwise, a list of available databases is shown

**Legacy support:** `--database` still works as an alias for `--database`.

---

## Migration from Legacy Config

If you have an existing `agent.yml` using the old `namespace` format:

```bash
# Preview migration (dry run)
python -m datus.configuration.config_migrator --config conf/agent.yml --dry-run

# Migrate (backs up original to agent.yml.bak)
python -m datus.configuration.config_migrator --config conf/agent.yml
```

The old format is also auto-migrated at runtime — no manual migration required for existing configs.

**Old format:**
```yaml
agent:
  namespace:
    my_ns:
      type: sqlite
      dbs:
        - name: db1
          uri: ./db1.sqlite
```

**New format:**
```yaml
agent:
  service:
    databases:
      db1:
        type: sqlite
        uri: ./db1.sqlite
        default: true
```
