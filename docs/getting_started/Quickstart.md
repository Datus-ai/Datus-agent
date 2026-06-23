# Quickstart

Get started with Datus Agent in minutes: install → configure → first query.

!!! tip "Need the full warehouse workflow?"
    For an end-to-end example covering layered warehouse design, ETL generation, Airflow scheduling, semantic assets, and Superset dashboards, see [Data Engineering Quickstart](./data_engineering_quickstart.md).

## 1. Install

One-liner for Linux / macOS (recommended):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

The script bootstraps `uv`, creates a dedicated venv at `~/.datus/venv` (Python 3.12 is downloaded automatically if missing), and writes `datus`, `datus-cli`, `datus-api`, `datus-mcp`, `datus-pip`, etc. into `~/.local/bin`. Open a new shell (or `source ~/.zshrc`) so the new PATH takes effect.

??? note "Other install methods"
    **Pin a released version** (the variable is passed to the receiving shell, not to `curl`):
    ```bash
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.2.6 sh
    ```

    **From GitHub source** (unreleased changes on `main`, or any branch / tag / commit):
    ```bash
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
    ```

    **Managed Python environment** (Python 3.12 required):
    ```bash
    # Pick one: Conda / virtualenv / uv venv. After activating:
    pip install datus-agent
    ```

    Other variables: `DATUS_HOME`, `DATUS_BIN_DIR`, `DATUS_FORCE=1`, `DATUS_NO_MODIFY_PATH=1`. To install extra Python packages into the same venv later, use `datus-pip install <package>`.

    Pre-release builds: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent`.

## 2. Configure & Init

Launch the REPL:

```bash
datus
```

Inside the REPL, run `/datasource`, `/model`, and `/init` in turn.

### Datasource

Run `/datasource`. The TUI prompts for name, type (DuckDB, SQLite, MySQL, PostgreSQL, Snowflake, StarRocks, …) and connection details, tests connectivity, and writes the result to `~/.datus/conf/agent.yml`. The same TUI also handles edit / delete / set-default / plugin install. Switch at runtime with `/datasource <name>`.

!!! tip "Demo Database"
    Datus ships with a pre-configured demo DuckDB database at `~/.datus/sample/duckdb-demo.duckdb`. In `/datasource`, pick `duckdb` and point at this path for an instant working datasource.

### Model

Run `/model`. The TUI lists all providers; selecting one prompts for the API key (auto-detecting common env vars). Direct shortcuts like `/model openai/gpt-4.1` are also supported.

Common providers:

| Provider | Default model | Env var |
|---|---|---|
| `openai` | `gpt-4.1` | `OPENAI_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `claude` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` |

Full provider list (Kimi / Qwen / GLM / MiniMax, Claude subscription, Codex OAuth, Coding Plan, …) lives in [Model Command](../cli/other_commands.md#model).

### Init (optional)

`cd` into your project, launch `datus`, and run `/init` to scan the directory and generate a project-level `AGENTS.md` using the model + datasource you saved above. To target a different datasource, run `/datasource <name>` before `/init`.

## 3. Start Using Datus

You'll see the startup banner and a `>` prompt. The prompt accepts three input modes:

- **Slash commands** — `/help`, `/datasource`, `/model`, `/exit`, …
- **SQL** — `SELECT …`, `DESCRIBE …`, `SHOW …` are detected automatically and executed against the active datasource
- **Natural language** — anything else goes to the agent

```text title="Examples"
> /tables
> desc gold_vs_bitcoin
> Detailed analysis of gold–Bitcoin correlation.
```

For natural-language turns, Datus streams thinking deltas, tool calls, SQL, and the final markdown report live, with a pinned status row at the bottom showing the currently running tool:

```text
● Let me check the schema of gold_vs_bitcoin and run a correlation analysis.
● describe_table({"table_name": "gold_vs_bitcoin"})  ✓ 3 columns (0.5s)
● read_query({"sql": "SELECT CORR(gold, bitcoin) ..."}) ✓ 1 row (0.5s)
○ Running read_query …
```

!!! tip "Trace details"
    Press **Ctrl+O** at any time to open the inline trace for the previous turn (full tool inputs, SQL, raw outputs). Press it again or `q` to close.

## Next Steps

- **[Data Engineering Quickstart](./data_engineering_quickstart.md)** — layered warehouse + Airflow + Superset, end-to-end
- **[Contextual Data Engineering](./contextual_data_engineering.md)** — `@` references, knowledge base, context management
- **[Configuration Guide](../configuration/introduction.md)** — connect your own databases and customize settings
- **[CLI Reference](../cli/introduction.md)** — all commands and options
- **[Semantic Adapters](../adapters/semantic_adapters.md)** — datus-semantic-metricflow
