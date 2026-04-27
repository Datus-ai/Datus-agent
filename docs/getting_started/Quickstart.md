# Quickstart

Get started with Datus Agent in just a few minutes. This guide will walk you through installation, setup, and your first interactions with Datus.

!!! tip "Need the full warehouse workflow?"
    For an end-to-end example that covers layered warehouse design, ETL generation, Airflow scheduling, semantic assets, and Superset dashboards, see [Data Engineering Quickstart](./data_engineering_quickstart.md).

## Step 1: Install

### Default — one-liner (Linux / macOS)

Stable install from PyPI — recommended for most users:

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

The script bootstraps `uv`, creates a dedicated venv at `~/.datus/venv` (Python 3.12 is downloaded automatically if missing), and writes `datus`, `datus-cli`, `datus-api`, `datus-mcp`, `datus-agent`, `datus-gateway`, and `datus-pip` shims into `~/.local/bin`. Open a new shell (or `source ~/.zshrc`) so the new PATH takes effect.

Pin a released version (the variable is passed to the receiving shell, not to `curl`):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.2.6 sh
```

Dev install from GitHub source (picks up unreleased changes on `main`, or any branch / tag / commit):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
# or pin to a specific ref
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
```

Other variables supported by both scripts: `DATUS_HOME`, `DATUS_BIN_DIR`, `DATUS_FORCE=1`, `DATUS_NO_MODIFY_PATH=1`. To install additional packages into the global venv later, use `datus-pip install <package>` (a shim for `~/.datus/venv/bin/pip`).

### Custom — managed Python environment

Use this path if you maintain your own Python and prefer to install `datus-agent` into an existing virtualenv / conda env. Datus requires Python 3.12.

=== "Conda"

    ```bash
    conda create -n datus python=3.12
    conda activate datus
    pip install datus-agent
    ```

=== "virtualenv"

    ```bash
    virtualenv datus --python=python3.12
    source datus/bin/activate
    pip install datus-agent
    ```

=== "uv"

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    uv pip install datus-agent
    ```

!!! note
    If `pip install` complains about an old pip, upgrade it first:

    ```bash
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip setuptools wheel
    ```

For pre-release builds: `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent`.

### Configure LLM & Datasource

Configuration lives entirely inside the REPL. Launch Datus first:

```bash
datus
```

Then use two slash commands — order doesn't matter, but `/datasource` first is convenient because the data preview reflects the active datasource:

**1. Add or switch a datasource** — Run `/datasource`:

```text
Datus> /datasource
```

`/datasource` opens a TUI that lists the datasources in `agent.yml`, plus an "Add" entry. Adding prompts for name, type (DuckDB, SQLite, Snowflake, MySQL, PostgreSQL, StarRocks, …) and connection details, tests connectivity, and writes the result to `~/.datus/conf/agent.yml`. The same TUI also handles edit / delete / set-default / plugin install. Switch at runtime with `/datasource <name>`.

!!! tip "Demo Database"
    Datus ships with a pre-configured demo DuckDB database at `~/.datus/sample/duckdb-demo.duckdb`. In `/datasource`, pick `duckdb` and point at this path to get a working datasource immediately.

**2. Pick an LLM** — Run `/model`:

```text
Datus> /model
```

`/model` opens a TUI picker covering the providers below. Selecting a provider prompts for the API key (auto-detecting common env vars such as `OPENAI_API_KEY`) and writes the result to `agent.yml`. Direct shortcuts like `/model openai/gpt-4.1` are also supported. See [Model Command](../cli/model_command.md) for full details.

#### Built-in LLM providers

| Provider | Typical models | Auth | Best for |
|---|---|---|---|
| `openai` | `gpt-5.2`, `gpt-4.1`, `o3` | API key with `OPENAI_API_KEY` auto-detection | General chat, reasoning, tool use |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` | API key with `DEEPSEEK_API_KEY` auto-detection | Cost-effective reasoning and SQL generation |
| `claude` | `claude-sonnet-4-5`, `claude-opus-4-5` | API key with `ANTHROPIC_API_KEY` auto-detection | Long context and complex reasoning |
| `kimi` | `kimi-k2.5`, `kimi-k2-thinking` | API key with `KIMI_API_KEY` auto-detection | Chinese-heavy workloads and long context |
| `qwen` | `qwen3-max`, `qwen3-coder-plus` | API key with `DASHSCOPE_API_KEY` auto-detection | Chinese workloads, general chat, coding |
| `gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` | API key with `GEMINI_API_KEY` auto-detection | Large-context analysis |
| `minimax` | `MiniMax-M2.7`, `MiniMax-M2.5` | API key | General reasoning |
| `glm` | `glm-5`, `glm-4.7` | API key | Chinese workloads and tool-calling |

There are also two special auth flows:

| Provider | Auth | Notes |
|---|---|---|
| `claude_subscription` | Claude subscription token | The wizard first tries to auto-detect a local Claude subscription credential, then falls back to manual token input |

!!! note
    `codex` (ChatGPT Plus/Pro via Codex OAuth) is also exposed by `/model`; selecting it triggers a browser-based OAuth flow. See [Model Command](../cli/model_command.md#codex-chatgpt-pluspro-oauth) for caveats.

#### What are Coding Plan providers

`/model` also exposes a **Plans** tab with coding/plan-oriented providers. These use Anthropic-compatible endpoints, but from Datus's perspective they are configured just like any other model entry in `agent.models`.

| Provider | Default model | Best for |
|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | One coding endpoint that can serve Qwen / GLM / Kimi / MiniMax models |
| `glm_coding` | `glm-5` | GLM coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | MiniMax coding endpoint |
| `kimi_coding` | `kimi-for-coding` | Kimi coding endpoint |

These are a good fit when:

- You want the default model to lean toward planning, coding, or structured decomposition
- You use [Plan Mode](../cli/plan_mode.md) frequently for multi-step tasks
- You want to separate general-purpose chat models from coding/plan models and bind them to different nodes later

!!! tip "Environment variables and model overrides"
    For OpenAI, DeepSeek, Claude, Kimi, Qwen, and Gemini, `/model` prompts with provider-specific environment variable hints.

    For `minimax`, `glm`, and the `*_coding` providers, you can still enter environment-variable references directly, such as `${MINIMAX_API_KEY}`, `${GLM_API_KEY}`, `${KIMI_API_KEY}`, or `${DASHSCOPE_API_KEY}`.

    The current implementation also auto-applies required parameter overrides for some models, such as `kimi-k2.5` and `qwen3-coder-plus`.

### Initialize Project (Optional)

`cd` into your project directory, launch `datus`, and run `/init` inside the REPL:

```text
Datus> /init
```

This generates an `AGENTS.md` file describing your project's architecture, directory structure, services, and data assets. The LLM analyzes your directory and README, and (when the REPL has a datasource selected) the datasource's table list, to produce the content. To target a different datasource, switch with `/datasource <name>` first, then run `/init`.

## Step 2: Launch Datus

Start the REPL — `datus` is the canonical entry (it's a shim for `datus-cli`):

```bash title="Terminal"
datus
# or pin to a specific datasource at launch
datus --datasource duckdb-demo
```

You'll see the unified startup banner inside a Rich panel:

```text title="Startup"
╭─ v0.2.7 ─────────────────────────────────────────────────╮
│                                                          │
│  ██████╗   █████╗  ████████╗ ██╗   ██╗ ███████╗          │
│  ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║   ██║ ██╔════╝          │
│  ██║  ██║ ███████║    ██║    ██║   ██║ ███████╗          │
│  ██║  ██║ ██╔══██║    ██║    ██║   ██║ ╚════██║          │
│  ██████╔╝ ██║  ██║    ██║    ╚██████╔╝ ███████║          │
│  ╚═════╝  ╚═╝  ╚═╝    ╚═╝     ╚═════╝  ╚══════╝          │
│                                                          │
│  Data engineering agent builds evolvable context for     │
│  your data system                                        │
│                                                          │
│  Datasource  duckdb-demo  (duckdb)                       │
│                                                          │
│  Type / for commands, /help for the full list, /exit     │
│  to quit                                                 │
│                                                          │
╰──────────────────────────────────────────────────────────╯
Datus>
```

The banner shows the active datasource (or `not selected` if none picked yet) and any preloaded context. Three input modes share the same prompt:

- **Slash commands** — start with `/` (e.g., `/help`, `/datasource`, `/model`, `/exit`)
- **SQL** — `SELECT …`, `DESCRIBE …`, `SHOW …` and friends are detected automatically and executed against the active datasource
- **Natural language** — anything else goes to the agent

## Step 3: Start Using Datus

!!! tip
    You can execute SQL in Datus just like in a SQL editor.

List all tables in the active datasource:

```text title="Terminal"
Datus> /tables
```

The result is rendered as a Rich table headed with the datasource name, e.g. `Tables in duckdb-demo` followed by rows of `Table Name`. Type `desc <table>` (or any standard SQL the dialect understands) to inspect a single table:

```text title="Terminal"
Datus> desc gold_vs_bitcoin
```

DuckDB returns the column metadata (`column_name`, `column_type`, `null`, `key`, `default`, `extra`) plus the row count and elapsed time below the table.

!!! tip
    To chat with the agent, just type your question in natural language — no prefix needed. Lines starting with `/` are reserved for slash commands.

Suppose we want to understand the correlation between gold and Bitcoin. We can ask the Datus agent directly:

```bash title="Terminal"
Datus> Detailed analysis of gold–Bitcoin correlation.
```

Datus streams the agent's progress live: thinking deltas, tool calls, SQL queries, and the final markdown answer all flow through the prompt area, while a pinned status row at the bottom shows the currently running tool. Concretely you will see something like:

```text title="Streaming output"
● Thinking
  Let me check the schema of gold_vs_bitcoin and run a correlation analysis.

● Tool call · describe_table({"table_name": "gold_vs_bitcoin"})  ✓ 3 columns (0.5s)

● Tool call · read_query({"sql": "SELECT CORR(gold, bitcoin) ..."}) ✓ 1 row (0.5s)

● Tool call · read_query({"sql": "WITH daily_aggregates AS (...)"}) ✓ 1 row (0.5s)

⠋ Running read_query …                                              ← pinned row
```

When the turn finishes, the pinned row clears and the agent's full markdown report is rendered inline.

**Generated SQL:**

```sql title="Generated Query"
-- Generated SQL (copied)
SELECT
    -- Overall correlation and basic stats
    CORR(gold, bitcoin) as correlation_coefficient,
    COUNT(*) as total_observations,
    AVG(gold) as avg_gold_price,
    AVG(bitcoin) as avg_bitcoin_price,
    STDDEV(gold) as gold_volatility,
    STDDEV(bitcoin) as bitcoin_volatility,
    COVAR_POP(gold, bitcoin) as covariance,

    -- Price movement patterns
    (SELECT COUNT(*) FROM (
        SELECT
            CASE
                WHEN gold_change > 0 AND bitcoin_change > 0 THEN 'Both Up'
                WHEN gold_change < 0 AND bitcoin_change < 0 THEN 'Both Down'
                WHEN gold_change > 0 AND bitcoin_change < 0 THEN 'Gold Up, Bitcoin Down'
                WHEN gold_change < 0 AND bitcoin_change > 0 THEN 'Gold Down, Bitcoin Up'
            END as pattern
        FROM (
            SELECT
                gold - LAG(gold) OVER (ORDER BY time) as gold_change,
                bitcoin - LAG(bitcoin) OVER (ORDER BY time) as bitcoin_change
            FROM gold_vs_bitcoin
        ) WHERE gold_change IS NOT NULL
    ) WHERE pattern = 'Both Up') as both_up_count,

    (SELECT COUNT(*) FROM (
        SELECT
            CASE
                WHEN gold_change > 0 AND bitcoin_change > 0 THEN 'Both Up'
                WHEN gold_change < 0 AND bitcoin_change < 0 THEN 'Both Down'
                WHEN gold_change > 0 AND bitcoin_change < 0 THEN 'Gold Up, Bitcoin Down'
                WHEN gold_change < 0 AND bitcoin_change > 0 THEN 'Gold Down, Bitcoin Up'
            END as pattern
        FROM (
            SELECT
                gold - LAG(gold) OVER (ORDER BY time) as gold_change,
                bitcoin - LAG(bitcoin) OVER (ORDER BY time) as bitcoin_change
            FROM gold_vs_bitcoin
        ) WHERE gold_change IS NOT NULL
    ) WHERE pattern = 'Both Down') as both_down_count
FROM gold_vs_bitcoin
```

**Analysis Results:**

??? success "Gold-Bitcoin Correlation Analysis"
    ```{ .yaml .no-copy }
    +===============================================================================================================+
    |                                        Gold-Bitcoin Correlation Analysis                                     |
    +===============================================================================================================+

                                                          📊 Overall Correlation

    • Correlation Coefficient: -0.225 (Weak negative correlation)
    • Total Observations: 954 data points
    • Time Period: Single day - December 1, 2024 (16:00 to 16:15)


                                                            💰 Price Statistics

    • Average Gold Price: 0.684
    • Average Bitcoin Price: 0.785
    • Gold Volatility: 0.200
    • Bitcoin Volatility: 0.233
    • Covariance: -0.0105


                                                        📈 Price Movement Patterns

    Based on 953 observed price changes:

    • Both Assets Moving Up: 216 times (22.67%)
    • Both Assets Moving Down: 203 times (21.30%)
    • Gold Up, Bitcoin Down: 189 times (19.83%)
    • Gold Down, Bitcoin Up: 190 times (19.94%)


                                                              🎯 Key Insights

                                                       1. Weak Negative Correlation

    The -0.225 correlation suggests these assets tend to move in opposite directions more often than together, though the relationship
    is weak.

                                                       2. Similar Movement Patterns

    Both assets show remarkably similar movement patterns:

    • Approximately 44% of the time they move in the same direction
    • Approximately 40% of the time they move in opposite directions

                                                         3. Volatility Comparison

    Bitcoin shows slightly higher volatility (0.233) compared to Gold (0.200), which aligns with Bitcoin's reputation as a more
    volatile asset.

                                                       4. Risk-Adjusted Performance

    • Gold Sharpe Ratio: 3.42
    • Bitcoin Sharpe Ratio: 3.37 Both assets show strong risk-adjusted returns during this period.


                                                               ⚠️ Limitations

    This analysis is based on a very short time window (15 minutes of data). For more robust correlation analysis, data spanning
    multiple days/weeks would be needed to observe:

    • Longer-term correlation trends
    • Market regime changes
    • Impact of macroeconomic events
    • Seasonal patterns


                                                        💡 Investment Implications

    The weak negative correlation suggests these assets could provide some diversification benefits, though the relationship is not
    strong enough for reliable hedging strategies based on this limited dataset.
    ```

!!! tip "Trace details"
    Press **Ctrl+O** at any time to open the inline trace details for the last chat turn — full tool inputs, full SQL, raw tool outputs. Press **Ctrl+O** again (or `q`) to close it. There is no `(y/n)` prompt at the end of a turn anymore.

## Step 4: Advanced Features

### Context Management

!!! note
    If you have initialized the knowledge base for metadata.

If you want to add items such as a file, metric or table to the context, use `@`. Manual retrieval gives the LLM more precise context, narrows the search scope, and improves answer accuracy.

```bash title="Terminal"
Datus> @Table duckdb-demo.main.bank_failures
```

Datus will automatically analyze the table and add its metadata to the context, then stream the answer the same way as a regular chat turn — thinking deltas, tool calls (`describe_table`, `read_query`, …), and a markdown report. Press **Ctrl+O** to inspect the raw tool inputs / outputs for the turn.

??? example "Context Analysis Output (excerpt)"
    ```text
    ● Tool call · describe_table({"table_name": "bank_failures"})  ✓ 7 columns (0.5s)
    ● Tool call · read_query({"sql": "-- 1. Basic overview ..."})  ✓ 1 row   (0.5s)
    ● Tool call · read_query({"sql": "-- 2. Yearly analysis ..."}) ✓ 14 rows (0.5s)
    ● Tool call · read_query({"sql": "-- 3. State-wise ..."})      ✓ 15 rows (0.5s)
    ● Tool call · read_query({"sql": "-- 7. Recent 2023-2024 ..."}) ✓ 8 rows (0.5s)
    ```

The agent then renders a markdown report (executive summary, geographic distribution, asset-size analysis, largest failures, recent trends, key insights). The exact shape depends on the LLM and the data; treat the example above as illustrative.

!!! tip
    For more command references and options, see [CLI](../cli/introduction.md) or simply type `/help`.

## Next Steps

Now that you're up and running with Datus, explore more advanced features:

- **[Data Engineering Quickstart](./data_engineering_quickstart.md)** - Build a layered warehouse from DAComp, schedule it in Airflow, and publish it to Superset
- **[Contextual Data Engineering](./contextual_data_engineering.md)** - Learn how to use data assets as context
- **[Configuration Guide](../configuration/introduction.md)** - Connect to your own databases and customize settings
- **[CLI Reference](../cli/introduction.md)** - Discover all available commands and options
- **[Semantic Adapters](../adapters/semantic_adapters.md)** - Generate and query metrics with datus-semantic-metricflow
