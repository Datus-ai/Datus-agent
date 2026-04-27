# 快速开始

几分钟内即可上手 Datus Agent。本指南将带你完成安装、配置和首次体验。

!!! tip "想直接走完整数仓链路？"
    如果你要体验分层建模、ETL 生成、Airflow 调度、语义资产生成和 Superset 仪表盘发布，请继续阅读 [数据工程快速开始](./data_engineering_quickstart.zh.md)。

## 步骤 1：安装

### 默认方式：一键安装（Linux / macOS）

从 PyPI 安装稳定版——大多数用户用这一种就够：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

脚本会自动 bootstrap `uv`，在 `~/.datus/venv` 下建一个独立 venv（缺 Python 3.12 时自动下载），并把 `datus`、`datus-cli`、`datus-api`、`datus-mcp`、`datus-agent`、`datus-gateway`、`datus-pip` 等 shim 写入 `~/.local/bin`。开新 shell（或 `source ~/.zshrc`）让 PATH 生效。

固定版本（变量传给接收脚本的 shell，不是 `curl`）：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.2.6 sh
```

直接装 GitHub 源（拿 `main` 上未发布的改动，或任意分支 / tag / commit）：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
# 或固定 ref
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
```

两个脚本都支持 `DATUS_HOME`、`DATUS_BIN_DIR`、`DATUS_FORCE=1`、`DATUS_NO_MODIFY_PATH=1` 等变量。后续要往这个 venv 里装别的 Python 包时，用 `datus-pip install <package>`（即 `~/.datus/venv/bin/pip` 的 shim）。

### 自定义方式：自管 Python 环境

如果你想自己管理 Python，把 `datus-agent` 装进已有 virtualenv / conda 环境，可以走这条路。Datus 需要 Python 3.12。

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
    如果 `pip install` 报 pip 版本太旧，先升级:

    ```bash
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip setuptools wheel
    ```

预发布版本：`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent`。

### 配置 LLM 与数据源

配置全部在 REPL 内完成。先启动 Datus：

```bash
datus
```

然后用两个斜杠命令——顺序无所谓，但建议先 `/datasource`，因为后续数据预览会跟随当前数据源：

**1. 添加或切换数据源** —— 运行 `/datasource`：

```text
Datus> /datasource
```

`/datasource` 会打开一个 TUI，列出 `agent.yml` 里的全部数据源,并提供顶部的"Add"入口。新增时会询问名称、类型（DuckDB、SQLite、Snowflake、MySQL、PostgreSQL、StarRocks 等）与连接信息,测试连通性,并写回 `~/.datus/conf/agent.yml`。同一个 TUI 还支持编辑 / 删除 / 设置默认 / 自动安装缺失的适配器插件。运行时切换可直接 `/datasource <name>`。

!!! tip "演示数据库"
    Datus 自带预配置的 DuckDB 演示库，路径为 `~/.datus/sample/duckdb-demo.duckdb`。在 `/datasource` 中选 `duckdb` 并指向该路径即可立即获得一个可用的数据源。

**2. 选 LLM** —— 运行 `/model`：

```text
Datus> /model
```

`/model` 会打开一个 TUI 选择器，覆盖下表中所有 provider。选中后会要求输入 API Key（自动识别常见环境变量如 `OPENAI_API_KEY`），并写入 `agent.yml`。也支持 `/model openai/gpt-4.1` 这类直接快捷写法。完整说明见 [Model 命令](../cli/model_command.zh.md)。

#### 内置 LLM provider

| Provider | 典型模型 | 认证方式 | 适用场景 |
|---|---|---|---|
| `openai` | `gpt-5.2`、`gpt-4.1`、`o3` | API Key（支持自动识别 `OPENAI_API_KEY`） | 通用对话、推理、工具调用 |
| `deepseek` | `deepseek-chat`、`deepseek-reasoner` | API Key（支持自动识别 `DEEPSEEK_API_KEY`） | 性价比较高的通用推理与 SQL 生成 |
| `claude` | `claude-sonnet-4-5`、`claude-opus-4-5` | API Key（支持自动识别 `ANTHROPIC_API_KEY`） | 长上下文、复杂推理 |
| `kimi` | `kimi-k2.5`、`kimi-k2-thinking` | API Key（支持自动识别 `KIMI_API_KEY`） | 中文场景、长上下文 |
| `qwen` | `qwen3-max`、`qwen3-coder-plus` | API Key（支持自动识别 `DASHSCOPE_API_KEY`） | 中文场景、通用对话与编码 |
| `gemini` | `gemini-2.5-flash`、`gemini-2.5-pro` | API Key（支持自动识别 `GEMINI_API_KEY`） | 超长上下文、多轮分析 |
| `minimax` | `MiniMax-M2.7`、`MiniMax-M2.5` | API Key | 通用对话与推理 |
| `glm` | `glm-5`、`glm-4.7` | API Key | 中文场景、推理与工具调用 |

另外还有两类特殊入口：

| Provider | 认证方式 | 说明 |
|---|---|---|
| `claude_subscription` | Claude 订阅 token | 向导会优先自动探测本地 Claude 订阅凭据，探测失败时可手动粘贴 `sk-ant-oat01-...` |

!!! note
    `codex`（ChatGPT Plus/Pro 通过 Codex OAuth）也由 `/model` 暴露，选择后会触发浏览器 OAuth 流程。注意事项详见 [Model 命令](../cli/model_command.zh.md)。

#### Coding Plan providers 是什么

`/model` 还提供一个 **Plans** Tab，列出面向编码/规划场景的 provider。它们底层使用 Anthropic-compatible endpoint，但在 Datus 里和普通模型一样，最终都会写入 `agent.models`，可以设为默认模型，也可以在节点级单独引用。

| Provider | 默认模型 | 适合场景 |
|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | 希望在同一个 coding endpoint 下使用 Qwen / GLM / Kimi / MiniMax 等模型 |
| `glm_coding` | `glm-5` | 使用 GLM 的 coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | 使用 MiniMax 的 coding endpoint |
| `kimi_coding` | `kimi-for-coding` | 使用 Kimi 的 coding endpoint |

这类 provider 适合以下场景：

- 你希望默认模型更偏向规划、编码或结构化任务拆解
- 你会频繁使用 [计划模式](../cli/plan_mode.zh.md) 处理复杂任务
- 你想把通用聊天模型和 coding/plan 模型分开配置，后续按节点指定

!!! tip "环境变量与参数覆盖"
    对 OpenAI、DeepSeek、Claude、Kimi、Qwen、Gemini，`/model` 会自动提示对应的环境变量。

    对 `minimax`、`glm` 以及各类 `*_coding` provider，即使没有内置自动提示，你仍然可以直接输入 `${MINIMAX_API_KEY}`、`${GLM_API_KEY}`、`${KIMI_API_KEY}`、`${DASHSCOPE_API_KEY}` 这类环境变量引用。

    其中 `kimi-k2.5` 和 `qwen3-coder-plus` 会自动附带当前实现要求的参数覆盖，例如 `temperature` 和 `top_p`。

### 初始化项目（可选）

`cd` 进入你的项目目录,启动 `datus`,然后在 REPL 中运行 `/init`：

```text
Datus> /init
```

`/init` 会读取上一步通过 `/model` 与 `/datasource` 保存到 `agent.yml` 的默认模型与数据源,扫描当前目录,并生成项目级 `AGENTS.md`。当 REPL 已选中数据源时,会自动把该数据源的表清单也加入 LLM 上下文。需要换数据源时,先用 `/datasource <name>` 切换,再运行 `/init`。

## 步骤 2：启动 Datus

REPL 的标准入口就叫 `datus`（它是 `datus-cli` 的 shim）：

```bash title="Terminal"
datus
# 或在启动时直接绑定数据源
datus --datasource duckdb-demo
```

启动时会显示统一 banner（Rich Panel）：

```text title="启动样例"
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

Banner 会展示当前数据源（未选择时为 `not selected`）以及预加载的上下文。提示符接受三种输入：

- **斜杠命令** —— 以 `/` 开头（如 `/help`、`/datasource`、`/model`、`/exit`）
- **SQL** —— `SELECT …`、`DESCRIBE …`、`SHOW …` 等会被自动识别并直接对当前数据源执行
- **自然语言** —— 其余输入都交给 agent

## 步骤 3：开始使用 Datus

列出当前数据源的全部表：

```text title="Terminal"
Datus> /tables
```

结果以 Rich 表格渲染，标题取自当前数据源（如 `Tables in duckdb-demo`），下方按行列出 `Table Name`。要看单表结构，输入 `desc <table>`（或对应方言的 SQL 即可）：

```text title="Terminal"
Datus> desc gold_vs_bitcoin
```

DuckDB 会返回列元数据（`column_name`、`column_type`、`null`、`key`、`default`、`extra`），表格下方显示行数与耗时。

!!! tip
    要与智能体对话，直接输入自然语言即可，无需任何前缀。`/` 开头的输入仅用于斜杠命令。

想要了解黄金与比特币之间的相关性，可以直接向 Datus 提问：

```bash title="Terminal"
Datus> Detailed analysis of gold–Bitcoin correlation.
```

Datus 会实时把 agent 的进度流式推送到提示框：思考增量、工具调用、SQL、最终 markdown 答案逐步出现，同时底部有一行 pinned 状态条提示当前正在跑的工具，例如：

```text title="流式输出"
● Thinking
  Let me check the schema of gold_vs_bitcoin and run a correlation analysis.

● Tool call · describe_table({"table_name": "gold_vs_bitcoin"})  ✓ 3 columns (0.5s)

● Tool call · read_query({"sql": "SELECT CORR(gold, bitcoin) ..."}) ✓ 1 row (0.5s)

● Tool call · read_query({"sql": "WITH daily_aggregates AS (...)"}) ✓ 1 row (0.5s)

⠋ Running read_query …                                              ← pinned row
```

整轮结束后，pinned 行清空,agent 完整的 markdown 报告会渲染在原位。

**生成的 SQL：**

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

**分析结果：**

??? success "黄金-比特币相关性分析"
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

!!! tip "查看 trace 详情"
    任何时候按 **Ctrl+O** 都能打开"上一轮对话"的 inline trace 详情——含完整工具入参、完整 SQL、原始工具输出。再按一次 **Ctrl+O**（或按 `q`）即可关闭。结束时不再有 `(y/n)` 提示。

## 步骤 4：进阶功能

### 上下文管理

!!! note
    前提是你已经初始化了用于元数据的知识库。

如果希望把文件、指标或数据表加入上下文，可使用 `@` 引用。手动检索不仅能为大模型提供更精确的上下文，还能缩小搜索范围，提高回答准确度。

```bash title="Terminal"
Datus> @Table duckdb-demo.main.bank_failures
```

Datus 会自动分析该表并把元数据加入上下文，然后像普通对话一样流式输出：思考增量、工具调用（`describe_table`、`read_query` 等）以及最终的 markdown 报告。按 **Ctrl+O** 可查看本轮工具调用的完整入参 / 原始输出。

??? example "上下文分析输出（节选）"
    ```text
    ● Tool call · describe_table({"table_name": "bank_failures"})  ✓ 7 columns (0.5s)
    ● Tool call · read_query({"sql": "-- 1. Basic overview ..."})  ✓ 1 row   (0.5s)
    ● Tool call · read_query({"sql": "-- 2. Yearly analysis ..."}) ✓ 14 rows (0.5s)
    ● Tool call · read_query({"sql": "-- 3. State-wise ..."})      ✓ 15 rows (0.5s)
    ● Tool call · read_query({"sql": "-- 7. Recent 2023-2024 ..."}) ✓ 8 rows (0.5s)
    ```

随后 agent 会渲染一份 markdown 报告（执行摘要、地区分布、规模分析、最大破产事件、近期趋势、关键洞察）。具体形态取决于 LLM 与数据，上述示例仅供参考。

!!! tip
    需要更多命令参考与用法，请查看 [CLI](../cli/introduction.md)，或在终端输入 `/help`。

## 下一步

在完成基础体验后，可以继续探索以下功能：

- **[数据工程快速开始](./data_engineering_quickstart.zh.md)** —— 使用 DAComp 构建分层数仓，并串起 Airflow 与 Superset
- **[上下文数据工程](./contextual_data_engineering.md)** —— 学习如何将数据资产用作上下文
- **[配置指南](../configuration/introduction.md)** —— 连接自有数据库并自定义设置
- **[CLI 参考手册](../cli/introduction.md)** —— 掌握全部命令与选项
- **[语义层适配器](../adapters/semantic_adapters.md)** —— 使用 datus-semantic-metricflow 构建与查询指标
