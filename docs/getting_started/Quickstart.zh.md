# 快速开始

几分钟上手 Datus Agent：安装 → 配置 → 第一次提问。

!!! tip "完整数仓链路"
    若需体验分层建模、ETL 生成、Airflow 调度、语义资产与 Superset 仪表盘，请阅读 [数据工程快速开始](./data_engineering_quickstart.zh.md)。

## 1. 安装

Linux / macOS 一键安装（推荐）：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

脚本会自动 bootstrap `uv`，在 `~/.datus/venv` 下建独立 venv（缺 Python 3.12 时自动下载），并把 `datus`、`datus-cli`、`datus-api`、`datus-mcp`、`datus-pip` 等 shim 写入 `~/.local/bin`。开新 shell（或 `source ~/.zshrc`）使 PATH 生效。

??? note "其他安装方式"
    **固定版本**（变量传给接收脚本的 shell，不是 `curl`）：
    ```bash
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.2.6 sh
    ```

    **从 GitHub 源安装**（拿 `main` 上未发布的改动，或任意 ref）：
    ```bash
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
    curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
    ```

    **自管 Python 环境**（需要 Python 3.12）：
    ```bash
    # Conda / virtualenv / uv 任选其一，激活后：
    pip install datus-agent
    ```

    其他变量：`DATUS_HOME`、`DATUS_BIN_DIR`、`DATUS_FORCE=1`、`DATUS_NO_MODIFY_PATH=1`。后续往该 venv 安装其它 Python 包请用 `datus-pip install <package>`。

    预发布版：`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ datus-agent`。

## 2. 配置与初始化

启动 REPL：

```bash
datus
```

在 REPL 内依次运行 `/datasource`、`/model`、`/init` 三条斜杠命令。

### Datasource

运行 `/datasource`。TUI 引导你填写名称、类型（DuckDB、SQLite、MySQL、PostgreSQL、Snowflake、StarRocks 等）和连接信息，自动测试连通性并写入 `~/.datus/conf/agent.yml`。同一 TUI 也支持编辑 / 删除 / 设默认 / 自动安装缺失的适配器插件。运行时切换可直接 `/datasource <name>`。

!!! tip "演示数据库"
    Datus 自带预配置的 DuckDB 演示库 `~/.datus/sample/duckdb-demo.duckdb`。在 `/datasource` 选 `duckdb` 并指向该路径即可立即获得可用数据源。

### Model

运行 `/model`。TUI 列出全部 provider，选中后输入 API Key（自动识别常见环境变量）即可。也支持快捷写法 `/model openai/gpt-4.1`。

常用 provider：

| Provider | 默认模型 | 环境变量 |
|---|---|---|
| `openai` | `gpt-4.1` | `OPENAI_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `claude` | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` |

完整 provider 列表（含 Kimi / Qwen / GLM / MiniMax、Claude 订阅、Codex OAuth、Coding Plan 等）见 [Model 命令](../cli/other_commands.zh.md#model)。

### Init（可选）

`cd` 进入项目目录后启动 `datus`，运行 `/init` 会读取上一步保存的默认模型与数据源，扫描当前目录并生成项目级 `AGENTS.md`。需要换数据源时先 `/datasource <name>` 再 `/init`。

## 3. 开始使用

启动后会看到 banner 与提示符 `>`，提示符接受三种输入：

- **斜杠命令** —— `/help`、`/datasource`、`/model`、`/exit` 等
- **SQL** —— `SELECT …`、`DESCRIBE …`、`SHOW …` 自动识别并对当前数据源执行
- **自然语言** —— 其余输入交给 agent

```text title="示例"
> /tables
> desc gold_vs_bitcoin
> Detailed analysis of gold–Bitcoin correlation.
```

自然语言提问后，Datus 实时流式展示思考、工具调用、SQL 与最终 markdown 报告，底部 pinned 行显示当前正在跑的工具：

```text
● Let me check the schema of gold_vs_bitcoin and run a correlation analysis.
● describe_table({"table_name": "gold_vs_bitcoin"})  ✓ 3 columns (0.5s)
● read_query({"sql": "SELECT CORR(gold, bitcoin) ..."}) ✓ 1 row (0.5s)
○ Running read_query …
```

!!! tip "查看 trace 详情"
    任何时候按 **Ctrl+O** 可打开上一轮对话的 inline trace（完整工具入参、SQL、原始输出），再次按下或 `q` 关闭。

## 下一步

- **[数据工程快速开始](./data_engineering_quickstart.zh.md)** —— 分层数仓 + Airflow + Superset 端到端
- **[上下文数据工程](./contextual_data_engineering.md)** —— `@` 引用、知识库与上下文管理
- **[配置指南](../configuration/introduction.md)** —— 自有数据库与高级配置
- **[CLI 参考](../cli/introduction.md)** —— 全部命令与选项
- **[语义层适配器](../adapters/semantic_adapters.md)** —— datus-semantic-metricflow
