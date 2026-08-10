<p align="center">
  <strong>Datus · 开源数据工程 Agent</strong>
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-blueviolet?style=for-the-badge" alt="License"></a>
  <a href="https://datus.ai"><img src="https://img.shields.io/badge/Website-5A0FC8?style=for-the-badge" alt="Website"></a>
  <a href="https://docs.datus.ai/"><img src="https://img.shields.io/badge/Docs-654FF0?style=for-the-badge" alt="Docs"></a>
  <a href="https://docs.datus.ai/zh/latest/getting_started/Quickstart/"><img src="https://img.shields.io/badge/Quick%20Start-3423A6?style=for-the-badge" alt="Quick Start"></a>
  <a href="https://docs.datus.ai/zh/latest/release_notes/"><img src="https://img.shields.io/badge/Release%20Notes-092540?style=for-the-badge" alt="Release Notes"></a>
  <a href="https://join.slack.com/t/datus-ai/shared_invite/zt-3g6h4fsdg-iOl5uNoz6A4GOc4xKKWUYg"><img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

---

## Datus 是什么?

**Datus** 是面向现代数据栈的开源数据工程 Agent:用一个 Agent 连接你的数据仓库、目录、语义层和 BI,底座是由你的团队自己拥有的**可演进上下文引擎(evolvable context engine)**。

Copilot 只回答问题,Datus 端到端地执行数据工作:规划、编写并验证 SQL,编写语义模型与指标,生成管道、报告和看板。这一切都以上下文为根基,它记录你的 schema、参考 SQL 和业务规则,并随每次交互持续改进。没有可演进上下文的 Agent,每周一都是个陌生人;Datus 记得团队教过它的一切。

这条路径是具体的:在类 Claude Code 的 CLI 中探索数据,将知识沉淀为上下文(知识库 + 语义层),把成熟领域打包为有边界的 Subagent,再通过 Web 聊天、REST API、MCP、Slack/飞书或 VS Code 交付给分析师。

![Datus 架构](docs/assets/datus_architecture.svg)

## 核心特性

### 可演进的上下文引擎,而非静态管道

NL2SQL 工具会幻觉出不存在的 join 和指标,因为它们对你的数据库一无所知。Datus 构建一个**上下文引擎**,把树状业务域结构与向量检索结合,将 schema 元数据、参考 SQL、参数化 SQL 模板、语义模型、指标和领域知识统一进一个由团队自己拥有的上下文层。`/init` 扫描项目生成 `AGENTS.md` 清单及文件型知识/记忆存储;`/build-kb` 在其上构建向量索引知识库,可按文件、表、数据源或业务域圈定范围。正是这些上下文让 Agent 生成的 SQL 准确可信,而每一次查询、修正和业务规则都会回流沉淀。→ [Contextual Data Engineering](https://docs.datus.ai/zh/latest/getting_started/contextual_data_engineering/)

### 指标与语义层:MetricFlow 与 OSI

通过可插拔的语义适配器超越裸 SQL。业务指标可用 [MetricFlow](https://docs.datus.ai/zh/latest/metricflow/introduction/) YAML 或 [OSI(Open Semantic Interchange)](https://docs.datus.ai/zh/latest/adapters/osi_semantic_adapter/) 规范文档编写,由 MetricFlow 或原生 Rust 引擎 [Dosi](docs/adapters/dosi_semantic_adapter.zh.md) 执行;Datus 还能从 schema 和 SQL 历史自动生成指标,涵盖累计、滚动窗口、环比等高级时间指标。AskMetrics subagent 直接基于指标层回答 KPI、趋势和归因问题,[Dashboard Copilot](https://docs.datus.ai/zh/latest/getting_started/dashboard_copilot/) 则把现有 BI 看板变成对话式分析。→ [语义适配器文档](https://docs.datus.ai/zh/latest/adapters/semantic_adapters/)

### 从探索到领域专属 Subagent

垂直 Agent 的胜负手是吃透领域上下文,而不是套一个更大的模型。从交互式 CLI 开始:与数据库对话,用 `@table` / `@file` 引用锚定上下文,按 `Tab` 在 chat / SQL / bash 三种输入模式间切换(或用 `!` 直接执行 agent 工具和插件 CLI),用 [Plan Mode](https://docs.datus.ai/zh/latest/cli/plan_mode/) 先审后行,自动会话压缩、记忆和 `/resume` 让长期工作保持连贯。当某个领域成熟后,打开 `/agent` 管理器将其打包为 **Subagent**:带精选上下文、工具和业务规则的领域聊天机器人,再通过 Web、API、MCP 或 IM 交付,让分析师在 Slack 里拿到的数字与管理层在看板上看到的一致。→ [Subagent 文档](https://docs.datus.ai/zh/latest/subagent/introduction/)

### 数据工程自动化

内置 subagent 覆盖 SQL 之外的工程工作:跨库迁移、ETL/作业生成、由 JOIN SQL 生成宽表、通过 Airflow 适配器进行调度编排。Superset 与 Grafana 的 BI 适配器让 Agent 能读写真实看板。→ [内置 subagent](https://docs.datus.ai/zh/latest/subagent/builtin_subagents/) · [调度适配器](https://docs.datus.ai/zh/latest/adapters/scheduler_adapters/) · [BI 适配器](https://docs.datus.ai/zh/latest/adapters/bi_adapters/)

### 可视化报告与看板

从对话直接生成自包含的 HTML 报告(KPI 卡片、图表、表格、叙述)和可交互看板。看板筛选器通过本地 Web 服务实时回查数据库,无需 SaaS 后端;所有产物支持按区块逐步精修。→ [可视化报告](https://docs.datus.ai/zh/latest/subagent/gen_visual_report/) · [可视化看板](https://docs.datus.ai/zh/latest/subagent/gen_visual_dashboard/)

### 企业级治理

Datus 内置权限三档(`normal` / `auto` / `dangerous`,可按请求切换)、按 SQL 语句类型细分的权限、命令级 bash 允许/拒绝规则与 OS 级沙箱、请求级 SQL policy 框架(行级改写)、只读多租户配置模式,以及可配置的 tracing(Langfuse、LangSmith、Datadog、Braintrust 或任意 OTLP collector)。→ [SQL Policy](https://docs.datus.ai/zh/latest/configuration/sql_policy/) · [可观测性](https://docs.datus.ai/zh/latest/develop/observability/)

### 开放平台

- **10+ LLM 提供商**:OpenAI、Claude、Gemini、DeepSeek、Qwen、Kimi、OpenRouter 等,支持订阅认证(Claude 订阅、OpenAI Codex OAuth)与 coding-plan 提供商;按节点分配模型,单个工作流内可混用。
- **13 种数据库**:内置 SQLite 和 DuckDB,插件式适配 PostgreSQL、MySQL、Snowflake、StarRocks、ClickHouse、Doris 等。
- **MCP 协议**:既是 MCP 服务端(向 Claude Desktop、Cursor 等暴露 Datus 工具),也是 MCP 客户端(CLI 中通过 `/mcp` 消费外部工具)。→ [MCP 文档](https://docs.datus.ai/zh/latest/integration/mcp/)
- **Skills 与插件**:以 [agentskills.io](https://agentskills.io) 风格的打包技能扩展 Datus(支持 marketplace),或以声明式 `datus-plugin.yml` 清单 + install/pack/export CLI 交付完整插件。→ [Skills 文档](https://docs.datus.ai/zh/latest/skills/introduction/) · [插件文档](https://docs.datus.ai/zh/latest/plugin/introduction/)

### 度量与改进

Datus 内置面向 BIRD 与 Spider 2.0-Snow 数据集的评测框架。为你的 Agent 测 SQL 准确率、对比配置,并随上下文演进跟踪提升。→ [Benchmark 文档](https://docs.datus.ai/zh/latest/benchmark/benchmark_manual/)

## 快速开始

### 安装

**要求:** Linux 或 macOS。使用一键脚本时会自动安装 Python 3.12。

#### 一键安装(Linux / macOS)

从 PyPI 安装稳定版:

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

脚本会在 `~/.datus/venv` 创建专用 venv,从 PyPI 安装 `datus-agent`,并在 `~/.local/bin` 写入 `datus`、`datus-cli`、`datus-api`、`datus-mcp`、`datus-agent`、`datus-gateway`、`datus-pip` 等 shim。打开新终端(或 `source ~/.zshrc`)使 PATH 生效后,运行 `datus` 启动 REPL:用 `/model` 配置 LLM,`/datasource` 添加数据源,(可选)`/init` 为当前项目生成 `AGENTS.md`。

后续向全局 venv 安装额外 Python 包用 `datus-pip install <package>`;升级 Datus 本体用 `datus upgrade`(`datus upgrade --check` 只查不装)。

从 GitHub 源码开发版安装(包含未发布改动):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | sh
# 或固定到某个分支 / tag / commit
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install-dev.sh | DATUS_REF=feature/foo sh
```

固定 PyPI 版本(仅稳定版脚本):

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | DATUS_VERSION=0.3.9 sh
```

两个安装脚本还支持:`DATUS_HOME`(默认 `~/.datus`)、`DATUS_BIN_DIR`(默认 `~/.local/bin`)、`DATUS_FORCE=1` 重建 venv、`DATUS_NO_MODIFY_PATH=1` 跳过 shell rc 修改。

#### 手动安装

```bash
pip install datus-agent
datus
```

REPL 启动后,运行 `/model` 配置 LLM、`/datasource` 添加数据源、(可选)`/init` 生成 `AGENTS.md`。详细指引见[快速开始指南](https://docs.datus.ai/zh/latest/getting_started/Quickstart/)。

### 接入方式

以下示例使用名为 `demo` 的数据源,请先在 REPL 中用 `/datasource` 创建(内置 DuckDB 示例库会作为默认选项提供)。

| 接入方式 | 命令 | 适用场景 |
|-----------|---------|----------|
| **CLI**(交互式 REPL) | `datus --datasource demo` | 数据工程师探索数据、构建上下文、创建 subagent |
| **Web 聊天**(FastAPI + React) | `datus --web --datasource demo` | 分析师通过浏览器与 subagent 对话(`http://localhost:8501`) |
| **REST API**(FastAPI) | `datus-api --datasource demo` | 应用通过 REST 消费数据服务(`http://localhost:8000`) |
| **MCP 服务端** | `datus-mcp --datasource demo` | MCP 客户端(Claude Desktop、Cursor 等) |
| **IM 网关** | `datus-gateway` | 分析师在 Slack 或飞书中与 subagent 对话 → [网关文档](https://docs.datus.ai/zh/latest/gateway/introduction/) |
| **VS Code**(Datus Studio) | 连接 `datus --web` | IDE 内的目录浏览器、聊天面板、SQL 结果与 AI 图表 → [Datus Studio 文档](https://docs.datus.ai/zh/latest/vscode_extension/introduction/) |

> **提示:** Print 模式向 stdout 流式输出 JSON 消息行,适合脚本与 CI:`datus -p "你的问题" --datasource demo`。加 `--resume <session_id>` 可多轮续跑,加 `--plan-mode` 自动确认生成的计划。

## 工作方式

![How It Works](docs/assets/how_it_works.svg)

**探索**:与数据库对话、测试查询,用 `@table` 或 `@file` 引用锚定上下文。

```bash
datus --datasource demo
Check the top 10 banks by assets lost @table duckdb-demo.main.bank_failures
```

**构建上下文**:扫描项目、生成语义模型与指标、索引 SQL 历史。每一份沉淀都成为后续查询的可复用上下文。

```bash
/init            # 项目清单(AGENTS.md)+ 知识与记忆存储
/bootstrap       # TUI:爬取 schema、导入参考 SQL、生成语义模型与指标
/build-kb        # 构建向量知识库,可按文件/表/业务域圈定范围
```

**创建 Subagent**:打开统一 agent 管理器,把成熟上下文打包为带精选工具和业务规则的领域聊天机器人。

```bash
/agent           # 在 TUI 中创建和管理 subagent
```

**交付**:通过 Web(`localhost:8501/?subagent=mychatbot`)、REST API、MCP 或 Slack/飞书把 subagent 交付给分析师,内置反馈收集(点赞、问题上报)。

**度量**:用 BIRD 或 Spider 2.0-Snow 基准测试跟踪 SQL 准确率随上下文演进的变化。

**迭代**:分析师反馈回流。工程师修正 SQL、补充规则、精修语义模型,并用 Skills、插件或 MCP 工具扩展。Agent 随时间越来越准。

→ [端到端教程](https://docs.datus.ai/zh/latest/getting_started/contextual_data_engineering/) · [CLI 文档](https://docs.datus.ai/zh/latest/cli/introduction/) · [知识库文档](https://docs.datus.ai/zh/latest/knowledge_base/introduction/) · [Subagent 文档](https://docs.datus.ai/zh/latest/subagent/introduction/)

## 架构

### 工作流引擎

在 agentic 对话层之下,Datus 使用可配置的**节点式工作流引擎**。benchmark 与批量运行直接使用它,也可组合出串行、并行、子工作流的自定义执行计划:

```yaml
workflow:
  plan: planA
  planA:
    - schema_linking     # 找到相关表
    - parallel:          # 并行执行
      - gen_sql          # SQL 生成
      - reasoning        # 思维链推理
    - selection          # 选出最优结果
    - execute_sql        # 执行查询
    - output             # 格式化返回
```

### 节点类型

| 类别 | 节点 |
|----------|-------|
| **核心** | `schema_linking`、`execute_sql`、`reasoning`、`reflect`、`output`、`fix`、`date_parser`、`doc_search` |
| **Agentic** | `chat`、`gen_sql`、`explore`、`semantic`(语义模型生成)、`sql_summary`、`search_metrics`、`ask_metrics`、`compare`、`gen_table`、`gen_job`、`gen_skill`、`gen_report`、`gen_visual_report`、`gen_visual_dashboard`、`gen_dashboard`、`scheduler`、`feedback` |
| **控制流** | `parallel`、`selection`、`subworkflow`、`hitl` |

### RAG 知识库

知识库默认由 **LanceDB** 驱动,支持可插拔的向量与关系型后端(含 PostgreSQL 存储)及可选的元数据全文检索。上下文分为多层:

- **Schema 元数据**:表和列的描述、关联关系
- **参考 SQL**:精选查询示例及摘要
- **参考模板**:参数化 Jinja2 SQL 模板,稳定可复用
- **语义模型**:业务逻辑与指标定义(MetricFlow 或 OSI)
- **指标**:通过语义层集成的可执行业务指标
- **平台文档**:从 GitHub 仓库、网站或本地文件摄取

交互式构建用 `/bootstrap` 和 `/build-kb`,批量构建:

```bash
datus-agent bootstrap-kb --datasource demo --components metadata reference_sql
```

## 配置

Datus 通过 `agent.yml` 配置。启动 `datus` 后用 `/model` 和 `/datasource` 交互式填充,或复制 [`conf/agent.yml.example`](conf/agent.yml.example) 手工编辑。

LLM 接入采用两层模型:**`agent.providers`** 存放已知提供商的凭据(模型目录来自 `conf/providers.yml`,`/model` 切换模型无需改 YAML);**`agent.models`** 用于自建/自托管端点。当前选择按项目持久化在 `./.datus/config.yml` 的 `target: { provider: ..., model: ... }`。

| 配置节 | 用途 |
|---------|---------|
| `agent.providers` | 内置 LLM 提供商凭据(模型目录来自 `providers.yml`) |
| `agent.models` | 自定义 / 自托管 LLM 端点定义 |
| `agent.nodes` | 按节点分配模型与调优参数 |
| `agent.services.datasources` | 数据库连接(SQLite、DuckDB、Snowflake 等) |
| `agent.storage` | Embedding 模型、向量库与 RAG 配置 |
| `agent.workflow` | 串行、并行、子工作流执行计划 |
| `agent.agentic_nodes` | Agentic 节点配置(语义模型生成、指标生成) |
| `agent.skills` | Skill 目录与权限 |
| `agent.document` | 平台文档来源(GitHub 仓库、网站、本地文件) |

API key 通过环境变量以 `${ENV_VAR}` 语法注入。

## 支持的 LLM 提供商

| 提供商 | 类型 | 说明 |
|----------|------|-------|
| OpenAI | `openai` | GPT-4o、GPT-4.1 等 |
| Anthropic Claude | `claude` | 直连 API |
| Claude 订阅 | `claude_subscription` | 通过 Claude Pro/Max 订阅 OAuth |
| OpenAI Codex | `codex` | 通过 ChatGPT 订阅 OAuth |
| Google Gemini | `gemini` | Gemini 2.0+ |
| DeepSeek | `deepseek` | DeepSeek-Chat、DeepSeek-Coder |
| 阿里通义千问 | `qwen` | Qwen 系列 |
| 月之暗面 Kimi | `kimi` | Kimi 模型 |
| MiniMax | `minimax` | MiniMax 模型 |
| 智谱 GLM | `glm` | GLM-4 系列 |
| OpenRouter | `openrouter` | 单个 API key 访问 300+ 模型 |
| Coding plans | `alibaba_coding`、`bigmodel_coding`、`zai_coding`、`minimax_coding`、`kimi_coding` | 复用各厂商 coding-plan 订阅 |

**Embedding 模型:** OpenAI、Sentence-Transformers、FastEmbed、Hugging Face。

按节点分配模型:不同工作流步骤可使用不同提供商(例如 schema linking 用便宜的模型,SQL 生成用更强的模型)。

## 支持的数据库

| 数据库 | 类型 | 包 |
|---------|------|---------|
| SQLite | `sqlite` | 内置 |
| DuckDB | `duckdb` | 内置 |
| PostgreSQL | `postgresql` | [`datus-postgresql`](https://github.com/Datus-ai/datus-db-adapters) |
| MySQL | `mysql` | [`datus-mysql`](https://github.com/Datus-ai/datus-db-adapters) |
| Snowflake | `snowflake` | [`datus-snowflake`](https://github.com/Datus-ai/datus-db-adapters) |
| StarRocks | `starrocks` | [`datus-starrocks`](https://github.com/Datus-ai/datus-db-adapters) |
| ClickHouse | `clickhouse` | [`datus-clickhouse`](https://github.com/Datus-ai/datus-db-adapters) |
| ClickZetta | `clickzetta` | [`datus-clickzetta`](https://github.com/Datus-ai/datus-db-adapters) |
| Apache Doris | `doris` | [`datus-doris`](https://github.com/Datus-ai/datus-db-adapters) |
| Hologres | `hologres` | [`datus-hologres`](https://github.com/Datus-ai/datus-db-adapters) |
| Hive | `hive` | [`datus-hive`](https://github.com/Datus-ai/datus-db-adapters) |
| Spark | `spark` | [`datus-spark`](https://github.com/Datus-ai/datus-db-adapters) |
| Trino | `trino` | [`datus-trino`](https://github.com/Datus-ai/datus-db-adapters) |

详见[数据库适配器文档](https://docs.datus.ai/zh/latest/adapters/db_adapters/)。

## 开发

```bash
uv sync                                                                    # 安装依赖
uv run python ci/run-pr-tests.py upstream/main                             # PR CI 测试(无外部依赖)
uv run ruff format datus/ tests/ && uv run ruff check --fix datus/ tests/  # 格式化与 Lint
```

CLI 命令加 `--save_llm_trace`,或在 `agent.yml` 中按模型设置 `save_llm_trace: true`,可持久化 LLM 输入/输出用于调试;也可配置 `agent.observability.tracing` 将运行 trace 导出到 Langfuse、LangSmith、Datadog、Braintrust 或 OTLP collector。→ [LLM Trace 文档](https://docs.datus.ai/zh/latest/training/llm_trace_usage/) · [可观测性文档](https://docs.datus.ai/zh/latest/develop/observability/)

完整开发规范、架构模式与测试规则见 [CLAUDE.md](CLAUDE.md)。

## 许可证

[Apache 2.0](LICENSE)
