# 介绍

**[Datus](https://github.com/Datus-ai/Datus-agent)** 是面向现代数据栈的开源数据工程 Agent：用一个 Agent 连接数据仓库、数据目录、语义层和 BI，底座是一套沉淀在团队自己手里的可演进上下文引擎(evolvable context engine)。

Datus 可以完成 SQL 编写与验证、语义模型与指标构建，以及数据管道、报告和看板的生成；每一次执行与修正都会沉淀为上下文，持续提升后续输出的准确性。整个体系在生态上保持开放与灵活：数据库、BI、调度、LLM 乃至团队自己的工具，都能以标准方式接入。

## 工作方式

Agent 回答的质量，取决于它拿到的上下文质量。Datus 因此把重点放在上下文的沉淀与复用上，下图是完整的循环：

![Datus 工作方式](assets/how_it_works.svg)

图分前后两半。前半段是数据工程师的工作：探索数据、构建上下文、完成语义建模，产出可复用的资产；后半段是组织对资产的消费：subagent 把它们变成任何人都能提问的服务。

两半之间也不是单向交付：分析师的每次修正都会回流，资产随使用不断变厚。

1. **探索**：不需要任何前置建设，在 [CLI](cli/chat_command.zh.md) 里直接与数据库对话，用 `@table` 引用表、`@file` 引用文件，边问边熟悉数据。
2. **构建上下文**：[`/init`](skills/init.zh.md) 扫描当前项目，`/bootstrap` 和 [`/build-kb`](skills/build_kb.zh.md) 把散落在 schema、历史 SQL 和文档里的知识收进[知识库](knowledge_base/introduction.zh.md)；这是后面一切准确性的原料。
3. **语义建模**：语义建模 subagent 从 schema 和历史 SQL 中挖掘数据集、[语义模型](knowledge_base/semantic_model.zh.md)与[指标](knowledge_base/metrics.zh.md)，校验后注册进语义层；业务口径从此有了唯一的、可执行的定义。
4. **创建 Subagent**：用 `/agent` 把配好的上下文、工具和规则打包成[面向单个业务域的 subagent](subagent/customized_subagent.zh.md)；资产从这一步开始变成别人可以直接使用的服务。
5. **交付**：分析师在自己习惯的地方提问，浏览器、Slack/飞书或 IDE 都可以(见[接入方式](#interfaces))；[AskMetrics](subagent/ask_metrics.zh.md) 依据指标定义回答，[报告和看板](subagent/gen_visual_report.zh.md)在对话里直接生成。
6. **度量**：用 [benchmark](benchmark/benchmark_manual.zh.md) 在 BIRD、Spider 2.0-Snow 或[自定义数据集](configuration/benchmark.zh.md)上度量 SQL 准确率，把上下文带来的提升变成可量化的数字。

第 5 步产生的修正、反馈和成功案例会回流进第 2 步的上下文。资产在使用中越来越完整，而不是建成之日就开始过时。

## 核心能力

准确率来自两处：语义层把业务口径变成可执行的定义，上下文引擎把使用中产生的知识留存下来。Subagent 负责把这些资产交付给使用的人，Plugin 生态与治理让整套体系能接入现有技术栈，并在生产环境中受控运行。

### 语义建模自动化

Agent 读取数据库 schema 和历史 SQL，自动生成 [OSI](https://dosi.datus.ai/) 格式的语义模型与指标定义，校验通过后注册进语义层，不需要手写 YAML。

执行由 [Dosi](https://dosi.datus.ai/) 引擎承担：同一份语义模型编译成 13+ 种数据库方言的 SQL。它是一个独立程序，也可以单独以 CLI、REST 服务或 MCP server 的方式部署，详见 [Dosi 语义适配器](adapters/dosi_semantic_adapter.zh.md)。

![语义建模：从 schema 与历史 SQL 到已校验的语义模型](assets/semantic_modeling_session.svg)

### 指标问答与归因

[AskMetrics](subagent/ask_metrics.zh.md) 依据指标定义回答业务问题，而不是临时拼 SQL；指标出现波动时，`attribution_analyze` 给出各维度贡献的量化归因。

![一次指标问答：提问、语义层工具调用、归因结果](assets/metric_qa_session.svg)

### 越用越准的上下文引擎

[上下文引擎](getting_started/contextual_data_engineering.zh.md)汇集 schema 元数据、参考 SQL 和业务规则，按业务域树组织，配合向量检索召回。使用中的每次修正都会写回[知识库](knowledge_base/introduction.zh.md)，让后续回答持续变准。

![构建上下文：schema 抓取、参考 SQL 索引、业务域树](assets/context_engine_session.svg)

### Subagent 交付

为一个业务领域配好上下文、工具和规则，打包成[专属聊天机器人](subagent/customized_subagent.zh.md)，交付给分析师直接使用。

- 分析师在浏览器、Slack/飞书或 IDE 里提问，[报告和看板](subagent/gen_visual_report.zh.md)在对话里生成，本地即可预览，不依赖任何 SaaS 后端。
- [内置 subagent](subagent/builtin_subagents.zh.md) 还覆盖跨库迁移、ETL 作业生成和宽表构建等工程任务，可编排 [Airflow](adapters/scheduler_adapters.zh.md) 调度。

![创建 subagent，并通过六种入口交付](assets/subagent_delivery_session.svg)

### Plugin 生态与治理

- [Plugin](plugin/introduction.zh.md) 框架把第三方平台和公司内部工具接入 Agent：一份 `datus-plugin.yml` 清单声明 CLI 命令、Skill 和 prompt 上下文，按项目启用。
- 适配器覆盖 [15 种数据库](adapters/db_adapters.zh.md)和 10+ LLM 提供商，另有 [MCP](integration/mcp.zh.md) 服务端与客户端；[Skill](skills/introduction.zh.md) 遵循 agentskills.io 约定，支持从 marketplace 安装。
- 治理上，权限分级，SQL 按语句类型授权并由 AI 预审，bash 运行在 OS 级沙箱中，[trace](develop/observability.zh.md) 可导出到任意 OTLP 平台。

![开放生态：安装 plugin，接入现有技术栈](assets/ecosystem_plugins.svg)

## 架构

![Datus 架构](assets/datus_architecture.svg)

整体架构自上而下分四层，与上图对应：

- **交付层**：CLI、Web 聊天、REST API、MCP、IM 网关和 VS Code 六个入口，共享同一个 Agent 后端。
- **智能层**：Chat Agent 负责规划和推理，subagent 处理专项任务，Skill 与 Plugin 提供扩展工具，治理机制也作用在这一层。交互请求走 Agentic 模式，步骤由 Agent 自行规划；benchmark 和批量任务走 [Workflow 模式](workflow/introduction.zh.md)，按预定义的节点计划执行。
- **语义层与上下文**：Agent 构建的资产层。一半是语义模型与指标，由 Dosi 或 MetricFlow 执行；另一半是[上下文](knowledge_base/introduction.zh.md)，包括 schema 元数据、参考 SQL、知识与记忆。检索用业务域树加向量召回；[存储](configuration/storage.zh.md)默认是内嵌的 LanceDB 和 SQLite，团队需要共享上下文时可换成 PostgreSQL。
- **数据与工具层**：经适配器连接的数据库、BI 平台、调度系统与 LLM 提供商。

## 快速开始

第一次运行不需要准备自己的数据库：安装自带 California Schools 示例数据集，数据源 `california_schools` 已预注册。Linux 或 macOS：

```bash
curl -fsSL https://raw.githubusercontent.com/datus-ai/datus-agent/main/install.sh | sh
```

打开新终端，运行 `datus`，然后：

1. `/model` 配置模型
2. `/datasource` 添加自己的数据源(只用内置示例可跳过)
3. `/init`(可选)扫描当前项目

也可以用 `pip install datus-agent` 手动安装(Python 3.12+)。配置分两级：全局 `agent.yml` 存放主配置，项目下的 `.datus/config.yml` 保存当前模型、默认数据源等项目级覆盖，详见[配置文档](configuration/introduction.zh.md)。

想先在示例数据上完整体验，[端到端教程](getting_started/contextual_data_engineering.zh.md)十分钟走完从构建上下文到指标问答的闭环：

!!! tip "继续深入"
    [:material-rocket-launch: **快速入门指南**](getting_started/Quickstart.zh.md){ .md-button .md-button--primary }
    [:material-school: **端到端教程**](getting_started/contextual_data_engineering.zh.md){ .md-button }

数据管道和迁移场景可以从[数据工程快速入门](getting_started/data_engineering_quickstart.zh.md)开始；以 BI 为主的场景见 [Dashboard Copilot](getting_started/dashboard_copilot.zh.md)。

## 接入方式 { #interfaces }

六个入口共享同一个 Agent 后端和同一份上下文：在 CLI 里沉淀的资产，分析师在浏览器或 Slack 里提问时同样生效。表中的 `demo` 是示例数据源名，适用于带 `--datasource` 参数的命令，请先用 `/datasource` 创建；使用内置示例时，把 `demo` 换成 `california_schools` 即可。

| 接入方式 | 命令 | 适用场景 |
|-----------|---------|----------|
| [**CLI**](cli/introduction.zh.md)(交互式 REPL) | `datus --datasource demo` | 数据工程师探索数据、构建上下文、创建 subagent |
| [**Web 聊天**](web_chatbot/introduction.zh.md)(FastAPI + React) | `datus --web --datasource demo` | 分析师通过浏览器与 subagent 对话(`http://localhost:8501`) |
| [**REST API**](API/introduction.zh.md)(FastAPI) | `datus-api --datasource demo` | 应用通过 REST 消费数据服务(`http://localhost:8000`) |
| [**MCP 服务端**](integration/mcp.zh.md) | `datus-mcp --datasource demo` | MCP 客户端(Claude Desktop、Cursor 等) |
| [**IM 网关**](gateway/introduction.zh.md) | `datus-gateway` | 分析师在 Slack 或飞书中与 subagent 对话 |
| [**VS Code**](vscode_extension/introduction.zh.md)(Datus Studio) | 连接 `datus --web` | IDE 内的目录浏览器、聊天面板、SQL 结果与 AI 图表 |

!!! note "Print 模式"
    Print 模式向 stdout 流式输出 JSON，适合脚本与 CI：`datus -p "你的问题" --datasource demo`。

## 深入了解

<div class="grid cards" markdown>

-   :material-layers-triple: **语义层**

    ---

    语义模型与指标如何生成、存储，并由 Dosi 或 MetricFlow 执行。

    [:octicons-arrow-right-24: 语义适配器](adapters/semantic_adapters.zh.md)

-   :material-robot-outline: **Subagent**

    ---

    为一个业务领域打包上下文、工具和规则，交付给分析师直接使用的聊天机器人。

    [:octicons-arrow-right-24: 探索 subagent](subagent/introduction.zh.md)

-   :material-database: **知识库**

    ---

    上下文引擎的存储：元数据、语义模型、指标、参考 SQL 与记忆。

    [:octicons-arrow-right-24: 浏览知识库](knowledge_base/introduction.zh.md)

-   :material-puzzle-outline: **Plugin**

    ---

    通过一份清单把第三方平台和公司内部工具接入 Agent。

    [:octicons-arrow-right-24: 了解 Plugin](plugin/introduction.zh.md)

-   :material-console-line: **CLI**

    ---

    交互式 REPL：对话、上下文与执行命令、MCP 扩展和 Plan 模式。

    [:octicons-arrow-right-24: CLI 参考](cli/introduction.zh.md)

-   :material-tools: **Skill**

    ---

    内置与可安装的 Skill：项目初始化、知识抽取、记忆整理等。

    [:octicons-arrow-right-24: 浏览 Skill](skills/introduction.zh.md)

-   :material-cog-outline: **配置**

    ---

    数据源、模型、语义层、SQL 策略、存储，以及 `agent.yml` 里的其他设置。

    [:octicons-arrow-right-24: 配置 Datus](configuration/introduction.zh.md)

-   :material-speedometer: **Benchmark**

    ---

    在 BIRD、Spider 2.0-Snow 或自定义数据集上度量 SQL 准确率。

    [:octicons-arrow-right-24: 运行 benchmark](benchmark/benchmark_manual.zh.md)

</div>
