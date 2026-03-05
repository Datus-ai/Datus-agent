# 介绍

**[Datus](https://github.com/Datus-ai/Datus-agent)** 是一个开源数据工程智能体，用于为你的数据系统构建 **可演化的上下文**。与只负责搬运数据的传统工具不同，Datus 会捕获、学习并进化围绕数据的知识，将元数据、Reference SQL、语义模型和指标转化为一个不断生长的知识库，为 AI 查询提供依据并消除幻觉。

借助 Datus，数据工程师能够把精力从编写重复的 SQL 转向构建可复用、面向 AI 的上下文。每一次查询、纠错和领域规则都会成为长期记忆帮助专用子代理为整个组织提供准确、具备领域认知的分析。

![Datus architecture](assets/home_architecture.png)

**面向不同用户的三个入口**

- **[Datus-CLI](cli/introduction.md)**：面向数据工程师的 AI 驱动命令行界面——就像“专为数据工程打造的 Claude Code”。你可以交互式地编写 SQL、构建子代理并维护上下文。
- **[Datus-Chat](web_chatbot/introduction.md)**：面向数据分析师的网页聊天机器人，支持多轮对话以及内置反馈机制（点赞、问题反馈、成功案例）。
- **[Datus-API](workflow/api.md)**：为其他智能体或应用提供稳定、准确的数据服务的 RESTful API。

**两种执行模式**

- **[Agentic 模式](subagent/introduction.md)**：适合临时开发和探索性流程。灵活、对话式，并通过专业化的子代理感知上下文。
- **[Workflow 模式](workflow/introduction.md)**：为需要高稳定性和编排能力的生产场景而设。工作流可以把子代理作为节点来构建复杂的管道。

**核心在于上下文引擎**

Datus 的核心是其 **[上下文引擎](knowledge_base/introduction.md)**，融合了人工专业知识与 AI 能力：

- 自动捕获[元数据](knowledge_base/metadata.zh.md)、[语义模型](knowledge_base/semantic_model.zh.md)、[指标](knowledge_base/metrics.zh.md)、[Reference SQL](knowledge_base/reference_sql.zh.md)、文档和成功案例
- 支持人在回路的策划与迭代
- 为子代理和工作流提供丰富、具领域特异性的上下文

**灵活的集成层**

Datus 可以无缝对接你现有的技术栈：

- **大模型**：支持 OpenAI、Claude、DeepSeek、Qwen、Kimi 等（参见[配置](configuration/agent.md)）
- **数据仓库**：支持 StarRocks、Snowflake、DuckDB、SQLite、PostgreSQL 等（参见[命名空间配置](configuration/namespace.md)、[数据库适配器](adapters/db_adapters.md)）
- **语义层**：兼容 [MetricFlow](metricflow/introduction.md) 的指标定义与查询
- **可扩展性**：通过 [MCP](cli/mcp_extensions.md)（Model Context Protocol）集成自定义扩展

## 快速入门

几分钟内即可启动你的 Datus Agent。

!!! tip "从这里开始"
    [:material-rocket-launch: **快速入门指南**](getting_started/Quickstart.md){ .md-button .md-button--primary }


了解 Datus 如何利用数据资产中的上下文数据工程不断学习和改进。

!!! info "掌握关键概念"
    [:material-book-open-variant: **上下文数据工程**](getting_started/contextual_data_engineering.md){ .md-button }


## 重点主题

<div class="grid cards" markdown>

-   :material-console-line: **Datus CLI**

    ---

    面向本地开发和实时预览数据工作流的命令行界面。

    [:octicons-arrow-right-24: 了解更多](cli/introduction.md)

-   :material-database: **知识库**

    ---

    用于组织和管理数据资产与文档的集中式存储库。

    [:octicons-arrow-right-24: 浏览知识库](knowledge_base/introduction.md)

-   :material-robot-outline: **子代理系统**

    ---

    通过专用子代理扩展 Datus，以处理不同的数据工程任务和流程。

    [:octicons-arrow-right-24: 探索子代理](subagent/introduction.md)

-   :material-sitemap-outline: **工作流管理**

    ---

    通过可配置的工作流构建器设计和编排复杂的数据管道。

    [:octicons-arrow-right-24: 探索工作流](workflow/introduction.md)

</div>


