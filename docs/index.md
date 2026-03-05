# Introduction

**[Datus](https://github.com/Datus-ai/Datus-agent)** is an open-source data engineering agent that builds **evolvable context** for your data systems. Unlike traditional tools that merely move data, Datus captures, learns, and evolves the knowledge surrounding your data—transforming metadata, reference SQL, semantic models, and metrics into a living knowledge base that grounds AI queries and eliminates hallucinations.

With Datus, data engineers shift from writing repetitive SQL to building reusable, AI-ready context. Every query, correction, and domain rule becomes long-term memory—enabling specialized subagents that deliver accurate, domain-aware analytics to your entire organization.

![Datus architecture](assets/home_architecture.png)

**Three Entry Points for Different Users**

- **[Datus-CLI](cli/introduction.md)**: An AI-powered command-line interface for data engineers—think "Claude Code for data engineers." Write SQL, build subagents, and construct context interactively.
- **[Datus-Chat](web_chatbot/introduction.md)**: A web chatbot providing multi-turn conversations with built-in feedback mechanisms (upvotes, issue reports, success stories) for data analysts.
- **[Datus-API](workflow/api.md)**: RESTful APIs for other agents or applications that need stable, accurate data services.

**Two Execution Modes**

- **[Agentic Mode](subagent/introduction.md)**: Ideal for ad-hoc development and exploratory workflows. Flexible, conversational, and context-aware through specialized subagents.
- **[Workflow Mode](workflow/introduction.md)**: Optimized for production scenarios requiring high stability and orchestration. Workflows can use subagents as nodes for complex pipelines.

**Context Engine at the Core**

The heart of Datus is its **[Context Engine](knowledge_base/introduction.md)**, which combines human expertise with AI capabilities:

- Automatically captures [metadata](knowledge_base/metadata.md), [semantic models](knowledge_base/semantic_model.md), [metrics](knowledge_base/metrics.md), [reference SQL](knowledge_base/reference_sql.md), documents, and success stories
- Supports human-in-the-loop curation and refinement
- Powers both subagents and workflows with rich, domain-specific context

**Flexible Integration Layer**

Datus integrates seamlessly with your existing stack:

- **LLMs**: OpenAI, Claude, DeepSeek, Qwen, Kimi, and more ([Configuration](configuration/agent.md))
- **Data Warehouses**: StarRocks, Snowflake, DuckDB, SQLite, PostgreSQL, and others ([Namespace Setup](configuration/namespace.md), [Database Adapters](adapters/db_adapters.md))
- **Semantic Layers**: [MetricFlow](metricflow/introduction.md) support for metric definitions and queries
- **Extensibility**: Add custom integrations via [MCP](cli/mcp_extensions.md) (Model Context Protocol)

## Getting Started

Get your Datus Agent up and running in minutes.

!!! tip "Start Here"
    [:material-rocket-launch: **Quickstart Guide**](getting_started/Quickstart.md){ .md-button .md-button--primary }

Follow a complete hands-on tutorial to build your first context-aware data agent

!!! example "Try the Tutorial"
    [:material-school: **Complete Tutorial**](getting_started/Datus_tutorial.md){ .md-button }

Discover how Datus leverages contextual data engineering from your data assets to continuously learn and improve

!!! info "Learn Key Concepts"
    [:material-book-open-variant: **Contextual Data Engineering**](getting_started/contextual_data_engineering.md){ .md-button }


## Important Topics

<div class="grid cards" markdown>

-   :material-console-line: **Datus CLI**

    ---

    Command-line interface for local development and real-time preview of your data workflows.

    [:octicons-arrow-right-24: Learn more](cli/introduction.md)

-   :material-database: **Knowledge Base**

    ---

    Centralized repository for organizing and managing your data assets and documentation.

    [:octicons-arrow-right-24: Browse knowledge base](knowledge_base/introduction.md)

-   :material-robot-outline: **Subagent System**

    ---

    Extend Datus with specialized subagents for different data engineering tasks and workflows.

    [:octicons-arrow-right-24: Explore subagents](subagent/introduction.md)

-   :material-sitemap-outline: **Workflow Management**

    ---

    Design and orchestrate complex data pipelines with configurable workflow builder.

    [:octicons-arrow-right-24: Explore workflows](workflow/introduction.md)

</div>
