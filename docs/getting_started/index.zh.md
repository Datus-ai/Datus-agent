# 选择上手路径

Datus 支持多种工作流，但不需要按顺序完成全部教程。先完成简短的安装指南，再根据想要构建的内容选择一条路径。

## 从这里开始

| 你的目标 | 推荐文档 | 最终会得到什么 |
| --- | --- | --- |
| 安装 Datus、连接 datasource 并完成第一次提问 | [安装并完成第一次提问](Quickstart.md) | 已配置 model 与 datasource 的本地 Datus REPL |
| 理解 metadata、semantic model、metric、Reference SQL 和 scoped subagent 如何协同 | [构建上下文增强 Agent](contextual_data_engineering.md) | 基于内置 California Schools 数据集的 Knowledge Base 和两个 subagent |
| 从源数据开始构建分层数仓、调度任务和 BI Dashboard | [端到端数据工程](data_engineering_quickstart.md) | DuckDB staging/intermediate/marts 表、Airflow DAG 和 Superset Dashboard |
| 把已有 BI Dashboard 转换为可复用 SQL、指标和分析 subagent | [将 Dashboard 变成 Copilot](dashboard_copilot.md) | Reference SQL、Dosi 语义模型，以及限定到 Superset Dashboard 的两个 subagent |

如果是第一次使用 Datus，请先完成[安装并完成第一次提问](Quickstart.md)。其他三个文档是相互独立的上手路径，按目标选择一个即可，不需要从头全部阅读。

## 几条路径有什么区别

```text
安装并完成第一次提问
├── 构建上下文增强 Agent
│   └── 使用内置样本数据学习核心 context 构建流程
├── 端到端数据工程
│   └── 构建数据 → ETL → Airflow → Superset Dashboard
└── 将 Dashboard 变成 Copilot
    └── 使用已有 Dashboard → SQL 和指标证据 → 分析 subagent
```

两个场景教程都会使用 Superset，但目标不同：

- **端到端数据工程**先构建数据管道，再创建并发布一个新 Dashboard。
- **Dashboard Copilot**从已有 Dashboard 出发，把其中的查询证据转换为可复用 context 和 subagent。

## 完成教程后继续了解

- [CLI](../cli/introduction.md)：命令、输入模式、session 和 agent 选择
- [Knowledge Base](../knowledge_base/introduction.md)：metadata、semantic model、metric 与 Reference SQL
- [Subagent](../subagent/introduction.md)：使用 scoped context 的内置与自定义 agent
- [Skills](../skills/introduction.md)：供 agent 和 plugin 使用的可复用工作流
- [配置](../configuration/introduction.md)：datasource、semantic adapter、storage 与 node
