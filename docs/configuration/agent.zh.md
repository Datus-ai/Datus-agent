# Agent 配置

`agent.agentic_nodes` 用于配置 chat 节点、内置 subagent，以及你自定义的 subagent。

适用范围：

- `chat`、`explore`、`gen_sql`、`gen_report`、`gen_dashboard`、`scheduler` 等内置 agentic 节点
- 通过 `.subagent` 创建的自定义 subagent
- 把自定义名称 alias 到某个内置节点类的高级手工配置

LLM provider 注册（`agent.models`）与默认模型选择（`agent.target`）请参见 **[模型](models.md)**。

## 常见字段

当前运行时会读取这些常见字段：

- `model`：引用 `agent.models` 中的 provider key
- `system_prompt`：subagent 名称 / 提示词模板基名
- `node_class`：要使用的节点实现，例如 `gen_sql`、`gen_report`、`explore`、`gen_table`、`gen_skill`、`gen_dashboard`、`scheduler`
- `prompt_version`、`prompt_language`
- `agent_description`
- `tools`、`mcp`、`skills`
- `rules`
- `max_turns`
- `workspace_root`
- `scoped_context`
- `subagents`
- semantic 节点使用的 `semantic_adapter`
- dashboard agent 使用的 `bi_platform`
- scheduler 节点使用的 `scheduler_service`

`scoped_kb_path` 已废弃。新配置使用共享的全局存储，并在查询时应用过滤，而不是为每个 subagent 持有独立 scoped KB 目录。

## `subagents` 委派

`subagents` 决定该节点是否暴露 `task()` 工具，以及允许委派到哪些 subagent 类型。

- `subagents: "*"` —— 允许委派到所有可发现的 subagent（排除自己）
- `subagents: explore, gen_sql` —— 只允许委派到指定子集
- 为空或省略时：
    - `chat` 默认是 `*`
    - 多数其他 agentic 节点默认是 `explore`
    - 显式写成空值时可禁用委派

## `scoped_context`

`scoped_context` 用来限制 subagent 在共享元数据和知识库里可见的范围：

```yaml
scoped_context:
  namespace: finance
  tables: mart.finance_daily, mart.finance_budget
  metrics: finance.revenue.daily_revenue
  sqls: finance.revenue.region_rollup
```

手工写 YAML 时请显式填写 `namespace`。`.subagent` 向导会自动从当前数据库填充该值。

## 示例

```yaml
agent:
  agentic_nodes:
    chat:
      model: claude
      max_turns: 50
      subagents: "*"

    finance_report:
      node_class: gen_report
      model: claude
      system_prompt: finance_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "财务分析助手"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      subagents: explore, gen_sql
      max_turns: 30
      scoped_context:
        namespace: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup

    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    semantic_metrics:
      node_class: gen_metrics
      model: claude
      semantic_adapter: metricflow
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      scheduler_service: airflow_prod
      max_turns: 30
```
