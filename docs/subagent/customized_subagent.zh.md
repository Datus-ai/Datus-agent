# 自定义 Subagent

## 概览

`/subagent` 用于管理保存在 `agent.yml` 的 `agent.agentic_nodes` 下的自定义 subagent。

当前 CLI 支持：

- `add`：通过交互式向导创建自定义 subagent
- `list`：列出已配置的自定义 subagent
- `update <agent_name>`：编辑已有自定义 subagent
- `remove <agent_name>`：删除自定义 subagent

来自 `SYS_SUB_AGENTS` 的内置系统 subagent 是保留名称，不能通过 `/subagent` 删除或修改。

## 向导会生成什么

向导会写入两部分内容：

1. `agent.agentic_nodes` 下的新配置项
2. 一个名为 `{agent_name}_system_{prompt_version}.j2` 的提示词模板文件

当前向导支持两类自定义节点：

- `gen_sql`（默认）
- `gen_report`

如果你希望创建更高级的 alias，例如 `explore`、`gen_table`、`gen_skill`、`gen_dashboard`、`scheduler`，需要手工编辑 `agent.yml`。

## 向导字段

### 第 1 步：基础信息

- `system_prompt`：subagent 名称，也是配置键名
- `node_class`：`gen_sql` 或 `gen_report`
- `agent_description`：简短描述，会出现在预览和 task-tool 描述里

### 第 2 步：Tools 与 MCP

必须至少选择一个原生工具或一个 MCP 工具。

- 原生工具会存成逗号分隔的类别或方法模式，例如 `db_tools`、`semantic_tools.*`、`context_search_tools.list_subject_tree`
- MCP 选择会存成逗号分隔的 server 或 `server.tool` 条目

### 第 3 步：范围化上下文

向导当前支持以下范围字段：

- `tables`
- `metrics`
- `sqls`

保存时，向导还会把当前数据库写入 `scoped_context.namespace`。

### 第 4 步：Rules

Rules 会以字符串列表形式保存在 `rules` 中，并追加到最终系统提示词里。

## 命令说明

### `/subagent add`

启动交互式向导并创建新的自定义 subagent。

```bash
/subagent add
```

![Add subagent](../assets/add_subagent.png)

### `/subagent list`

列出已配置的自定义 subagent。

```bash
/subagent list
```

当前表格会显示：

- `Name`
- `Scoped Context`
- `Scoped KB`
- `Tools`
- `MCPs`
- `Rules`

其中 `Scoped KB` 是遗留展示列，对新配置通常显示为 `—`。

当 subagent 配置了 scoped context 时，列表会按当前数据库进行过滤。

![List subagent](../assets/list_subagents.png)

### `/subagent update <agent_name>`

把现有配置加载到向导中，修改后再写回 `agent.yml`。

```bash
/subagent update finance_report
```

![Update subagent](../assets/update_subagent.png)

### `/subagent remove <agent_name>`

删除配置项以及对应生成的提示词模板。

```bash
/subagent remove finance_report
```

## 配置示例

生成后的配置通常类似这样：

```yaml
agent:
  agentic_nodes:
    finance_report:
      system_prompt: finance_report
      node_class: gen_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "财务分析助手"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      mcp: ""
      rules:
        - 优先复用已有财务指标，再决定是否编写新 SQL
      scoped_context:
        namespace: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup
```

## Scoped Context 的当前语义

当前代码已经不再为每个 subagent 构建单独的 scoped knowledge-base 目录。

现在的行为是：

- 范围信息保存在 `agentic_nodes.<name>.scoped_context`
- Datus 在查询时对共享的全局存储施加过滤
- 数据库工具也可能根据当前 subagent 缩小可见表范围

这意味着：

- 当前 CLI 没有 `/subagent bootstrap` 子命令
- `scoped_kb_path` 已废弃，新保存的配置不会持久化该字段
- 全局知识仍然需要通过 `datus-agent bootstrap-kb` 单独构建

## 高级手工配置

向导覆盖的是最常见的 `gen_sql` 和 `gen_report` 场景。更高级的配置请直接编辑 `agent.yml`。

例如：

```yaml
agent:
  agentic_nodes:
    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      max_turns: 30
```

支持的节点类别以及运行时行为，见 [Subagent 指南](./introduction.zh.md) 和 [内置 subagent](./builtin_subagents.zh.md)。
