# 自定义 Subagent

## 概览

`/agent` 和 `/subagent` 都会打开**统一的 agent 管理 TUI**，内含两个 Tab：

- **Built-in** — 列出 `SYS_SUB_AGENTS` 中的系统 subagent。任意行按 `Enter` 把该 agent 设为当前会话默认；按 `e` 打开单字段表单覆写 `max_turns`（落盘到 `agent.agentic_nodes.<name>`）。`model` 不在 UI 中暴露——模型选择由全局 `/model` 负责，如果确实需要按节点指定 model 请直接编辑 `agent.yml`。
- **Custom** — 列出 `agent.yml` 的 `agent.agentic_nodes` 下自定义 subagent。`Enter` 设为当前 agent，`e` 打开向导修改，`a` 启动向导新建，`d d`（两次）删除。

原来的 `/subagent add|list|remove|update` 文本子命令已全部移除，所有操作都在 TUI 内完成。

`/agent <name>` 仍保留为不启动 TUI 的快捷路径，直接把默认 agent 切到 `<name>`。

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

保存时，向导还会把当前数据库写入 `scoped_context.datasource`。

### 第 4 步：Rules

Rules 会以字符串列表形式保存在 `rules` 中，并追加到最终系统提示词里。

## TUI 操作说明

用 `/agent`（落在 Built-in Tab）或 `/subagent`（落在 Custom Tab）打开管理界面。

列表视图快捷键：

| 按键 | 动作 |
|------|------|
| ↑ ↓ / PageUp / PageDown | 上下移动 |
| `Tab` 或 ← → | 切换 Built-in ↔ Custom |
| `Enter` | 将高亮行设为当前 agent（Custom Tab 尾部的 `+ Add agent…` 行则进入向导新建态） |
| `e` | Built-in：打开 model / max_turns 覆写表单；Custom：对高亮项启动向导 |
| `a` | （Custom）启动向导创建新的自定义 subagent |
| 连按两次 `d` | （Custom）删除高亮行（第一次预警、第二次确认） |
| `Esc` / `Ctrl+C` | 取消 |

Built-in 编辑表单快捷键：

| 按键 | 动作 |
|------|------|
| 输入整数 | 直接在输入框编辑 `max_turns` |
| `Enter` / `Ctrl+S` | 保存覆写 |
| `Esc` | 返回 agent 列表 |

覆写只会把 `max_turns` 写入 `agent.agentic_nodes.<name>`，同一节点下的 `scoped_context`、`rules`、`tools` 以及 YAML 中手工写入的 `model` 覆写都会被保留。清空输入（留空 `max_turns`）会移除覆写；若该节点下没有其他字段，整个节点配置项也会从 YAML 中删除。

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
        datasource: finance
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
