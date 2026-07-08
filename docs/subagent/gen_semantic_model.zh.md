# 语义模型生成指南

## 概览

语义模型生成功能帮助你通过 AI 助手从数据库表创建语义模型。具体 YAML 格式由配置的 semantic adapter 决定：`metricflow` 生成 MetricFlow YAML，`osi` 生成 strict OSI core YAML。助手会分析表结构，并按所选适配器生成配置文件。

## 什么是语义模型？

语义模型是定义以下内容的 YAML 配置：

- **度量（Measures）**：指标和聚合（SUM、COUNT、AVERAGE 等）
- **维度（Dimensions）**：分类和时间属性
- **标识符（Identifiers）**：用于关系的主键和外键
- **数据源（Data Source）**：与数据库表的连接

## 工作原理

使用 `datus --datasource <datasource>` 启动 Datus CLI，然后使用子代理命令：

```text
  /gen_semantic_model generate a semantic model for table <table_name>
```


### 交互式生成

当你请求语义模型时，AI 助手会：

1. 检索你的表的 DDL（结构）
2. 检查是否已存在语义模型
3. 生成全面的 YAML 文件
4. 使用配置的 semantic adapter 验证配置
5. 验证通过后同步到知识库

### 生成工作流

```text
用户请求 → DDL 分析 → YAML 生成 → 验证 → 存储
```

### 验证和同步

发布前，agent 会调用 `validate_semantic()`。如果验证失败，会修改 YAML 并重试；验证通过后，`end_semantic_model_generation` 会自动把语义模型同步到知识库。

## 配置

大部分配置是内置的。在 `agent.yml` 中，最小化设置即可：

```yaml
agent:
  services:
    semantic_layer:
      metricflow: {}     # key 必须等于 adapter type（例如 `metricflow`）。
                         # 如果同时写了 `type:` 字段，必须与 key 一致，否则 Datus 会在启动时抛出配置错误。

  agentic_nodes:
    gen_semantic_model:
      model: claude      # 可选：默认使用已配置的模型
      max_turns: 30      # 可选：默认为 30
      semantic_adapter: metricflow   # 当仅配置了一个 semantic layer 时可省略
      # 可选：在生成 YAML 前启用历史 SQL profiling。
      # 覆盖 skills 时，需要同时保留默认的 MetricFlow 语义模型 skill。
      skills: "metricflow-semantic-authoring, semantic-sql-history-profiler"
```

完整配置项见 [语义层配置](../configuration/semantic_layer.zh.md)。

OSI 生成见 [OSI 语义适配器](../adapters/osi_semantic_adapter.zh.md)。

### 可选的历史 SQL Profiling

`semantic-sql-history-profiler` 是 `gen_semantic_model` 使用的内部 skill，不是聊天命令，也不能由用户直接调用。需要让语义模型生成基于历史 SQL 或 success-story SQL 做建模证据分析时，可以在 `gen_semantic_model` 节点上启用它。

当该 skill 可用，并且请求中包含历史 SQL 或 success-story SQL 时，subagent 会在生成 YAML 前加载它，并调用 `profile_semantic_model_evidence`。这些证据会用于推断 JOIN 关系、常见过滤或分组维度、聚合候选、时间字段、简洁的数据分布说明，以及关系可靠性提示。

MetricFlow 生成场景下，如果覆盖 `skills`，需要把默认语义模型 skill 一起列出：

```yaml
agent:
  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow
      skills: "metricflow-semantic-authoring, semantic-sql-history-profiler"
```

OSI 生成场景下，除非明确需要其他 skill，通常只启用 profiler：

```yaml
agent:
  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: osi
      skills: "semantic-sql-history-profiler"
```

**内置配置**（自动启用）：
- **工具**：数据库工具、生成工具和文件系统工具
- **Hooks**：验证证据记录和知识库同步
- **Semantic Adapter**：通过配置的语义层进行验证
- **系统提示**：内置模板；未显式设置 `prompt_version` 时使用最新可用版本
- **工作空间**：`~/.datus/data/{datasource}/semantic_models`

## 语义模型结构

### 基本模板

```yaml
data_source:
  name: table_name                    # 必需：小写加下划线
  description: "Table description"

  sql_table: schema.table_name        # 对于有 schema 的数据库
  # OR
  sql_query: |                        # 对于自定义查询
    SELECT * FROM table_name

  measures:
    - name: total_amount              # 必需
      agg: SUM                        # 必需：SUM|COUNT|AVERAGE|etc.
      expr: amount_column             # 列或 SQL 表达式
      create_metric: true             # 自动创建可查询指标
      description: "Total transaction amount"

  dimensions:
    - name: created_date
      type: TIME                      # 必需：TIME|CATEGORICAL
      type_params:
        is_primary: true              # 需要一个主时间维度
        time_granularity: DAY         # TIME 必需：DAY|WEEK|MONTH|etc.

    - name: status
      type: CATEGORICAL
      description: "Order status"

  identifiers:
    - name: order_id
      type: PRIMARY                   # PRIMARY|FOREIGN|UNIQUE|NATURAL
      expr: order_id

    - name: customer
      type: FOREIGN
      expr: customer_id
```

## 总结

语义模型生成功能提供：

- ✓ 从表 DDL 自动生成 YAML
- ✓ 交互式验证和错误修复
- ✓ 验证通过后自动同步
- ✓ 知识库集成
- ✓ 防止重复
- ✓ Semantic adapter 兼容性
