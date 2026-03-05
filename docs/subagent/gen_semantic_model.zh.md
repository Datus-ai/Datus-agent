# 语义模型生成指南

## 概览

语义模型生成功能帮助你通过 AI 助手从数据库表创建 MetricFlow 语义模型。助手分析你的表结构并生成全面的 YAML 配置文件，定义指标、维度和关系。

## 什么是语义模型？

语义模型是定义以下内容的 YAML 配置：

- **度量（Measures）**：指标和聚合（SUM、COUNT、AVERAGE 等）
- **维度（Dimensions）**：分类和时间属性
- **标识符（Identifiers）**：用于关系的主键和外键
- **数据源（Data Source）**：与数据库表的连接

## 工作原理

使用 `datus --namespace <namespace>` 启动 Datus CLI，然后使用子代理命令：

```text
  /gen_semantic_model generate a semantic model for table <table_name>
```


### 交互式生成

当你请求语义模型时，AI 助手会：

1. 检索你的表的 DDL（结构）
2. 检查是否已存在语义模型
3. 生成全面的 YAML 文件
4. 使用 MetricFlow 验证配置
5. 提示你保存到知识库

### 生成工作流

```text
用户请求 → DDL 分析 → YAML 生成 → 验证 → 用户确认 → 存储
```

### 用户确认

生成语义模型后，你会看到：

```text
=============================================================
Generated YAML: table_name.yml
Path: /path/to/file.yml
=============================================================
[带语法高亮的 YAML 内容]

SYNC TO KNOWLEDGE BASE?

1. Yes - Save to Knowledge Base
2. No - Keep file only

Please enter your choice: [1/2]
```

**选项**：
- **选项 1**：将语义模型保存到你的知识库（RAG 存储）用于 AI 驱动的查询
- **选项 2**：仅保留 YAML 文件，不同步到知识库

## 配置

大部分配置是内置的。在 `agent.yml` 中，最小化设置即可：

```yaml
agentic_nodes:
  gen_semantic_model:
    model: claude        # 可选：默认使用已配置的模型
    max_turns: 30        # 可选：默认为 30
```

**内置配置**（自动启用）：
- **工具**：数据库工具、生成工具和文件系统工具
- **Hooks**：交互模式下的用户确认工作流
- **MCP 服务器**：MetricFlow 验证服务器
- **系统提示**：内置模板版本 1.0
- **工作空间**：`~/.datus/data/{namespace}/semantic_models`

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
- ✓ 存储前用户确认
- ✓ 知识库集成
- ✓ 防止重复
- ✓ MetricFlow 兼容性

