# 语义模型

从 **0.2.4 版本**开始，语义模型作为**模式扩展（schema extensions）**，为数据库表提供语义信息的增强。它们定义表结构（列、维度、度量、实体）以帮助 agent 更好地理解数据，从而生成临时查询。

## 核心价值

增强数据库 schema 理解以改进 SQL 生成：
- **丰富的列描述**：使用模式和过滤示例增强 DDL 注释
- **维度和度量分类**：明确标记列为维度（用于分组）或度量（用于聚合）
- **实体关系**：定义表之间的外键关系以准确生成 JOIN
- **查询模式洞察**：从历史查询中捕获列的典型过滤方式（LIKE、IN、FIND_IN_SET 等）

## 工作原理

语义模型定义基础 schema 层。从 0.2.4 版本开始，它们独立运行：

- **语义模型**（本文档）：包含维度、度量和实体关系的 schema 扩展
  - 存储：LanceDB 中的 `semantic_model` 表
  - 目的：帮助 agent 理解表结构以生成临时 SQL

- **指标**（参见 [metrics.zh.md](metrics.zh.md)）：构建在语义模型之上的业务计算
  - 存储：LanceDB 中的 `metrics` 表
  - 目的：通过 MetricFlow 查询的标准化 KPI

语义模型提供指标引用的构建块（维度、度量）。

## 存储结构

语义模型对象按字段级别存储：

```python
# 存储的对象（kind 字段）:
- "table": 表级元数据
- "column": 列级元数据，带语义标记
- "entity": 关系的实体定义

# 列的语义标记:
- is_dimension: 用于分组/过滤的列
- is_measure: 用于聚合的列
- is_entity_key: 用于表连接的列
```

## 使用方法

**前置条件**：此命令依赖 [datus-semantic-metricflow](../adapters/semantic_adapters.md)，请先运行 `pip install datus-semantic-metricflow` 安装。

### 基本命令

```bash
# 从 CSV（历史 SQLs）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components semantic_model \
    --success_story path/to/success_story.csv

# 从 YAML（现有语义模型文件）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components semantic_model \
    --semantic_yaml path/to/semantic_model.yaml
```

### 关键参数

| 参数 | 必需 | 描述 | 示例 |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | 数据库命名空间 | `sales_db` |
| `--components` | ✅ | 要初始化的组件 | `semantic_model` |
| `--success_story` | ⚠️ | 包含历史 SQLs 的 CSV 文件（如果没有 `--semantic_yaml` 则必需） | `success_story.csv` |
| `--semantic_yaml` | ⚠️ | 语义模型 YAML 文件（如果没有 `--success_story` 则必需） | `semantic_model.yaml` |
| `--kb_update_strategy` | ✅ | 更新策略 | `overwrite`/`incremental` |

## 数据源格式

### CSV 格式

```csv
question,sql
How many orders per customer?,SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;
What is the average order amount?,SELECT AVG(amount) FROM orders WHERE status = 'completed';
```

Agent 分析这些 SQL 以：
- 提取表结构
- 识别维度（GROUP BY 列）和度量（聚合列）
- 发现列使用模式（WHERE 子句过滤器）
- 推断实体关系（JOIN 模式）

### YAML 格式

兼容 MetricFlow 的语义模型 YAML：

```yaml
data_source:
  name: orders
  description: "订单交易表"

  identifiers:
    - name: order
      type: PRIMARY
      expr: id
      description: "唯一订单标识符"

    - name: customer
      type: FOREIGN
      expr: customer_id
      description: "引用 customers 表"

  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
      description: "订单状态。使用 IN 进行过滤。常见值：'pending', 'completed', 'cancelled'"

    - name: created_at
      type: TIME
      expr: created_at
      description: "订单创建时间戳"

  measures:
    - name: amount
      agg: sum
      expr: amount
      description: "订单金额（美元）"

    - name: order_count
      agg: count
      expr: id
      description: "订单数量"
```

## 语义模型如何增强 SQL 生成

生成 SQL 时，agent 使用语义模型来：

1. **Schema Linking**：使用描述的向量搜索找到相关表和列
2. **正确的 JOIN 构造**：使用实体关系生成正确的 JOIN 条件
3. **智能过滤**：应用使用模式（例如，对逗号分隔的标签列使用 `FIND_IN_SET()`）
4. **准确的聚合**：为聚合函数选择适当的度量

查询流程示例：

```text
用户："按客户状态显示总收入"

Agent 处理流程：
1. 搜索语义对象："revenue", "customer", "status"
2. 找到：orders.amount (measure, agg=sum), customers.status (dimension)
3. 使用实体关系：orders.customer_id → customers.id
4. 生成 SQL:
   SELECT c.status, SUM(o.amount) as revenue
   FROM orders o
   JOIN customers c ON o.customer_id = c.id
   GROUP BY c.status
```

## 与上下文搜索集成

语义模型可通过 `@subject` 上下文命令搜索：

```bash
# 在 CLI 聊天模式中
@subject <domain>/<layer1>/<layer2>

# 搜索语义对象
search_semantic_objects(query="customer revenue", kinds=["table", "column"])
```

## 总结

语义模型通过丰富的语义信息扩展数据库 schema，实现更准确的临时 SQL 生成。从 0.2.4 版本开始，它们独立于指标运行：

主要特点：
- **Schema 扩展**：用业务语义增强物理 schema
- **字段级存储**：单独存储表、列、实体以实现灵活搜索
- **使用模式增强**：从历史 SQLs 捕获实际查询模式
- **独立于指标**：专注于 schema 理解，而非指标计算

这种分离确保语义模型专注于其核心目的：帮助 agent 理解数据结构，以生成正确、高效的 SQL 查询。
