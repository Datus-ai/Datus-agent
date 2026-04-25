# 指标生成指南

## 概览

指标生成功能帮助你将 SQL 查询转换为可复用的 MetricFlow 指标定义。使用 AI 助手，你可以分析 SQL 业务逻辑并自动生成标准化的 YAML 指标配置，组织内可一致查询。

## 什么是指标？

**指标**是基于语义模型构建的可复用业务计算。指标提供：

- **一致的业务逻辑**：一次定义，到处使用
- **类型安全**：已验证的结构和度量引用
- **元数据**：显示名称、格式、业务上下文
- **可组合性**：从简单指标构建复杂指标

**示例**：与其重复编写 `SELECT SUM(revenue) / COUNT(DISTINCT customer_id)`，不如定义一次 `avg_customer_revenue` 指标。

## 重要限制

**⚠️ 仅支持单表查询**

当前版本**仅支持从单表 SQL 查询生成指标**。不支持多表 JOIN。

**支持**：
```sql
SELECT SUM(revenue) FROM transactions WHERE status = 'completed'
SELECT COUNT(DISTINCT customer_id) / COUNT(*) FROM orders
```

**不支持**：
```sql
SELECT SUM(o.amount)
FROM orders o
JOIN customers c ON o.customer_id = c.id  -- ❌ 不支持 JOIN
```

## 工作原理

使用 `datus --database <datasource>` 启动 Datus CLI，然后使用指标生成子代理：

```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, the coresponding question is total amount of all transactions
```

### 生成工作流

```
用户提供 SQL 和问题 → 智能体分析逻辑 → 查找语义模型 → 读取度量 →
检查重复 → 生成指标 YAML → 追加到文件 → 验证 →
生成 dry-run SQL → 同步到知识库
```

### 验证和同步

发布前会经过两项检查：

- `validate_semantic()` 校验语义模型和指标 YAML。
- `query_metrics(..., dry_run=True)` 确认 MetricFlow 能为生成的指标编译 SQL。

两项检查都通过后，`end_metric_generation` 会自动把指标同步到知识库。

## 配置

大部分配置是内置的。在 `agent.yml` 中，最小化设置即可：

```yaml
agent:
  services:
    semantic_layer:
      metricflow: {}     # key 必须等于 adapter type（例如 `metricflow`）。
                         # 如果同时写了 `type:` 字段，必须与 key 一致，否则 Datus 会在启动时抛出配置错误。

  agentic_nodes:
    gen_metrics:
      model: claude      # 可选：默认使用已配置的模型
      max_turns: 30      # 可选：默认为 30
      semantic_adapter: metricflow   # 当仅配置了一个 semantic layer 时可省略
```

完整配置项见 [语义层配置](../configuration/semantic_layer.zh.md)。

**内置配置**（自动启用）：
- **工具**：生成工具和文件系统工具
- **Hooks**：验证证据记录和知识库同步
- **MCP 服务器**：MetricFlow 验证服务器
- **系统提示**：内置模板；未显式设置 `prompt_version` 时使用最新可用版本
- **工作空间**：`~/.datus/data/{datasource}/semantic_models`

### 主题树分类

在 CLI 模式下通过问题中包含主题树来组织指标：

**带主题树示例：**
```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, subject_tree: finance/revenue/transactions
```

**不带主题树示例：**
```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions
```

未提供时，agent 会基于知识库中的现有指标自动建议分类。



## 使用示例

### 示例 1：简单聚合

**用户输入**：
```text
/gen_metrics Generate a metric for total order count
```

**智能体操作**：
1. 查找 `orders.yml` 语义模型
2. 读取文件以发现 `order_count` 度量
3. 生成 measure proxy 指标：

```yaml
---
metric:
  name: total_orders
  description: "Total number of orders"
  type: measure_proxy
  type_params:
    measure: order_count
  locked_metadata:
    tags:
      - "subject_tree: finance/orders/core"
```

### 示例 2：收入指标

**用户输入**：
```
/gen_metrics Create a metric from this SQL:
SELECT SUM(amount) as total_revenue FROM transactions WHERE status = 'completed'
```

**智能体操作**：
1. 分析 SQL 聚合（带过滤的 SUM）
2. 查找或创建 `transactions.yml` 语义模型
3. 在语义模型中增加带过滤逻辑的 measure，并生成引用它的指标：

```yaml
---
data_source:
  name: transactions
  measures:
    - name: completed_revenue
      description: "Revenue from completed transactions"
      agg: SUM
      expr: "CASE WHEN status = 'completed' THEN amount ELSE 0 END"
---
metric:
  name: total_revenue
  description: "Total revenue from completed transactions"
  type: measure_proxy
  type_params:
    measure: completed_revenue
  locked_metadata:
    tags:
      - "subject_tree: finance/revenue/transactions"
```

### 示例 3：计数指标

**用户输入**：
```text
/gen_metrics Generate unique customer count metric:
SELECT COUNT(DISTINCT customer_id) FROM orders
```

**智能体操作**：
1. 定位 `orders.yml` 语义模型
2. 识别 COUNT DISTINCT 聚合
3. 生成 measure proxy 指标：

```yaml
---
metric:
  name: unique_customer_count
  description: "Total number of unique customers"
  type: measure_proxy
  type_params:
    measure: unique_customers
  locked_metadata:
    tags:
      - "subject_tree: sales/customers/core"
```

## 指标存储方式

### 文件组织

指标与语义模型分开存储在不同文件中：

- **语义模型**：`{table_name}.yml` - 包含 data_source 定义，含度量和维度
- **指标**：`metrics/{table_name}_metrics.yml` - 包含指标定义

**语义模型文件** (`transactions.yml`)：
```yaml
data_source:
  name: transactions
  sql_table: transactions
  measures:
    - name: revenue
      agg: SUM
      expr: amount
    - name: transaction_count
      agg: COUNT
      expr: "1"
  dimensions:
    - name: transaction_date
      type: TIME
```

**指标文件** (`metrics/transactions_metrics.yml`)：
```yaml
metric:
  name: total_revenue
  description: "Total revenue from all transactions"
  type: measure_proxy
  type_params:
    measure: revenue

---
metric:
  name: total_transactions
  description: "Total number of transactions"
  type: measure_proxy
  type_params:
    measure: transaction_count
```

**为什么分开存储？**
- 清晰分离 schema 定义和业务指标
- 更易于独立管理指标
- 指标文件内使用 YAML 文档分隔符 `---` 分隔多个指标

### 知识库存储

验证和 dry-run SQL 通过后，指标会同步到知识库，包含：

1. **元数据**：名称、描述、类型、域/层级分类
2. **LLM 文本**：用于语义搜索的自然语言表示
3. **引用**：关联的语义模型名称
4. **时间戳**：创建日期

## 总结

指标生成功能提供：

✅ **SQL 到指标转换**：分析 SQL 查询并生成 MetricFlow 指标
✅ **智能类型检测**：自动选择正确的指标类型
✅ **防止重复**：生成前检查现有指标
✅ **主题树支持**：按 domain/layer1/layer2 组织，支持预定义或学习模式
✅ **验证**：MetricFlow 验证确保正确性
✅ **发布门禁**：语义验证和 dry-run SQL 通过后才同步
✅ **知识库集成**：语义搜索以发现指标
✅ **文件管理**：将指标组织在独立于语义模型的专用文件中
