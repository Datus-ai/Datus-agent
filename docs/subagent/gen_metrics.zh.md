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

使用 `datus --namespace <namespace>` 启动 Datus CLI，然后使用指标生成子代理：

```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, the coresponding question is total amount of all transactions
```

### 生成工作流

```
用户提供 SQL 和问题 → 智能体分析逻辑 → 查找语义模型 → 读取度量 →
检查重复 → 生成指标 YAML → 追加到文件 → 验证 →
用户确认 → 同步到知识库
```

### 交互式确认

生成后，你会看到：

```
==========================================================
Generated YAML: transactions.yml
Path: /Users/you/.datus/data/semantic_models/transactions.yml
==========================================================
[带语法高亮的 YAML 内容，显示新指标]

  SYNC TO KNOWLEDGE BASE?

  1. Yes - Save to Knowledge Base
  2. No - Keep file only

Please enter your choice: [1/2]
```

**选项**：
- **选项 1**：将指标同步到你的知识库，用于 AI 驱动的语义搜索
- **选项 2**：仅保留 YAML 文件，不同步到知识库

## 配置

大部分配置是内置的。在 `agent.yml` 中，最小化设置即可：

```yaml
agentic_nodes:
  gen_metrics:
    model: claude        # 可选：默认使用已配置的模型
    max_turns: 30        # 可选：默认为 30
```

**内置配置**（自动启用）：
- **工具**：生成工具和文件系统工具
- **Hooks**：交互模式下的用户确认工作流
- **MCP 服务器**：MetricFlow 验证服务器
- **系统提示**：内置模板版本 1.0
- **工作空间**：`~/.datus/data/{namespace}/semantic_models`

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
3. 生成 simple 指标：

```yaml
---
metric:
  name: total_orders
  description: Total number of orders
  type: simple
  type_params:
    measure: order_count
  locked_metadata:
    display_name: "Total Orders"
    increase_is_good: true
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
3. 生成 simple 指标：

```yaml
---
metric:
  name: total_revenue
  description: Total revenue from completed transactions
  type: simple
  type_params:
    measure: total_amount
  filter: "status = 'completed'"
  locked_metadata:
    display_name: "Total Revenue"
    value_format: "$,.2f"
    increase_is_good: true
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
3. 生成 simple 指标：

```yaml
---
metric:
  name: unique_customer_count
  description: Total number of unique customers
  type: simple
  type_params:
    measure: unique_customers
  locked_metadata:
    display_name: "Unique Customers"
    increase_is_good: true
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
  description: Total revenue from all transactions
  type: simple
  type_params:
    measure: revenue

---
metric:
  name: total_transactions
  description: Total number of transactions
  type: simple
  type_params:
    measure: transaction_count
```

**为什么分开存储？**
- 清晰分离 schema 定义和业务指标
- 更易于独立管理指标
- 指标文件内使用 YAML 文档分隔符 `---` 分隔多个指标

### 知识库存储

当你选择 "1. Yes - Save to Knowledge Base" 时，指标会存储到向量数据库中，包含：

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
✅ **交互式工作流**：同步前审阅和批准
✅ **知识库集成**：语义搜索以发现指标
✅ **文件管理**：将指标组织在独立于语义模型的专用文件中

