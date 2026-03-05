# 业务指标智能化

从 **0.2.4 版本**开始，指标组件专注于创建标准化、可查询的业务指标，作为独立的语义查询层。指标可通过 MetricFlow 直接执行查询，而不仅仅作为 LLM 生成 SQL 的参考。

## 核心价值

解决常见的企业挑战：
- **重复的 SQL 查询**：直接查询指标，而非重写相似的 SQL
- **不一致的定义**：通过可执行的规范跨团队标准化指标定义
- **手动分类**：使用层级主题树分类体系组织指标
- **临时 SQL 复杂性**：对常见指标使用语义查询（`query_metrics`）而非生成 SQL

## 工作原理

指标是构建在语义模型之上的业务级计算。从 0.2.4 版本开始，它们独立运行：

- **指标**（本文档）：通过 MetricFlow 查询的标准化 KPI
- **语义模型**（参见 [semantic_model.zh.md](semantic_model.zh.md)）：用于临时 SQL 生成的 schema 扩展

两者都可以从历史 SQLs 生成，但指标专注于可复用的业务逻辑，而语义模型专注于 schema 理解。

## 查询指标

定义指标后，可使用 MetricFlow 工具直接查询：

```python
# 在 agent 对话或工作流中
# 搜索相关指标
search_semantic_objects(query="daily active users", kinds=["metric"])

# 执行指标查询
query_metrics(
    metrics=["daily_active_users"],
    group_by=["platform", "country"],
    start_time="2024-01-01",
    end_time="2024-01-31"
)
```

**指标优先策略**：当用户查询涉及 KPI（例如 "按平台展示 DAU"）时，agent 将：
1. 使用 `search_semantic_objects` 搜索匹配的指标
2. 如果找到，通过 `query_metrics` 执行（首选）
3. 仅当不存在指标时才回退到临时 SQL 生成

这确保了组织内指标定义的一致性。

## 使用方法

**前置条件**：此命令依赖 [datus-semantic-metricflow](../adapters/semantic_adapters.md)，请先运行 `pip install datus-semantic-metricflow` 安装。

### 基本命令

```bash
# 从 CSV（历史 SQLs）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components metrics \
    --success_story path/to/success_story.csv

# 从 YAML（语义模型）
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components metrics \
    --semantic_yaml path/to/semantic_model.yaml
```

### 关键参数

| 参数 | 必需 | 描述 | 示例 |
|-----------|----------|-------------|------------|
| `--namespace` | ✅ | 数据库命名空间 | `sales_db` |
| `--components` | ✅ | 要初始化的组件 | `metrics` |
| `--success_story` | ⚠️ | 包含历史 SQLs 和问题的 CSV 文件（如果没有 `--semantic_yaml` 则必需） | `success_story.csv` |
| `--semantic_yaml` | ⚠️ | 语义模型 YAML 文件（如果没有 `--success_story` 则必需） | `semantic_model.yaml` |
| `--kb_update_strategy` | ❌ | 更新策略 | `overwrite`/`incremental` |
| `--subject_tree` | ❌ | 预定义分类（逗号分隔） | `Sales/Reporting/Daily,Finance/Revenue/Monthly` |
| `--pool_size` | ❌ | 并发线程数 | `4` |

### 主题树分类

使用层级分类法组织指标：`domain/layer1/layer2`（例如 `Sales/Reporting/Daily`）

**两种模式：**
- **预定义**：使用 `--subject_tree` 强制指定特定分类
- **学习**：省略 `--subject_tree` 以复用现有分类或创建新分类

```bash
# 预定义模式示例
--subject_tree "Sales/Reporting/Daily,Finance/Revenue/Monthly"

# 学习模式：省略 --subject_tree 参数
```

**生成的标签格式：**

指标生成后，主题树分类存储在 `locked_metadata.tags` 中，格式为 `"subject_tree: {domain}/{layer1}/{layer2}"`：

```yaml
metric:
  name: daily_revenue
  type: simple
  type_params:
    measure: revenue
  locked_metadata:
    tags:
      - "Finance"
      - "subject_tree: Sales/Reporting/Daily"
```

**YAML 导入注意事项：**

使用 `--semantic_yaml` 从 YAML 文件同步指标到 lancedb 时，必须在 YAML 文件中手动添加包含 subject_tree 格式的 `locked_metadata.tags` 才能成功分类。系统不会自动对从 YAML 导入的指标进行分类，需要自行添加标签：

```yaml
metric:
  name: your_metric
  # ... 其他字段
  locked_metadata:
    tags:
      - "YourDomain"
      - "subject_tree: Domain/Layer1/Layer2"
```

## 数据源格式

### CSV 格式

```csv
question,sql
How many customers have been added per day?,"SELECT ds AS date, SUM(1) AS new_customers FROM customers GROUP BY ds ORDER BY ds;"
What is the total transaction amount?,SELECT SUM(transaction_amount_usd) as total_amount FROM transactions;
```

### YAML 格式（仅指标）

从 YAML 文件导入指标时，指标定义引用已存在的语义模型：

```yaml
metric:
  name: total_revenue
  description: "Total revenue from all transactions"
  type: simple
  type_params:
    measure: amount  # 引用语义模型中的 measure
  filter: "amount > 0"
  locked_metadata:
    tags:
      - "Finance"
      - "subject_tree: Finance/Revenue/Total"
```

**注意**：底层的语义模型（包含 dimensions/measures 的 `data_source`）应该已经存在。参见 [semantic_model.zh.md](semantic_model.zh.md) 了解如何定义语义模型。

## 总结

指标组件建立了一个**语义查询层**，将历史 SQLs 转换为标准化、可执行的指标定义。与传统的仅作为 LLM 参考的语义层不同，Datus 指标可通过 MetricFlow 直接查询，无需为常见 KPI 生成临时 SQL。

主要特点：
- **可执行指标**：通过 `query_metrics` 查询而非生成 SQL
- **指标优先策略**：Agent 优先使用指标查询而非临时 SQL
- **独立于语义模型**：指标作为独立的查询工具运行，而非嵌入在 schema 定义中
- **层级组织**：主题树分类法提高可发现性

这种方法确保了团队之间指标定义的一致性，同时降低了查询复杂性并提高了性能。
