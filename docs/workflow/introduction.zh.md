# 工作流概览

Datus Agent 工作流是一个将自然语言问题转换为 SQL 并执行的智能系统。可把它类比为一位懂业务的数据分析师：理解你的意图，自动生成合适的 SQL，拿回结果。

## 内置工作流

### 1. 固定工作流（Fixed）
- **适用场景**：通用 SQL 生成与执行
- **特性**：性能快、路径可预期
- **示例**：
  - “列出加州的所有客户”
  - “展示 2023 年总销售额”
- **用途**：直接取数、聚合、过滤和多表查询

### 2. Agentic 工作流
- **适用场景**：需要迭代调用工具的复杂查询
- **特性**：可按需检索预定义业务指标及其他上下文
- **示例**：
  - “展示上季度 MAU”
  - “计算客户流失率”
- **用途**：KPI 报表、时间分析、语义层查询和复杂 SQL 生成

## 组成：节点（Nodes）

工作流由若干专用“[节点](nodes.md)”组成，每个节点聚焦一件事：

- **[Schema Linking](nodes.md#schema-linking-node)**：为问题找对表
- **[Generate SQL](nodes.md#generate-sql-node)**：生成查询
- **[Execute SQL](nodes.md#execute-sql-node)**：执行查询
- **[Output](nodes.md#output-node)**：友好呈现结果

## 快速上手

### 命令行
```bash
# 简单问题
datus-agent run --datasource your_db --task_db_name analytics --task "Show me monthly sales"

# 指定工作流类型
datus-agent run --datasource your_db --task_db_name analytics --task "Show me complex revenue trends" --workflow fixed

# 使用业务指标
datus-agent run --datasource your_db --task_db_name analytics --task "Calculate customer lifetime value" --workflow gen_sql_agentic
```

### 通过 API
```python
import requests

response = requests.post(
    "http://localhost:8000/workflows/run",
    headers={"Authorization": "Bearer your_token"},
    json={
        "workflow": "fixed",
        "datasource": "your_db",
        "task": "Show me quarterly revenue trends",
        "mode": "sync"
    }
)

result = response.json()
print(result["sql"])    # 生成的 SQL
print(result["result"]) # 查询结果
```

## 下一步
- 了解[工作流编排](orchestration.md)的执行方式
- 查看[API 文档](api.md)以便程序化调用
- 深入[节点说明](nodes.md)理解各组件职责
