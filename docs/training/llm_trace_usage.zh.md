# LLM Trace 使用指南

## 概览

`--save_llm_trace` 参数允许你将 LLM 输入和输出保存到 YAML 文件，用于调试、分析和审计目的。

## 如何启用

### 1. 命令行参数

在任何 Datus 命令中添加 `--save_llm_trace` 参数：

```bash
# CLI 模式
datus-cli --namespace spidersnow --save_llm_trace

# 主程序模式
python datus/main.py run --namespace bird_sqlite --task "Query all users" --task_db_name test.db --save_llm_trace

# Benchmark 模式
python datus/main.py benchmark --namespace bird_sqlite --benchmark bird_dev --save_llm_trace
```

### 2. 配置文件

你也可以在 `agent.yml` 配置文件中为特定模型启用跟踪：

```yaml
agent:
  models:
    deepseek-v3:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
      save_llm_trace: true  # 为此模型启用跟踪
```

## 输出格式

跟踪文件保存在 `trajectory_dir/{task_id}/` 目录中，每个节点生成一个 `{node_id}.yml` 文件。

### YAML 文件结构

```yaml
system_prompt: "You are a helpful SQL assistant."
user_prompt: "Generate a SQL query to select all users"
reason_content: "This query selects all columns from the users table..."
output_content: "SELECT * FROM users;"
```

### 字段说明

- **system_prompt**：系统提示词（如果使用消息格式）
- **user_prompt**：用户输入提示词
- **reason_content**：推理内容（仅适用于推理模型，如 deepseek-reasoner）
- **output_content**：模型输出内容

## 支持的模型

目前，LLM trace 功能主要在 DeepSeek 模型中实现：

- ✅ DeepSeek Chat (deepseek-chat)
- ✅ DeepSeek Reasoner (deepseek-reasoner) - 包含推理内容
- 🔄 其他模型（OpenAI、Claude、Qwen）有基本接口，可以扩展

## 文件组织

```
trajectory_dir/
└── task_123_20240101/
    ├── node_1.yml          # Schema Linking 节点
    ├── node_2.yml          # Generate SQL 节点
    ├── node_3.yml          # Execute SQL 节点
    └── node_4.yml          # Output 节点
```

## 使用场景

### 1. 调试和开发

```bash
# 在开发期间启用跟踪以调试提示词
datus-cli --namespace local_duckdb --save_llm_trace
```

### 2. 性能分析

```bash
# 分析不同节点的 LLM 输入/输出
python datus/main.py benchmark --namespace bird_sqlite --benchmark bird_dev --save_llm_trace
```

### 3. 审计和合规

```bash
# 记录所有 LLM 交互以进行审计
python datus/main.py run --namespace prod --task "sensitive query" --save_llm_trace
```

### 4. MCP 调用跟踪

```bash
# 启用 MCP 调用的详细跟踪，包括整个函数调用过程
python datus/main.py run --namespace bird_sqlite --task "Complex query" --save_llm_trace
```

## 重要说明

1. **存储空间**：启用跟踪会增加存储使用量，特别是对于长对话
2. **敏感信息**：跟踪文件可能包含敏感数据，请谨慎处理
3. **性能影响**：文件写入操作可能会略微影响性能
4. **目录权限**：确保 `trajectory_dir` 目录具有写入权限

## 输出示例

### 简单提示

```yaml
system_prompt: ""
user_prompt: "Create a query to find all active users"
reason_content: ""
output_content: "SELECT * FROM users WHERE status = 'active';"
```

### 带推理的输出（DeepSeek Reasoner）

```yaml
system_prompt: "You are an expert SQL developer."
user_prompt: "Find the top 5 customers by revenue"
reason_content: "I need to join the customers and orders tables, sum the order amounts, group by customer, and limit to top 5..."
output_content: "SELECT c.name, SUM(o.amount) as total_revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total_revenue DESC LIMIT 5;"
```

### MCP 调用跟踪（DeepSeek 模型）

```yaml
system_prompt: "You are a snowflake expert."
user_prompt: "Instruction: Generate SQL for data analysis\n\nUser Prompt: Find recent transactions"
reason_content: |
  === MCP Conversation Started ===
  Instruction: You are a snowflake expert. Generate SQL queries and execute them.
  User Prompt: Find recent transactions
  Max Turns: 10

  === MCP Server Started ===
  Agent created: MCP_Agent
  Output type: <class 'str'>

  === Agent Execution Started ===

  === Result Analysis ===
  Result type: <class 'agents.runner.Result'>
  Result attributes: ['final_output', 'messages', 'run_id', 'iteration_count', 'total_cost']

  === Conversation Messages ===
  Message 1 (user):
    Content: Find recent transactions from the database

  Message 2 (assistant):
    Content: I'll help you find recent transactions. Let me query the database.
    Tool Calls:
      1. read_query({"query": "SELECT * FROM transactions WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' ORDER BY created_at DESC LIMIT 100"})

  Message 3 (user):
    Content: Tool result: [transaction data...]

  === Tool Execution Results ===
  Tool Output 1:
    Content: SELECT * FROM transactions WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' ORDER BY created_at DESC LIMIT 100
    Results: [{"id": 1, "amount": 100.00, "created_at": "2024-01-15"}...]

  === Agent Execution Completed ===
  Final Output: Here are the recent transactions from the last 7 days: [results summary]

  Run ID: run_123456
  Iteration count: 3
  Total cost: 0.025
output_content: "Here are the recent transactions from the last 7 days: [results summary]"
```
