# LLM Trace Usage Guide

## Overview

The `--save_llm_trace` parameter allows you to save LLM input and output to YAML files for debugging, analysis, and auditing purposes.

## How to Enable

### 1. Command Line Parameters

Add the `--save_llm_trace` parameter to any Datus command:

```bash
# CLI mode
datus-cli --namespace spidersnow --save_llm_trace

# Main program mode
python datus/main.py run --namespace bird_sqlite --task "Query all users" --task_db_name test.db --save_llm_trace

# Benchmark mode
python datus/main.py benchmark --namespace bird_sqlite --benchmark bird_dev --save_llm_trace
```

### 2. Configuration File

You can also enable tracing for specific models in the `agent.yml` configuration file:

```yaml
agent:
  models:
    deepseek-v3:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
      save_llm_trace: true  # Enable tracing for this model
```

## Output Format

Trace files are saved in the `trajectory_dir/{task_id}/` directory, with each node generating a `{node_id}.yml` file.

### YAML File Structure

```yaml
system_prompt: "You are a helpful SQL assistant."
user_prompt: "Generate a SQL query to select all users"
reason_content: "This query selects all columns from the users table..."
output_content: "SELECT * FROM users;"
```

### Field Descriptions

- **system_prompt**: System prompt (if using message format)
- **user_prompt**: User input prompt
- **reason_content**: Reasoning content (only applicable to reasoning models like deepseek-reasoner)
- **output_content**: Model output content

## Supported Models

Currently, LLM trace functionality is primarily implemented in DeepSeek models:

- ✅ DeepSeek Chat (deepseek-chat)
- ✅ DeepSeek Reasoner (deepseek-reasoner) - includes reasoning content
- 🔄 Other models (OpenAI, Claude, Qwen) have basic interfaces and can be extended

## File Organization

```
trajectory_dir/
└── task_123_20240101/
    ├── node_1.yml          # Schema Linking node
    ├── node_2.yml          # Generate SQL node
    ├── node_3.yml          # Execute SQL node
    └── node_4.yml          # Output node
```

## Use Cases

### 1. Debugging and Development

```bash
# Enable tracing during development to debug prompts
datus-cli --namespace local_duckdb --save_llm_trace
```

### 2. Performance Analysis

```bash
# Analyze LLM input/output for different nodes
python datus/main.py benchmark --namespace bird_sqlite --benchmark bird_dev --save_llm_trace
```

### 3. Auditing and Compliance

```bash
# Record all LLM interactions for auditing
python datus/main.py run --namespace prod --task "sensitive query" --save_llm_trace
```

### 4. MCP Call Tracing

```bash
# Enable detailed tracing of MCP calls, including the entire function call process
python datus/main.py run --namespace bird_sqlite --task "Complex query" --save_llm_trace
```

## Important Notes

1. **Storage Space**: Enabling tracing will increase storage usage, especially for long conversations
2. **Sensitive Information**: Trace files may contain sensitive data, please handle with care
3. **Performance Impact**: File writing operations may slightly affect performance
4. **Directory Permissions**: Ensure the `trajectory_dir` directory has write permissions

## Example Outputs

### Simple Prompt

```yaml
system_prompt: ""
user_prompt: "Create a query to find all active users"
reason_content: ""
output_content: "SELECT * FROM users WHERE status = 'active';"
```

### Output with Reasoning (DeepSeek Reasoner)

```yaml
system_prompt: "You are an expert SQL developer."
user_prompt: "Find the top 5 customers by revenue"
reason_content: "I need to join the customers and orders tables, sum the order amounts, group by customer, and limit to top 5..."
output_content: "SELECT c.name, SUM(o.amount) as total_revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total_revenue DESC LIMIT 5;"
```

### MCP Call Tracing (DeepSeek Model)

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