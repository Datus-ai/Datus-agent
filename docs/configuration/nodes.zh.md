---
title: '节点（Nodes）'
description: '为结构关联、SQL 生成、推理等处理任务配置工作流节点'
---

## 概览

节点是 Datus Agent 工作流的构件。每个节点负责数据处理链路中的一环，从结构关联与 SQL 生成，到推理与结果输出。本文介绍如何按需配置各类节点。

## 配置结构
```yaml
nodes:
  node_name:
    model: provider_name
    prompt_version: "1.0"
    # 其它节点参数
```

!!! tip
    `model` 引用在 [`agent.models`](agent.md#models-configuration) 中定义的提供方键名。

## 核心节点

### Schema Linking
```yaml
schema_linking:
  model: openai
  matching_rate: fast        # fast/medium/slow/from_llm
  prompt_version: "1.0"
```
**参数**：

- `model`：引用 agent.models 的键
- `matching_rate`：匹配范围/速度（fast/medium/slow/from_llm）
- `prompt_version`：提示模板版本

### Generate SQL
```yaml
generate_sql:
  model: deepseek_v3
  prompt_version: "1.0"
  max_table_schemas_length: 4000
  max_data_details_length: 2000
  max_context_length: 8000
  max_value_length: 500
```
**参数**：同上，侧重控制上下文长度上限。

### Reasoning
```yaml
reasoning:
  model: anthropic
  prompt_version: "1.0"
  max_table_schemas_length: 4000
  max_data_details_length: 2000
  max_context_length: 8000
  max_value_length: 500
```
**参数**：同 `generate_sql`，用于迭代优化。

### Search Metrics
```yaml
search_metrics:
  model: openai
  matching_rate: medium
  prompt_version: "1.0"
```
**参数**：同 `schema_linking`，针对指标检索。

## 处理节点

### Reflect
```yaml
reflect:
  prompt_version: "1.0"
```
**参数**：`prompt_version`

### Output
```yaml
output:
  model: anthropic
  prompt_version: "1.0"
  check_result: true
```
**参数**：格式化/校验相关设置。

## 交互节点

### Chat
```yaml
chat:
  workspace_root: sql2
  model: anthropic
  max_turns: 25
```
**参数**：工作目录、对话模型、最大轮数。

## 实用节点

### Date Parser
```yaml
date_parser:
  prompt_version: "1.0"
```

### Compare
```yaml
compare:
  prompt_version: "1.0"
```

### Fix
```yaml
fix:
  model: openai
  prompt_version: "1.0"
```

## 完整示例
```yaml
nodes:
  schema_linking:
    model: openai
    matching_rate: fast
    prompt_version: "1.0"

  search_metrics:
    model: openai
    matching_rate: medium
    prompt_version: "1.0"

  generate_sql:
    model: deepseek_v3
    prompt_version: "1.0"
    max_table_schemas_length: 4000
    max_data_details_length: 2000
    max_context_length: 8000
    max_value_length: 500

  reasoning:
    model: anthropic
    prompt_version: "1.0"
    max_table_schemas_length: 4000
    max_data_details_length: 2000
    max_context_length: 8000
    max_value_length: 500

  reflect:
    prompt_version: "1.0"

  output:
    model: anthropic
    prompt_version: "1.0"
    check_result: true

  chat:
    workspace_root: workspace
    model: anthropic
    max_turns: 25

  date_parser:
    prompt_version: "1.0"

  fix:
    model: openai
    prompt_version: "1.0"
```

## 模型分配建议
- 结构关联：`gpt-3.5-turbo`、`deepseek-chat`；复杂结构用 `gpt-4`、`claude-4-sonnet`
- SQL 生成：建议 `deepseek-chat`、`gpt-4-turbo`、`claude-4-sonnet`
- 推理：`claude-4-sonnet`、`gpt-4-turbo`、`claude-4-opus`，或 `gemini-2.5-flash`
- 输出/对话：`claude-4-sonnet`、`gpt-4-turbo`
