---
title: '节点（Nodes）'
description: '为结构关联、SQL 生成等处理任务配置工作流节点'
---

## 概览

节点是 Datus Agent 工作流的构件。每个节点负责数据处理链路中的一环，从结构关联与 SQL 生成到结果输出。本文介绍如何按需配置各类节点。

## 配置结构
```yaml
nodes:
  node_name:
    model: provider_name
    prompt_version: "1.0"
    # 其它节点参数
agentic_nodes:
  gen_sql:
    model: provider_name
    system_prompt: gen_sql
    max_turns: 30
```

!!! tip
    `model` 引用在顶层 [`models`](agent.md#models-configuration) 中定义的提供方键名。

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
- `prompt_version`：SQL模板版本

### GenSQL
```yaml
agentic_nodes:
  gen_sql:
    model: deepseek_v3
    system_prompt: gen_sql
    prompt_version: "1.2"
    tools: db_tools.*, context_search_tools.*
    max_turns: 30
```
**参数**：`gen_sql` 是 agentic 节点，`system_prompt` 统一使用 `gen_sql`，对应 `gen_sql_system` prompt，可通过 `tools` 和 `max_turns` 控制工具范围和推理轮数。

## 处理节点

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
agentic_nodes:
  chat:
    workspace_root: sql2
    model: anthropic
    max_turns: 25
```
**参数**：工作目录、对话模型、最大轮数。

## 实用节点

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

  output:
    model: anthropic
    prompt_version: "1.0"
    check_result: true

  fix:
    model: openai
    prompt_version: "1.0"

agentic_nodes:
  gen_sql:
    model: deepseek_v3
    system_prompt: gen_sql
    prompt_version: "1.2"
    tools: db_tools.*, context_search_tools.*, date_parsing_tools.*
    max_turns: 30

  chat:
    workspace_root: workspace
    model: anthropic
    max_turns: 25

# 日期解析工具配置（独立于工作流节点）
date_parsing:
  language: zh                     # en/zh
```

指标检索、日期解析和平台文档检索现在都是函数工具，而不是工作流节点。可分别通过 `context_search_tools.*`、`date_parsing_tools.*` 和 `platform_doc_tools.*` 为 agentic 节点启用。

## 模型分配建议
- 结构关联：`gpt-3.5-turbo`、`deepseek-v4-flash`；复杂结构用 `gpt-4`、`claude-4-sonnet`
- SQL 生成：建议 `deepseek-v4-flash`、`gpt-4-turbo`、`claude-4-sonnet`
- Agentic 工作流：`claude-4-sonnet`、`gpt-4-turbo`、`claude-4-opus`，或 `gemini-2.5-flash`
- 输出/对话：`claude-4-sonnet`、`gpt-4-turbo`
