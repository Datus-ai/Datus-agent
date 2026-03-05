# 工作流编排

在 Datus Agent 中，工作流编排用于定义、管理并执行一组节点以完成数据分析任务。本文介绍工作流的结构、配置与执行方式，帮助你把自然语言请求自动转换为 SQL 与结果。

## 核心概念

### 1. 工作流定义
工作流是一组按序执行的节点：

- **目标明确**：每个工作流解决某类问题
- **顺序清晰**：节点按预设顺序执行
- **数据共享**：通过共享上下文在节点之间传递信息
- **可自适应**：部分工作流能在运行时自我调整

### 2. 工作流配置
Datus 提供多种内置工作流模板：

```yaml
workflow:
  reflection:
    - schema_linking
    - generate_sql
    - execute_sql
    - reflect
    - output

  fixed:
    - schema_linking
    - generate_sql
    - execute_sql
    - output

  metric_to_sql:
    - schema_linking
    - search_metrics
    - date_parser
    - generate_sql
    - execute_sql
    - output
```

> 说明：以上为内置模板。要自定义或新增工作流，请在 `agent.yml` 配置（见下文“自定义工作流”）。

## 内置工作流类型

### 1. Reflection（反思）
**目的**：具自我改进能力的智能 SQL 生成

**节点序列**：
```
Schema Linking → Generate SQL → Execute SQL → Reflect → Output
```

**特性**：
- 自我评估：反思节点根据结果决定后续动作
- 自适应：可按需要动态加入节点
- 稳健：适配需要多次尝试的复杂查询

**适用**：复杂业务查询、首次 SQL 难以一次到位、需要领域知识的场景

**示例**：
```
User: "Show me quarterly revenue trends by product category, excluding returns and considering seasonal adjustments"
Process:
1. Schema Linking：定位 orders/products/categories
2. Generate SQL：生成季度收入初版
3. Execute SQL：执行
4. Reflect：发现缺少季节性调整
5. Add Fix：补充季节性计算
6. Output：返回修正后的结果
```

### 2. Fixed（固定）
**目的**：确定性路径、可预期的 SQL 生成

**节点序列**：
```
Schema Linking → Generate SQL → Execute SQL → Output
```

**特性**：
- 可预期：始终遵循同一执行路径
- 快速：无反思开销
- 简单：易理解、易排错
- 可靠：适合明确问题

**适用**：简单查询、需求明确、性能敏感

**示例**：
```
User: "List all customers from California"
Process:
1. Schema Linking：定位 customers（含 state）
2. Generate SQL：SELECT * FROM customers WHERE state = 'CA'
3. Execute SQL：返回结果
4. Output：展示
```

### 3. Metric-to-SQL（指标到 SQL）
**目的**：从业务指标生成 SQL

**节点序列**：
```
Schema Linking → Search Metrics → Date Parser → Generate SQL → Execute SQL → Output
```

**特性**：
- 指标驱动：以业务指标为起点
- 时间感知：包含时间解析
- 可复用：复用既有指标定义
- 标准化：保证口径一致

**适用**：BI/报表、标准 KPI、时序分析、看板与例行报告

**示例**：
```
User: "Show monthly active users for the last quarter"
Process:
1. Schema Linking：定位 user_activity
2. Search Metrics：找到 monthly_active_users 定义
3. Date Parser：解析“last quarter”范围
4. Generate SQL：按指标口径生成查询
5. Execute SQL：执行
6. Output：展示分月 MAU
```

## 工作流配置

### 自定义工作流
在 `agent.yml` 中新增模板：
```yaml
agent:
  workflow:
    plan: custom_analytics

    custom_analytics:
      - schema_linking
      - search_metrics
      - generate_sql
      - execute_sql
      - compare
      - output

    data_exploration:
      - schema_linking
      - doc_search
      - generate_sql
      - execute_sql
      - reflect
      - output
```

### 高级特性

#### 并行执行
```yaml
agent:
  workflow:
    plan: bird_para

    bird_para:
      - schema_linking
      - parallel:
        - generate_sql
        - reasoning
      - selection
      - execute_sql
      - output
```

#### 子工作流
```yaml
agent:
  workflow:
    plan: main_workflow

    main_workflow:
      - schema_linking
      - parallel:
        - subworkflow1
        - subworkflow2
      - selection
      - execute_sql
      - output

    subworkflow1:
      - search_metrics
      - generate_sql

    subworkflow2:
      - search_metrics
      - reasoning
```

#### 子工作流（独立配置）
```yaml
agent:
  workflow:
    plan: multi_agent

    multi_agent:
      - schema_linking
      - parallel:
        - agent1_workflow
        - agent2_workflow
      - selection
      - output

    agent1_workflow:
      steps:
        - search_metrics
        - generate_sql
      config: multi/agent1.yaml

    agent2_workflow:
      steps:
        - reasoning
        - reflect
      config: multi/agent2.yaml
```

### 工作流参数
```bash
# 使用内置工作流
datus run --namespace <your_namespace> --task "your query" --plan reflection
# 使用自定义工作流
datus run --namespace <your_namespace> --task "your query" --plan custom_analytics
```

**可用参数**：

| 参数 | 描述 | 默认 | 取值 |
|---|---|---|---|
| `--plan` | 执行的工作流类型 | `reflection` | `reflection`、`fixed`、`metric_to_sql` 或自定义 |
| `--namespace` | 数据库命名空间 | 必填 | 已配置命名空间 |
| `--task` | 自然语言任务 | 必填 | 文本 |
| `--max_iterations` | 最大反思轮数 | `3` | 整数 |
| `--save_dir` | 保存工作流状态目录 | `./save` | 路径 |

## 调试与监控
```bash
# 详细日志
datus run --namespace <your_namespace> --task "your query" --debug
# 保存工作流状态
datus run --namespace <your_namespace> --task "your query" --save_dir ./debug_session
# 从保存状态恢复
datus resume --save_dir ./debug_session
```

## 结语

工作流编排是 Datus Agent 智能 SQL 生成能力的基石。理解不同工作流类型与适用场景，能帮助你高效、可靠地解决复杂的数据分析问题。
