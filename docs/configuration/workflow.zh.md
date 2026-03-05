# 工作流配置

在 Datus Agent 中配置工作流的执行计划、节点顺序、并行与子工作流组合。

!!! tip "快速上手"
    初学者可先阅读下方的“基础配置”。

## 结构
```yaml title="agent.yml"
workflow:
  plan: planA

planA:
  - schema_linking
  - generate_sql
  - output

planB:
  - schema_linking
  - generate_sql
  - execute_sql
  - reflect
  - output
```

## 基础配置

### 顺序执行
```yaml title="Simple Sequential Workflow"
workflow:
  plan: basic_sql

basic_sql:
  - schema_linking
  - generate_sql
  - output
```

### 含执行与反思
```yaml title="Workflow with Execution and Reflection"
workflow:
  plan: with_execution

with_execution:
  - schema_linking
  - generate_sql
  - execute_sql
  - reflect
  - output
```

## 高级特性

!!! warning "注意"
    高级特性依赖节点间契合度，建议在开发环境充分测试。

### 并行执行
```yaml title="Parallel Execution Example"
workflow:
  plan: parallel_generation

parallel_generation:
  - schema_linking
  - parallel:
      - generate_sql
      - reasoning
  - selection
  - execute_sql
  - output
```

### 子工作流
```yaml title="Sub-workflows Example"
workflow:
  plan: multi_approach

multi_approach:
  - schema_linking
  - parallel:
      - subworkflow1
      - subworkflow2
      - subworkflow3
  - selection
  - execute_sql
  - output

subworkflow1:
  - search_metrics
  - generate_sql

subworkflow2:
  - search_metrics
  - reasoning

subworkflow3:
  - reasoning
  - reflect
```

### 子工作流独立配置
```yaml title="Sub-workflows with Custom Configuration"
workflow:
  plan: multi_agent

multi_agent:
  - schema_linking
  - parallel:
      - subworkflow1
      - subworkflow2
      - subworkflow3
  - selection
  - execute_sql
  - output

subworkflow1:
  steps:
    - search_metrics
    - generate_sql
  config: multi/agent1.yaml

subworkflow2:
  steps:
    - search_metrics
    - reasoning
  config: multi/agent2.yaml

subworkflow3:
  steps:
    - reasoning
    - reflect
  config: multi/agent3.yaml
```

## 内置计划

=== "reflection"
```yaml
reflection:
  - schema_linking
  - generate_sql
  - execute_sql
  - reflect
  - output
```

=== "fixed"
```yaml
fixed:
  - schema_linking
  - generate_sql
  - execute_sql
  - output
```

=== "metric_to_sql"
```yaml
metric_to_sql:
  - schema_linking
  - search_metrics
  - date_parser
  - generate_sql
  - execute_sql
  - reflect
  - output
```

## 反思触发的备用计划
```yaml title="Reflection Nodes Configuration"
reflection_nodes:
  schema_linking:
    - schema_linking
    - generate_sql
    - execute_sql
    - reflect

  doc_search:
    - doc_search
    - generate_sql
    - execute_sql
    - reflect

  simple_regenerate:
    - execute_sql
    - reflect

  reasoning:
    - reasoning
    - execute_sql
    - reflect
```
