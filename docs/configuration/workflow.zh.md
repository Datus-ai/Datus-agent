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
    - gen_sql
    - output

```

## 基础配置

### 顺序执行
```yaml title="Simple Sequential Workflow"
workflow:
  plan: basic_sql

basic_sql:
  - schema_linking
  - gen_sql
  - output
```

### 含执行步骤
```yaml title="Workflow with Execution"
workflow:
  plan: with_execution

with_execution:
  - schema_linking
  - gen_sql
  - execute_sql
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
      - gen_sql
      - gen_sql
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
  - gen_sql

subworkflow2:
  - gen_sql

subworkflow3:
  - gen_sql
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
    - gen_sql
  config: multi/agent1.yaml

subworkflow2:
  steps:
    - gen_sql
  config: multi/agent2.yaml

subworkflow3:
  steps:
    - gen_sql
  config: multi/agent3.yaml
```

## 内置计划

=== "fixed"
```yaml
fixed:
  - schema_linking
  - gen_sql
  - execute_sql
  - output
```

=== "chat_agentic"
```yaml
chat_agentic:
  - chat
  - execute_sql
  - output
```

=== "gen_sql_agentic"
```yaml
gen_sql_agentic:
  - gen_sql
  - execute_sql
  - output
```

=== "empty"
```yaml
empty: []
```
