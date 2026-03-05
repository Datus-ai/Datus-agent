# Workflow

Configure workflow orchestration in Datus Agent to define execution plans, node sequences, parallel processing, and sub-workflow composition. Workflows control how nodes are executed to process user queries and generate SQL.

!!! tip "Quick Start"
    New to workflows? Start with the [Basic Workflow Configuration](#basic-workflow-configuration) section to understand the fundamentals.

## Workflow Structure

Workflows are defined in the configuration file under the `workflow` section:

```yaml title="agent.yml"
workflow:
  plan: planA                        # Active execution plan

# Define execution plans
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

## Basic Workflow Configuration

### Simple Sequential Workflow

```yaml title="Simple Sequential Workflow"
workflow:
  plan: basic_sql

basic_sql:
  - schema_linking                   # Find relevant tables
  - generate_sql                     # Create SQL query
  - output                          # Format results
```

### Workflow with Execution and Reflection

```yaml title="Workflow with Execution and Reflection"
workflow:
  plan: with_execution

with_execution:
  - schema_linking                   # Find relevant tables
  - generate_sql                     # Create SQL query
  - execute_sql                     # Run the query
  - reflect                         # Analyze results
  - output                          # Format final output
```

## Advanced Workflow Features

!!! warning "Advanced Features"
    The following features require careful planning and understanding of node dependencies. Test thoroughly in development environments.

### Parallel Execution

Execute multiple nodes simultaneously and then select the best result:

```yaml title="Parallel Execution Example"
workflow:
  plan: parallel_generation

parallel_generation:
  - schema_linking
  - parallel:                       # Execute in parallel
      - generate_sql
      - reasoning
  - selection                       # Choose best result
  - execute_sql
  - output
```

### Sub-workflows

Break complex workflows into reusable sub-components:

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

# Define sub-workflows
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

### Sub-workflows with Custom Configuration

Each sub-workflow can use its own configuration file:

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
  config: multi/agent1.yaml          # Custom config file

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

## Built-in Workflow Plans

Datus Agent provides three built-in workflow plans if you don't configure custom ones:

!!! info "Default Workflows"
    These workflows are automatically available and can be referenced without additional configuration.

=== "Reflection Workflow"

    ```yaml
    # Built-in: reflection
    reflection:
      - schema_linking
      - generate_sql
      - execute_sql
      - reflect
      - output
    ```

=== "Fixed Workflow"

    ```yaml
    # Built-in: fixed
    fixed:
      - schema_linking
      - generate_sql
      - execute_sql
      - output
    ```

=== "Metric-to-SQL Workflow"

    ```yaml
    # Built-in: metric_to_sql
    metric_to_sql:
      - schema_linking
      - search_metrics
      - date_parser
      - generate_sql
      - execute_sql
      - reflect
      - output
    ```

## Reflection Nodes Configuration

Configure alternative workflows that can be triggered during reflection:

```yaml title="Reflection Nodes Configuration"
reflection_nodes:
  # When schema linking needs improvement
  schema_linking:
    - schema_linking
    - generate_sql
    - execute_sql
    - reflect
    
  # When document search is needed
  doc_search:
    - doc_search
    - generate_sql
    - execute_sql
    - reflect
    
  # Simple regeneration
  simple_regenerate:
    - execute_sql
    - reflect
    
  # Enhanced reasoning
  reasoning:
    - reasoning
    - execute_sql
    - reflect
```
