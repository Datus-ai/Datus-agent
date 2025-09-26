# Workflow Nodes

Nodes are the fundamental building blocks of Datus Agent's workflow system. Each node performs a specific task in the process of understanding user requests, generating SQL queries, executing them, and providing results. This document explains the different types of nodes, their purposes, and how they work together in workflows.

## Node Categories

### 1. Control Nodes

Control nodes manage workflow execution flow and decision-making.

#### Reflect Node
- **Purpose**: Evaluate results and decide next steps
- **Key Feature**: Core intelligence that enables adaptive SQL generation
- **Common Strategies**:
  - Simple regeneration (retry SQL generation)
  - Document search (find relevant documentation)
  - Schema re-analysis (re-examine database structure)
  - Deep reasoning analysis

#### Parallel Node
- **Purpose**: Execute multiple child nodes simultaneously
- **Use Case**: Testing multiple SQL generation strategies for comparison

#### Selection Node
- **Purpose**: Choose the best result from multiple candidates
- **Use Case**: Selecting the best SQL query from multiple generated options

#### Subworkflow Node
- **Purpose**: Execute nested workflows
- **Use Case**: Reusing complex workflow patterns and modular composition

### 2. Action Nodes

Action nodes perform specific data processing and SQL-related tasks.

#### Schema Linking Node
- **Purpose**: Understand user queries and find relevant database schemas
- **Key Activities**:
  - Parse user intent from natural language
  - Search knowledge base for relevant tables
  - Extract table schemas and sample data
  - Update workflow context with schema information
- **Output**: List of relevant table schemas with sample data

#### Generate SQL Node
- **Purpose**: Generate SQL queries based on user requirements
- **Key Features**:
  - Uses LLM to understand business requirements
  - Leverages historical SQL patterns
  - Incorporates business metrics when available
  - Handles complex query logic
- **Output**: Generated SQL query with execution plan

#### Execute SQL Node
- **Purpose**: Execute SQL queries against databases
- **Key Activities**:
  - Connect to target database
  - Execute SQL safely with error handling
  - Return query results or error messages
  - Update execution context
- **Output**: Query results, execution time, error information

#### Output Node
- **Purpose**: Present final results to users
- **Features**:
  - Result formatting and presentation
  - Error message clarity
  - Performance metrics display
- **Output**: User-friendly result presentation

#### Reasoning Node
- **Purpose**: Provide deep analysis and reasoning
- **Use Case**: Complex business logic explanation and validation

#### Fix Node
- **Purpose**: Repair problematic SQL queries
- **Key Features**:
  - Error pattern recognition
  - Automated SQL correction
  - Validation of fixed queries
- **Use Case**: Automatically correcting failed SQL executions

#### Generate Metrics Node
- **Purpose**: Create business metrics from SQL queries
- **Key Activities**:
  - Analyze SQL query patterns
  - Identify business metrics
  - Generate metric definitions
  - Store metrics in knowledge base
- **Output**: Business metric definitions and calculations

#### Generate Semantic Model Node
- **Purpose**: Create semantic models for database tables
- **Key Features**:
  - Identifies business dimensions and measures
  - Defines table semantics
  - Creates reusable data models
- **Output**: Semantic model definitions for business intelligence

#### Search Metrics Node
- **Purpose**: Find relevant business metrics
- **Use Case**: Reusing existing business calculations and ensuring consistency

#### Compare Node
- **Purpose**: Compare SQL results with expected outcomes
- **Use Case**: Testing, validation, and quality assurance scenarios

#### Date Parser Node
- **Purpose**: Parse temporal expressions in user queries
- **Examples**:
  - "last month" → specific date range
  - "Q3 2023" → quarter date boundaries
  - "past 7 days" → rolling date window

#### Document Search Node
- **Purpose**: Find relevant documentation and context
- **Use Case**: Providing additional context for complex queries and domain knowledge

### 3. Agentic Nodes

Advanced AI-powered nodes with conversational and adaptive capabilities.

#### Chat Agentic Node
- **Purpose**: Conversational AI interactions with tool calling
- **Key Features**:
  - Multi-turn conversations
  - Tool calling capabilities
  - Context maintenance
  - Adaptive responses
- **Use Case**: Interactive SQL generation and refinement

## Node Implementation Details

### Input/Output Structure

Each node follows a consistent interface pattern:

```python
class BaseNode:
    def setup_input(self, context: Context) -> NodeInput
    def run(self, input: NodeInput) -> NodeOutput
    def update_context(self, context: Context, output: NodeOutput) -> Context
```

### Context Management

Nodes share information through a unified context object:

```python
class Context:
    sql_contexts: List[SQLContext]      # Generated SQL and results
    table_schemas: List[TableSchema]    # Database schema information
    metrics: List[BusinessMetric]       # Available business metrics
    reflections: List[Reflection]       # Reflection results
    documents: List[Document]           # Retrieved documentation
```

### Error Handling

Nodes implement comprehensive error handling:

- **Input Validation**: Check required parameters and context
- **Execution Safety**: Handle database errors and timeouts
- **Output Validation**: Ensure output format compliance
- **Recovery Mechanisms**: Automatic retry and fallback strategies

## Node Configuration

### Model Assignment

Different nodes can use different LLM models:

```yaml
nodes:
  schema_linking:
    model: "claude-3-sonnet"
    temperature: 0.1

  generate_sql:
    model: "gpt-4"
    temperature: 0.2

  reasoning:
    model: "claude-3-opus"
    temperature: 0.3
```

### Prompt Templates

Nodes use configurable prompt templates:

```yaml
nodes:
  generate_sql:
    prompt_template: "generate_sql_system.j2"
    user_template: "generate_sql_user.j2"
```

### Resource Limits

Configure execution constraints:

```yaml
nodes:
  execute_sql:
    timeout: 30
    max_rows: 10000
    memory_limit: "1GB"
```

## Best Practices

### Node Selection

1. **Use Schema Linking First**: Always start workflows with schema linking for context
2. **Combine Complementary Nodes**: Use reasoning and generate_sql together for complex queries
3. **Add Reflection for Robustness**: Include reflection nodes for adaptive behavior
4. **Use Parallel for Experimentation**: Run multiple strategies in parallel for comparison

### Performance Optimization

- **Cache Schema Information**: Reuse schema linking results across workflows
- **Optimize SQL Generation**: Use appropriate model sizes for different complexity levels
- **Limit Result Sets**: Configure reasonable limits for SQL execution
- **Monitor Resource Usage**: Track memory and CPU usage for long-running workflows

### Error Recovery

- **Graceful Degradation**: Provide useful partial results when possible
- **Automatic Retry**: Implement retry logic for transient failures
- **User Feedback**: Surface actionable error messages to users
- **Logging**: Maintain detailed logs for debugging and improvement

## Advanced Usage

### Custom Nodes

Create custom nodes for specific business logic:

```python
class CustomValidationNode(BaseNode):
    def run(self, input: ValidationInput) -> ValidationOutput:
        # Custom validation logic
        return ValidationOutput(is_valid=True, message="Validation passed")
```

### Dynamic Workflows

Nodes can modify workflow execution dynamically:

```python
# In reflection node
if complexity_score > threshold:
    workflow.add_node("reasoning", after="current")

if needs_validation:
    workflow.add_node("compare", before="output")
```

### Node Composition

Combine multiple nodes for complex operations:

```python
# Parallel SQL generation strategies
parallel_node = ParallelNode([
    GenerateSQLNode(strategy="conservative"),
    GenerateSQLNode(strategy="aggressive"),
    GenerateSQLNode(strategy="metric_based")
])

# Select best result
selection_node = SelectionNode(criteria="accuracy")
```

## Conclusion

Nodes are the powerful, modular components that make Datus Agent's workflow system efficient and intelligent. By understanding each node's purpose and how they work together, users can create sophisticated SQL generation workflows that adapt to complex requirements and deliver accurate results.

The modular design allows for flexible composition, enabling both simple linear workflows and complex adaptive systems that can handle the full spectrum of data analysis challenges.