# Builtin Subagent

## Overview

The **Builtin Subagent** are specialized AI assistants integrated within the Datus Agent system. Each subagent focuses on a specific aspect of data engineering automation — analyzing SQL, generating semantic models, and converting queries into reusable metrics — together forming a closed-loop workflow from raw SQL to knowledge-aware data products.

This document covers four core subagents:

1. **[gen_sql_summary](#gen_sql_summary)** — Summarizes and classifies SQL queries
2. **[gen_semantic_model](#gen_semantic_model)** — Generates MetricFlow semantic models
3. **[gen_metrics](#gen_metrics)** — Generates MetricFlow metric definitions
4. **[gen_ext_knowledge](#gen_ext_knowledge)** — Generates business concept definitions

## Configuration

Builtin subagents work out of the box with minimal configuration. Most settings (tools, hooks, MCP servers, system prompts) are built-in. You can optionally customize them in your `agent.yml` file:

```yaml
agent:
  agentic_nodes:
    gen_semantic_model:
      model: claude     # Optional: defaults to configured model
      max_turns: 30     # Optional: defaults to 30

    gen_metrics:
      model: claude     # Optional: defaults to configured model
      max_turns: 30     # Optional: defaults to 30

    gen_sql_summary:
      model: deepseek   # Optional: defaults to configured model
      max_turns: 30     # Optional: defaults to 30

    gen_ext_knowledge:
      model: claude     # Optional: defaults to configured model
      max_turns: 30     # Optional: defaults to 30
```

**Optional configuration parameters:**

- `model`: The AI model to use (e.g., `claude`, `deepseek`). Defaults to your configured model.
- `max_turns`: Maximum conversation turns (default: 30)

**Built-in configurations** (no setup needed):
- **Tools**: Automatically configured based on subagent type
- **Hooks**: User confirmation workflow in interactive mode
- **MCP Servers**: MetricFlow validation (for gen_semantic_model and gen_metrics)
- **System Prompts**: Built-in templates version 1.0
- **Workspace**: `~/.datus/data/{namespace}/` with subagent-specific subdirectories

---

## gen_sql_summary

### Overview

The SQL Summary feature helps you analyze, classify, and catalog SQL queries for knowledge reuse. It automatically generates structured YAML summaries that are stored in a searchable Knowledge Base, making it easy to find and reuse similar queries in the future.

### What is a SQL Summary?

A **SQL summary** is a structured YAML document that captures:

- **Query Text**: The complete SQL query
- **Business Context**: Domain, categories, and tags
- **Semantic Summary**: Detailed explanation for vector search
- **Metadata**: Name, comment, file path

### Quick Start

Launch the SQL summary generation subagent:

```bash
/gen_sql_summary Analyze this SQL: SELECT SUM(revenue) FROM sales GROUP BY region. (You can also add some description on this SQL)
```

### Generation Workflow

```mermaid
graph LR
    A[User provides SQL + description] --> B[Agent analyzes query]
    B --> C[Retrieves context]
    C --> D[Generates unique ID]
    D --> E[Creates YAML]
    E --> F[Saves file]
    F --> G[User confirms]
    G --> H[Syncs to Knowledge Base]
```

**Detailed Steps:**

1. **Understand SQL**: The AI analyzes your query structure and business logic
2. **Get Context**: Automatically retrieves from Knowledge Base:
   - Existing subject trees (domain/layer1/layer2 combinations)
   - Similar SQL summaries (top 5 most similar queries) for classification reference
3. **Generate Unique ID**: Uses `generate_sql_summary_id()` tool based on SQL + comment
4. **Create Unique Name**: Generates a descriptive name (max 20 characters)
5. **Classify Query**: Assigns domain, layer1, layer2, and tags following existing patterns
6. **Generate YAML**: Creates structured summary document
7. **Save File**: Writes YAML to workspace using `write_file()` tool
8. **User Confirmation**: Shows the generated YAML and prompts for approval
9. **Sync to Knowledge Base**: Stores in LanceDB for semantic search

### Interactive Confirmation

After generation, you'll see:

```
==========================================================
Generated Reference SQL YAML
File: /path/to/sql_summary.yml
==========================================================
[YAML content with syntax highlighting]

  SYNC TO KNOWLEDGE BASE?

  1. Yes - Save to Knowledge Base
  2. No - Keep file only

Please enter your choice: [1/2]
```

### Subject Tree Categorization

Subject tree allows organizing SQL summaries by domain and layers. In CLI mode, include it in your question:

**Example with subject_tree:**
```bash
/gen_sql_summary Analyze this SQL: SELECT SUM(revenue) FROM sales, subject_tree: sales/reporting/revenue_analysis
```

**Example without subject_tree:**
```bash
/gen_sql_summary Analyze this SQL: SELECT SUM(revenue) FROM sales
```

When not provided, the agent suggests categories based on existing subject trees and similar queries in the Knowledge Base.

### YAML Structure

The generated SQL summary follows this structure:

```yaml
id: "abc123def456..."                      # Auto-generated MD5 hash
name: "Revenue by Region"                  # Descriptive name (max 20 chars)
sql: |                                     # Complete SQL query
  SELECT
    region,
    SUM(revenue) as total_revenue
  FROM sales
  GROUP BY region
comment: "Calculate total revenue grouped by region"
summary: "This query aggregates total revenue from the sales table, grouping results by geographic region. It uses SUM aggregation to calculate revenue totals for each region."
filepath: "/Users/you/.datus/data/reference_sql/revenue_by_region.yml"
domain: "Sales"                            # Business domain
layer1: "Reporting"                        # Primary category
layer2: "Revenue Analysis"                 # Secondary category
tags: "revenue, region, aggregation"       # Comma-separated tags
```

#### Field Descriptions

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `id` | Yes | Unique hash (auto-generated) | `abc123def456...` |
| `name` | Yes | Short descriptive name (max 20 chars) | `Revenue by Region` |
| `sql` | Yes | Complete SQL query | `SELECT ...` |
| `comment` | Yes | Brief one-line description | User's message or generated summary |
| `summary` | Yes | Detailed explanation (for search) | Comprehensive query description |
| `filepath` | Yes | Actual file path | `/path/to/file.yml` |
| `domain` | Yes | Business domain | `Sales`, `Marketing`, `Finance` |
| `layer1` | Yes | Primary category | `Reporting`, `Analytics`, `ETL` |
| `layer2` | Yes | Secondary category | `Revenue Analysis`, `Customer Insights` |
| `tags` | Optional | Comma-separated keywords | `revenue, region, aggregation` |

---

## gen_semantic_model

### Overview

The semantic model generation feature helps you create MetricFlow semantic models from database tables through an AI-powered assistant. The assistant analyzes your table structure and generates comprehensive YAML configuration files that define metrics, dimensions, and relationships.

### What is a Semantic Model?

A semantic model is a YAML configuration that defines:

- **Measures**: Metrics and aggregations (SUM, COUNT, AVERAGE, etc.)
- **Dimensions**: Categorical and time-based attributes
- **Identifiers**: Primary and foreign keys for relationships
- **Data Source**: Connection to your database table

### Quick Start

Start Datus CLI with `datus --namespace <namespace>`, and begin with a subagent command:

```bash
/gen_semantic_model generate a semantic model for table <table_name>
```

### How It Works

#### Interactive Generation

When you request a semantic model, the AI assistant:

1. Retrieves your table's DDL (structure)
2. Checks if a semantic model already exists
3. Generates a comprehensive YAML file
4. Validates the configuration using MetricFlow
5. Prompts you to save it to the Knowledge Base

#### Generation Workflow

```mermaid
graph LR
    A[User Request] --> B[DDL Analysis]
    B --> C[YAML Generation]
    C --> D[Validation]
    D --> E[User Confirmation]
    E --> F[Storage]
```

### Interactive Confirmation

After generating the semantic model, you'll see:

```text
=============================================================
Generated YAML: table_name.yml
Path: /path/to/file.yml
=============================================================
[YAML content with syntax highlighting]

SYNC TO KNOWLEDGE BASE?

1. Yes - Save to Knowledge Base
2. No - Keep file only

Please enter your choice: [1/2]
```

**Options:**

- **Option 1**: Saves the semantic model to your Knowledge Base (RAG storage) for AI-powered queries
- **Option 2**: Keeps the YAML file only without syncing to the Knowledge Base

### Semantic Model Structure

#### Basic Template

```yaml
data_source:
  name: table_name                    # Required: lowercase with underscores
  description: "Table description"

  sql_table: schema.table_name        # For databases with schemas
  # OR
  sql_query: |                        # For custom queries
    SELECT * FROM table_name

  measures:
    - name: total_amount              # Required
      agg: SUM                        # Required: SUM|COUNT|AVERAGE|etc.
      expr: amount_column             # Column or SQL expression
      create_metric: true             # Auto-create queryable metric
      description: "Total transaction amount"

  dimensions:
    - name: created_date
      type: TIME                      # Required: TIME|CATEGORICAL
      type_params:
        is_primary: true              # One primary time dimension required
        time_granularity: DAY         # Required for TIME: DAY|WEEK|MONTH|etc.

    - name: status
      type: CATEGORICAL
      description: "Order status"

  identifiers:
    - name: order_id
      type: PRIMARY                   # PRIMARY|FOREIGN|UNIQUE|NATURAL
      expr: order_id

    - name: customer
      type: FOREIGN
      expr: customer_id
```

### Summary

The semantic model generation feature provides:

- ✅ Automated YAML generation from table DDL
- ✅ Built-in tools, hooks, and MCP server integration
- ✅ Interactive validation and error fixing
- ✅ User confirmation before storage
- ✅ Knowledge Base integration
- ✅ Duplicate prevention
- ✅ MetricFlow compatibility

---

## gen_metrics

### Overview

The metrics generation feature helps you convert SQL queries into reusable MetricFlow metric definitions. Using an AI assistant, you can analyze SQL business logic and automatically generate standardized YAML metric configurations that can be queried consistently across your organization.

### What is a Metric?

A **metric** is a reusable business calculation built on top of semantic models. Metrics provide:

- **Consistent Business Logic**: One definition, used everywhere
- **Type Safety**: Validated structure and measure references
- **Metadata**: Display names, formats, business context
- **Composability**: Build complex metrics from simpler ones

**Example**: Instead of writing `SELECT SUM(revenue) / COUNT(DISTINCT customer_id)` repeatedly, define an `avg_customer_revenue` metric once.

### Quick Start

Start Datus CLI with `datus --namespace <namespace>`, and use the metrics generation subagent:

```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, the corresponding question is total amount of all transactions
```

### How It Works

#### Generation Workflow

```mermaid
graph LR
    A[User provides SQL and question] --> B[Agent analyzes logic]
    B --> C[Finds semantic model]
    C --> D[Reads measures]
    D --> E[Checks for duplicates]
    E --> F[Generates metric YAML]
    F --> G[Appends to file]
    G --> H[Validates]
    H --> I[User confirms]
    I --> J[Syncs to Knowledge Base]
```

#### Important Limitations

> **⚠️ Single Table Queries Only**
>
> The current version **only supports generating metrics from single-table SQL queries**. Multi-table JOINs are not supported.

**Supported:**
```sql
SELECT SUM(revenue) FROM transactions WHERE status = 'completed'
SELECT COUNT(DISTINCT customer_id) / COUNT(*) FROM orders
```

**Not Supported:**
```sql
SELECT SUM(o.amount)
FROM orders o
JOIN customers c ON o.customer_id = c.id  -- ❌ JOIN not supported
```

### Interactive Confirmation

After generation, you'll see:

```
==========================================================
Generated YAML: transactions.yml
Path: /Users/you/.datus/data/semantic_models/transactions.yml
==========================================================
[YAML content with syntax highlighting showing the new metric]

  SYNC TO KNOWLEDGE BASE?

  1. Yes - Save to Knowledge Base
  2. No - Keep file only

Please enter your choice: [1/2]
```

**Options:**
- **Option 1**: Syncs the metric to your Knowledge Base for AI-powered semantic search
- **Option 2**: Keeps the YAML file only without syncing to the Knowledge Base

### Subject Tree Categorization

Subject tree allows organizing metrics by domain and layers. In CLI mode, include it in your question:

**Example with subject_tree:**
```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, subject_tree: finance/revenue/transactions
```

**Example without subject_tree:**
```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions
```

When not provided, the agent operates in learning mode and suggests categories based on existing metrics in the Knowledge Base.

### Usage Examples

#### Example 1: Simple Aggregation

**User Input:**
```bash
/gen_metrics Generate a metric for total order count
```

**Agent Actions:**
1. Finds `orders.yml` semantic model
2. Reads file to discover `order_count` measure
3. Generates MEASURE_PROXY metric:

```yaml
---
metric:
  name: total_orders
  description: Total number of orders
  type: measure_proxy
  type_params:
    measure: order_count
  locked_metadata:
    display_name: "Total Orders"
    increase_is_good: true
```

#### Example 2: Conversion Rate

**User Input:**
```bash
/gen_metrics Create a metric from this SQL:
SELECT
  COUNT(DISTINCT CASE WHEN status = 'completed' THEN order_id END) /
  COUNT(DISTINCT order_id) AS completion_rate
FROM orders
```

**Agent Actions:**
1. Analyzes SQL logic (ratio with conditional counting)
2. Finds `orders.yml` and reads measures
3. Generates RATIO metric with constraint:

```yaml
---
metric:
  name: order_completion_rate
  description: Percentage of orders that reached completed status
  type: ratio
  type_params:
    numerator:
      name: order_count
      constraint: status = 'completed'
    denominator: order_count
  locked_metadata:
    display_name: "Order Completion Rate"
    value_format: ".2%"
    increase_is_good: true
```

#### Example 3: Complex Calculation

**User Input:**
```bash
/gen_metrics Generate average basket size metric:
SELECT SUM(total_amount) / COUNT(DISTINCT order_id)
FROM order_items
```

**Agent Actions:**
1. Locates `order_items.yml`
2. Identifies this as a RATIO (average)
3. Generates metric:

```yaml
---
metric:
  name: avg_basket_size
  description: Average order value (basket size)
  type: ratio
  type_params:
    numerator: total_amount
    denominator: order_count
  locked_metadata:
    display_name: "Average Basket Size"
    value_format: "$$,.2f"
    unit: "dollars"
    increase_is_good: true
```

### How Metrics Are Stored

#### File Organization

Metrics are appended to existing semantic model files using the YAML document separator `---`:

```yaml
# Existing semantic model
data_source:
  name: transactions
  sql_table: transactions
  measures:
    - name: revenue
      agg: SUM
      expr: amount
  dimensions:
    - name: transaction_date
      type: TIME

---
# First metric (appended)
metric:
  name: total_revenue
  type: measure_proxy
  type_params:
    measure: revenue

---
# Second metric (appended)
metric:
  name: avg_transaction_value
  type: ratio
  type_params:
    numerator: revenue
    denominator: transaction_count
```

**Why append instead of separate files?**
- Keeps related metrics close to their semantic model
- Easier maintenance and validation
- MetricFlow can validate all definitions together

#### Knowledge Base Storage

When you choose "1. Yes - Save to Knowledge Base", the metric is stored in a Vector Database with:

1. **Metadata**: Name, description, type, domain/layer classification
2. **LLM Text**: Natural language representation for semantic search
3. **References**: Associated semantic model name
4. **Timestamp**: Creation date

### Summary

The metrics generation feature provides:

- ✅ **SQL-to-Metric Conversion**: Analyze SQL queries and generate MetricFlow metrics
- ✅ **Intelligent Type Detection**: Automatically selects the right metric type
- ✅ **Duplicate Prevention**: Checks for existing metrics before generation
- ✅ **Subject Tree Support**: Organize by domain/layer1/layer2 with predefined or learned categories
- ✅ **Validation**: MetricFlow validation ensures correctness
- ✅ **Interactive Workflow**: Review and approve before syncing
- ✅ **Knowledge Base Integration**: Semantic search for metric discovery
- ✅ **File Management**: Appends to existing semantic model files safely

---

## gen_ext_knowledge

### Overview

The external knowledge generation feature helps you create and manage business concepts and domain-specific definitions. Using an AI assistant, you can document business knowledge in a structured format that becomes searchable in the Knowledge Base, enabling better context retrieval for SQL generation and data analysis tasks.

### What is External Knowledge?

**External knowledge** captures business-specific information that isn't directly stored in database schemas:

- **Business Rules**: Calculation logic and business constraints
- **Domain Concepts**: Industry or company-specific knowledge
- **Data Interpretations**: How to understand specific data fields or values

This knowledge helps the AI agent understand your business context when generating SQL queries or analyzing data.

### Quick Start

Launch the external knowledge generation subagent:


```bash
/gen_ext_knowledge Extract knowledge from this sql
-- Question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?
-- SQL: 
SELECT 
  `Free Meal Count (K-12)` / `Enrollment (K-12)` 
FROM 
  frpm 
WHERE 
  `County Name` = 'Alameda' 
ORDER BY 
  (
    CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`
  ) DESC 
LIMIT 1
```

### Generation Workflow

The workflow follows a **knowledge gap discovery** approach: the agent first attempts to solve the problem independently, then compares with the reference SQL to identify implicit business knowledge.

```mermaid
graph LR
    A[User provides Question + SQL] --> B[Agent attempts to solve]
    B --> C[Compare with reference SQL]
    C --> D[Identify knowledge gaps]
    D --> E[Check for duplicates]
    E --> F[Generate YAML]
    F --> G[Save file]
    G --> H[User confirms]
    H --> I[Sync to Knowledge Base]
```

**Detailed Steps:**

1. **Understand the Problem**: Read the question from SQL comments and understand the goal
2. **Attempt to Solve**: The agent uses available tools to try solving the problem
3. **Compare with Reference SQL**: Find gaps between attempt and reference sql
4. **Extract Knowledge from Gaps**: Discovering hidden business concepts in gaps
5. **Check for Duplicates**: Use `search_knowledge` to verify extracted knowledge doesn't already exist
6. **Generate YAML**: Create structured knowledge entries with unique IDs via `generate_ext_knowledge_id()`
7. **Save File**: Write YAML using `write_file(path, content, file_type="ext_knowledge")`
8. **User Confirmation**: Review generated YAML and approve
9. **Sync to Knowledge Base**: Store in vector database for semantic search

> **Important**: If no knowledge gaps are found (agent's attempt matches reference SQL), no knowledge file is generated.

### Interactive Confirmation

After generation, you'll see:

```
============================================================
Generated External Knowledge YAML
File: /Users/liuyufei/DatusProject/bird/datus/ext_knowledge/bird_sqlite_with_knowledge/sat_school_administration_knowledge.yaml
============================================================
                                                                                                                         

  SYNC TO KNOWLEDGE BASE?

  1. Yes - Save to Knowledge Base
  2. No - Keep file only

Please enter your choice: [1/2] 1
✓ Syncing to Knowledge Base...
✓ Successfully synced external knowledge to Knowledge Base
```

### Subject Path Categorization

Subject path allows organizing external knowledge hierarchically. In CLI mode, include it in your question:

**Example with subject_path:**
```bash
/gen_ext_knowledge Extract knowledge from this sql
Question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?
subject_tree: education/schools/data_integration
SQL: ***
```

**Example without subject_path:**
```bash
/gen_ext_knowledge Extract knowledge from this sql
Question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?
SQL: ***
```

When not provided, the agent operates in learning mode and suggests categories based on existing subject trees in the Knowledge Base.

### YAML Structure

The generated external knowledge follows this structure:

```yaml
id: education/schools/data_integration/CDS code school identifier California Department of Education join tables                                                                
name: CDS Code as School Identifier                                                                                                                                             
search_text: CDS code school identifier California Department of Education join tables                                                                                          
explanation: |                                                                                                                                                                  
  The CDS (California Department of Education) code serves as the primary identifier for linking educational datasets in California. Use `cds` field in SAT scores table to join
subject_path: education/schools/data_integration       
```

#### Field Descriptions

| Field | Required | Description                                                         | Example |
|-------|----------|---------------------------------------------------------------------|---------|
| `id` | Yes | Unique ID `generate_ext_knowledge_id()`)                            | `education/schools/data_integration/CDS code school identifier California Department of Education join tables` |
| `name` | Yes | Short identifier name (max 30 chars)                                | `Free Meal Rate`, `GMV` |
| `search_text` | Yes | Search keywords for retrieval (vector/inverted index)               | `eligible free rate K-12` |
| `explanation` | Yes | Concise explanation (2-4 sentences): what it is + when/how to apply | Business rule, calculation logic |
| `subject_path` | Yes | Hierarchical classification (slash-separated)                       | `Education/School Metrics/FRPM` |

---

## Summary

| Subagent | Purpose | Output | Stored In | Key Features                                        |
|----------|---------|--------|-----------|-----------------------------------------------------|
| `gen_sql_summary` | Summarize and classify SQL queries | YAML (SQL summary) | `/data/reference_sql` | Subject tree categorization, auto context retrieval |
| `gen_semantic_model` | Generate semantic model from tables | YAML (semantic model) | `/data/semantic_models` | DDL → MetricFlow model, built-in validation         |
| `gen_metrics` | Generate metrics from SQL | YAML (metric) | `/data/semantic_models` | SQL → MetricFlow metric, subject tree support       |
| `gen_ext_knowledge` | Generate business concepts | YAML (external knowledge) | `/data/ext_knowledge` | Question&SQL → knowledge, subject tree support      |

**Built-in Features Across All Subagents:**
- Minimal configuration required (only `model` and `max_turns` optional)
- Automatic tool setup, hooks, and MCP server integration
- Built-in system prompts (version 1.0)
- User confirmation workflow in interactive mode
- Knowledge Base integration for semantic search
- Automatic workspace management

Together, these subagents automate the **data engineering knowledge pipeline** — from **query understanding → model definition → metric generation → business knowledge capture → searchable Knowledge Base**.