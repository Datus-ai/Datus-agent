---
title: 'Nodes'
description: 'Configure workflow nodes for schema linking, SQL generation, reasoning, and other processing tasks'
---

## Overview

Nodes are the building blocks of Datus Agent workflows. Each node performs a specific task in the data processing pipeline, from schema linking and SQL generation to reasoning and output formatting. This guide covers how to configure each node type for optimal performance.

## Configuration Structure

Nodes are configured within the `nodes` section of your configuration file:

```yaml
nodes:
  node_name:
    model: provider_name
    prompt_version: "1.0"
    # Additional node-specific parameters
```

!!! tip
    The `model` parameter in node configurations references the provider names defined in [`agent.models`](agent.md#models-configuration).

## Core Nodes

### Schema Linking

The schema linking node uses vector search to match table metadata, sample data, and extended knowledge definitions related to user questions.

```yaml
schema_linking:
  model: openai                    # LLM model for schema selection
  matching_rate: fast              # fast/medium/slow/from_llm
  prompt_version: "1.0"            # Prompt version to use
```

**Configuration Parameters:**

- **model**: LLM model key from agent.models configuration
- **matching_rate**: Controls how many matching results to return
    - `fast`: Top 5 matching data (fastest, least comprehensive)
    - `medium`: Top 10 matching data (balanced)
    - `slow`: Top 20 matching data (most comprehensive)
    - `from_llm`: Use LLM to select the most relevant tables from all available metadata
- **prompt_version**: Version of the prompt template to use

### Generate SQL

Generates SQL statements based on user questions and matching table information.

```yaml
generate_sql:
  model: deepseek_v3                    # LLM for SQL generation
  prompt_version: "1.0"                 # Prompt template version
  max_table_schemas_length: 4000        # Max length for table metadata
  max_data_details_length: 2000         # Max length for sample data
  max_context_length: 8000              # Max context length
  max_value_length: 500                 # Max length per sample value
```

**Configuration Parameters:**

- **model**: LLM model for SQL generation
- **prompt_version**: Prompt template version (latest used by default)
- **max_table_schemas_length**: Maximum character length for table metadata provided to LLM
- **max_data_details_length**: Maximum character length for table sample data
- **max_context_length**: Maximum character length for context information
- **max_value_length**: Maximum character length for individual sample values

### Reasoning

Iteratively generates, executes, and optimizes SQL queries based on database feedback.

```yaml
reasoning:
  model: anthropic                      # LLM for reasoning
  prompt_version: "1.0"                 # Prompt template version
  max_table_schemas_length: 4000        # Max length for table metadata
  max_data_details_length: 2000         # Max length for sample data
  max_context_length: 8000              # Max context length
  max_value_length: 500                 # Max length per sample value
```

**Configuration Parameters:**
- Same as `generate_sql` node - focuses on iterative improvement of SQL queries

### Search Metrics

Matches relevant metrics through vector search based on user questions.

```yaml
search_metrics:
  model: openai                    # LLM model for metric selection
  matching_rate: medium            # fast/medium/slow/from_llm
  prompt_version: "1.0"            # Prompt version to use
```

**Configuration Parameters:**
- Same as `schema_linking` node - specialized for metric discovery

## Processing Nodes

### Reflect

Evaluates SQL execution results and provides improvement suggestions.

```yaml
reflect:
  prompt_version: "1.0"            # Prompt template version
```

**Configuration Parameters:**
- **prompt_version**: Version of reflection prompt template to use

### Output

Formats and outputs SQL results to files and provides final responses.

```yaml
output:
  model: anthropic                 # LLM for output formatting
  prompt_version: "1.0"            # Prompt template version
  check_result: true               # Enable result validation
```

**Configuration Parameters:**

- **model**: LLM model for result formatting and validation
- **prompt_version**: Output formatting prompt version
- **check_result**: When true, LLM validates generated SQL and results for completeness and accuracy

## Interactive Nodes

### Chat

Enables multi-turn conversations with access to databases, files, metrics, and knowledge bases.

```yaml
chat:
  workspace_root: sql2             # Root directory for file operations
  model: anthropic                 # LLM for conversation
  max_turns: 25                    # Maximum conversation turns
```

**Configuration Parameters:**

- **workspace_root**: Root directory where file tools can operate
- **model**: LLM model for multi-turn dialogue
- **max_turns**: Maximum number of tool-assisted reasoning turns

## Utility Nodes

### Date Parser

Parses and interprets date-related queries in user questions.

```yaml
date_parser:
  # Typically uses default configuration
  prompt_version: "1.0"
```

### Compare

Compares generated SQL with reference SQL for benchmarking purposes.

```yaml
compare:
  # Used primarily in benchmark scenarios
  prompt_version: "1.0"
```

### Fix

Analyzes and fixes SQL queries using dialect-specific rules.

```yaml
fix:
  model: openai                    # LLM for SQL fixing
  prompt_version: "1.0"            # Prompt version
```

## Complete Node Configuration Example

```yaml
nodes:
  # Schema discovery and linking
  schema_linking:
    model: openai
    matching_rate: fast
    prompt_version: "1.0"
    
  # Metric discovery
  search_metrics:
    model: openai
    matching_rate: medium
    prompt_version: "1.0"
    
  # SQL generation
  generate_sql:
    model: deepseek_v3
    prompt_version: "1.0"
    max_table_schemas_length: 4000
    max_data_details_length: 2000
    max_context_length: 8000
    max_value_length: 500
    
  # Advanced reasoning
  reasoning:
    model: anthropic
    prompt_version: "1.0"
    max_table_schemas_length: 4000
    max_data_details_length: 2000
    max_context_length: 8000
    max_value_length: 500
    
  # Result reflection and improvement
  reflect:
    prompt_version: "1.0"
    
  # Output formatting and validation
  output:
    model: anthropic
    prompt_version: "1.0"
    check_result: true
    
  # Interactive chat
  chat:
    workspace_root: workspace
    model: anthropic
    max_turns: 25
    
  # Date parsing
  date_parser:
    prompt_version: "1.0"
    
  # SQL fixing
  fix:
    model: openai
    prompt_version: "1.0"
```

## Model Assignment Strategy

**For Schema Linking:**

- Use fast, cost-effective models: `gpt-3.5-turbo`, `deepseek-chat`
- For complex schemas: `gpt-4`, `claude-4-sonnet`

**For SQL Generation:**

- Recommended: `deepseek-chat`, `gpt-4-turbo`, `claude-4-sonnet`
- Avoid: Basic models that struggle with complex SQL

**For Reasoning:**

- Best: `claude-4-sonnet`, `gpt-4-turbo`, `claude-4-opus`
- Good: `gemini-2.5-flash`

**For Output and Chat:**

- Recommended: `claude-4-sonnet`, `gpt-4-turbo`
- Good for formatting: `anthropic` models
