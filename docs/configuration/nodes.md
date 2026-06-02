---
title: 'Nodes'
description: 'Configure workflow nodes for schema linking, SQL generation, reasoning, and other processing tasks'
---

## Overview

Nodes are the building blocks of Datus Agent workflows. Each node performs a specific task in the data processing pipeline, from schema linking and SQL generation to reasoning and output formatting. This guide covers how to configure each node type for optimal performance.

## Configuration Structure

Fixed workflow nodes are configured in `nodes`; agentic workflow nodes such as `gen_sql` are configured in `agentic_nodes`:

```yaml
nodes:
  node_name:
    model: provider_name
    prompt_version: "1.0"
    # Additional node-specific parameters
agentic_nodes:
  gen_sql:
    model: provider_name
    system_prompt: gen_sql
    max_turns: 30
```

!!! tip
    The `model` parameter in node configurations references provider keys defined in [`models`](agent.md#models-configuration).

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

- **model**: LLM model key from top-level `models` configuration
- **matching_rate**: Controls how many matching results to return
    - `fast`: Top 5 matching data (fastest, least comprehensive)
    - `medium`: Top 10 matching data (balanced)
    - `slow`: Top 20 matching data (most comprehensive)
    - `from_llm`: Use LLM to select the most relevant tables from all available metadata
- **prompt_version**: Version of the prompt template to use

### GenSQL

Uses the agentic SQL generator to produce and validate SQL with database, semantic, and context-search tools.

```yaml
agentic_nodes:
  gen_sql:
    model: deepseek_v3                    # LLM for SQL generation
    system_prompt: gen_sql                # Uses the canonical gen_sql_system prompt
    prompt_version: "1.2"                 # Prompt template version
    tools: db_tools.*, context_search_tools.*
    max_turns: 30
```

**Configuration Parameters:**

- **model**: LLM model for SQL generation
- **system_prompt**: Prompt family for SQL generation; use `gen_sql`, which maps to the canonical `gen_sql_system` template
- **prompt_version**: Prompt template version (latest used by default)
- **tools**: Tool patterns available to the SQL generator
- **max_turns**: Maximum number of tool-assisted reasoning turns

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
- Reasoning keeps its own fixed-node limits and may fall back to `gen_sql` when regeneration is needed

### Search Metrics

Matches relevant metrics through vector search based on user questions.

```yaml
search_metrics:
  model: openai                    # LLM model for metric selection
  matching_rate: medium            # fast/medium/slow
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
agentic_nodes:
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

  # Date parsing
  date_parser:
    prompt_version: "1.0"

  # SQL fixing
  fix:
    model: openai
    prompt_version: "1.0"

agentic_nodes:
  # SQL generation
  gen_sql:
    model: deepseek_v3
    system_prompt: gen_sql
    prompt_version: "1.2"
    tools: db_tools.*, context_search_tools.*
    max_turns: 30

  # Interactive chat
  chat:
    workspace_root: workspace
    model: anthropic
    max_turns: 25
```

## Model Assignment Strategy

The model names below refer to the `model` field inside each `models.<key>` provider entry (not the `nodes.*.model` key, which should use the provider key name like `openai` or `anthropic`).

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
