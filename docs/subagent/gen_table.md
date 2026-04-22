# Table Generation (gen_table)

## Overview

The `gen_table` subagent is a built-in agent for creating database tables. It supports two input modes:

1. **SQL mode (CTAS)**: Provide a JOIN/SELECT SQL query to create a wide table for query acceleration
2. **Description mode**: Describe the table structure in natural language to generate CREATE TABLE DDL

The agent analyzes the input, generates DDL, asks for user confirmation, executes the DDL, and verifies the result.

## Quick Start

Start Datus CLI and use the gen_table subagent:

```bash
# SQL mode: create wide table from a JOIN query
/gen_table CREATE TABLE wide_orders AS SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id

# Description mode: describe what you want
/gen_table Create a user_profiles table with id (int), name (varchar), email (varchar), created_at (timestamp)
```

The chat agent can also automatically delegate to gen_table when it detects a table creation task.

## Built-in Tools

| Tool | Purpose |
|------|---------|
| `list_tables` | List tables in the database |
| `list_databases` | Discover available databases |
| `list_schemas` | List schemas |
| `describe_table` | Get column definitions and metadata |
| `read_query` | Execute read-only SQL (validation, row count) |
| `get_table_ddl` | Get existing CREATE TABLE statement |
| `execute_ddl` | Execute DDL (CREATE/ALTER/DROP TABLE/SCHEMA) |
| `read_file` / `write_file` | Read/write SQL artifact files |
| `ask_user` | Interactive confirmation (interactive mode only) |

Note: gen_table uses `DBFuncTool.create_dynamic()` for multi-connector support, allowing table creation across different databases.

## Workflow

### SQL Mode (CTAS)

```
1. Parse the input SQL, identify source tables and columns
2. Call describe_table for source tables to understand types
3. Optionally call read_query with LIMIT to validate output
4. Generate CTAS DDL: CREATE TABLE schema.table AS (SELECT ...)
5. Present full DDL to user via ask_user → [Execute / Modify / Cancel]
6. Execute DDL and verify (row count + schema check)
7. Output summary
```

### Description Mode (CREATE TABLE)

```
1. Parse natural language description for table name, columns, types
2. Call describe_table for referenced existing tables
3. If critical info is missing, ask_user for clarification
4. Generate CREATE TABLE DDL with column definitions
5. Present full DDL to user via ask_user → [Execute / Modify / Cancel]
6. Execute DDL and verify (schema check)
7. Output summary
```

## User Confirmation

gen_table **always** asks for user confirmation before executing DDL. The complete DDL SQL is displayed in the `ask_user` widget:

- **Execute**: Proceed with DDL execution
- **Modify**: Adjust the DDL and re-confirm
- **Cancel**: Stop immediately, no table is created

If a DDL execution fails, the agent parses the error, fixes the SQL, and re-confirms with the user (up to 3 retries).

## Optional Configuration

gen_table works out of the box. You can optionally customize it in `agent.yml`:

```yaml
agent:
  agentic_nodes:
    gen_table:
      max_turns: 20       # Default: 20
```

## Skills Used

- **gen-table** — Table creation workflow: input analysis, DDL generation, user confirmation, execution, verification

## Important Rules

- DDL execution requires user confirmation — tables are never created silently
- If the target table already exists, the agent warns and asks whether to DROP + recreate or abort
- gen_table only creates tables — for semantic model generation on the new table, use `gen_semantic_model` separately
- Source tables are never modified

## Output Format

```json
{
  "table_name": "created_table_name",
  "output": "markdown summary with row count, column list, and original SQL/description"
}
```

## Examples

### Example 1: Wide Table from JOIN

```
User: Create a wide_order_customer table from:
      SELECT o.order_id, o.amount, c.name, c.region
      FROM orders o JOIN customers c ON o.customer_id = c.id

gen_table:
  1. Parses SQL, identifies orders and customers tables
  2. Calls describe_table for both source tables
  3. Generates DDL: CREATE TABLE wide_order_customer AS (SELECT ...)
  4. Shows DDL in ask_user → user selects "Execute"
  5. Executes DDL, verifies 1000 rows created
  6. Returns summary with table name and column list
```

### Example 2: New Table from Description

```
User: Create a user_events table with:
      - event_id (bigint, primary key)
      - user_id (int, not null)
      - event_type (varchar 50)
      - payload (text, nullable)
      - created_at (timestamp, default now)

gen_table:
  1. Parses description into column definitions
  2. Generates CREATE TABLE DDL with types and constraints
  3. Shows DDL in ask_user → user selects "Execute"
  4. Executes DDL, verifies schema matches
  5. Returns summary with column list
```

## Comparison with gen_job

| Feature | gen_table | gen_job |
|---------|-----------|---------|
| Table creation (CTAS) | Yes | Yes |
| Table creation (DDL) | Yes | Yes |
| Data writes (INSERT/UPDATE/DELETE) | No | Yes |
| Cross-database migration | No | Yes |
| User confirmation (ask_user) | Always required | Available |
| Default max_turns | 20 | 40 |

Use **gen_table** when you need interactive table creation with user confirmation. Use **gen_job** for single-database ETL pipelines or cross-database migration (gen_job now covers both).
