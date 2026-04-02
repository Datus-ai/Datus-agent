---
name: gen-table
description: Create database tables from SQL (CTAS) or natural language descriptions
tags: "wide-table, CTAS, DDL, create-table, query-acceleration"
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

## Phase 1: Understand Intent (MANDATORY ask_user)

Detect input mode:
- **SQL mode**: User provides a JOIN SQL or other SELECT statement → CTAS path
- **Description mode**: User describes table structure in natural language (columns, types, purpose) → CREATE TABLE path

### SQL Mode (CTAS)

1. **Parse the input SQL**: Identify all source tables, JOIN conditions, selected columns, and any transformations.
2. **Call `describe_table`** for each source table to understand column types and relationships.
3. **Optionally call `read_query`** with `LIMIT 10` to sample the result and validate column output.
4. **Propose wide table schema**:
   - Table name: `{prefix}_{descriptive_name}` (e.g., `wide_order_customer`)
   - List all output columns with types
   - Identify primary time dimension (for future metric definition)

### Description Mode (CREATE TABLE)

1. **Parse user description**: Extract table name, columns, types, constraints from natural language.
2. **Call `describe_table`** for any referenced existing tables to infer column types.
3. **Propose table schema**:
   - Table name and target schema
   - Column definitions with types and constraints (NOT NULL, DEFAULT, etc.)
   - Primary key if applicable

### Both Modes

5. **MUST call `ask_user`** to confirm:
   - Proposed table name
   - Column list and types
   - Target schema/database (if ambiguous)

## Phase 2: Execute DDL

### SQL Mode
1. **Generate CTAS SQL**: `CREATE TABLE {schema}.{table_name} AS ({select_sql})`
2. **Call `execute_ddl(sql)`** to create the table.
3. **Verify**: Call `read_query("SELECT COUNT(*) FROM {schema}.{table_name}")` to confirm row count.

### Description Mode
1. **Generate CREATE TABLE SQL**: `CREATE TABLE {schema}.{table_name} ({column_defs})`
2. **Call `execute_ddl(sql)`** to create the table.
3. **Verify**: Call `describe_table("{schema}.{table_name}")` to confirm schema matches.

### Both Modes
4. **Call `describe_table("{schema}.{table_name}")`** to confirm the created schema.

If DDL fails:
- Parse the error message
- Fix the SQL and retry (up to 3 attempts)
- If still failing, report the error to the user via `ask_user`

## Phase 3: Summary

Output a summary including:
- Created table name and location
- Row count (for CTAS) or column count (for CREATE TABLE)
- Column list with types
- Original SQL (for CTAS) or user description (for CREATE TABLE)
- Hint: if the user needs a semantic model, suggest `task(type="gen_semantic_model", prompt="{table_name}")`

## Important Rules

- **MUST call `ask_user`** before executing any DDL — never create tables without user confirmation
- **DDL is irreversible** — always show the exact DDL SQL to the user before execution
- If the target table already exists, warn the user and ask whether to DROP and recreate or abort
- Language: match user's language (Chinese input → Chinese output)
- Do NOT modify the source tables — only create new tables
- **Single responsibility** — gen-table only creates tables, does not generate semantic model YAML. For semantic model, suggest using `gen_semantic_model`
