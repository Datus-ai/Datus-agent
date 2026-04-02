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

## Phase 2: Confirm DDL (MANDATORY — two-step display)

Generate the exact DDL SQL statement. **Display it as a separate assistant message first**, then ask for confirmation in a follow-up `ask_user` call. This ensures the DDL is always visible to the user and not collapsed inside the question UI.

### SQL Mode
1. **Generate CTAS SQL**: `CREATE TABLE {schema}.{table_name} AS ({select_sql})`

### Description Mode
1. **Generate CREATE TABLE SQL**: `CREATE TABLE {schema}.{table_name} ({column_defs})`

### Both Modes — Two-Step Confirmation

**Step A — Display DDL (Turn 1, NO tool calls)**: Output the complete DDL statement as a normal assistant message. Do NOT call any tool (including `ask_user`) in this turn. Example:

> Here is the DDL statement I generated based on your request:
>
> ```sql
> CREATE TABLE {schema}.{table_name} AS (
>   SELECT ...
> );
> ```

**Step B — Ask for confirmation (Turn 2, call `ask_user`)**: In the NEXT turn, call `ask_user` with a short confirmation question:

```
ask_user(questions=[{
  "question": "Confirm execution of the DDL statement above?",
  "options": ["Execute", "Modify", "Cancel"]
}])
```

- If **Execute**: proceed to Phase 3
- If **Modify**: ask what to change, regenerate the DDL, and repeat Step A + B
- If **Cancel**: stop and do not execute any DDL

**CRITICAL**: Step A and Step B MUST be in separate turns. If DDL text and `ask_user` are in the same turn, the DDL will be hidden in the UI and the user cannot see it. This is a UI limitation — always split into two turns.

## Phase 3: Execute and Verify

1. **Call `execute_ddl(sql)`** with the confirmed DDL statement.
2. **Verify**:
   - SQL Mode: Call `read_query("SELECT COUNT(*) FROM {schema}.{table_name}")` to confirm row count
   - Description Mode: Call `describe_table("{schema}.{table_name}")` to confirm schema matches
3. **Call `describe_table("{schema}.{table_name}")`** to confirm the created schema.

If DDL fails:
- Parse the error message
- Fix the SQL, show the updated DDL to the user via `ask_user`, and retry (up to 3 attempts)
- If still failing, report the error to the user via `ask_user`

## Phase 4: Summary

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
