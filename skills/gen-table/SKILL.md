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

### Both Modes — DDL Confirmation via ask_user

**Include the full DDL SQL inside the `ask_user` question text.** This is required because when running as a sub-agent, all intermediate assistant messages are collapsed in the UI — the user can ONLY see the `ask_user` interaction widget.

Call `ask_user` with the complete DDL embedded in the question:

```
ask_user(questions=[{
  "question": "Generated DDL:\n\nCREATE TABLE {schema}.{table_name} AS (\n  SELECT ...\n);\n\nConfirm execution?",
  "options": ["Execute", "Modify", "Cancel"]
}])
```

**Formatting rules for the question text:**
- Start with a label: "Generated DDL:" or "DDL to execute:"
- Include the COMPLETE DDL statement — do NOT abbreviate or truncate
- Use `\n` for line breaks to keep the SQL readable
- End with a short confirmation prompt: "Confirm execution?"

**Based on user response:**
- If **Execute**: proceed to Phase 3
- If **Modify**: ask what to change, regenerate the DDL, and call `ask_user` again with the updated DDL
- If **Cancel**: stop and do not execute any DDL

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
