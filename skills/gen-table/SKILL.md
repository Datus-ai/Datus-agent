---
name: gen-table
description: Create database tables from SQL (CTAS) or natural language descriptions
tags:
  - wide-table
  - CTAS
  - DDL
  - create-table
  - query-acceleration
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

## CRITICAL: Cancel = Immediate Stop

**If the user selects "Cancel" at ANY point (any `ask_user` response), you MUST immediately stop ALL work.** Do NOT:
- Ask follow-up questions
- Regenerate DDL
- Continue to any subsequent phase
- Propose alternatives

Return immediately with:
```json
{"table_name": "", "output": "Cancelled by user."}
```

## Phase 1: Analyze Input

Detect input mode:
- **SQL mode**: User provides a JOIN SQL or other SELECT statement → CTAS path
- **Description mode**: User describes table structure in natural language → CREATE TABLE path

### SQL Mode (CTAS) — Go Directly to DDL

The user's SQL already fully defines the output schema. Do NOT ask the user about table usage, purpose, or column selection — the SQL is the spec.

1. **Parse the input SQL**: Identify source tables, JOIN conditions, selected columns, and transformations.
2. **Call `describe_table`** for each source table to understand column types.
3. **Optionally call `read_query`** with `LIMIT 10` to validate the query output.
4. **Determine table name**: Derive from the SQL context (e.g., `wide_order_customer`). If the user specified a name, use it.
5. **Go directly to Phase 2** — do NOT call `ask_user` here. The DDL confirmation in Phase 2 is the only user interaction needed.

### Description Mode (CREATE TABLE) — Confirm Schema First

Natural language is ambiguous, so clarification may be needed before generating DDL.

1. **Parse user description**: Extract table name, columns, types, constraints.
2. **Call `describe_table`** for any referenced existing tables to infer column types.
3. **If critical information is missing** (e.g., no column names or types specified), call `ask_user` to clarify. Only ask about genuinely missing information — do NOT ask about table usage or purpose if the user already described the schema.
4. **Go to Phase 2** once the schema is clear.

## Phase 2: Generate DDL and Confirm (MANDATORY ask_user)

Generate the exact DDL SQL statement and present it to the user for confirmation.

**Include the full DDL SQL inside the `ask_user` question text.** This is required because when running as a sub-agent, all intermediate assistant messages are collapsed in the UI — the user can ONLY see the `ask_user` interaction widget.

### SQL Mode
Generate CTAS: `CREATE TABLE {schema}.{table_name} AS ({select_sql})`

### Description Mode
Generate: `CREATE TABLE {schema}.{table_name} ({column_defs})`

### Both Modes — DDL Confirmation via ask_user

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
- **Execute**: proceed to Phase 3
- **Modify**: ask what to change, regenerate DDL, call `ask_user` again with the updated DDL
- **Cancel**: **STOP IMMEDIATELY.** Return `{"table_name": "", "output": "Cancelled by user."}`. Do NOT continue.

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
