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
allowed_agents:
  - gen_table
  - gen_job
---

## CRITICAL: Interactive vs Workflow Mode

- When `ask_user` is available, use it for DDL confirmation and clarification.
- When `ask_user` is not available (workflow, batch, or print mode), never call `ask_user` and never wait for user input. Treat the original request as authorization only for the specific non-destructive `CREATE TABLE` / CTAS it explicitly asks for.
- If critical schema details, target table, target database, or destructive authorization are missing and `ask_user` is unavailable, stop and report exactly what is missing.

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
5. **Go directly to Phase 2** — do NOT ask about table usage, purpose, or column selection.

### Description Mode (CREATE TABLE) — Confirm Schema First

Natural language is ambiguous, so clarification may be needed before generating DDL.

1. **Parse user description**: Extract table name, columns, types, constraints.
2. **Call `describe_table`** for any referenced existing tables to infer column types.
3. **If critical information is missing** (e.g., no column names or types specified), clarify with `ask_user` when available. If `ask_user` is unavailable, stop and report the missing fields instead of guessing.
4. **Go to Phase 2** once the schema is clear.

## Phase 2: Generate DDL and Authorize

Generate the exact DDL SQL statement.

### SQL Mode
Generate CTAS: `CREATE TABLE {schema}.{table_name} AS ({select_sql})`

### Description Mode
Generate: `CREATE TABLE {schema}.{table_name} ({column_defs})`

### When `ask_user` is available — DDL Confirmation

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

### When `ask_user` is unavailable — Workflow Authorization

- Do not call `ask_user`.
- Proceed to Phase 3 only when the request explicitly asks to create this table or CTAS result and the target database/table is unambiguous.
- If the target table already exists, proceed only if the request explicitly authorizes replacement, overwrite, drop/recreate, or equivalent destructive behavior.
- If the DDL includes `DROP`, `ALTER`, `TRUNCATE`, `CREATE OR REPLACE`, or any existing-object replacement, require explicit authorization in the original request. Otherwise stop and report the required authorization.
- Include the final DDL in the output summary so the caller can audit what was executed.

## Phase 3: Execute and Verify

1. **Call `execute_ddl(sql)`** with the confirmed or workflow-authorized DDL statement.
2. **Verify**:
   - SQL Mode: Call `read_query("SELECT COUNT(*) FROM {schema}.{table_name}")` to confirm row count
   - Description Mode: Call `describe_table("{schema}.{table_name}")` to confirm schema matches
3. **Call `describe_table("{schema}.{table_name}")`** to confirm the created schema.

If DDL fails:
- Parse the error message
- If `ask_user` is available, fix the SQL, show the updated DDL to the user via `ask_user`, and retry (up to 3 attempts)
- If `ask_user` is unavailable, fix and retry directly up to 3 attempts when the intent remains the same and no new destructive action is introduced
- If still failing, report the final error and the last attempted DDL in the output

## Phase 4: Summary

Output a summary including:
- Created table name and location
- Row count (for CTAS) or column count (for CREATE TABLE)
- Column list with types
- Original SQL (for CTAS) or user description (for CREATE TABLE)
- Hint: if the user needs a semantic model, suggest `task(type="gen_semantic_model", prompt="{table_name}")`

## Important Rules

- Use `ask_user` before executing DDL only when the tool is available.
- In workflow mode, execute only explicitly requested non-destructive table creation; block ambiguous or destructive work instead of guessing.
- **DDL is irreversible** — always include the exact DDL SQL in `ask_user` confirmation when interactive, or in the final output when workflow mode executes.
- If the target table already exists, ask whether to drop/recreate/abort when interactive; require explicit replacement authorization when workflow mode.
- Language: match user's language (Chinese input → Chinese output)
- Do NOT modify the source tables — only create new tables
- **Single responsibility** — gen-table only creates tables, does not generate semantic model YAML. For semantic model, suggest using `gen_semantic_model`
