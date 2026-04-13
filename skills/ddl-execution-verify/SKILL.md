---
name: ddl-execution-verify
description: Execute or review CREATE or ALTER DDL changes with immediate schema verification against an expected contract, returning existence, type, nullability, and extra or missing column diffs before downstream writes continue
tags:
  - data-engineering
  - ddl
  - schema
  - validation
  - sql
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# DDL Execution & Verify

Use this skill when a table or view definition must be created or altered and immediately verified before downstream stages continue. This skill is for **schema-changing SQL plus schema verification**, not data-quality checks on written rows.

## When to use this skill

Activate when you need to:

- execute or review `CREATE TABLE`, `CREATE VIEW`, or `ALTER TABLE`
- verify the created object exists
- compare actual columns against an expected schema contract
- block downstream writes if schema validation fails

## Core workflow

1. Identify the exact target object and DDL statement.
2. Execute the DDL in a safe environment if execution is allowed.
3. Convert the expected schema into a structured verification spec.
4. Compare existence, missing columns, extra columns, type mismatches, and nullability mismatches.
5. Return a compact verification report and stop the pipeline if validation fails.

## Bundled resources

- For the verification checklist, read [references/checklist.md](references/checklist.md).
- To start a schema verification spec, copy [assets/table_schema_contract.template.json](assets/table_schema_contract.template.json).
- To render deterministic schema diff SQL, run:

```bash
python skills/ddl-execution-verify/scripts/render_ddl_verify_checks.py --spec skills/ddl-execution-verify/assets/table_schema_contract.template.json
```

## Output expectations

At minimum, return:

- target object
- DDL execution status
- existence check result
- missing columns
- extra columns
- type or nullability mismatches

