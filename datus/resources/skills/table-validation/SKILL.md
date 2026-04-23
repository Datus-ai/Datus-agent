---
name: table-validation
description: Validate that a newly written table matches its expected schema contract — object exists, columns match, types and nullability align. Data-content assertions (row counts, null ratios, value ranges) are out of scope and belong to project-level validator skills.
tags:
  - data-engineering
  - validation
  - schema
  - contract
version: "2.1.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_table
  - gen_job
kind: validator
trigger:
  - on_tool_end
severity: blocking
mode: llm
targets: []
---

# Table Validation

Verify the **schema contract** of a table that was just created or written. This
skill is deliberately narrow: it confirms the object exists and its columns
match expectations, and nothing else. Data-content checks (row counts, null
ratios, value ranges, accepted values, regex / format rules, duplicates) are
**out of scope** — CTAS of an empty source, idempotent upserts, schema-only
bootstrapping, and partition scaffolding are all legitimate patterns that
produce zero-row tables, and blocking on that would cause false positives.

If you need data-content assertions for a specific table, author a
project-level validator skill under `./.datus/skills/` or `~/.datus/skills/`
with your rules and a `targets:` filter scoped to the relevant table /
schema.

## Checks in scope

1. **Object exists** — call `describe_table` and confirm it returns a
   non-empty column list.
2. **Column set** — compare column names against the expected contract. Flag
   missing columns and extra columns.
3. **Types** — each expected column's declared type matches.
4. **Nullability** — each expected column's nullability matches.

If no explicit column contract is supplied (the caller did not pass one), run
only check 1 (`object exists`).

## Tools

Use `describe_table` and `get_table_ddl` to introspect the target. Do **not**
run `read_query` for counting rows or sampling data — that's out of scope.

## Output

Emit the standard validator JSON output block (see the output contract
appended by the hook). Set `severity: "blocking"` only for genuine schema
contract violations. Any observation that merely reflects data content should
be omitted — it belongs in another skill.
