---
name: table-validation
description: Validate the column contract of a newly written table — column set, types, and nullability match expectations. Object existence and row counts are handled by the builtin layer and are out of scope. Data-content assertions belong to project-level validator skills.
tags:
  - data-engineering
  - validation
  - schema
  - contract
version: "3.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_table
  - gen_job
kind: validator
severity: blocking
mode: llm
targets:
  - type: table
---

# Table Validation

Verify the **column contract** of a table that was just created or written:
columns present, declared types correct, nullability correct. This skill is
deliberately narrow.

## Target shape (important)

`ValidationHook.on_end` invokes this skill with the **whole session**, not
a single table. The target you receive is a `SessionTarget` whose
`.targets` is a list of table records matching this skill's `targets:
[{type: table}]` filter. When a node writes multiple tables (CTAS
scaffolding, layered ETL), **loop over `session.targets`** and run the
checks below independently for each `TableTarget`. Transfer targets are
covered by `transfer-reconciliation`. Emit one `CheckResult` per (target,
check) pair so the retry prompt can tell the agent which specific table
failed.

**Explicitly out of scope**:

- *Object exists* and *row count > 0* — already checked by the builtin
  validation layer before this skill runs. The hook supplies you with those
  results in the precheck context; do not re-run `describe_table` just to
  confirm the table exists.
- Data-content assertions (null ratios, value ranges, accepted values, regex
  format, duplicates, uniqueness). CTAS from an empty source, idempotent
  upserts, schema-only bootstrapping, and partition scaffolding are legitimate
  patterns that produce zero-row tables; blocking on those would cause false
  positives. If you need data-content rules for a specific table, author a
  project-level validator skill under `./.datus/skills/` or
  `~/.datus/skills/` with a `targets:` filter.

## Checks in scope

1. **Column set** — every expected column name appears in the actual table,
   and (when strict match is requested) no unexpected columns appear.
2. **Types** — each expected column's declared type matches.
3. **Nullability** — each expected column's nullability matches.

## Execution checklist

Run the column-contract checks in this order. Stop on the first blocking
failure for a given table, then continue with the next target table if the
session contains multiple table targets.

1. **Expected columns present** — when the caller supplied an expected column
   set, every expected column must appear in `describe_table` output.
2. **No unexpected columns** — when the caller requires exact matching, flag
   any actual column that is not in the contract.
3. **Types match** — compare each expected column's declared type with the
   contract. Treat widening as acceptable only when the contract explicitly
   allows it.
4. **Nullability matches** — compare each expected column's nullable /
   `NOT NULL` setting with the contract.

For every executed check, report the check name, observed value, expected
value or threshold, pass/fail decision, and a short failure reason.

## When there is no explicit column contract

If the caller did not supply an expected column set / type map, there is
**nothing for this skill to check** — emit the JSON block with
`"checks": []` and return without calling tools. The builtin layer has
already confirmed existence and row count; duplicating that check here only
produces false negatives when catalog/database/schema identifiers are
ambiguous.

## Tools

Use `describe_table` and `get_table_ddl` to introspect the target. Do **not**
run `read_query` for counting rows or sampling data — out of scope.

## Project-level validation examples

The following checks are intentionally not bundled here. Add them in a
project-level validator skill under `./.datus/skills/<name>/` or
`~/.datus/skills/<name>/` with `kind: validator` and a `targets:` filter when
the table actually needs them:

- null ratios per column
- numeric ranges / min-max checks
- accepted value sets / enum membership
- regex / format validation
- uniqueness / duplicate-key detection
- cross-column assertions

## Output

Emit the standard validator JSON block (see the output contract appended by
the hook). Use `severity: "blocking"` only for column contract violations
that would break downstream consumers. Mismatches that are cosmetic or
widening-safe should be `severity: "advisory"`.
