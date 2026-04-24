---
name: migration-reconciliation
description: Post-transfer cross-database reconciliation — row count, null ratio, min/max, distinct count, duplicate keys, sample diff, and numeric aggregate comparison between source and target.
tags:
  - data-engineering
  - migration
  - validation
  - reconciliation
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_job
# Driven by ValidationHook at the end of the agent run. Scoped to transfer
# targets only so intra-DB gen_job runs (no cross-DB path) skip this cleanly.
kind: validator
trigger:
  - on_end
severity: blocking
mode: llm
targets:
  - type: transfer
---

# Migration Reconciliation

Use this skill at the end of a cross-database migration run. It compares the
source (read-only) and target (just-written) tables across seven axes so we
catch row-level data drift or silent column mistranslation that the transfer
tool alone cannot detect.

## When this skill fires

`ValidationHook` triggers this skill automatically at the end of any `gen_job`
agent run that produced one or more `TransferTarget` deliverables. When the
run stayed intra-database (no `transfer_query_result` calls), this skill is
**not** invoked.

You never call this skill directly. The hook starts a read-only sub-agent
whose only job is to execute the checks below against the run's transfer
targets and emit a structured JSON report.

## What you receive

The hook hands you a `SessionTarget` containing one or more
`TransferTarget` records. Each record carries only:

- `source.name` — the source connector key (route `read_query` here for
  source-side probes).
- `target.datasource` / `target.database` / `target.db_schema` /
  `target.table` — the target coordinates.
- `source_row_count`, `transferred_row_count` — tool-reported counts
  (authoritative for check 1).

**You do NOT receive the original `source_sql`.** For checks 2–7 that
need to read source data, probe the source table directly with
`describe_table` + `read_query` on the source datasource. When the
transfer used a derived / joined source query, the session's tool-call
history in your context window shows the `transfer_query_result` call
that ran — use the `source_sql` argument recorded there. If your run
was started without a parent session (workflow / sessionless mode, or
interactive mode with `parent_session=None`) the tool-call history is
empty; in that case, fall back to the simple table-vs-table comparison
and rely on `source_row_count` / `transferred_row_count` for check 1.
Any check you cannot run should be emitted as
`{"passed": true, "severity": "advisory", "observed": {"skipped": "<reason>"}}`
— never silently drop it.

## Core workflow

For **every** `TransferTarget` in the session (the hook will tell you which
tables were transferred), execute all seven checks **in this order** and then
emit the output JSON:

1. **Row count** — total rows in source vs. target. Tool-reported
   `source_row_count` / `transferred_row_count` are the authoritative values;
   do **not** re-run the source query.
2. **Null ratio** — per-column null rate should match within a small
   tolerance. Use `read_query` on both sides with the same `COUNT(CASE WHEN
   col IS NULL THEN 1 END) / COUNT(*)` expression.
3. **Min / max** — for numeric and date columns, min and max should agree.
4. **Distinct count** — cardinality of key columns should agree.
5. **Duplicate key** — target must not have duplicates on the declared key
   (no source check needed — we assert the invariant in the target).
6. **Sample diff** — pick up to 10 rows ordered by the key column, compare
   source vs. target row by row.
7. **Numeric aggregate** — `SUM` / `AVG` of numeric columns should agree
   within float tolerance.

## Tools

You have read-only access to: `list_databases`, `list_schemas`, `list_tables`,
`describe_table`, `get_table_ddl`, `search_table`, `read_query`. Any
write tool is explicitly excluded.

## Critical rules

- `read_query` takes a `datasource=<connector key>` kwarg — it must be the
  **concrete connector key** from the TransferTarget, not the literal words
  "source" or "target". Read it from the target payload:
    - For source-side queries: use the value of `TransferTarget.source.name`
      (e.g. `datasource="pg_prod"`).
    - For target-side queries: use the value of `TransferTarget.target.datasource`
      (e.g. `datasource="ch_prod"`).
  If the SQL needs to disambiguate a database or schema inside that
  connector, qualify it in the query (e.g. `FROM <db>.<schema>.<table>`).
  Never pass the strings `"source"`/`"target"` — those are not real
  datasource keys and `read_query` will route to the wrong connector.
- Source database is read-only. Do **not** attempt any DDL or write.
- Report every check even when some fail — the hook's retry logic needs to
  see the full picture.
- For tolerances: exact match for row counts and distinct counts; float
  comparisons may use `1e-6` relative tolerance.

## Checklist

Detailed per-check SQL templates: [references/checklist.md](references/checklist.md)

## Output

Emit the JSON report block that `ValidationHook` expects — see the output
contract appended to this skill at invocation time by the hook.
