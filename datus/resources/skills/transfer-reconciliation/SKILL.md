---
name: transfer-reconciliation
description: Lightweight post-transfer reconciliation example — verify tool-reported row count parity and run a small target-side sanity check without re-scanning the source.
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
# targets only so intra-DB gen_job runs skip this cleanly.
kind: validator
severity: blocking
mode: llm
targets:
  - type: transfer
---

# Transfer Reconciliation

Use this skill at the end of a `transfer_query_result` run. It is intentionally
lightweight: the built-in layer already compares tool-reported source and
target row counts, so this skill only demonstrates a small target-side sanity
check. Project-specific or strict source-vs-target checks should live in a
project/user validator skill.

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
  source-side probes only when a project-specific validator explicitly needs
  them).
- `target.datasource` / `target.database` / `target.db_schema` /
  `target.table` — the target coordinates.
- `source_row_count`, `transferred_row_count` — tool-reported counts
  (authoritative for check 1).

Default validator behavior must avoid expensive source re-scans. Do not infer
or search for the original source table; many transfers are derived queries,
joins, or aggregations. Treat `source_row_count` / `transferred_row_count` as
the source-side evidence.

## Core workflow

For **every** `TransferTarget` in the session, run only these example checks:

1. **Row count parity** — report whether tool-reported `source_row_count` and
   `transferred_row_count` match. This should normally already be present in
   the precheck context from the built-in layer; summarize it instead of
   querying the source.
2. **Target row count** — run one target-side `COUNT(*)` query using
   `TransferTarget.target.datasource` and the target table coordinate. Compare
   it with `transferred_row_count`.
3. **Target sample** — optionally read up to 5 target rows to confirm the table
   is queryable. This is advisory and should not become blocking unless the
   target query itself fails.

Stop after these checks. Do not run null-ratio, min/max, distinct-count,
duplicate-key, sample-diff, or numeric-aggregate checks in the built-in skill.
Those are examples for user-defined strict validators.

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
- Source database is read-only. The default validator should not query it.
- Report the row-count and target-query checks only. Keep failures concrete.

## Checklist

Run these lightweight checks for each `TransferTarget`.

### 1. Row Count Parity

Compare the tool-reported counts:

- `source_row_count`
- `transferred_row_count`

Fail if both are present and differ. Do not re-run the source query.

### 2. Target Row Count

Run one target-side count query using the concrete target datasource and table
coordinates from the `TransferTarget`.

```sql
SELECT COUNT(*) AS row_count
FROM {target_table};
```

Compare the result with `transferred_row_count`. This is blocking when the
target count cannot be read or does not match the transferred count.

### 3. Target Sample

Optionally read a small sample to confirm the target table is queryable.

```sql
SELECT *
FROM {target_table}
LIMIT 5;
```

Treat this as advisory unless the query itself fails.

### Out Of Scope For The Built-In Skill

Do not run null-ratio, min/max, distinct-count, duplicate-key, sample-diff, or
numeric-aggregate checks here. Those checks can be added later as
project-specific validator skills with their own cost profile and table
contracts.

## Output

Emit the JSON report block that `ValidationHook` expects — see the output
contract appended to this skill at invocation time by the hook.
