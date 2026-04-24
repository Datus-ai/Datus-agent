# Migration Reconciliation Checklist

Run checks in this order against each `TransferTarget`. Replace
`{src}`, `{tgt}`, `{col}`, and `{key}` with the concrete names from the
target. Pick the right connector by reading the concrete keys out of the
TransferTarget payload: `read_query(datasource=<TransferTarget.source.name>)`
for the source side and `read_query(datasource=<TransferTarget.target.datasource>)`
for the target side. Never pass the literal words `"source"`/`"target"` —
those are not real datasource keys. Every statement is a single read-only
SELECT. If the SQL needs to disambiguate a database or schema inside a
connector, qualify it inside the query (e.g. `FROM <db>.<schema>.<table>`).

## 1. Row count

Already authoritative via tool-reported `source_row_count` /
`transferred_row_count`. Fail if they differ. Do **not** re-run the source
query.

## 2. Null ratio (per column)

For each column `{col}` returned by `describe_table`:

```sql
-- Run once on source (datasource=<TransferTarget.source.name>),
-- once on target (datasource=<TransferTarget.target.datasource>)
SELECT
  SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
  COUNT(*) AS total,
  SUM(CASE WHEN {col} IS NULL THEN 1.0 ELSE 0.0 END) / NULLIF(COUNT(*), 0) AS null_ratio
FROM {table};
```

Expect `null_ratio_source ≈ null_ratio_target` (abs diff <= 1e-6).

## 3. Min / max (numeric & date columns)

```sql
SELECT MIN({col}) AS min_val, MAX({col}) AS max_val FROM {table};
```

Expect exact equality on both min and max.

## 4. Distinct count (key columns)

```sql
SELECT COUNT(DISTINCT {key}) AS distinct_count FROM {table};
```

Expect exact equality.

## 5. Duplicate key (target only)

```sql
-- run on target
SELECT {key}, COUNT(*) AS occurrences
FROM {table}
GROUP BY {key}
HAVING COUNT(*) > 1
LIMIT 10;
```

Expect empty result.

## 6. Sample diff (top 10 by key)

```sql
SELECT *
FROM {table}
ORDER BY {key}
LIMIT 10;
```

Run on both sides with the same key ordering; compare row by row.

## 7. Numeric aggregate

For each numeric column `{col}`:

```sql
SELECT SUM({col}) AS sum_val, AVG({col}) AS avg_val FROM {table};
```

Expect `abs(sum_src - sum_tgt) / max(abs(sum_src), 1) < 1e-6` and similarly
for `avg`.

## What counts as a blocking failure

- Row count mismatch
- Duplicate keys in target
- Distinct count mismatch for the declared key column
- Any sample row that differs materially

Null ratio / min-max / numeric aggregate mismatches within float tolerance
are **advisory** (reported but not blocking). The hook will inject blocking
failures back into the gen_job agent loop for retry.
