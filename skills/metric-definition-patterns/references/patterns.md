# Metric Patterns

## Safe division

Use explicit guards:

```sql
CASE
    WHEN denominator IS NULL OR denominator = 0 THEN 0
    ELSE numerator * 1.0 / denominator
END
```

## Grain discipline

Do not compute a per-user metric directly from event grain if the contract expects user grain. Aggregate to the expected grain first.

## Precision discipline

Avoid `CAST(... AS BIGINT)` on values that are logically continuous metrics unless the contract explicitly requires an integer.

## Time-window metrics

Use a pinned reference date when the environment requires deterministic outputs.

