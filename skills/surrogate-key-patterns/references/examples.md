# Surrogate Key Review Guide

Before finalizing a surrogate key expression, check:

1. Does the chosen field set uniquely identify the target grain?
2. Are the fields ordered consistently?
3. Is null handling deterministic?
4. Would the same source row generate the same key across reruns?
5. Are you matching the contract's canonical naming and hashing pattern?

Example shape:

```sql
md5(concat(visitor_id, event_time, account_id, feature_id))
```

Only use separators if the governing convention explicitly requires them.

