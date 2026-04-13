# DDL Verification Checklist

Run these checks after a `CREATE` or `ALTER` statement:

1. Confirm the target object exists.
2. Compare actual columns against the expected contract.
3. Report missing columns.
4. Report unexpected extra columns.
5. Report type mismatches.
6. Report nullability mismatches if the warehouse exposes them reliably.
7. Do not continue to write data if the schema contract fails.

Use deterministic metadata queries where possible. Prefer `information_schema.columns` over free-form descriptions.

