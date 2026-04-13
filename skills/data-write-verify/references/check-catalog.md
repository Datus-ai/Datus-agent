# Data Write Verification Check Catalog

This skill focuses on checks that are cheap, deterministic, and appropriate immediately after a write:

- `row_count_range`
- `null_ratio`
- `numeric_range`
- `accepted_values`
- `regex`
- `uniqueness`

Recommended order:

1. row-count gate
2. cheap aggregate checks
3. duplicate checks
4. expensive row-level diagnostics only if a gate fails

Do not treat write verification as a replacement for long-running downstream data-quality monitoring. This is a promotion gate.

