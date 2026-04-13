# Check Catalog

This skill converts a table contract into deterministic validation SQL.

## Supported checks

### 1. `required_columns`

Checks that the spec lists the columns expected in the output. This is mainly a metadata/reporting field for the current renderer and should be paired with runtime schema inspection.

Required fields:

- `columns`

### 2. `not_null`

Checks that listed columns contain no nulls.

Required fields:

- `columns`

### 3. `unique_key`

Checks that one or more columns uniquely identify rows.

Required fields:

- `columns`

### 4. `grain`

Checks a coarse grain assumption by requiring one row per key set.

Required fields:

- `columns`

This renders the same duplicate-row style check as `unique_key`, but should be interpreted as a business-grain guard rather than a primary-key contract.

## Reporting guidance

Return each check as a separate named result. If a uniqueness or grain check fails, include the duplicate-row count and a sample duplicate query when possible.

