# Check Catalog

This skill translates a structured quality contract into deterministic SQL checks. Keep the contract focused on checks that can be validated directly from table contents.

## Supported checks

### 1. `null_ratio`

Use when a column may contain some nulls, but the null proportion must stay below a threshold.

Required fields:

- `column`
- `max_ratio`

Example:

```json
{
  "type": "null_ratio",
  "name": "email_null_ratio",
  "column": "email",
  "max_ratio": 0.01
}
```

### 2. `range`

Use when a numeric column must stay within expected lower and/or upper bounds.

Required fields:

- `column`

Optional fields:

- `min_value`
- `max_value`

Example:

```json
{
  "type": "range",
  "name": "amount_range",
  "column": "amount",
  "min_value": 0,
  "max_value": 100000
}
```

### 3. `accepted_values`

Use when a categorical column must stay inside a known whitelist.

Required fields:

- `column`
- `values`

Optional fields:

- `max_fail_ratio`

Example:

```json
{
  "type": "accepted_values",
  "name": "status_values",
  "column": "status",
  "values": ["active", "inactive", "paused"],
  "max_fail_ratio": 0.0
}
```

### 4. `regex`

Use when a string column must match a format such as email, phone number, postal code, or ISO timestamp text.

Required fields:

- `column`
- `pattern`

Optional fields:

- `max_fail_ratio`

Example:

```json
{
  "type": "regex",
  "name": "phone_format",
  "column": "phone_number",
  "pattern": "^\\d{10}$",
  "max_fail_ratio": 0.0
}
```

### 5. `uniqueness`

Use when one or more columns should uniquely identify rows.

Required fields:

- `columns`

Optional fields:

- `max_duplicate_rows`

Example:

```json
{
  "type": "uniqueness",
  "name": "customer_key_unique",
  "columns": ["customer_id"],
  "max_duplicate_rows": 0
}
```

## Recommended execution order

Run checks in this order when possible:

1. row-count sanity checks
2. null-ratio checks
3. uniqueness checks
4. accepted-values checks
5. regex / format checks
6. numeric range checks

This ordering finds the cheapest structural problems first.

## Reporting guidance

For every failed check, include:

- the observed metric
- the allowed threshold
- a short reason
- a follow-up query or failing-sample filter if useful

The report should stay structured enough that another agent or pipeline can consume it directly.

