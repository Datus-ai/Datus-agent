# Table Validation Checklist

Run checks in this order when possible:

1. confirm the target object exists
2. compare actual columns against the expected contract
3. report missing or extra columns
4. report type or nullability mismatches
5. gate on row count
6. run null-ratio checks
7. run uniqueness / duplicate checks
8. run accepted-values and regex checks
9. run numeric range checks

Prefer deterministic metadata queries and cheap aggregate SQL before row-level diagnostics.

For every failing check, include:

- observed value
- expected value or threshold
- pass / fail decision
- a short reason
- a follow-up sample query when useful

## Example Rules From `impl-001`

Use rules like the following as natural-language inputs that the agent should
translate into `describe_table` / `get_table_ddl` / `read_query` checks.

### `staging.stg_lever__user`

- The table should contain exactly these business columns:
  `user_id`, `access_role`, `email`, `user_name`, `username`, `created_at`,
  `deactivated_at`, `external_directory_user_id`, `photo`.
- `user_id` must be unique, not null, trimmed, and non-empty.
- `access_role` must be one of `admin`, `team_admin`, `limited`,
  `super_admin`, or `interviewer`.
- `email` must be present, lowercase, trimmed, and match a basic email format.
- `user_name` must be present, trimmed, and non-empty.
- `username` should be lowercase and trimmed.
- If `deactivated_at` is present and `created_at` is present, then
  `deactivated_at` must be greater than or equal to `created_at`.
- `photo` may be null, but if it is present it should start with `http://` or
  `https://`.

### `staging.stg_lever__requisition`

- The table should contain exactly the requisition columns defined in the
  contract, including user id fields, compensation fields, headcount fields,
  and the descriptive requisition fields.
- `requisition_id` must be unique, not null, trimmed, and non-empty.
- `requisition_code` must be unique, not null, and start with `REQ-`.
- `requisition_name`, `status`, `employment_status`, `creator_user_id`,
  `owner_user_id`, and `backfill` must be present.
- `status` must be one of `open`, `closed`, `cancelled`, `draft`, or `pending`.
- `employment_status` must be one of `full-time`, `part-time`, `contract`,
  `internship`, or `temporary`.
- `compensation_band_currency`, when present, must be one of `USD`, `EUR`,
  `GBP`, `CNY`, `CAD`, or `AUD`.
- `compensation_band_interval`, when present, must be one of `hourly`,
  `monthly`, or `yearly`.
- `compensation_band_min`, when present, must be non-negative.
- `compensation_band_max`, when present, must be greater than or equal to
  `compensation_band_min`.
- `backfill` should be a real boolean flag, not a text surrogate.
- `headcount_hired`, when present, must be non-negative.
- `headcount_total`, when present, must be non-negative.
- When both are present, `headcount_hired` must be less than or equal to
  `headcount_total`.
- If `headcount_infinite` is true, `headcount_total` should be null.

### `intermediate.int_lever__requisition_users`

- Grain should stay at one row per requisition row from
  `stg_lever__requisition`; if the user dimension is unique on `user_id`, the
  join should not multiply requisitions.
- All base requisition columns should pass through unchanged from
  `stg_lever__requisition`.
- `owner_name` must come from joining `owner_user_id` to `stg_lever__user.user_id`.
- `creator_name` must come from joining `creator_user_id` to
  `stg_lever__user.user_id`.
- The output should add only `owner_name` and `creator_name` on top of the base
  requisition columns.
- A missing user match is acceptable and should produce a null `owner_name` or
  `creator_name` instead of dropping the requisition row.
