# Table Schema Contract Checklist

Run checks in this order, stopping on the first blocking failure:

1. **Object exists** — `describe_table(target)` returns a non-empty column
   list. If not, the DDL did not actually produce the table.
2. **Expected columns present** — when the caller passed an expected column
   set, every expected column appears in `describe_table` output.
3. **No unexpected columns** — when the caller requires exact match, flag
   any column in `describe_table` output that's not in the contract.
4. **Types match** — per expected column, declared type in `describe_table`
   matches the contract. Widening is acceptable only when the contract
   explicitly allows it.
5. **Nullability matches** — per expected column, `NOT NULL` / nullable in
   `describe_table` matches the contract.

## Not in scope

These belong to **project-level validator skills**, not this bundled skill:

- Row counts (> 0, ranges, minimums)
- Null ratios per column
- Numeric ranges / min-max
- Accepted value sets / enum membership
- Regex / format validation
- Uniqueness / duplicate key detection
- Cross-column assertions

To add such rules for your tables, create a new skill under
`./.datus/skills/<name>/` (project-level) or `~/.datus/skills/<name>/`
(user-level) with `kind: validator`, `targets:` scoping to the tables it
applies to, and the rules in its body. The ValidationHook will fire it
automatically alongside this bundled contract check.

## Output shape

For each check executed, report:

- check name
- observed value
- expected value / threshold
- pass / fail decision
- short reason on failure

Set `severity: "blocking"` only for genuine schema contract violations.
