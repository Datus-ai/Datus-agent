# Table Column Contract Checklist

This skill checks the **column contract** only. Object existence and row
count are handled by the builtin validation layer before this skill runs —
do not re-check them here.

Run checks in this order, stopping on the first blocking failure:

1. **Expected columns present** — when the caller passed an expected column
   set, every expected column appears in `describe_table` output.
2. **No unexpected columns** — when the caller requires exact match, flag
   any column in `describe_table` output that's not in the contract.
3. **Types match** — per expected column, declared type in `describe_table`
   matches the contract. Widening is acceptable only when the contract
   explicitly allows it.
4. **Nullability matches** — per expected column, `NOT NULL` / nullable in
   `describe_table` matches the contract.

If no expected column contract was supplied, there is nothing to check —
emit an empty `checks` list and return.

## Not in scope

Already covered by the builtin layer (do **not** duplicate):

- Object existence (whether the table was created)
- Row count > 0

Belongs in **project-level validator skills**, not this bundled skill:

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

Set `severity: "blocking"` only for genuine column contract violations that
break downstream consumers. Use `severity: "advisory"` for cosmetic or
widening-safe mismatches.
