# Data Migration Checklist

## Pre-Migration

1. Confirm source table exists and has data
2. Document source schema: columns, types, nullable, row count
3. Identify primary key candidates in source
4. Check if target database/schema exists; create if needed
5. Verify target database connectivity and write permissions

## DDL Generation

6. Map source column types to target dialect
7. Handle unsupported types (LIST, STRUCT, MAP, BLOB) — report and exclude or error
8. For StarRocks: determine DUPLICATE KEY and DISTRIBUTED BY HASH columns
9. Generate target CREATE TABLE DDL
10. Execute target DDL and verify table created with correct schema

## Data Transfer

11. Execute data transfer using transfer_query_result
12. Verify transfer completed: check rows_transferred matches source row count
13. If transfer fails partially, report rows written and error details

## Post-Migration Reconciliation

Execute all checks in order. Each check compares source vs target:

14. Row count comparison
15. Null ratio comparison for nullable columns
16. Min/max comparison for numeric and date columns
17. Distinct count comparison for key columns
18. Duplicate key check on target table
19. Key-based sample diff (top 10 rows by key)
20. Numeric aggregate comparison (SUM, AVG) for numeric columns

## Reporting

21. Compile all check results with pass/fail status
22. Flag any blocking issues
23. Output final migration summary
