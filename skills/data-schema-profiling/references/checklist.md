# Profiling Checklist

Use this checklist when profiling a table for downstream SQL generation:

1. Confirm the physical table name and schema.
2. List all columns and source types.
3. Identify target columns likely to need transformation.
4. For timestamp/date candidates:
   - sample a few values
   - try plain casts
   - note whether a salvage strategy is needed
5. For ids / dimensions:
   - inspect nulls
   - inspect distinctness only if it affects joins or keys
6. For enumerations or categories used in business logic:
   - sample distinct values with a cap
7. Record only the findings that downstream SQL generation must preserve.

