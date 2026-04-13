# Pipeline Sanity Checklist

Use these checks in order:

1. Is every direct upstream table present?
2. If a direct upstream is in the same layer, has it already been materialized?
3. Are direct upstream schemas compatible with the current SQL?
4. Does the final SELECT match the expected output contract?
5. If execution fails, is the error local or caused by a missing upstream?
6. If many downstream tables fail, identify the earliest missing or malformed dependency first.

