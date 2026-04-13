---
name: data-write-verify
description: Validate insert, CTAS, merge, or load operations after execution using row-count gates and deterministic post-write data assertions such as null ratios, ranges, accepted values, and duplicate checks
tags:
  - data-engineering
  - write
  - validation
  - sql
  - data-quality
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Data Write & Verify

Use this skill when a write operation such as `INSERT`, `CTAS`, `MERGE`, or `COPY` must be validated immediately after execution. This skill is for **post-write verification**, not schema-only checks.

## When to use this skill

Activate when you need to:

- gate writes by expected row-count range
- validate the written table after `INSERT` or `CTAS`
- return a compact write summary plus quality assertions
- decide whether a write should be accepted or rolled back

## Core workflow

1. Identify the target table and write statement.
2. Define the expected row-count range.
3. Convert post-write assertions into a structured validation spec.
4. Run the row-count gate first.
5. Run aggregate quality checks such as null ratio, numeric range, accepted values, regex, and duplicates.
6. Return a pass / fail report before promoting the write.

## Bundled resources

- For the supported check families, read [references/check-catalog.md](references/check-catalog.md).
- To start a write validation spec, copy [assets/write_validation.template.json](assets/write_validation.template.json).
- To render deterministic verification SQL, run:

```bash
python skills/data-write-verify/scripts/render_write_checks.py --spec skills/data-write-verify/assets/write_validation.template.json
```

## Output expectations

At minimum, return:

- target table
- observed row count
- expected row-count range
- executed checks
- pass / fail status per check

