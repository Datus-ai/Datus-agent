---
name: contract-validation
description: Validate generated SQL outputs against table contracts such as expected columns, not-null constraints, uniqueness, and coarse grain assumptions using deterministic SQL checks and structured validation specs
tags:
  - data-engineering
  - contract
  - validation
  - sql
  - data-quality
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Contract Validation

Use this skill when a generated table or SQL model must be checked against an explicit table contract. This skill is for **table contract conformance**, not repository-wide dependency analysis.

## When to use this skill

Activate when you need to validate any of:

- expected output columns
- not-null constraints
- uniqueness constraints
- coarse grain assumptions such as one row per key

Use this after a table has been materialized, or when a validation query can run against the candidate output.

## Core workflow

1. Identify the target table and its contract.
2. Convert the contract into a structured validation spec.
3. Render deterministic SQL checks from the spec.
4. Run schema and data checks separately.
5. Report pass / fail per contract rule.

## Bundled resources

- For the supported contract checks, read [references/check-catalog.md](references/check-catalog.md).
- To start a validation spec, copy [assets/table_contract_validation.template.json](assets/table_contract_validation.template.json).
- To render validation SQL, run:

```bash
python skills/contract-validation/scripts/render_contract_checks.py --spec skills/contract-validation/assets/table_contract_validation.template.json
```

## Output expectations

At minimum, return:

- target table
- expected columns
- executed checks
- pass / fail status per rule
- failing keys or counts where available

