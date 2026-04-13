---
name: data-governance-validation
description: Validate newly generated tables against data quality and governance expectations such as null ratios, numeric ranges, accepted values, format constraints, and uniqueness using deterministic SQL checks and structured check specs
tags:
  - data-engineering
  - data-governance
  - data-quality
  - validation
  - sql
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Data Governance Validation

Use this skill when a generated table must be checked against explicit data quality requirements before downstream use. This skill is for **post-generation data validation**, not for writing the transformation itself.

## When to use this skill

Activate when you need to validate one or more of:

- null-value ratios
- minimum / maximum numeric ranges
- accepted categorical values
- regex / format constraints
- uniqueness or duplicate-row expectations

Use this after a table has been materialized or when you can run deterministic validation SQL against it.

## Core workflow

1. Identify the exact target table and its expected grain.
2. Convert requirements into a structured quality contract.
3. Generate deterministic SQL checks from that contract.
4. Run cheap aggregate checks first before deeper row-level inspection.
5. Return a compact report with:
   - check name
   - observed value
   - threshold or expectation
   - pass / fail
   - failing sample query if needed

## Supported check families

- null-ratio checks
- numeric range checks
- accepted-values checks
- regex / format checks
- uniqueness checks

## Bundled resources

- For the check catalog and how to encode each rule, read [references/check-catalog.md](references/check-catalog.md).
- To start a new validation contract, copy [assets/table_quality_contract.template.json](assets/table_quality_contract.template.json).
- To render deterministic SQL from a contract, run:

```bash
python skills/data-governance-validation/scripts/render_quality_checks.py --spec skills/data-governance-validation/assets/table_quality_contract.template.json
```

## Output expectations

The useful output of this skill is a structured validation result, not a prose summary. The minimum useful report should include:

- table name
- list of executed checks
- observed metrics
- pass / fail status
- next action for failing checks

