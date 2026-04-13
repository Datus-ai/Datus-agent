---
name: surrogate-key-patterns
description: Generate deterministic surrogate keys for data engineering pipelines using stable input fields, canonical ordering, and reproducible hashing patterns
tags:
  - data-engineering
  - surrogate-key
  - hashing
  - staging
  - events
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Surrogate Key Patterns

Use this skill when a table requires a synthetic primary key or event key that is not provided directly by the source.

## When to use this skill

Activate when:

- the contract asks for a generated key
- the source has no stable natural primary key
- the output grain is event-level and uniqueness depends on combining several columns

## Key design rules

- Use stable business fields only.
- Keep field ordering deterministic.
- Avoid including volatile helper fields unless the contract explicitly requires them.
- Prefer canonical string concatenation followed by hashing when a fixed-width key is expected.
- Reuse the same field set everywhere the same entity/event key is produced.

## Common failure modes

- omitting one required input column
- changing field order between tables
- including separators when the canonical pattern expects raw concatenation
- using pre-cleaned values in one table and raw values in another

## Workflow

1. Confirm the intended grain.
2. Enumerate the exact source columns that define uniqueness.
3. Normalize null handling consistently.
4. Build the key expression once and reuse it.

For examples and review questions, read [references/examples.md](references/examples.md).

