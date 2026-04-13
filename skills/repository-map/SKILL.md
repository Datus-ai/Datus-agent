---
name: repository-map
description: Map layered SQL repositories into staging, intermediate, and marts inventories with inferred output tables, short file responsibilities, and direct dependency summaries for faster repository orientation
tags:
  - data-engineering
  - repository
  - sql
  - navigation
  - impact-analysis
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Repository Map

Use this skill when you need to understand how a SQL repository is organized before editing, debugging, or tracing lineage. This skill is for **repository orientation**, not row-level data validation.

## When to use this skill

Activate for tasks that require any of:

- finding where staging, intermediate, and marts logic lives
- locating the likely file(s) to edit for a target table
- building a concise codebase map for a layered SQL project
- summarizing repository structure before impact analysis

## Core workflow

1. Identify the SQL repository root.
2. Scan SQL files and classify them by layer from the directory layout.
3. Infer each file's output table.
4. Extract direct upstream tables.
5. Produce a compact map with:
   - files grouped by layer
   - one-line role summaries
   - direct upstream counts
   - likely entry points for the requested target

## Bundled resources

- For the expected output structure, read [references/output-format.md](references/output-format.md).
- To generate a repository map from a SQL root, run:

```bash
python skills/repository-map/scripts/build_repository_map.py --root <repo_root>
```

## Output expectations

The useful output of this skill is a compact repository inventory, not a long narrative. At minimum, return:

- files grouped by layer
- inferred output tables
- direct upstream tables
- a short role summary per file

