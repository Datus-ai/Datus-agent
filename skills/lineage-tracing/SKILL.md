---
name: lineage-tracing
description: Trace upstream and downstream SQL lineage in layered repositories by parsing SQL ASTs and building deterministic dependency graphs for tables, files, and target nodes
tags:
  - data-engineering
  - lineage
  - sql
  - repository
  - impact-analysis
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Lineage Tracing

Use this skill when you need a deterministic view of which tables or SQL files feed a target node and which downstream nodes depend on it. This skill is for **repository-level SQL lineage**, not database metadata search.

## When to use this skill

Activate for tasks that require any of:

- understanding what feeds a target table
- identifying which downstream files will break after a change
- building a repository dependency graph
- checking whether a proposed edit has hidden ripple effects

Use this before impact analysis, repository-level edits, or cascade-failure debugging.

## Core workflow

1. Identify the SQL repository root.
2. Parse SQL files with an AST-based approach.
3. Infer repository outputs from file paths.
4. Build direct edges:
   - file -> upstream tables
   - output table -> upstream tables
5. For a target table, return:
   - direct upstreams
   - transitive upstreams
   - direct downstreams
   - transitive downstreams

## Bundled resources

- For the expected output format and interpretation rules, read [references/output-format.md](references/output-format.md).
- To build lineage from a SQL repository, run:

```bash
python skills/lineage-tracing/scripts/trace_sql_lineage.py --root <repo_root> --target intermediate.orders_enriched
```

## Output expectations

The useful output of this skill is a structured lineage graph, not a prose summary. At minimum, return:

- discovered output tables
- direct dependency edges
- upstream/downstream sets for the target
- unresolved external references

