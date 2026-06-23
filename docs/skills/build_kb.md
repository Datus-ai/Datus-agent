# Build KB

`build-kb` builds the project's vector-indexed knowledge base — semantic models, metrics, and reference SQL — so you can run semantic search over your data. It is the heavyweight counterpart to [`init`](init.md).

Run it with the `/build-kb` command inside the REPL.

## What it does

- Scans the configured datasources (tables, schemas, relationships).
- Proposes a **generation manifest** — what it plans to build — and waits for you to confirm or adjust it.
- After confirmation, generates and indexes each artifact: semantic models, metrics, and reference SQL.
- Refreshes the knowledge-base index section of `AGENTS.md`.

It is the **heavyweight** tier: slower than `init` (minutes), and it always asks for confirmation before generating.

## When to use it

- You want semantic search over your data (semantic models, metrics, reference SQL).
- The lightweight inventory from [`/init`](init.md) isn't enough.
- You want to build or rebuild the knowledge base.

## How to use it

```text
/build-kb
```

You can add optional free-text hints to focus generation on specific files, tables, or domains:

```text
/build-kb only the orders and customers tables
```

A typical run looks like this:

1. `build-kb` scans your datasources and **proposes a manifest** of what it will generate.
2. You review the manifest and confirm or adjust it.
3. It generates and indexes the confirmed artifacts, then refreshes the `AGENTS.md` index.
4. It reports what was generated and indexed (counts per artifact type).

With no hints, it proposes a manifest covering the main datasources. Re-running updates existing artifacts rather than duplicating them.

## Build KB vs. Init

Run [`/init`](init.md) first for an instant, lightweight inventory, then `/build-kb` for the vector-indexed knowledge base. See [Init](init.md#init-vs-build-kb) for the full comparison.

## Notes

- Generation only covers configured datasources.
- The confirmation gate always runs — you decide what gets generated before any work happens.
