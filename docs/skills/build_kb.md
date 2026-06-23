# Build KB

`build-kb` builds the project's vector-indexed knowledge base — semantic models, metrics, and reference SQL — so you can run semantic search over your data. It is the heavyweight counterpart to [`init`](init.md).

Run it with the `/build-kb` command inside the REPL.

## What it does

- Scans the configured datasources (tables, schemas, relationships).
- Proposes a **generation manifest** — what it plans to build — and waits for you to confirm or adjust it.
- After confirmation, generates and indexes each artifact: semantic models, metrics, and reference SQL.
- Refreshes the knowledge-base index section of `AGENTS.md`.

It is the **heavyweight** tier: slower than `init` (minutes), and by default it asks you to confirm a manifest before generating.

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

## Examples

The text after `/build-kb` is forwarded to the agent as instructions, so you can shape the run in plain language.

### Limit the scope

Restrict scanning and generation to specific tables, files, or business domains instead of all datasources:

```text
/build-kb only the orders and order_items tables
/build-kb the sales domain, plus queries/*.sql
```

### Choose what to generate

Build only some artifact types. Here it generates semantic models and metrics but leaves reference SQL out:

```text
/build-kb semantic models and metrics only, skip reference SQL
```

### Skip the confirmation

Bypass the manifest review and generate right away — handy when you already know the scope and don't need to adjust the plan:

```text
/build-kb the orders table, skip the manifest confirmation and generate directly
```

These can be combined — for example, *"only the orders domain, semantic models only, skip confirmation"*.

## Build KB vs. Init

Run [`/init`](init.md) first for an instant, lightweight inventory, then `/build-kb` for the vector-indexed knowledge base. See [Init](init.md#init-vs-build-kb) for the full comparison.

## Notes

- Generation only covers configured datasources.
- By default the manifest confirmation runs so you decide what gets generated before any work happens; you can ask it to skip that gate (see [Examples](#skip-the-confirmation)).
