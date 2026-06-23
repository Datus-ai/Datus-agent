# Init

`init` is the lightweight way to bootstrap a project workspace. It scans your working directory and configured datasources, then writes a concise, durable picture of the project so future sessions start with context — without the heavier vector knowledge base.

Run it with the `/init` command inside the REPL.

## What it does

- Scans the working directory and the configured datasources (names, types, table counts).
- Writes an `AGENTS.md` inventory at the project root: architecture, directory map, services, data assets, and a knowledge index.
- Persists durable business facts as file-based knowledge, and agent-bound preferences as memory.

It is the **lightweight** tier: fast (seconds to a minute), no vector index, and no confirmation gate. For the vector-indexed knowledge base, use [`/build-kb`](build_kb.md).

## When to use it

- A new project workspace has no `AGENTS.md` yet.
- You explicitly want to initialize or re-scan the project.
- Files or schemas changed materially and the inventory is stale.

You can skip it when an up-to-date `AGENTS.md` already exists and nothing material has changed.

## How to use it

```text
/init
```

You can add optional free-text hints after the command — a goal, a scope, or specific files or tables to focus on:

```text
/init focus on the sales and finance schemas
```

With no hints, `init` scans broadly. Re-running it refreshes the inventory without duplicating knowledge entries.

## Init vs. Build KB

| | `init` | [`/build-kb`](build_kb.md) |
|---|---|---|
| Speed | Fast (seconds to a minute) | Slower (minutes) |
| Output | `AGENTS.md` + knowledge / memory | Vector KB: semantic models, metrics, reference SQL |
| Confirmation | None | Manifest confirmation gate |
| Vector index | No | Yes |
| When | First, always | After init, when you want semantic search |

Typical flow: run `/init` first for instant context, then `/build-kb` when you want the vector-indexed knowledge base.

## Notes

- Knowledge is stored as files under `knowledge/`; memory is agent-bound and persists across sessions.
- `init` only inventories — it does not run SQL queries.
- Generating semantic models, metrics, and reference SQL is the job of [`/build-kb`](build_kb.md).
