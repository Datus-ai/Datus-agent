# Memory Organization

`memory-organization` audits and tidies up the project's persistent stores — knowledge, memory, semantic models, and the `AGENTS.md` index. It finds duplicates, contradictions, misfiled entries, and gaps, then proposes fixes and applies the ones you approve.

Run it periodically, or whenever the stores feel cluttered.

## What it does

Over time, knowledge and memory accumulate: the same fact gets recorded twice, two entries start to conflict, something filed as memory really belongs in knowledge, or an obvious fact never got written down. `memory-organization` reviews the stores as a whole and brings them back into a clean, consistent state.

## What it checks

- **Duplicates** — the same fact recorded in more than one place.
- **Contradictions** — entries that conflict with each other.
- **Misfiled** — knowledge that should be memory, or vice versa.
- **Gaps** — obvious facts that should be recorded but aren't.
- **Staleness** — entries that are no longer true.

## When to use it

- The stores feel cluttered or contradictory.
- After heavy usage that added many entries.
- Knowledge and memory have drifted or started to overlap.
- You want to clean up or reorganize.

## How to use it

Run `memory-organization` as a skill from the REPL. You can optionally tell it which stores to focus on; with no hint, it audits all of them. It reports what it found and proposes fixes before applying the ones you approve — so you stay in control of any change.

## Example

After a few weeks of heavy use, your project has the same "fiscal year starts in April" rule written in two `knowledge` files, a `memory` note that contradicts a newer preference, and a stale entry referring to a table you dropped. Running `memory-organization` flags all three, proposes merging the duplicate, removing the stale entry, and reconciling the conflict — and applies them once you confirm.
