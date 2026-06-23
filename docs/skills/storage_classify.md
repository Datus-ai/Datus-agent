# Storage Classify

`storage-classify` decides **where** the content produced during your work should be saved, so that everything lands in the store that fits its shape and reuse pattern. It runs automatically as part of [`/init`](init.md), [`/build-kb`](build_kb.md), and session summarization — you don't invoke it directly.

## What it does

When Datus produces something worth keeping — a business fact, a validated SQL, a metric definition, a preference — `storage-classify` works out which store it belongs to and routes it there. The result is that your knowledge base stays organized: facts go where you'll find them, definitions are indexed for search, and one-off noise isn't persisted at all.

## Where content goes

| Content | Store |
|---------|-------|
| A reusable business rule — a field encoding, a mandatory filter, a term-to-field mapping, a join trap | **knowledge** (`knowledge/*.md`) |
| The structure of a table — its identifiers, measures, and dimensions | **semantic models** |
| A named business metric built on a measure | **metrics** |
| A complete, validated SQL worth keeping as an example for future reuse | **reference SQL** |
| A high-level project overview — architecture, services, data assets, KB index | **`AGENTS.md`** |
| A lightweight, cross-session preference bound to one agent | **memory** |
| A reusable, multi-step workflow you'll run again | **skills** |
| One-off content, or anything already obvious from the schema | **not persisted** |

## What you'll observe

- After `/init` or `/build-kb`, the produced content is spread across these stores rather than dumped into one place.
- Business rules become searchable `knowledge` entries; reusable queries become `reference SQL` examples; table structures become semantic models — each retrievable later.
- Trivial or non-reusable content is intentionally left out, so the knowledge base doesn't fill up with noise.

## Example

Suppose you finish an analysis that answers *"monthly active users by region"* with a validated query, and along the way you learn that `status = 'A'` means an active account. After the run, `storage-classify` routes the pieces separately:

- the validated query → **reference SQL** (a reusable example),
- the `status = 'A'` rule → **knowledge** (the reason behind the query),
- the regional user-count metric, if defined → **metrics**.

Each ends up in the store built for it, so the next similar question can reuse all three.

## Relationship to Init and Build KB

`storage-classify` is the routing layer that [`/init`](init.md) and [`/build-kb`](build_kb.md) rely on when deciding where their findings and generated artifacts belong. It only classifies and routes — the actual generation and indexing is done by the matching parts of those commands.
