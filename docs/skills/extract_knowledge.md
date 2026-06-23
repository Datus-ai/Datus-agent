# Extract Knowledge

`extract-knowledge` mines reusable business facts from a finished analysis and saves them as `knowledge/*.md` files. It runs automatically after a successful query when the reasoning revealed a non-obvious rule worth remembering — you don't invoke it directly.

## What it does

After you reach a working query, the path to it often teaches something that isn't visible in the schema: a field encoding, a filter you must always apply, a join that has to go through a mapping table. `extract-knowledge` captures that as an atomic, reusable fact and stores it in the project's `knowledge`, so the next similar question already knows the rule.

## What it captures

- **Field encodings / enums** — e.g. `status = 'A'` means an active account.
- **Mandatory filters** — e.g. always exclude `rtype = 'D'`.
- **Join traps** — e.g. two tables must be joined through a mapping table.
- **Business-term-to-field mappings** — which column a business term actually refers to.

It skips trivial queries and facts that are already recorded.

## When it runs

It is triggered as part of other flows — for example during [`/init`](init.md), [`/build-kb`](build_kb.md), and [Session Summarize](session_summarize.md) — whenever a finished analysis exposes a durable rule. You normally don't run it by hand.

## What you'll observe

New entries appear under `knowledge/` (indexed in `AGENTS.md`), each a short, atomic fact tied to your data. The next time a similar question comes up, the agent retrieves these facts instead of rediscovering — or missing — the rule.

## Example

You ask *"total active subscriptions last month"*. Getting it right requires knowing that `status = 'A'` means active and that cancelled rows carry `rtype = 'D'` and must be excluded. Once the query validates, `extract-knowledge` records two facts:

- `status = 'A'` → active account
- always exclude `rtype = 'D'` for active counts

Both become `knowledge` entries, so a later question about active users reuses them automatically.
