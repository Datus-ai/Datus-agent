# Session Summarize

`session-summarize` reviews a chat session and persists what matters. It walks the conversation, identifies durable takeaways, and routes each to the right store. Use it at the end of a working session so the next one starts where you left off.

## What it does

A working session often produces things worth keeping — a business rule you uncovered, a preference you stated, a query you validated. `session-summarize` reviews the whole conversation, picks out those durable takeaways, and saves each to the store that fits (via [Storage Classify](storage_classify.md)): facts to knowledge, preferences to memory, validated queries to reference SQL, and so on.

## When to use it

- The conversation produced durable facts, preferences, or validated SQL.
- You want to save or remember the session.
- Before switching to a different project or task.

## How to use it

Run `session-summarize` as a skill at the end of a session. You can optionally give it focus hints; with no hint, it reviews the whole session. It captures only the takeaways worth keeping — the one-off back-and-forth is left out.

## Example

You spend a session figuring out a churn query. Along the way you confirm a validated SQL, learn that `plan_tier = 0` means a free account, and decide you prefer results rounded to whole percentages. At the end, `session-summarize` saves:

- the churn query → **reference SQL**,
- the `plan_tier = 0` rule → **knowledge**,
- the rounding preference → **memory**.

The next session picks all three up automatically.
