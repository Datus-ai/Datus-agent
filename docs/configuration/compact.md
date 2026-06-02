# Context Compaction

As a chat session grows, its history eventually approaches the model's context window limit. Datus manages this automatically with two complementary compaction passes configured under `agent.compact`, so you can keep working in one long session without manually clearing it or hitting token limits.

| | Minor compact | Major compact |
|---|---|---|
| What it does | Archives old tool I/O to disk | Summarizes the whole session |
| Driven by | A rule (user-turn count) | The LLM (a summarization call) |
| Execution | Synchronous, but fast (local, no LLM call) | Synchronous, blocks the run loop |
| Touches recent turns | No | Yes — replaces all history |
| Recoverable | Yes (archive files) | Yes (full-history JSONL) |

## Minor compact

**What triggers it** — At the start of each user turn, if the session has more than `keep_recent_user_turns` user turns (default 4), minor compact runs. Its gate — the user-turn count — only changes between turns, so it is evaluated once per turn rather than after every tool call (that per-tool-call check is reserved for major, whose token ratio actually grows mid-turn). It is synchronous but, being a purely local, rule-based archive with no LLM call, finishes quickly and barely delays the agent.

**What it does** — For every turn *older* than the kept window, any tool-call argument or output longer than `archive_threshold` characters (default 1000) is moved out of the live conversation and written to an on-disk archive. A short inline preview (`archive_preview_chars`, default 1000; 2× for error outputs) is left behind with a `[DATUS_ARCHIVED]` marker.

**Resulting behavior**

- The most recent `keep_recent_user_turns` turns keep their full tool I/O — the active part of the conversation is never degraded.
- Older bulky outputs shrink to a preview plus a pointer; the model can still `read_file` the archive to recover the full content when it needs the detail.
- Because it only archives (never summarizes), nothing is lost and no LLM call is spent. It is cheap and fast, and runs often.

## Major compact

**What triggers it** — When the input tokens of the most recent model call reach `token_threshold` of the context window (default `0.9`, i.e. 90%), a major compact is forced. The check runs at the start of each user turn and after each tool call, so it can fire **mid-turn** — right when the context is about to overflow — instead of waiting for the next turn. `/compact` triggers it manually at any time.

**What it does** — The model is asked to summarize the **entire** session into a single recap. The session history is then cleared and replaced by that summary as the new starting point, and the conversation continues from there. The complete pre-compact history is dumped to a JSONL file, and a pointer to it is appended to the summary so the agent can `read_file` it to recover any specific detail.

**Resulting behavior**

- It is **synchronous and blocking**: the run loop waits for the summary to be written before continuing, because the next model call must see the compacted history rather than the over-limit one.
- The visible conversation is collapsed — earlier turns are replaced by the recap. In the CLI this shows a `Compacting context…` hint followed by a summary panel; over the API/print stream it arrives as a `compact_summary` markdown message.
- Some fidelity is traded for room: the summary is concise, so fine detail now lives only in the JSONL dump (still reachable via `read_file`).
- A major compact spends one extra LLM call (the summarization), so it is rarer and only triggers near the context limit.

## Auto vs. manual

- **Automatic** — Both passes run on their own during a session; you don't need to do anything. Major fires near the context limit, minor as the turn count grows.
- **Manual `/compact`** — Always runs a **major** pass immediately, regardless of current usage. Useful before starting a big new task when you want a clean, summarized starting point.

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `compact.major.enabled` | bool | `true` | Enable the LLM-driven full-history summarization pass. |
| `compact.major.token_threshold` | float | `0.9` | Fraction of the context window above which a major compact is forced (overrides minor selection in the auto dispatcher). |
| `compact.minor.enabled` | bool | `true` | Enable the rule-based archiving pass. |
| `compact.minor.keep_recent_user_turns` | int | `4` | Keep the original tool I/O of the most recent N user turns intact; older turns are eligible for archiving. |
| `compact.minor.archive_threshold` | int | `1000` | Tool-call arguments/outputs longer than this many characters are offloaded to disk. |
| `compact.minor.archive_preview_chars` | int | `1000` | Inline preview length kept in the archive marker (error outputs get a 2× preview). |

```yaml title="agent.yml"
agent:
  compact:
    major:
      enabled: true
      token_threshold: 0.9
    minor:
      enabled: true
      keep_recent_user_turns: 4
      archive_threshold: 1000
      archive_preview_chars: 1000
```

To disable automatic compaction entirely, set both `major.enabled` and `minor.enabled` to `false`. You can still run `/compact` manually even when `major.enabled` is `false`.

## Notes

- Major compact's trigger reads the **live** token usage of the current turn (the same figure shown in the CLI status bar). On a brand-new session or right after `resume`, that signal starts at zero, so a major compact won't fire on the very first turn until at least one model call has reported usage.
- Minor compact reads its eligibility from the session's user-turn count, so it keeps working correctly after a resume.
- Archived tool I/O and the full-history JSONL live under the session's data directory; see [Storage](storage.md) for paths.
