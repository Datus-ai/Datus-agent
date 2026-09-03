# Context Compaction

As a chat session grows, its history eventually approaches the model's context window limit. Datus manages this automatically with two complementary compaction passes configured under `agent.compact`, so you can keep working in one long session without manually clearing it or hitting token limits.

| | Minor compact | Major compact |
|---|---|---|
| What it does | Archives old tool I/O to disk | Summarizes the whole session |
| Driven by | A rule (user-turn count at turn start; tool-output age mid-turn) | The LLM (a summarization call) |
| Execution | Synchronous, but fast (local, no LLM call) | Synchronous, blocks the run loop |
| Touches recent turns | No (turn start) / only this turn's older tool outputs (mid-turn) | Yes — replaces all history |
| Recoverable | Yes (archive files) | Yes (full-history JSONL) |

Both passes run at two points: **at the start of a user turn**, before the model is called, and **in the middle of a turn**, right before the next model call of an agent loop that is still running tools. The mid-turn variant is described in its own section below.

## Minor compact

**What triggers it** — At the start of each user turn, if the session has more than `keep_recent_user_turns` user turns (default 4), minor compact runs. Its gate — the user-turn count — only changes between turns, so it is evaluated once per turn. It is synchronous but, being a purely local, rule-based archive with no LLM call, finishes quickly and barely delays the agent.

**What it does** — For every turn *older* than the kept window, any tool output longer than `archive_threshold` characters (default 1000) is moved out of the live conversation and written to an on-disk archive. A short inline preview (`archive_preview_chars`, default 1000; 2× for error outputs) is left behind with a `[DATUS_ARCHIVED]` marker.

**Resulting behavior**

- The most recent `keep_recent_user_turns` turns keep their full tool I/O — the active part of the conversation is never degraded at turn start.
- Older bulky outputs shrink to a preview plus a pointer; the model can still `read_file` the archive to recover the full content when it needs the detail.
- Because it only archives (never summarizes), nothing is lost and no LLM call is spent. It is cheap and fast, and runs often.

## Major compact

**What triggers it** — When the context occupancy reaches `token_threshold` of the context window (default `0.9`, i.e. 90%), a major compact runs. At turn start the occupancy is the input-token count of the most recent model call. Mid-turn it is estimated before each model call as *last call's real input tokens + an estimate of the items appended since (the model's reply and the tool outputs) + the output headroom reserved for the next reply*, so the pass fires **before** the request that would overflow, not after it fails. `/compact` triggers it manually at any time.

**What it does** — The model is asked to summarize the **entire** transcript into a single recap. The history is then replaced by that summary as the new starting point, and the conversation continues from there. The complete pre-compact history is dumped to a JSONL file, and a pointer to it is appended to the summary so the agent can `read_file` it to recover any specific detail.

**Resulting behavior**

- It is **synchronous and blocking**: the loop waits for the summary before issuing the next model call, because that call must see the compacted history rather than the over-limit one.
- The visible conversation is collapsed — earlier turns are replaced by the recap. In the CLI this shows a `Compacting context…` hint followed by a summary panel; over the API/print stream it arrives as a `compact_summary` markdown message.
- Some fidelity is traded for room: the summary is concise, so fine detail now lives only in the JSONL dump (still reachable via `read_file`).
- A major compact spends one extra LLM call (the summarization), so it is rarer and only triggers near the context limit.

## Mid-turn compaction

A single user request can drive dozens of tool calls, and the context can fill up long before the turn ends. Datus therefore re-evaluates the occupancy **before every model call** of a running turn — at the point where the previous round's tool results have all been appended and no tool is still running — and rewrites the history the model is about to receive. The very next model call already uses the compacted context; the turn continues without any user interaction.

**Two stages, cheapest first**

1. **Archive stage (minor)** — at `minor.mid_turn_token_threshold` (default `0.75`): the tool outputs of the turn in progress that are older than the most recent `keep_recent_tool_results` (default 5) are moved to the on-disk archive, exactly like the turn-start minor pass does for older turns. Outputs whose marker (path + preview) would not be shorter than the text are left alone. No LLM call. If this brings the estimate back under the major threshold, nothing else happens.
2. **Summary stage (major)** — if the estimate is still at or above `major.token_threshold`: the (archived) transcript is summarized and the history is rebuilt as

   ```
   [user]      your original request for this turn (verbatim)
   [user]      any message you typed while the turn was running (verbatim)
   [assistant] the summary, plus the JSONL recovery pointer
   [user]      a short instruction to continue from the summary without asking or repeating work
   ```

   The rewrite starts with your request and ends with an instruction, so the model resumes the task instead of replying to the summary. The instruction is synthetic and is hidden from the conversation history shown in the CLI and API.

**What you will notice**

- The status bar's context usage drops right after the rewrite, and the `Compacting context…` hint / summary panel appear mid-turn.
- Later calls of the same turn keep the compacted view and append new tool rounds after it. The pre-compaction tool outputs are never sent again, but the full transcript is in the JSONL dump and archived outputs are in the archive directory.
- Pressing ESC to cancel after a mid-turn compaction rolls the session back to the compacted view (the request plus summary), not to an empty session.
- The session file is rewritten to match the compacted view, so a later `resume` starts from it.

**Safety limits** — Mid-turn compaction gives up for the rest of the turn after three consecutive failures (for example a failing summary call) or when the compacted view is itself still above the threshold; the turn then continues uncompacted. A summary failure never discards the result of the archive stage. Either stage can be switched off independently with its `mid_turn_enabled` flag without affecting turn-start compaction.

## Auto vs. manual

- **Automatic** — Both passes run on their own during a session; you don't need to do anything. Major fires near the context limit, minor as the turn count grows or as a long turn accumulates tool output.
- **Manual `/compact`** — Always runs a **major** pass immediately, regardless of current usage. Useful before starting a big new task when you want a clean, summarized starting point.

## Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `compact.major.enabled` | bool | `true` | Enable the LLM-driven full-history summarization pass. |
| `compact.major.token_threshold` | float | `0.9` | Fraction of the context window at which a major compact runs (turn start and mid-turn). |
| `compact.major.mid_turn_enabled` | bool | `true` | Allow the summary stage to run in the middle of a turn. |
| `compact.minor.enabled` | bool | `true` | Enable the rule-based archiving pass. |
| `compact.minor.keep_recent_user_turns` | int | `4` | Turn start: keep the original tool I/O of the most recent N user turns intact; older turns are eligible for archiving. |
| `compact.minor.archive_threshold` | int | `1000` | Tool outputs longer than this many characters are offloaded to disk. |
| `compact.minor.archive_preview_chars` | int | `1000` | Inline preview length kept in the archive marker (error outputs get a 2× preview). |
| `compact.minor.mid_turn_enabled` | bool | `true` | Allow the archive stage to run in the middle of a turn. |
| `compact.minor.mid_turn_token_threshold` | float | `0.75` | Mid-turn: estimated occupancy at which the archive stage runs. |
| `compact.minor.keep_recent_tool_results` | int | `5` | Mid-turn: the most recent N tool outputs of the running turn are never archived. |

```yaml title="agent.yml"
agent:
  compact:
    major:
      enabled: true
      token_threshold: 0.9
      mid_turn_enabled: true
    minor:
      enabled: true
      keep_recent_user_turns: 4
      archive_threshold: 1000
      archive_preview_chars: 1000
      mid_turn_enabled: true
      mid_turn_token_threshold: 0.75
      keep_recent_tool_results: 5
```

To disable automatic compaction entirely, set both `major.enabled` and `minor.enabled` to `false`. You can still run `/compact` manually even when `major.enabled` is `false`.

## Notes

- The turn-start major trigger reads the **live** token usage of the previous model call (the same figure shown in the CLI status bar). On a brand-new session or right after `resume`, that signal starts at zero, so a major compact won't fire before the first model call. Mid-turn, the first model call of a turn is never compacted; the check starts from the second call, once real usage is available.
- Mid-turn occupancy estimates the newly appended items at roughly four characters per token, and reserves the model's reply budget on top. Dense code or CJK text is under-estimated by that rule; the default thresholds leave room for it.
- Every rewrite cools the provider's prompt cache for one call. This is why the archive stage waits for 75% rather than running continuously.
- Models whose context window is unknown to Datus never trigger compaction automatically; `/compact` still works.
- Minor compact reads its turn-start eligibility from the session's user-turn count, so it keeps working correctly after a resume.
- Archived tool I/O and the full-history JSONL live under the session's data directory; see [Storage](storage.md) for paths.
