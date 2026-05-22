# Session Compact

Datus has two compact strategies that work together to keep the LLM session
within the model's context window:

| Mode  | Driver        | When                                          | Cost                              | Result                                                                |
| ----- | ------------- | --------------------------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| Major | LLM           | History near context limit (default ≥ 90%)    | One LLM round (`max_turns=1`)     | Whole session replaced by a structured 10-section summary             |
| Minor | Rules only    | Rolling window matures + cache TTL / token %  | A few filesystem writes           | Long tool I/O offloaded to disk; original short messages untouched    |

Both are orchestrated by `AgenticNode.compact()` — the single public entry
point exposed to the CLI, the SDK `RunHooks`, and the model overflow
fallback.

## Configuration

Add an `agent.compact` block to your `agent.yml`. All fields are optional;
the defaults below are applied when an entry is missing.

```yaml
agent:
  compact:
    major:
      enabled: true
      token_threshold: 0.9           # >= 90% of context_length → run major
    minor:
      enabled: true
      # Rolling window (tool-call rounds)
      window_size: 10                # Keep the last N rounds; compact [_compacted_until, T-N)
      # Trigger conditions (OR)
      cache_ttl_seconds: 270         # Anthropic ephemeral cache TTL ~5 min; act below that
      token_threshold: 0.6           # History tokens / context_length above this → trigger
      # Archive behavior
      archive_threshold: 1000        # Args/output >= this many characters land on disk
      archive_preview_chars: 200     # Inline preview kept in the session pointer
      archive_error_preview_chars: 1000  # Error outputs get a longer preview
      archive_dir: null              # null → path_manager.session_data_dir(session_id)
```

### Parameter notes

* `major.token_threshold` is also the hard upper bound used by the auto
  dispatcher; once history crosses it the next opportunity (CLI, user turn,
  or tool-end hook) forces a major compact, regardless of any minor signal.
* `minor.window_size` controls **two** things at once: how many recent rounds
  are exempt from compaction, and the minimum amount of new history that
  must accumulate before the next minor pass fires. Picking a single
  parameter for both keeps the rolling window self-consistent.
* `minor.cache_ttl_seconds` should sit below the model provider's prompt
  cache TTL (Anthropic ephemeral ≈ 5 min). When the last LLM call is older
  than this, the prefix cache is already invalidated, so rewriting history
  costs nothing — a free window to compact.
* `minor.token_threshold` is the back-pressure trigger: even if cache is
  still hot, when history takes more than this fraction of context we
  compact anyway to keep the major threshold out of reach.
* `minor.archive_threshold` is the **only** size knob for what lands on disk;
  there is no per-tool whitelist. Long write_file content, large read_query
  outputs, sub-agent task results — all go through the same threshold check.

## When does compact fire?

There are four entry points, all funneled through `AgenticNode.compact()`:

1. **CLI `/compact`** — runs `compact(mode="major")` synchronously. Minor is
   not exposed because it is condition-driven and a manual minor call would
   either be a no-op or duplicate work already scheduled by the hook.
2. **Before each user turn** — `execute_stream()` calls
   `compact(mode="auto", reason="pre_user_turn")` at the top of every
   interactive turn. Picks major when history is above
   `major.token_threshold`, minor when the rolling window says to, noop
   otherwise.
3. **Inside the tool-call loop** — `CompactHook` (an Agents SDK
   `RunHooks`) fires `on_tool_end` after every successful tool call. It
   increments `_tool_call_turn_count`, runs the same auto dispatcher, and:

   * if major is selected, **blocks** the run loop until the summary is
     persisted (otherwise the next turn would still overflow);
   * if minor is selected, schedules the pass with `asyncio.create_task`
     so the SDK loop doesn't stall on disk I/O.
4. **Model overflow fallback** — when `openai_compatible` catches
   `MaxTurnsExceeded`, if the caller provided a `compact_callback` (the
   node's `compact` method) it invokes `compact_callback(mode="major", reason="overflow")`
   and retries the run **once**. A single-retry guard prevents a degenerate
   loop if the compact itself fails to free enough space.

The dispatcher (`_decide_compact_mode`) uses the priority order:

1. `_history_token_ratio_sync() >= major.token_threshold` → **major**
2. `_should_minor_compact()` returns a reason → **minor**
3. otherwise → **noop**

`_should_minor_compact()` requires the rolling-window precondition plus an
OR-trigger:

* `cache_expired` — `_seconds_since_last_llm_call() > cache_ttl_seconds`
* `token_ratio` — `_history_token_ratio_sync() >= token_threshold`

The precondition `T - _compacted_until_turn >= 2 * window_size` guarantees
there is at least one round of "matured" new history to compact, and is
what makes the rolling window naturally rate-limit itself.

## How does compact run?

### Major

```
_major_compact(reason)
├─ ① _dump_session_history_jsonl()        ── writes the entire session to
│                                              {sessions_dir}/{sid}/data/history_{ts}.jsonl
├─ ② render_major_compact_prompt(...)     ── j2 template `compact_major_1.0`
│                                              fills in node_role, history path,
│                                              and archive_dir for the model.
├─ ③ model.generate_with_tools(prompt, instruction=system_prompt, max_turns=1)
│                                          ── LLM produces a structured 10-section
│                                              markdown summary.
├─ ④ session.clear_session()
└─ ⑤ session.add_items([continuation_user_message])
                                           ── single user message wrapping the
                                              summary + recovery pointers.
```

Resulting session is exactly one user message; the next turn sees the
summary, can `read_file(<history_jsonl>)` to fetch any original item, and
can `read_file(<archive_dir>/...)` to fetch archived tool I/O.

`_compacted_until` is reset to 0; the rolling window starts fresh.

### Minor

```
_minor_compact(reason)
├─ items = await session.get_items()
├─ (lo, hi) = _resolve_window_bounds(items)
│                                          ── lo = _compacted_until,
│                                              hi positions the latest
│                                              window_size function_calls
│                                              in items[hi:].
├─ for idx, item in enumerate(items[lo:hi]):
│       rewritten[idx] = maybe_truncate_item(item, archive, threshold, idx)
│                                          ── archives arguments / output
│                                              that exceed archive_threshold.
├─ if no item changed → advance _compacted_until and return
├─ session.clear_session()
├─ session.add_items(rewritten)
└─ _compacted_until = hi
```

#### What gets archived

For each item in the window:

* `function_call.arguments` whose serialized length ≥ `archive_threshold` →
  whole text persisted to `{idx:06d}_args_{hash8}.json`; the field is
  replaced with a pointer dict.
* `function_call_output.output` whose length ≥ `archive_threshold` → whole
  text persisted to `{idx:06d}_output_{hash8}.txt`; same pointer
  replacement.
* Reasoning / message items pass through unchanged.

#### Pointer format

```json
{
  "_archived": true,
  "path": "/Users/.../sessions/<project>/<sid>/data/000042_args_5c0e0ea4.json",
  "original_len": 12345,
  "content_hash": "sha256:5c0e0ea4...",
  "preview": "...first 200 chars of the original...",
  "recovery_hint": "read_file('/.../000042_args_5c0e0ea4.json') to load the original args"
}
```

Errors detected via `FuncToolResult.success == 0` (or, when the output is
not valid JSON, a string match on `\"success\": 0` / `\"error\":` /
`Traceback`) widen the preview to `archive_error_preview_chars` so the LLM
can read the error inline.

## Storage layout

Everything compact-related lives next to the session db:

```
~/.datus/sessions/<project_name>/<session_id>.db        # session SQLite
~/.datus/sessions/<project_name>/<session_id>/data/     # compact data
    ├── history_20260521T180153Z.jsonl                  # major dump
    ├── 000042_args_5c0e0ea4.json                       # minor archive
    └── 000048_output_a1b2c3d4.txt
```

Paths are resolved through `path_manager.session_data_dir(session_id)`;
nothing in compact code hardcodes the layout. `session_manager.delete_session`
rmtrees `~/.datus/sessions/<project_name>/<session_id>/` when the session
is removed.

## Filesystem permissions

For the recovery flow to work, the LLM needs `read_file` access to the
archive directory **without** a permission prompt. `fs_path_policy.classify_path`
accepts a `session_data_dir` argument; paths under the current session's
data dir resolve to `WHITELIST` (read-only). Other sessions' data
directories stay `EXTERNAL`, so a session cannot read another's archive
even if the path is guessable.

The wiring happens in two places:

* `AgenticNode._make_filesystem_func_tool()` injects `session_data_dir =
  path_manager.session_data_dir(self.session_id)` into `FilesystemFuncTool`.
* `AgenticNode._make_filesystem_policy()` injects the same value into the
  `FilesystemPolicy` used by `PermissionHooks`.

## Observability

The major and minor compact passes both log a single line at `INFO` level
when they finish:

```
INFO  Starting major compact for session <sid> (reason=<reason>)
INFO  Major compact complete: <N> chars summary, <K> output tokens, history=<path>
INFO  Minor compact done: window=[<lo>,<hi>) archived=<n> reason=<reason>
```

To audit on-disk archives directly:

```bash
ls -l ~/.datus/sessions/<project>/<session_id>/data/
cat  ~/.datus/sessions/<project>/<session_id>/data/history_*.jsonl | head
```

`session_id` is shown in the CLI status bar and in the `/compact` command
output.

## Disabling / tuning

* Disable minor compact entirely: `agent.compact.minor.enabled: false`.
  Major still fires at the 90% threshold (and via CLI `/compact`).
* Disable major: `agent.compact.major.enabled: false`. Overflow fallback
  still runs because it explicitly requests `mode="major"`; if you want
  to turn that off too, set the threshold to `1.0` and rely on minor +
  overflow.
* Trade off cache hit rate vs. token pressure by adjusting
  `minor.token_threshold`. Lower values trigger minor sooner (more disk
  archives, more frequent cache invalidation from the rewritten window
  position); higher values let history grow longer (better cache reuse,
  more risk of hitting the major threshold).
* Trade off archive granularity vs. archive count with
  `minor.archive_threshold`. A higher threshold keeps more content
  inline (less disk traffic, larger session db). A lower threshold sends
  more content to disk (smaller session db, more `read_file` calls when
  the LLM needs to recover detail).
