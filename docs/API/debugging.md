# API Debugging

This page collects tools for diagnosing a misbehaving API server. There is one so far.

## SIGUSR1 task dump

### When to use it

Sometimes an API request never returns even though the server is still alive: `GET /health` responds, the
process is not pinned at 100% CPU, but a `/chat/stream` call just sits there. This is almost always a
**parked coroutine**, not a blocked event loop — some `await` is waiting on a result that never arrives. A
common cause is a tool that awaits an interaction confirmation (e.g. a permission prompt) that no one can
answer on a non-interactive surface like the REST API.

Because the loop itself keeps running, you can ask the server to walk every live `asyncio` task and print
where each one is currently suspended.

### Triggering the dump

At startup the API server installs a `SIGUSR1` handler. Sending that signal to the server process writes a
snapshot of all live async tasks to the log:

```bash
# Find the server PID (daemon mode writes it to the PID file)
cat ~/.datus/run/datus-agent-api.pid

# Trigger the dump
kill -USR1 <pid>
```

The dump is emitted to the server log at `WARNING` level. In daemon mode that is the daemon log file
(`logs/datus-agent-api.log` by default, or whatever `--daemon-log-file` points at); in foreground mode it
goes to the same place the server logs everything else.

`SIGUSR1` is not used by uvicorn or Datus for anything else, so it is safe to send repeatedly — each signal
just produces a fresh snapshot.

### Reading the dump

Each live task is rendered with two views:

```
=== async task dump: 3 live task(s) ===

--- Task 'Task-42' done=False coro=chat_stream ---
  File ".../datus/api/service.py", line 312, in chat_stream
    async for event in agent.run(...):
  ...
await chain (innermost last):
  File ".../datus/tools/permission/broker.py", line 88, in request
    decision = await self._future
=== end async task dump ===
```

- **`Task.print_stack` output** — the caller chain up to the directly-suspended coroutine. This stops at the
  first `await` and does not descend across `async for` (async-generator) boundaries.
- **`await chain (innermost last)`** — the diagnostically useful view. It follows the
  `cr_await` / `ag_await` / `gi_yieldfrom` links across async-generator and nested-await boundaries to the
  **innermost parked frame** — the actual line where the coroutine is blocked (e.g. `await broker.request(...)`).

The innermost frame of the await chain tells you what the request is really waiting on. In the example above
the request is parked on a permission broker future that will never resolve, confirming the interaction-never-
answered diagnosis.

If a task is awaiting a bare `Future` with no descendable frames, the await chain shows
`<no descendable frames; coroutine may be awaiting a bare Future>` — still useful, since it rules out an
async-generator stall.

### Platform notes

- `SIGUSR1` does not exist on Windows. The handler install is skipped gracefully there (it returns `False`
  and logs at `DEBUG`), and server startup is unaffected. Use the foreground logs or an attached debugger on
  Windows instead.
- The dump never raises: any introspection error is folded into the dumped text, so triggering it can never
  crash the server.
