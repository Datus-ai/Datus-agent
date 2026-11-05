# Host chat execution extensions

Hosts can keep their own authorization and evidence rules while sharing Agent
chat history, SSE, cancellation, usage accounting and policy execution.
These are trusted Python extension points, not fields accepted from chat clients.

## Construct the service once

`DatusService(..., task_manager_factory=...)` calls a `ChatTaskManagerFactory`
with `default_source`, `default_interactive` and `stream_thinking`. It must return
a fresh manager for that project. The lazy ChatService receives that same manager.
Never assign a manager to a service after publication or copy a running manager.

For API hosts, pass the factory to `init_deps`. Background jobs should use
`create_datus_service(config, project_id)` inside the same service-cache factory;
this uses the startup defaults and extension. The caller owns caching/shutdown.
Calling `init_deps` without a factory restores ordinary Agent construction.

A manager can accept `node_factory: ChatNodeFactory`. It runs on a worker thread
with a per-turn configuration clone, user scope, session ID and execution mode.
Return a node for the host's sessions, or `None` for standard node dispatch.
An exception fails the turn; it never falls back to an unrestricted node.
Override the public `start_chat` only when the host must constrain continuation
parameters before dispatch. Keep authorization in that host layer.

## Accept a durable opening

Use `manager.session_manager(config, user_id)` to obtain the user-scoped directory.
`AcceptedTurnStore(directory, session_id, namespace=...)` owns the normal SQLite
history schema. The host maps its stable request ID to a stable session ID and
chooses a metadata namespace. No separate message database is created.

```python
store = AcceptedTurnStore(manager.session_manager(config, user_id).session_dir,
                          session_id, namespace="host_analysis_v1")
receipt = store.accept(request_id, opening, metadata={"evidence": evidence},
                       validate_existing=validate_immutable_evidence)
```

`accept` atomically commits the opening, its turn structure, title and receipt.
Concurrent/retried acceptance returns the same session/message/execution IDs.
It rejects a different request ID, opening or an existing unrelated chat.
`validate_existing` runs synchronously under the transaction lock; it must
validate immutable host metadata without I/O. Older host receipts without an
opening digest rely on this validator, preserving pre-extension sessions.

Read/update the receipt with `read`/`update`. Use `execution_lock` for the entire
execution, including terminal status persistence. This is a nonblocking POSIX
process fence on a separate file: SQLite history remains readable while running.
The host rechecks authorization and decides recovery: pending acceptance may be
dispatched after a restart, but an interrupted model run must not be silently
replayed. This interface does not promise exactly-once external model side effects.
The host is responsible for retaining the session directory for the receipt's life.

For the opening execution, install `AcceptedOpeningSession` on the host node and
pass `accepted_opening=True` to `start_chat`. The session avoids feeding/storing
the already accepted input twice, including SDK history rewrites. The manager
suppresses the duplicate opening SSE event only; normal history already contains
it. Use ordinary sessions and the default flag on subsequent turns.

## Execute metrics under policy

`DosiRuntime.from_agent_config(config)` constructs the same datasource/profile
mapping without opening RAG/navigation tools. `PolicyMetricExecutor(config,
runtime)` provides `query_metrics` and `attribute`, each taking fresh trusted
`policy_context`. Hosts supply business filters and authorized windows/dimensions;
metric policy precedes compilation, and SQL policy precedes reads. Query results
pass through the result policy and omit internal SQL metadata.

Native attribution currently has no per-query Python callback. The executor's
compatibility adapter applies row policies to every dataset in an in-memory
model, then lets Dosi perform the full analysis. Shared model files are unchanged.
Only the known SQL-policy row-filter and metric-row-filter forms are supported;
unknown plugins, column/result policies and ontology attribution fail closed.
`validate_attribution_support` provides this check before offering analysis.
This is not a generic callback for arbitrary native queries. Future native engine
hooks can replace the adapter behind this public interface.

Attribution results retain engine metadata for trusted callers. A host exposing
them to a model or user must project allowed business evidence and redact SQL,
policy details and exceptions. The host must reauthorize every tool call and
refresh its policy context; the executor does not own user identity or token refresh.
