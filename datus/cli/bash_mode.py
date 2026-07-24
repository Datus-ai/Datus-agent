# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Bash-mode execution for the REPL input bar.

Runs a user-typed shell command through the same :class:`BashTool` +
:class:`PermissionHooks` pipeline the agent uses, then hands the result back
to the REPL, which packs the command + output into a chat turn (see
``manual_exec``). The permission ASK prompt still surfaces through a per-run
:class:`InteractionBroker` (merged in via ``merge_interaction_stream``) and is
answered by the embedded :class:`InteractionApp` wizard — identical to an
agent-run permission prompt — but no tool frame is committed to the
scrollback (the styled bash block does the rendering instead).

Direct Python calls to ``BashTool.bash()`` bypass ``PermissionHooks`` (hooks
only fire through the SDK Runner), so this module re-invokes the real
``on_tool_start`` gate with a minimal context/tool shim — the hook reads only
``context.tool_arguments`` and ``tool.name`` (pinned by unit tests).
"""

import asyncio
import inspect
import json
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional

from pydantic import BaseModel
from rich.console import Console

from datus.cli.execution_state import InteractionBroker, InteractionCancelled, merge_interaction_stream
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI
    from datus.tools.func_tool.bash_tool import BashTool
    from datus.tools.permission.permission_hooks import PermissionHooks

logger = get_logger(__name__)


class _BashContextShim:
    """Minimal stand-in for the SDK ``RunContextWrapper``.

    ``PermissionHooks`` reads only ``context.tool_arguments`` (via
    ``_parse_tool_args``, which accepts a JSON string or a dict).
    """

    def __init__(self, command: str) -> None:
        self.tool_arguments = json.dumps({"command": command})


class _BashToolStub:
    """Minimal stand-in for the SDK ``Tool``: ``on_tool_start`` reads only ``.name``."""

    name = "bash"


class _SqlContextShim:
    """Context shim for the ``execute_sql`` permission gate (``_handle_sql_permission``)."""

    def __init__(self, sql: str) -> None:
        self.tool_arguments = json.dumps({"sql": sql})


class _SqlToolStub:
    name = "execute_sql"


class _ToolContextShim:
    """Generic context shim for the ``!<tool>`` permission gate.

    ``PermissionHooks.on_tool_start`` reads only ``context.tool_arguments`` (a
    JSON string of the call arguments), so a manual tool invocation gets the same
    category / SQL-class / bash-command gating an LLM-driven call would.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        self.tool_arguments = json.dumps(args)


class _ToolStub:
    """Generic ``Tool`` stand-in: ``on_tool_start`` reads only ``.name``."""

    def __init__(self, name: str) -> None:
        self.name = name


async def _apply_plugin_transformers(
    cli: "DatusCLI", tool_name: str, category: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Run plugin tool-transformers over *args* — the plugin arg-rewrite layer.

    Mirrors what the wrapped ``FunctionTool.on_invoke_tool`` does for LLM-driven
    calls (see :mod:`datus.tools.middleware.tool_middleware`), so a manual
    execution honours the same plugin hooks. Returns the possibly-rewritten
    args dict; raises to DENY (fail closed, exactly like the middleware).
    """
    agent_config = getattr(cli, "agent_config", None)
    if agent_config is None or not getattr(agent_config, "plugins_enabled", True):
        return args
    try:
        from datus.plugins.registry import collect_plugin_tool_transformers
        from datus.tools.proxy.proxy_tool import parse_tool_patterns, tool_name_matches
    except Exception as exc:  # pragma: no cover - defensive (optional plugin deps)
        logger.debug("Plugin transformer collection unavailable: %s", exc)
        return args

    active = agent_config.active_plugin_names() if hasattr(agent_config, "active_plugin_names") else None
    by_pattern = collect_plugin_tool_transformers(active)
    if not by_pattern:
        return args

    registry = {tool_name: category}
    for pattern, transformers in by_pattern.items():
        if not tool_name_matches(tool_name, registry, parse_tool_patterns([pattern])):
            continue
        for transformer in transformers:
            result = transformer(tool_name, args, {})
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                name = getattr(transformer, "__name__", repr(transformer))
                raise DatusException(
                    ErrorCode.TOOL_EXECUTION_FAILED,
                    message=f"tool transformer {name} returned {type(result).__name__} instead of an arg dict",
                )
            args = result
    return args


class SqlGateResult(BaseModel):
    """Outcome of the SQL permission/policy gate; ``sql`` is the effective (possibly
    transformer/policy-rewritten) statement to execute."""

    sql: str
    approved: bool = False
    error: Optional[str] = None


class ToolGateResult(BaseModel):
    """Outcome of the generic ``!<tool>`` permission gate."""

    approved: bool = False
    error: Optional[str] = None


class BashModeRun(BaseModel):
    """Outcome of one bash-mode execution, packed into a chat turn by the REPL."""

    command: str
    executed: bool = False
    success: bool = False
    output: str = ""
    error: Optional[str] = None
    duration: float = 0.0


def _make_input_collector(cli: "DatusCLI") -> Callable[[ActionHistory, Console], Optional[List[List[str]]]]:
    """Synchronous INTERACTION collector for the streaming context.

    Mirrors ``ChatCommands._make_input_collector`` minus the escape guard —
    bash mode runs without the termios ESC listener, so there is nothing to
    pause around the wizard.

    Never returns ``None``: a ``None`` makes the renderer skip the
    interaction WITHOUT submitting, which would leave the permission gate's
    ``broker.request()`` awaiting forever (the chat flow can break out via
    its interrupt path; this standalone stream cannot). Any collector
    failure degrades to the ESC/cancel answer ``[[""]]`` — the bash gate
    treats it as a rejection, so errors fail closed instead of hanging.
    """

    def collect(action: ActionHistory, console: Console) -> List[List[str]]:
        cancelled: List[List[str]] = [[""]]
        try:
            from datus.cli.interaction_app import InteractionApp
            from datus.schemas.interaction_event import InteractionEvent

            events = InteractionEvent.from_broker_input(action.input or {})
            if not events:
                return cancelled
            app = InteractionApp(events)
            result = app.run(tui_app=getattr(cli, "tui_app", None))
            answers = getattr(result, "answers", None)
            return answers if answers else cancelled
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error collecting bash permission input: {e}")
            return cancelled

    return collect


def _resolve_permission_components(cli: "DatusCLI"):
    """Resolve (permission_manager, node_name, tool_registry) for the gate.

    Prefers the live chat node's manager/registry so session approvals are
    shared with agent runs; before the first chat turn a CLI-owned manager is
    built from ``agent_config`` with the same inputs the node uses
    (``AgenticNode._setup_permission_manager``) and cached for reuse.
    """
    from datus.tools.registry.tool_registry import ToolRegistry

    node = getattr(getattr(cli, "chat_commands", None), "current_node", None)
    pm = getattr(node, "permission_manager", None) if node is not None else None
    if pm is not None:
        node_name = node.get_node_name()
        registry = getattr(node, "tool_registry", None)
    else:
        node_name = "chat"
        registry = None
        pm = getattr(cli, "_cli_bash_permission_manager", None)
        if pm is None:
            from datus.tools.permission.permission_manager import PermissionManager

            pm = PermissionManager(
                global_config=cli.agent_config.permissions_config,
                active_profile=getattr(cli.agent_config, "active_profile_name", None) or "normal",
                plugin_bash_rules=getattr(cli.agent_config, "plugin_bash_rules", None),
                project_bash_allows=getattr(cli.agent_config, "project_bash_allow", None),
            )
            cli._cli_bash_permission_manager = pm

    # ``on_tool_start`` routes to the bash / SQL gates via the registry's
    # category mapping (``bash -> bash_tools``, ``execute_sql -> db_tools``);
    # guarantee both regardless of node state, otherwise a manual call falls
    # through to the coarse category check (and can wrongly ASK/DENY).
    required = {"bash": "bash_tools", "execute_sql": "db_tools"}
    if registry is None or any(name not in registry for name in required):
        base = registry.to_dict() if registry is not None else {}
        for name, category in required.items():
            base.setdefault(name, category)
        registry = ToolRegistry(base)
    return pm, node_name, registry


def _resolve_bash_tool(cli: "DatusCLI") -> "BashTool":
    """Reuse the chat node's ``BashTool`` (session-dir output archiving) or a cached CLI-owned one."""
    node = getattr(getattr(cli, "chat_commands", None), "current_node", None)
    tool = getattr(node, "bash_tool", None) if node is not None else None
    if tool is not None:
        return tool
    tool = getattr(cli, "_cli_bash_tool", None)
    if tool is None:
        import os

        from datus.tools.func_tool.bash_tool import BashTool

        # ``allowed_patterns=["*"]`` defers all gating to the permission
        # pipeline, which has already run by the time the tool executes —
        # the same posture agentic nodes use.
        tool = BashTool(
            workspace_root=getattr(cli.agent_config, "project_root", None) or os.getcwd(),
            allowed_patterns=["*"],
            identity="cli-bash-mode",
        )
        cli._cli_bash_tool = tool
    return tool


def _build_permission_hooks(cli: "DatusCLI", broker: InteractionBroker) -> "PermissionHooks":
    """Fresh ``PermissionHooks`` wired to this run's broker.

    A fresh instance per run keeps the node's own broker/hook untouched;
    session approvals still persist because they live on the shared
    ``PermissionManager``.
    """
    from datus.tools.permission.bash_classifier import create_bash_classifier
    from datus.tools.permission.permission_hooks import PermissionHooks

    pm, node_name, registry = _resolve_permission_components(cli)
    bash_rules = getattr(cli.agent_config.permissions_config, "bash_commands", None)
    return PermissionHooks(
        broker=broker,
        permission_manager=pm,
        node_name=node_name,
        tool_registry=registry,
        non_interactive=False,
        project_root=getattr(cli.agent_config, "project_root", None),
        config_mutable=bool(getattr(cli.agent_config, "config_mutable", True)),
        bash_classifier=create_bash_classifier(bash_rules, cli.agent_config),
    )


async def _gate_and_execute(
    cli: "DatusCLI",
    run: BashModeRun,
    hooks: "PermissionHooks",
    bash_tool: "BashTool",
) -> AsyncGenerator[ActionHistory, None]:
    """Run the permission gate and, on approval, execute the command.

    Yields NO display actions — the execution result is surfaced by the
    downstream chat turn (the styled bash block), not a tool frame. The gate's
    ASK prompt still flows through the broker as an INTERACTION action, which
    the ``merge_interaction_stream`` consumer in :func:`run_bash_mode_command`
    answers. Written as an async generator so ``ActionBus.merge`` can drive it
    as the primary stream and close the broker when it completes.
    """
    from datus.tools.permission.permission_hooks import PermissionDeniedException

    approved = False
    command = run.command
    try:
        await hooks.on_tool_start(_BashContextShim(command), None, _BashToolStub())
        # Plugin transformers (the arg-rewrite layer) run after the permission
        # gate, matching the LLM call chain (on_tool_start → on_invoke_tool).
        args = await _apply_plugin_transformers(cli, "bash", "bash_tools", {"command": command})
        command = args.get("command", command)
        run.command = command
        approved = True
    except PermissionDeniedException as exc:
        run.error = str(exc)
    except InteractionCancelled:
        run.error = "Permission prompt cancelled"
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Bash permission gate failed: {exc}", exc_info=True)
        run.error = f"Permission check failed: {exc}"

    if approved:
        exec_start = time.monotonic()
        result = await asyncio.to_thread(bash_tool.bash, command)
        run.duration = time.monotonic() - exec_start
        run.executed = True
        run.success = bool(result.success)
        run.output = str(result.result or "")
        run.error = result.error

    # Make this coroutine an async generator without emitting display actions.
    return
    yield  # pragma: no cover - unreachable, marks the function as a generator


def run_bash_mode_command(cli: "DatusCLI", command: str) -> BashModeRun:
    """Execute one bash-mode command through the permission-gated pipeline.

    Called from the REPL dispatch path (TUI worker thread, or the main thread
    in the PromptSession fallback). A plain ``asyncio.run`` drives the gate;
    the permission ASK prompt surfaces as a broker INTERACTION action, merged
    in via :func:`merge_interaction_stream` and answered by the embedded
    wizard (:class:`InteractionApp`) — identical to an agent-run permission
    prompt, but without committing any tool frame to the scrollback. The
    caller packs the resulting :class:`BashModeRun` into a chat turn.
    """
    run = BashModeRun(command=command)

    if not getattr(cli.agent_config, "bash_tool_enabled", True):
        run.error = "Bash mode is disabled (agent.bash.enabled=false)"
        return run
    if getattr(cli.agent_config, "permissions_config", None) is None:
        # Fail closed: without a permission config the gate cannot run, and
        # executing unchecked shell commands would bypass the profile.
        run.error = "Bash mode unavailable: no permission configuration loaded"
        return run

    try:
        broker = InteractionBroker()
        hooks = _build_permission_hooks(cli, broker)
        bash_tool = _resolve_bash_tool(cli)
    except Exception as exc:
        logger.error(f"Failed to prepare bash mode pipeline: {exc}", exc_info=True)
        run.error = f"Bash mode unavailable: {exc}"
        return run

    collector = _make_input_collector(cli)

    async def _drive() -> None:
        # ``reset_queue`` binds the broker queue to THIS loop; the gate runs as
        # the merge's primary stream, and any INTERACTION request is answered
        # here (the wizard runs off-loop via ``to_thread`` so the event loop
        # stays responsive for the pending ``broker.submit``).
        broker.reset_queue()
        async for action in merge_interaction_stream(_gate_and_execute(cli, run, hooks, bash_tool), broker):
            if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                answers = await asyncio.to_thread(collector, action, cli.console)
                await broker.submit(action.action_id, answers)

    try:
        asyncio.run(_drive())
    except Exception as exc:
        logger.error(f"Bash mode gate failed: {exc}", exc_info=True)
        if run.error is None:
            run.error = str(exc)
    return run


# ── SQL permission / policy gate ──────────────────────────────────────────


def _enabled_sql_policy(agent_config):
    """Return the enabled ``SqlPolicyConfig`` on *agent_config*, else ``None``."""
    if agent_config is None:
        return None
    policy = getattr(agent_config, "sql_policy_config", None)
    try:
        from datus.tools.sql_policy import SqlPolicyConfig
    except Exception:  # pragma: no cover - defensive (optional deps)
        return None
    if isinstance(policy, SqlPolicyConfig) and getattr(policy, "enabled", False):
        return policy
    return None


def _enforce_sql_policy_if_read(cli: "DatusCLI", sql: str) -> str:
    """Apply the SQL data-governance policy to a read statement (rewrite/deny),
    matching ``DBFuncTool``'s read path. No-op for writes/DDL (only reads are
    governed).

    Enforcement is resolved independently of the chat node so governance is not
    silently skipped before the first chat turn: the live node's tool is reused
    when present (so its request-scoped principal applies, identical to an agent
    ``execute_sql`` call); otherwise the policy is applied directly off
    ``agent_config`` (the enforcement only needs the policy config + principal,
    not connector / RAG state — avoiding a heavyweight ``DBFuncTool`` build).

    Raises ``DatusException`` when the policy denies the query (same as the tool).
    """
    from datus.utils.constants import SQLType
    from datus.utils.sql_utils import parse_sql_type

    connector = getattr(cli, "db_connector", None)
    dialect = (getattr(connector, "dialect", "") or "") if connector is not None else ""
    if parse_sql_type(sql, dialect) not in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
        return sql

    agent_config = getattr(cli, "agent_config", None)
    datasource = getattr(agent_config, "current_datasource", "") or ""

    node = getattr(getattr(cli, "chat_commands", None), "current_node", None)
    db_func_tool = getattr(node, "db_func_tool", None) if node is not None else None
    if db_func_tool is not None:
        return db_func_tool._enforce_sql_policy(sql, datasource=datasource, dialect=dialect)

    policy = _enabled_sql_policy(agent_config)
    if policy is None:
        return sql
    from datus.tools.sql_policy import load_sql_policy_enforcer

    principal_source = getattr(agent_config, "principal", None)
    principal = dict(principal_source) if isinstance(principal_source, dict) else {}
    enforced = load_sql_policy_enforcer(policy).enforce_read(
        sql, datasource=datasource, dialect=dialect, principal=principal
    )
    if not enforced.allowed:
        raise DatusException(
            ErrorCode.TOOL_INVALID_INPUT,
            message=enforced.reason or "SQL policy denied the query",
        )
    return sql if enforced.sql is None else enforced.sql


async def _sql_gate_stream(
    result: SqlGateResult, hooks: "PermissionHooks", cli: "DatusCLI"
) -> AsyncGenerator[ActionHistory, None]:
    """Run the SQL permission gate + plugin transformers + sql_policy.

    Fires the same enforcement an LLM ``execute_sql`` call gets: ``on_tool_start``
    → ``_handle_sql_permission`` (read auto-allow / write·DDL ASK), then plugin
    transformers, then ``_enforce_sql_policy``. Sets ``result.approved`` and the
    effective ``result.sql``; yields nothing (execution + rendering happen in
    the REPL). Written as an async generator so ``ActionBus.merge`` can drive it.
    """
    from datus.tools.permission.permission_hooks import PermissionDeniedException

    try:
        await hooks.on_tool_start(_SqlContextShim(result.sql), None, _SqlToolStub())
        args = await _apply_plugin_transformers(cli, "execute_sql", "db_tools", {"sql": result.sql})
        sql = args.get("sql", result.sql)
        sql = await asyncio.to_thread(_enforce_sql_policy_if_read, cli, sql)
        result.sql = sql
        result.approved = True
    except PermissionDeniedException as exc:
        result.error = str(exc)
    except InteractionCancelled:
        result.error = "Permission prompt cancelled"
    except Exception as exc:
        logger.error(f"SQL permission gate failed: {exc}", exc_info=True)
        result.error = str(exc)

    return
    yield  # pragma: no cover - unreachable, marks the function as a generator


def run_sql_gate(cli: "DatusCLI", sql: str) -> SqlGateResult:
    """Gate a manual SQL statement through the same hooks an LLM call uses.

    Returns a :class:`SqlGateResult` with ``approved`` and the effective
    (transformer/policy-rewritten) ``sql``. Reads auto-allow with no prompt,
    writes/DDL prompt via the embedded wizard. Fails closed (``approved=False``
    with an actionable error) when no permission config is loaded or the hook
    pipeline cannot be built — an ungoverned write/DDL must never slip through,
    matching bash mode's posture.
    """
    result = SqlGateResult(sql=sql)
    if getattr(cli.agent_config, "permissions_config", None) is None:
        result.error = "SQL mode unavailable: no permission configuration loaded"
        return result

    try:
        broker = InteractionBroker()
        hooks = _build_permission_hooks(cli, broker)
    except Exception as exc:
        logger.error(f"Failed to prepare SQL gate: {exc}", exc_info=True)
        result.error = f"SQL gate unavailable: {exc}"
        return result

    collector = _make_input_collector(cli)

    async def _drive() -> None:
        broker.reset_queue()
        async for action in merge_interaction_stream(_sql_gate_stream(result, hooks, cli), broker):
            if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                answers = await asyncio.to_thread(collector, action, cli.console)
                await broker.submit(action.action_id, answers)

    try:
        asyncio.run(_drive())
    except Exception as exc:
        logger.error(f"SQL gate failed: {exc}", exc_info=True)
        if result.error is None:
            result.error = str(exc)
    return result


# ── Generic tool permission gate (``!<tool> [args...]``) ───────────────────


async def _tool_gate_stream(
    result: ToolGateResult,
    hooks: "PermissionHooks",
    tool_name: str,
    args: Dict[str, Any],
) -> AsyncGenerator[ActionHistory, None]:
    """Run the permission gate for a manual ``!<tool>`` call.

    Fires the same ``on_tool_start`` gate an LLM-driven call gets — the hook only
    reads ``tool.name`` + ``context.tool_arguments``, so read tools auto-allow and
    write/``execute_sql``/``bash`` calls hit their fine-grained confirmation. Sets
    ``result.approved``; yields nothing (the tool executes + renders in the
    caller). Plugin transformers are NOT applied here: unlike the SQL/bash gates
    (which bypass the ``FunctionTool`` wrapper), ``!<tool>`` invokes the real
    ``FunctionTool.on_invoke_tool``, which already runs any transformer middleware.
    """
    from datus.tools.permission.permission_hooks import PermissionDeniedException

    try:
        await hooks.on_tool_start(_ToolContextShim(args), None, _ToolStub(tool_name))
        result.approved = True
    except PermissionDeniedException as exc:
        result.error = str(exc)
    except InteractionCancelled:
        result.error = "Permission prompt cancelled"
    except Exception as exc:
        logger.error(f"Tool permission gate failed for '{tool_name}': {exc}", exc_info=True)
        result.error = str(exc)

    return
    yield  # pragma: no cover - unreachable, marks the function as a generator


def run_tool_gate(cli: "DatusCLI", tool_name: str, args: Dict[str, Any]) -> ToolGateResult:
    """Gate a manual ``!<tool>`` invocation through the same hooks an LLM call uses.

    Returns a :class:`ToolGateResult` with ``approved`` set. Read-only tools
    auto-allow with no prompt; write / ``execute_sql`` / ``bash`` tools prompt via
    the embedded wizard. Fails closed (``approved=False`` with an actionable
    error) when no permission config is loaded or the hook pipeline cannot be
    built, matching the SQL/bash gates' posture.
    """
    result = ToolGateResult()
    if getattr(cli.agent_config, "permissions_config", None) is None:
        result.error = "Tool execution unavailable: no permission configuration loaded"
        return result

    try:
        broker = InteractionBroker()
        hooks = _build_permission_hooks(cli, broker)
    except Exception as exc:
        logger.error(f"Failed to prepare tool gate: {exc}", exc_info=True)
        result.error = f"Tool gate unavailable: {exc}"
        return result

    collector = _make_input_collector(cli)

    async def _drive() -> None:
        broker.reset_queue()
        async for action in merge_interaction_stream(_tool_gate_stream(result, hooks, tool_name, args), broker):
            if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                answers = await asyncio.to_thread(collector, action, cli.console)
                await broker.submit(action.action_id, answers)

    try:
        asyncio.run(_drive())
    except Exception as exc:
        logger.error(f"Tool gate failed for '{tool_name}': {exc}", exc_info=True)
        if result.error is None:
            result.error = str(exc)
    return result


# ── Live execution frame (blinking ``running`` → result block) ─────────────


def _trim_denial(message: str) -> str:
    """Drop the model-facing boilerplate from a permission-denied message."""
    message = (message or "").removeprefix("PERMISSION_DENIED: ")
    cut = message.find("STOP retrying")
    if cut > 0:
        message = message[:cut].rstrip(" —-")
    return message.strip()


def _run_with_live_frame(cli: "DatusCLI", kind: str, command: str, work: Callable[[], tuple]):
    """Render a live ``manual_exec`` frame around a blocking gate+execute call.

    Shows the mode-coloured command line with a ``· running Ns`` indicator that
    ticks while ``work()`` runs (the streaming context's refresh loop repaints
    it), then replaces it with the bordered result block. ``work`` returns
    ``(payload, dispatch)``: ``payload`` renders as the terminal block (``None``
    → empty terminal, e.g. an infra failure that already printed), ``dispatch``
    tells the caller whether to feed the result to the model.
    """
    import uuid
    from datetime import datetime

    from datus.cli.action_display.display import ActionHistoryDisplay
    from datus.cli.manual_exec import MANUAL_EXEC_ACTION_TYPE

    call_id = f"cli_{kind}_{uuid.uuid4().hex[:8]}"
    start = datetime.now()
    actions: List[ActionHistory] = [
        ActionHistory(
            action_id=call_id,
            role=ActionRole.TOOL,
            action_type=MANUAL_EXEC_ACTION_TYPE,
            messages=f"{kind}> {command}",
            input={"kind": kind, "command": command},
            output=None,
            status=ActionStatus.PROCESSING,
            start_time=start,
        )
    ]
    display = ActionHistoryDisplay(cli.console, live_state=getattr(cli, "live_state", None))
    ctx = display.display_streaming_actions(actions)

    payload = None
    dispatch = False
    with ctx:
        try:
            payload, dispatch = work()
        finally:
            status = ActionStatus.SUCCESS if (payload and payload.get("success")) else ActionStatus.FAILED
            actions.append(
                ActionHistory(
                    action_id=f"complete_{call_id}",
                    role=ActionRole.TOOL,
                    action_type=MANUAL_EXEC_ACTION_TYPE,
                    messages=f"{kind}> {command}",
                    input={"kind": kind, "command": command},
                    output={"payload": payload},
                    status=status,
                    start_time=start,
                    end_time=datetime.now(),
                )
            )
    return payload, dispatch


def run_manual_bash_live(cli: "DatusCLI", command: str) -> tuple:
    """Run a bash-mode command with a live frame; return ``(payload, dispatch)``.

    Reuses :func:`run_bash_mode_command` for the permission gate + execution,
    wrapped in the live ``manual_exec`` frame. Denied/failed-to-start commands
    render their reason and are not dispatched to the model.
    """
    from datus.cli.manual_exec import build_bash_payload

    def _work():
        run = run_bash_mode_command(cli, command)
        if not run.executed:
            if run.error:
                return build_bash_payload(command, False, "", _trim_denial(run.error), 0.0), False
            return None, False
        payload = build_bash_payload(run.command, run.success, run.output, run.error, run.duration)
        return payload, True

    return _run_with_live_frame(cli, "bash", command, _work)


def run_manual_sql_live(cli: "DatusCLI", command: str, execute_fn: Callable[[str], Optional[dict]]) -> tuple:
    """Run a SQL-mode statement with a live frame; return ``(payload, dispatch)``.

    Reuses :func:`run_sql_gate` for permission + transformers + sql_policy, then
    calls ``execute_fn`` (the REPL's connector-backed executor) with the
    effective SQL. Denials render their reason and are not dispatched; an infra
    failure (``execute_fn`` → ``None``) yields an empty terminal and no dispatch.
    """
    from datus.cli.manual_exec import build_sql_error_payload

    def _work():
        gate = run_sql_gate(cli, command)
        if not gate.approved:
            return build_sql_error_payload(command, _trim_denial(gate.error or "SQL not permitted")), False
        payload = execute_fn(gate.sql)
        if payload is None:
            return None, False
        return payload, True

    return _run_with_live_frame(cli, "sql", command, _work)


def run_manual_tool_live(
    cli: "DatusCLI",
    tool_name: str,
    command: str,
    args: Dict[str, Any],
    invoke_fn: Callable[[], tuple],
) -> tuple:
    """Run a ``!<tool>`` call with a live frame; return ``(payload, dispatch)``.

    Reuses :func:`run_tool_gate` for the permission gate (identical to an LLM
    ``on_tool_start``), then calls ``invoke_fn`` — a ``() -> (success, output)``
    closure that performs ``on_invoke_tool`` and renders the result to text.
    Denials render their reason and are not dispatched; a successful or
    tool-failed execution (and an unexpected invocation error) is packed into a
    ``tool`` payload and dispatched to the model so the run enters the
    conversation and triggers a reply.
    """
    from datus.cli.manual_exec import build_tool_payload

    def _work():
        gate = run_tool_gate(cli, tool_name, args)
        if not gate.approved:
            denial = _trim_denial(gate.error or f"Tool '{tool_name}' not permitted")
            return build_tool_payload(command, False, "", denial, 0.0), False
        start = time.monotonic()
        try:
            success, output = invoke_fn()
        except Exception as exc:  # noqa: BLE001 - surface the failure to the model
            logger.exception("Manual tool invocation failed for '%s'", tool_name)
            return build_tool_payload(command, False, "", str(exc), time.monotonic() - start), True
        duration = time.monotonic() - start
        return build_tool_payload(command, success, output, None if success else output, duration), True

    return _run_with_live_frame(cli, "tool", command, _work)
