# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Invocation-local metadata cache, tool restrictions and execution outcome for /init.

Published Markdown and lineage are the only init artifacts reused by later
calls. No task directory, execution identifier or recovery queue is persisted.
"""

from __future__ import annotations

import asyncio
import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator

from agents import FunctionTool

from datus.tools.func_tool.base import parse_tool_args, write_back_tool_args
from datus.utils.sql_utils import parse_table_name_parts
from datus.utils.tool_result_store import ToolResultStore, parse_result

_READS = {
    "load_skill",
    "read_file",
    "glob",
    "grep",
    "list_databases",
    "list_schemas",
    "list_tables",
    "describe_table",
    "get_table_ddl",
    "search_table",
    "query_lineage",
    "read_tool_result",
}
_WRITES = {"write_file", "edit_file"}
_CACHEABLE = {"list_databases", "list_schemas", "list_tables", "describe_table", "get_table_ddl", "search_table"}
_TODOS = {"todo_list", "todo_read", "todo_write", "todo_update"}


def _error(message: str) -> dict:
    return {"success": 0, "error": message, "result": None}


def _without_samples(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_samples(item) for key, item in value.items() if key not in {"sample_rows", "sample_data"}}
    if isinstance(value, list):
        return [_without_samples(item) for item in value]
    return value


class InitContext:
    """Protect one ordinary chat invocation without tracking business completion."""

    def __init__(self, root: Path, binding: dict, scope: str = ""):
        self.root = root.expanduser().resolve()
        self.binding = binding
        self.state = {
            "scope": scope,
            "outputs": [],
            "status": "running",
            "stop_reason": "",
        }
        self.calls = 0
        self._cache: dict[str, Any] = {}
        self._serial = asyncio.Lock()
        self._inflight: set[asyncio.Task] = set()
        self._closed = False
        self._returned_normally = False

    @contextmanager
    def activate(self, node):
        """Temporarily constrain this chat node without replacing its session."""
        missing = object()
        fields = (
            "tools",
            "mcp_servers",
            "_builtin_web_tools",
            "_init_context",
            "_tool_transformers_applied",
        )
        saved = {name: getattr(node, name, missing) for name in fields}
        try:
            node._init_context = self
            node.tools = list(node.tools or [])
            if node.db_func_tool is not None and not any(t.name == "get_table_ddl" for t in node.tools):
                node.tools.append(node.db_func_tool.to_function_tool(node.db_func_tool.get_table_ddl))
            node.mcp_servers = {}
            node._builtin_web_tools = None
            yield
        finally:
            for name, value in saved.items():
                if value is missing:
                    if hasattr(node, name):
                        delattr(node, name)
                else:
                    setattr(node, name, value)

    def user_context(self) -> list[dict]:
        return self.state.get("user_instructions", [])

    def record_user_updates(self, messages: list[str]) -> None:
        updates = [{"source": "user_update", "text": message} for message in messages if message.strip()]
        if updates:
            self.state.setdefault("user_instructions", []).extend(updates)

    def prompt_context(self, store: ToolResultStore | None) -> str:
        """Keep binding and user instructions recoverable, without a second plan.

        Clarification answers are tool results, so generic retention of user
        messages alone is insufficient. Reuse the session result archive for
        bounded recovery when these instructions grow; never truncate them.
        """
        context = {
            "binding": self.binding,
            "request": {"success": 1, "result": self.state["scope"]},
            "user_clarifications": {"success": 1, "result": list(reversed(self.user_context()))},
            "stop_reason": self.state["stop_reason"],
        }
        if store is not None:
            for field in ("request", "user_clarifications"):
                context[field] = store.project(f"init-{field}", context[field])
        return (
            "\n<init_context>\n" + json.dumps(context, ensure_ascii=False) + "\n</init_context>\n"
            "This /init runs in the current chat loop. Choose useful business topics and continue here. "
            "Use todo_list/read/write/update for work steps; they do not require plan approval. "
            "User clarifications are newest first; later corrections win. Use read_tool_result for "
            "archived instructions and results when needed. "
            "Use metadata and project sources to resolve structural questions. Do not query business rows. "
            "Read existing knowledge before incremental edits. Keep unsupported interpretations in "
            "knowledge/pending-questions.md; they are not confirmed facts. "
            "After compaction, recover the goal, completed work, remaining work and next step from "
            "the summary, todos and relevant knowledge. Do not reload entire archived results. "
            "End with findings, changed files, unresolved questions and unfinished scope. "
            "A structure lookup or a normal return does not establish initialization completeness."
        )

    def finish_execution(self, result, *, interrupted: bool = False, pending_input: bool = False):
        """Record whether execution returned, never whether business work is complete."""
        if interrupted:
            self._stop("Init interrupted; already published knowledge is preserved")
        elif getattr(result, "success", None) is not True:
            self._stop(f"Init chat did not complete successfully: {getattr(result, 'error', '') or 'No final result'}")
        else:
            self._returned_normally = True
            if pending_input:
                self._stop("User input remains unprocessed in the chat queue; continue on the next user turn")
        self._finish()

    def _path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Init evidence and outputs must be inside the project")
        return path

    def _output_path(self, name: str) -> Path:
        path = self._path(name)
        relative = path.relative_to(self.root)
        if relative != Path("AGENTS.md") and not (
            relative.parts and relative.parts[0] == "knowledge" and path.suffix == ".md"
        ):
            raise ValueError("Init writes only AGENTS.md and knowledge/*.md")
        return path

    def _stop(self, reason: str) -> dict:
        if not self.state["stop_reason"]:
            self.state["stop_reason"] = reason
        self.state["status"] = "stopped"
        return _error(
            self.state["stop_reason"] + ". Report the remaining scope; already published knowledge is preserved."
        )

    def wrap_tools(self, tools: list) -> list:
        """Keep project readers, metadata tools and knowledge writers for this run."""
        wrapped = []
        for tool in tools:
            if not isinstance(tool, FunctionTool) or tool.name not in _READS | _WRITES | _TODOS | {
                "upsert_lineage",
                "ask_user",
            }:
                continue
            if getattr(tool, "_init_context", None) is self:
                wrapped.append(tool)
                continue
            original = tool.on_invoke_tool

            async def invoke(
                context, arguments, *, _name=tool.name, _original=original, _schema=tool.params_json_schema
            ):
                _, canonical, error = parse_tool_args(arguments, tool_name=_name, params_schema=_schema)
                write_back_tool_args(context, canonical)
                if error:
                    return _error(error)
                return await self.invoke(_name, _original, context, canonical)

            wrapped_tool = replace(tool, on_invoke_tool=invoke)
            wrapped_tool._init_context = self
            wrapped_tool._init_original_tool = tool
            wrapped.append(wrapped_tool)
        return wrapped

    async def invoke(self, name: str, original, context, arguments: str):
        async with self._serial:
            return await self._invoke(name, original, context, arguments)

    async def _invoke(self, name: str, original, context, arguments: str):
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict):
                raise ValueError("Expected a JSON object")
            if name in {"describe_table", "get_table_ddl"} and (
                not isinstance(args.get("table_name"), str) or not args["table_name"].strip()
            ):
                raise ValueError("table_name must be a nonempty string")
            resolved = self._metadata_arguments(args) if name in {"describe_table", "get_table_ddl"} else args
            for field in ("datasource", "catalog", "database", "schema_name"):
                if args.get(field) is not None and not isinstance(args.get(field, ""), str):
                    raise ValueError(f"{field} must be a string")
                expected = self.binding.get(field, "")
                if expected and resolved.get(field) and resolved[field] != expected:
                    raise ValueError(f"Keep {field} within this init run's database context")
            if name in _WRITES:
                self._output_path(args.get("path", ""))
            if self.state["stop_reason"]:
                return _error(self.state["stop_reason"] + ". Report unfinished work and end this run.")
        except Exception as exc:
            return _error(str(exc))
        if name == "ask_user":
            try:
                result = parse_result(await self._tracked_call(original, context, arguments))
                answers = parse_result(result.get("result")) if isinstance(result, dict) else None
                if not (
                    isinstance(result, dict)
                    and result.get("success", 1) == 1
                    and isinstance(answers, list)
                    and answers
                    and all(isinstance(a, dict) and a.get("answer") for a in answers)
                ):
                    return self._stop("Initialization clarification was cancelled or left unanswered")
                self.state.setdefault("user_instructions", []).extend(
                    {**answer, "source": "scope_answer"} for answer in answers
                )
                return self._register_result(context, result)
            except TimeoutError:
                return self._stop("Init interaction timed out; the operation may still be running")
        key = name + ":" + json.dumps(resolved, sort_keys=True)
        if key in self._cache:
            result = copy.deepcopy(self._cache[key])
            return self._register_result(context, result)
        if self.state.get("stop_reason"):
            return _error(self.state["stop_reason"] + ". Report unfinished work and finish this invocation.")
        self.calls += 1
        try:
            result = parse_result(await self._tracked_call(original, context, arguments))
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json")
            if name == "search_table":
                result = _without_samples(result)
            success = isinstance(result, dict) and result.get("success", 1) == 1
            if success:
                if name in _WRITES:
                    output = self._output_path(args["path"]).relative_to(self.root).as_posix()
                    if output not in self.state["outputs"]:
                        self.state["outputs"].append(output)
            if success and name in _CACHEABLE:
                self._cache[key] = copy.deepcopy(result)
            return self._register_result(context, result)
        except TimeoutError:
            # A timed-out driver thread may still run. Keep the lease until it
            # exits and stop further exploration rather than pile up queries.
            return self._stop("Init tool timed out; the database operation may still be running")
        except Exception as exc:
            return _error(f"{name} failed: {str(exc)[:1200]}")

    def _metadata_arguments(self, args: dict) -> dict:
        """Resolve identifiers with the same dialect parser used by DB tools.

        Validate and cache the effective target, but pass the original spelling
        to the tool so quoted identifiers containing dots are not reparsed.
        """
        dialect = self.binding.get("dialect")
        if not dialect:
            return args
        parts = parse_table_name_parts(args["table_name"], dialect)
        resolved = dict(args)
        for field, part in (
            ("catalog", "catalog_name"),
            ("database", "database_name"),
            ("schema_name", "schema_name"),
            ("table_name", "table_name"),
        ):
            resolved[field] = parts.get(part) or args.get(field) or self.binding.get(field, "")
        return resolved

    def _task_done(self, task: asyncio.Task):
        self._inflight.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve exceptions after a caller timed out.
        if self._closed and not self._inflight:
            self._finish()

    async def _tracked_call(self, original, context, arguments: str):
        from datus.utils.tool_result_store import ACTIVE_RESULT_STORE

        # Only this wrapper publishes sanitized results. A late driver result
        # must not register unsanitized samples after its caller timed out.
        token = ACTIVE_RESULT_STORE.set(None)
        try:
            task = asyncio.create_task(original(context, arguments))
        finally:
            ACTIVE_RESULT_STORE.reset(token)
        self._inflight.add(task)
        task.add_done_callback(self._task_done)
        return await asyncio.shield(task)

    @staticmethod
    def _register_result(context, result):
        # The underlying generic invoker registers raw results for A's result
        # archive. Replace that registration after hiding cached sample rows.
        from datus.utils.tool_result_store import ACTIVE_RESULT_STORE

        store = ACTIVE_RESULT_STORE.get()
        call_id = getattr(context, "tool_call_id", None)
        if store is not None and isinstance(call_id, str):
            store.register(call_id, result)
        return result

    async def stream(self, actions: AsyncIterator):
        """Use the ordinary chat limits and report partial results on interruption."""
        try:
            async for action in actions:
                if getattr(action, "action_type", None) == "user_insert":
                    self.record_user_updates([action.messages])
                yield action
        except asyncio.CancelledError:
            self._stop("Init interrupted; already published knowledge is preserved")
            raise
        finally:
            await actions.aclose()

    def close(self):
        self._closed = True
        self._finish()

    def _finish(self):
        if not self.state["stop_reason"]:
            if self._inflight:
                self._stop("Init tool work is still in flight; execution has not finished")
            elif not self._returned_normally:
                self._stop("Init ended without a final chat result")
        self.state["status"] = "stopped" if self.state["stop_reason"] else "returned"
