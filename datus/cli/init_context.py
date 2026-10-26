# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Ephemeral scope, metadata cache and tool restrictions for one /init call.

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
from datus.utils.tool_result_store import parse_result

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
_CONTROLS = {"get_init_status", "plan_init_scope", "report_init_progress"}
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
    """One invocation's scope, tool cache and model-reported progress."""

    def __init__(self, root: Path, binding: dict, scope: str = ""):
        self.root = root.expanduser().resolve()
        self.binding = binding
        self.state = {
            "scope": scope,
            "coverage": "selected",
            "tables": {},
            "selected_tables": [],
            "sources": [],
            "outputs": [],
            "summary": "",
            "scope_complete": False,
            "remaining": "",
            "next_step": "",
            "uncovered": "",
            "status": "running",
            "stop_reason": "",
        }
        self.calls = 0
        self._cache: dict[str, Any] = {}
        self._table_aliases: dict[str, str] = {}
        self._serial = asyncio.Lock()
        self._inflight: set[asyncio.Task] = set()
        self._closed = False
        self._returned_normally = False

    @contextmanager
    def activate(self, node):
        """Temporarily constrain this chat node without replacing its session."""
        from datus.tools.func_tool.init_tools import InitTools

        missing = object()
        fields = (
            "tools",
            "mcp_servers",
            "_builtin_web_tools",
            "_init_context",
            "init_tools",
            "_tool_transformers_applied",
        )
        saved = {name: getattr(node, name, missing) for name in fields}
        try:
            node._init_context = self
            node.init_tools = InitTools(self)
            node.tools = list(node.tools or []) + node.init_tools.available_tools()
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
            self.state["scope_complete"] = False

    @staticmethod
    def _excerpt(text: str, limit: int = 2000) -> str:
        return text if len(text) <= limit else text[:limit] + " [continued in get_init_status]"

    def prompt_context(self) -> str:
        """Small live reminder, regenerated after compaction; details remain queryable."""
        entries = self._selected_entries()
        latest = self.user_context()[-1] if self.user_context() else None
        context = {
            "scope": self._excerpt(self.state["scope"]),
            "binding": self.binding,
            "selected": len(entries),
            "completed": sum(e["status"] == "completed" for e in entries),
            "summary": self.state["summary"],
            "remaining": self.state["remaining"],
            "next_step": self.state["next_step"],
            "stop_reason": self.state["stop_reason"],
            "user_updates": len(self.user_context()),
            "latest_user_update": self._excerpt(json.dumps(latest, ensure_ascii=False)) if latest else None,
        }
        return (
            "\n<init_context>\n" + json.dumps(context, ensure_ascii=False) + "\n</init_context>\n"
            "This /init runs in the current chat loop. Choose useful business topics and continue here. "
            "Use todo_list/read/write/update for work steps; they do not require plan approval. "
            "plan_init_scope records coverage only; report_init_progress records outcomes, never ends a run. "
            "Read get_init_status for paged scope, sources and full user clarifications; later corrections win. "
            "Use metadata and project sources to resolve structural questions. Do not query business rows. "
            "Read existing knowledge before incremental edits. Keep unsupported interpretations in "
            "knowledge/pending-questions.md; they are not confirmed facts. "
            "Use archived result references and bounded reads after compaction. Update progress with actual "
            "completed work, remaining work and next step. Do not infer completion from a structure lookup. "
            "End normally when done, or explain the unfinished scope and stopping reason."
        )

    def select_scope(self, tables: list[str], source_paths: list[str], uncovered: str, exclude_tables: list[str]):
        """Register coverage without scheduling work or creating an execution plan."""
        if len(uncovered) > 1500:
            raise ValueError("Keep uncovered within 1500 characters")
        names = [self.table_name(name) for name in tables]
        excluded = {self.table_name(name) for name in exclude_tables}
        sources = [self._path(name).relative_to(self.root).as_posix() for name in source_paths]
        selected = self.state["selected_tables"]
        for name in names:
            self.state["tables"].setdefault(name, {"table": name, "status": "pending"})
            if name not in selected:
                selected.append(name)
        self.state["selected_tables"] = [name for name in selected if name not in excluded]
        self.state["sources"] = list(dict.fromkeys([*self.state["sources"], *sources]))
        self.state.update(uncovered=uncovered, scope_complete=False)
        return self.progress()

    def report_progress(
        self, summary: str, completed_tables: list[str], scope_complete: bool, remaining: str, next_step: str
    ):
        if not summary.strip() or any(len(value) > 1500 for value in (summary, remaining, next_step)):
            raise ValueError("Provide a nonempty summary; keep each progress field within 1500 characters")
        for name in [self.table_name(name) for name in completed_tables]:
            self.state["tables"].setdefault(name, {"table": name})["status"] = "completed"
            if name not in self.state["selected_tables"]:
                self.state["selected_tables"].append(name)
        self.state.update(summary=summary, remaining=remaining, next_step=next_step, scope_complete=scope_complete)
        return self.progress()

    def finish_execution(
        self, result, *, interrupted: bool = False, plan_mode: bool = False, pending_input: bool = False
    ):
        """Runtime outcome is independent of the model's coverage report."""
        if interrupted:
            self._stop("Init interrupted; already published knowledge is preserved")
        elif getattr(result, "success", None) is not True:
            self._stop(f"Init chat did not complete successfully: {getattr(result, 'error', '') or 'No final result'}")
        else:
            self._returned_normally = not plan_mode
            if pending_input:
                self._stop("User input remains unprocessed in the chat queue; continue on the next user turn")
        self._finish()

    def _path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Init evidence and outputs must be inside the project")
        return path

    def progress(self, offset: int = 0, limit: int = 10) -> dict:
        if offset < 0 or not 1 <= limit <= 20:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 20")
        entries = self._selected_entries()
        pending = [entry for entry in entries if entry["status"] != "completed"]
        sources, updates = self.state["sources"], self.user_context()
        return {
            "scope": self.state["scope"],
            "coverage": "selected",
            "status": self.state["status"],
            "stop_reason": self.state["stop_reason"],
            "selected": len(entries),
            "completed": len(entries) - len(pending),
            "pending": len(pending),
            "items": pending[offset : offset + limit],
            "has_more": offset + limit < len(pending),
            "selected_items": entries[offset : offset + limit],
            "selected_has_more": offset + limit < len(entries),
            "next_offset": offset + limit,
            "summary": self.state["summary"],
            "remaining": self.state["remaining"],
            "next_step": self.state["next_step"],
            "scope_complete": self.state["scope_complete"],
            "uncovered": self.state["uncovered"],
            "calls_used": self.calls,
            "structure_checked": sum(e.get("metadata_status") == "checked" for e in self.state["tables"].values()),
            "metadata_failed": sum(e.get("metadata_status") == "failed" for e in self.state["tables"].values()),
            "source_paths": sources[offset : offset + limit],
            "sources_has_more": offset + limit < len(sources),
            "user_context": updates[offset : offset + limit],
            "user_context_has_more": offset + limit < len(updates),
            "modified_files": self.state["outputs"][offset : offset + limit],
            "modified_files_total": len(self.state["outputs"]),
        }

    def _selected_entries(self) -> list[dict]:
        return [self.state["tables"][table] for table in self.state["selected_tables"]]

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
        return _error(
            self.state["stop_reason"] + ". Report the remaining scope; already published knowledge is preserved."
        )

    def wrap_tools(self, tools: list) -> list:
        """Keep project readers, metadata tools and knowledge writers for this run."""
        wrapped = []
        for tool in tools:
            if not isinstance(tool, FunctionTool) or tool.name not in _READS | _WRITES | _CONTROLS | _TODOS | {
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
            if self.state["stop_reason"] and name != "get_init_status":
                return _error(self.state["stop_reason"] + ". Report unfinished work and end this run.")
            table = self._qualify_metadata_table(args) if name in {"describe_table", "get_table_ddl"} else ""
        except Exception as exc:
            return _error(str(exc))
        if name in _CONTROLS | {"ask_user"}:
            if name == "ask_user":
                self.state["scope_questions"] = args.get("questions", [])
            try:
                result = await self._tracked_call(original, context, arguments)
                if name == "ask_user":
                    result = parse_result(result)
                    answers = parse_result(result.get("result")) if isinstance(result, dict) else None
                    if not (
                        isinstance(result, dict)
                        and result.get("success", 1) == 1
                        and isinstance(answers, list)
                        and answers
                        and all(isinstance(a, dict) and a.get("answer") for a in answers)
                    ):
                        return self._stop("Initialization clarification was cancelled or left unanswered")
                    self.state["scope_complete"] = False
                    self.state.setdefault("scope_answers", []).extend(answers)
                    self.state.setdefault("user_instructions", []).extend(
                        {**answer, "source": "scope_answer"} for answer in answers
                    )
                    self.state.pop("scope_questions", None)
                return result
            except TimeoutError:
                return self._stop("Init control or interaction timed out; the operation may still be running")
        key = name + ":" + json.dumps(resolved, sort_keys=True)
        if key in self._cache:
            result = copy.deepcopy(self._cache[key])
            if name in {"describe_table", "get_table_ddl"}:
                self._record_metadata(args, result)
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
            if table:
                self._record_metadata(args, result)
            if success and name in _CACHEABLE:
                self._cache[key] = copy.deepcopy(result)
            return self._register_result(context, result)
        except TimeoutError:
            if table:
                self.state["tables"][table]["metadata_status"] = "failed"
                self.state["tables"][table]["error"] = "Tool timed out; the database operation may still be running"
            # A timed-out driver thread may still run. Keep the lease until it
            # exits and stop further exploration rather than pile up queries.
            return self._stop("Init tool timed out; the database operation may still be running")
        except Exception as exc:
            if table:
                self.state["tables"][table]["metadata_status"] = "failed"
                self.state["tables"][table]["error"] = str(exc)[:1200]
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

    def _qualify_metadata_table(self, args: dict) -> str:
        """Keep paged scope references aligned when discovery qualifies a name."""
        table, raw = self._table_key(args), args["table_name"]
        entries = self.state["tables"]
        if table != raw and raw in entries:
            existing = entries.get(table, {})
            completed = entries[raw]["status"] == "completed" or existing.get("status") == "completed"
            entries[table] = {**entries.pop(raw), **existing, "table": table}
            if completed:
                entries[table]["status"] = "completed"
            if raw in self.state["selected_tables"]:
                self._table_aliases = {
                    name: table if target == raw else target for name, target in self._table_aliases.items()
                }
                self._table_aliases[raw] = table
            self.state["selected_tables"] = list(
                dict.fromkeys(table if name == raw else name for name in self.state["selected_tables"])
            )
        entries.setdefault(table, {"table": table, "status": "pending"})
        return table

    def table_name(self, name: str) -> str:
        """Resolve qualified aliases without changing selection or knowledge status."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Table names must be nonempty strings")
        return self._table_aliases.get(name, name)

    def _record_metadata(self, args: dict, result: Any) -> None:
        table = self._table_key(args)
        entry = self.state["tables"].setdefault(table, {"table": table, "status": "pending"})
        success = isinstance(result, dict) and result.get("success", 1) == 1
        entry["metadata_status"] = "checked" if success else "failed"
        if success:
            entry["metadata_status"] = "checked"
            entry.pop("error", None)
        else:
            entry["error"] = (
                str(result.get("error", "Metadata check failed"))[:1200]
                if isinstance(result, dict)
                else "Metadata check failed"
            )
        if isinstance(result, dict):
            result["init_table_key"] = table

    def _table_key(self, args: dict) -> str:
        args = self._metadata_arguments(args)
        table = args.get("table_name", "")
        # The run is already bound to its selected namespaces. Distinguish
        # additional namespace filters when exploring an unbound schema/db.
        namespaces = [
            args[field]
            for field in ("catalog", "database", "schema_name")
            if args.get(field) and not self.binding.get(field)
        ]
        if self.binding.get("dialect"):
            # A literal dot inside a quoted table name is not a namespace
            # separator. Keep those names distinct in coverage as well as cache.
            return ".".join(
                '"' + part.replace('"', '""') + '"' if "." in part or '"' in part else part
                for part in [*namespaces, table]
            )
        prefix = ".".join(namespaces)
        return f"{prefix}.{table}" if prefix and not table.startswith(prefix + ".") else table

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
        # Completion is explicit; remaining is handoff prose, which may say "None".
        completed = (
            self._returned_normally
            and self.state["scope_complete"]
            and not self._inflight
            and not self.state["stop_reason"]
            and all(item["status"] == "completed" for item in self._selected_entries())
        )
        self.state["status"] = "completed_scope" if completed else "partial"
