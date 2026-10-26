"""Web-only review of frozen Ontology/Mapping filesystem proposals."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import FunctionTool

from datus.storage.semantic_model.artifact_file import path_mutation_lock
from datus.tools.func_tool.base import FuncToolResult, parse_tool_args
from datus.tools.func_tool.fs_path_policy import PathZone
from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.proxy.tool_result_channel import DEFAULT_RESULT_TIMEOUT
from datus.utils.memory_loader import apply_single_replacement
from datus.utils.ontology_utils import is_ontology_file_path

REVIEW_TOOLS = {"write_file", "edit_file"}


@dataclass(frozen=True)
class PreparedOntologyFile:
    tool_name: str
    path: str
    target: Path
    before: bytes | None
    after: str
    old_string: str = ""
    new_string: str = ""

    def preview(self) -> dict[str, Any]:
        return {
            "filePath": self.path,
            "oldText": self.before.decode("utf-8") if self.before is not None else "",
            "newText": self.after,
            "reviewType": "ontology",
        }


def _prepare(fs: Any, tool_name: str, args: dict) -> PreparedOntologyFile:
    path = args["path"]
    resolved = fs._classify(path)
    target = resolved.resolved
    if resolved.zone != PathZone.INTERNAL or resolved.read_only or not is_ontology_file_path(target):
        raise ValueError("Ontology review requires a writable project-local ontology file")
    guard = fs._mutation_guard_error(target)
    if guard is not None:
        raise ValueError(guard.error)
    with path_mutation_lock(target):
        before = target.read_bytes() if target.exists() else None
        original = before.decode("utf-8") if before is not None else ""
        if tool_name == "write_file":
            after = args["content"]
            if not isinstance(after, str):
                raise ValueError("content must be a string")
            return PreparedOntologyFile(tool_name, path, target, before, after)
        if before is None:
            raise ValueError("Cannot edit a file that does not exist")
        old, new = args["old_string"], args["new_string"]
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old_string must be nonempty and new_string must be a string")
        after, error = apply_single_replacement(original, old, new)
        if error is not None:
            raise ValueError(error)
        return PreparedOntologyFile(tool_name, path, target, before, after, old, new)


def _commit(fs: Any, prepared: PreparedOntologyFile) -> FuncToolResult:
    # Lock the reviewed path, then re-resolve the caller's path as well: a
    # symlink/parent swap must never redirect an already approved proposal.
    with path_mutation_lock(prepared.target):
        resolved = fs._classify(prepared.path)
        if resolved.resolved != prepared.target or resolved.zone != PathZone.INTERNAL or resolved.read_only:
            return FuncToolResult(success=0, error="Ontology path changed during review; request a fresh preview")
        current = prepared.target.read_bytes() if prepared.target.exists() else None
        if current != prepared.before:
            return FuncToolResult(success=0, error="Ontology file changed during review; request a fresh preview")
        # Keep the original filesystem policy, domain validation and mutation
        # callbacks. The shared lock is reentrant for these nested writers.
        if prepared.tool_name == "write_file":
            return fs.write_file(path=prepared.path, content=prepared.after)
        return fs.edit_file(path=prepared.path, old_string=prepared.old_string, new_string=prepared.new_string)


def apply_ontology_file_review(node: Any, *, context: Any) -> None:
    """Install after generic proxy wrapping; ordinary file tools keep their executor."""
    fs = getattr(node, "filesystem_func_tool", None)
    if fs is None:
        return

    def wrap(original: FunctionTool) -> FunctionTool:
        async def invoke(tool_ctx: Any, args_str: str):
            args, _, error = parse_tool_args(args_str, tool_name=original.name)
            if error or not isinstance(args.get("path"), str) or not is_ontology_file_path(args["path"]):
                return await original.on_invoke_tool(tool_ctx, args_str)
            # ASK is fulfilled by the review below; ALLOW still shows the Web
            # normal preview, while an explicit global/node DENY stays binding.
            manager = getattr(node, "permission_manager", None)
            if (
                manager is not None
                and manager.check_permission("filesystem_tools", original.name, node.get_node_name())
                == PermissionLevel.DENY
            ):
                return FuncToolResult(success=0, error="Ontology write denied by permission rules").model_dump()
            try:
                prepared = await asyncio.to_thread(_prepare, fs, original.name, args)
                call_id = tool_ctx.tool_call_id
                context.previews[call_id] = prepared.preview()
                try:
                    decision = await context.channel.wait_for(call_id, timeout=DEFAULT_RESULT_TIMEOUT)
                    if not isinstance(decision, dict) or decision.get("success") != 1:
                        return FuncToolResult(
                            success=0,
                            error="User cancelled the ontology change; no file was written. Wait for feedback before proposing another edit.",
                        ).model_dump()
                    result = await asyncio.to_thread(_commit, fs, prepared)
                    return result.model_dump()
                finally:
                    context.previews.pop(call_id, None)
            except (OSError, ValueError, KeyError, TimeoutError, RuntimeError) as exc:
                return FuncToolResult(success=0, error=f"Ontology review failed: {exc}").model_dump()

        invoke._datus_ontology_review = True
        return FunctionTool(
            name=original.name,
            description=original.description,
            params_json_schema=original.params_json_schema,
            on_invoke_tool=invoke,
            strict_json_schema=original.strict_json_schema,
        )

    node.tools = [
        wrap(tool) if isinstance(tool, FunctionTool) and tool.name in REVIEW_TOOLS else tool for tool in node.tools
    ]
