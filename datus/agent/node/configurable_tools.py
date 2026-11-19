# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Instantiate explicitly selected tools using existing tool contracts and factories."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from agents import FunctionTool

from datus.cli.execution_state import InteractionCancelled
from datus.schemas.agent_spec import MCPToolSelection
from datus.tools.func_tool.base import trans_to_function_tool
from datus.tools.permission.permission_hooks import PermissionDeniedException
from datus.tools.registry.capabilities import TOOL_CAPABILITIES
from datus.tools.registry.contracts import definition_from_callable, tool_definition
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.mcp_decorators import get_tool_registry

if TYPE_CHECKING:
    from datus.agent.node.configurable_agentic_node import ConfigurableAgenticNode


def _tool_owners() -> dict[str, type]:
    """Use the registered tool classes, retaining their real declaration identities."""
    from datus.agent.semantic_authoring_session import SemanticAuthoringSession
    from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools
    from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
    from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

    owners = {}
    classes = [config.tool_class for config in get_tool_registry()]
    classes.extend(
        [SemanticAuthoringSession, SemanticModelEditingTools, SemanticPlanFileTools, SemanticVerificationTools]
    )
    for cls in classes:
        category = getattr(cls, "permission_category", None)
        if not isinstance(category, str):
            continue
        for name, method in inspect.getmembers(cls, callable):
            tool_id = f"{category}.{name}"
            if tool_id not in TOOL_CAPABILITIES or name.startswith("_"):
                continue
            definition = definition_from_callable(method, owner=cls)
            if definition.tool_id != tool_id:
                continue
            previous = owners.get(tool_id)
            if previous is not None and previous is not cls:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"Multiple native tool factories declare {tool_id!r}.",
                )
            owners[tool_id] = cls
    return owners


def configurable_tool_catalog() -> dict[str, dict[str, Any]]:
    """Describe constructible native tools without opening databases or creating a node."""
    return {
        tool_id: {"name": tool_id, "access": TOOL_CAPABILITIES[tool_id].access} for tool_id in sorted(_tool_owners())
    }


def resolve_native_tool_ids(selections: list[str]) -> list[str]:
    """Resolve explicit canonical IDs or unambiguous method names, with no default expansion."""
    owners = _tool_owners()
    result = []
    for selection in selections:
        matches = [tool_id for tool_id in owners if tool_id == selection or tool_id.split(".", 1)[1] == selection]
        if len(matches) != 1:
            qualifier = "ambiguous" if matches else "unknown or unavailable"
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"Native tool {selection!r} is {qualifier}; select a canonical tool ID from the catalog.",
            )
        if matches[0] not in result:
            result.append(matches[0])
    return result


class ConfigurableToolSet:
    """Own tool instances without exposing unselected methods to the model."""

    def __init__(self, node: ConfigurableAgenticNode):
        self.node = node
        self.groups: dict[type, Any] = {}

    def _create(self, cls: type) -> Any:
        """Reuse special node context only where the existing factory cannot receive it."""
        node = self.node
        category = cls.permission_category
        from datus.agent.semantic_authoring_session import SemanticAuthoringSession
        from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools
        from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
        from datus.tools.func_tool.semantic_verification_tools import SemanticVerificationTools

        if cls is SemanticAuthoringSession:
            return node._get_semantic_authoring_session()
        if cls is SemanticModelEditingTools:
            return node._get_semantic_authoring_session().semantic_model_editing_tools
        if cls is SemanticVerificationTools:
            return node._get_semantic_authoring_session().semantic_verification_tools
        if cls is SemanticPlanFileTools:
            instance = (
                node._get_semantic_authoring_session().semantic_plan_file_tool
                if node._uses_semantic_authoring
                else SemanticPlanFileTools(
                    node.agent_config.project_root, path_allowed=node.execution_scope.allows_file
                )
            )
            node.semantic_plan_file_tool = instance
            return instance
        if category == "db_tools" and node._uses_semantic_authoring:
            instance = node._get_semantic_authoring_session().db_func_tool
            instance.sql_file_allowed = node.execution_scope.allows_file
            node.db_func_tool = instance
            return instance
        if category == "filesystem_tools":
            if node._uses_semantic_authoring:
                instance = node._get_semantic_authoring_session().filesystem_func_tool
            else:
                instance = node._make_filesystem_tool()
            node.filesystem_func_tool = instance
        elif category == "semantic_tools":
            from datus.tools.func_tool.semantic_tools import SemanticTools

            if node._uses_semantic_authoring:
                instance = node._get_semantic_authoring_session().semantic_tools
                node.semantic_tools = instance
                return instance
            instance = SemanticTools(
                node.agent_config,
                sub_agent_name=node.get_node_name(),
                runtime_db_context_provider=node._semantic_runtime_db_context,
                initialize_runtime=False,
                enforce_metric_scope=True,
            )
            node.semantic_tools = instance
        else:
            instance = cls.create_dynamic(node.agent_config, sub_agent_name=node.get_node_name())
            if category == "db_tools":
                instance.read_only = node.node_config.get("db_read_only", False)
                instance.sql_file_allowed = node.execution_scope.allows_file
                node.db_func_tool = instance
            elif category == "context_search_tools":
                node.context_search_tools = instance
            elif category == "reference_template_tools":
                from datus.tools.func_tool.database import DBFuncTool

                if DBFuncTool not in self.groups:
                    self.groups[DBFuncTool] = self._create(DBFuncTool)
                instance.db_func_tool = self.groups[DBFuncTool]
                node.db_func_tool = instance.db_func_tool
        return instance

    def select(self, selections: list[str]) -> list[FunctionTool]:
        """Build only declared methods, once per owning tool class and execution instance."""
        from datus.tools.func_tool.database import DBFuncTool
        from datus.tools.func_tool.semantic_model_filesystem_tools import SemanticModelFilesystemTools

        owners = _tool_owners()
        tools = []
        for tool_id in resolve_native_tool_ids(selections):
            self.node.execution_scope.require_tool_scope(tool_id, {})
            cls = owners[tool_id]
            if cls not in self.groups:
                with self.node.execution_scope.activate():
                    self.groups[cls] = self._create(cls)
            instance = self.groups[cls]
            method_name = tool_id.split(".", 1)[1]
            method = getattr(instance, method_name)
            if isinstance(instance, SemanticModelFilesystemTools):
                tool = next((item for item in instance.available_tools() if item.name == method_name), None)
                if tool is None:
                    raise DatusException(
                        ErrorCode.COMMON_CONFIG_ERROR,
                        message=f"Tool {tool_id!r} is not available in a semantic authoring session.",
                    )
            elif isinstance(instance, DBFuncTool):
                tool = instance.to_function_tool(method)
            else:
                tool = trans_to_function_tool(method)
            if tool_definition(tool).tool_id != tool_id:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=f"Native factory returned a different tool contract for {tool_id!r}.",
                )
            original = tool.on_invoke_tool

            def guard(invoke, identity):
                async def checked(context, arguments):
                    error = getattr(context, "_datus_configurable_argument_error", None)
                    if error:
                        return {"success": 0, "error": error, "result": None}
                    try:
                        with self.node.execution_scope.activate():
                            self.node.execution_scope.require_tool_scope(identity, json.loads(arguments or "{}"))
                            if self.node._uses_semantic_authoring and identity in {
                                "semantic_tools.validate_semantic",
                                "semantic_tools.query_metrics",
                                "semantic_tools.get_metric",
                                "semantic_tools.list_metrics",
                                "semantic_tools.search_metrics",
                            }:
                                self.node._get_semantic_authoring_session()._selected_osi_artifact()
                            draft = self.node._semantic_draft
                            if draft is not None and not draft.published:
                                args = json.loads(arguments or "{}")
                                for key in ("path", "semantic_model_file"):
                                    value = args.get(key)
                                    if isinstance(value, str) and value:
                                        from pathlib import Path

                                        path = Path(value)
                                        if path.is_absolute() and path.is_relative_to(draft.live_root):
                                            args[key] = str(draft.root / path.relative_to(draft.live_root))
                                arguments = json.dumps(args)
                            return await invoke(context, arguments)
                    except (InteractionCancelled, PermissionDeniedException):
                        raise
                    except Exception as exc:
                        # Keep scope and target failures visible to the model, just
                        # like a native tool's failed result. Cancellation still
                        # propagates, and a failed guard never invokes the tool.
                        return {"success": 0, "error": str(exc), "result": None}

                return checked

            tool.on_invoke_tool = guard(original, tool_id)
            tool.__datus_tool_local_only__ = True
            tools.append(tool)
        return tools


def build_mcp_servers(
    agent_config: Any,
    selections: list[MCPToolSelection],
    *,
    permission_manager: Any,
    node_name: str,
) -> dict[str, Any]:
    """Create per-agent server configurations without mutating project-owned connections."""
    if not selections:
        return {}
    from datus.tools.mcp_tools.mcp_config import ToolFilterConfig
    from datus.tools.mcp_tools.mcp_manager import MCPManager
    from datus.tools.permission.permission_config import PermissionLevel

    manager = MCPManager(agent_config=agent_config)
    servers = {}
    for selection in selections:
        config = manager.get_server_config(selection.connection_ref)
        if config is None:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"MCP connection {selection.connection_ref!r} is not available in this project.",
            )
        project_filter = config.tool_filter
        if project_filter and any(not project_filter.is_tool_allowed(tool) for tool in selection.tools):
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"MCP selection exceeds project connection {selection.connection_ref!r}'s tool scope.",
            )
        allowed = [
            tool
            for tool in selection.tools
            if permission_manager is None
            or permission_manager.check_permission(f"mcp.{selection.connection_ref}", tool, node_name)
            != PermissionLevel.DENY
        ]
        # An empty effective filter must never become an unfiltered connection.
        if not allowed:
            continue
        scoped = config.model_copy(deep=True)
        scoped.tool_filter = ToolFilterConfig(allowed_tool_names=allowed)
        server, _details = manager._create_server_instance(scoped)
        if server is None:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"MCP connection {selection.connection_ref!r} could not be initialized.",
            )
        servers[selection.connection_ref] = server
    return servers
