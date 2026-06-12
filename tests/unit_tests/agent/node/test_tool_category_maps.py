# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool registry population coverage for the permission system.

Categories are declared once at each tool class's definition site via
``permission_category``; ``AgenticNode._populate_tool_registry`` introspects
node attributes and registers every group's full name surface
(``all_tools_name()`` preferred over ``available_tools()``). These tests pin
both halves of that contract:

* the declaration table — a tool class drifting to the wrong category would
  silently re-route its profile rules;
* the introspection — a mounted group that the scanner misses falls back to
  the ``tools`` catch-all at hook time (default ASK on normal/auto), which is
  exactly the historical ``task``-tool bug this design removes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.tools.registry.tool_registry import ToolRegistry


def _named(name):
    t = MagicMock()
    t.name = name
    return t


def _group(category, *names, with_all_tools_name=False):
    """Tool-group stand-in with an explicit string category.

    ``permission_category`` must be a plain string for the scanner to pick
    the group up — Mock auto-attributes are excluded by design.
    """
    group = SimpleNamespace(
        permission_category=category,
        available_tools=lambda: [_named(n) for n in names],
    )
    if with_all_tools_name:
        group.all_tools_name = lambda: list(names)
    return group


def _bare_node():
    node = AgenticNode.__new__(AgenticNode)
    node.tool_registry = ToolRegistry()
    return node


class TestPermissionCategoryDeclarations:
    """Every tool class declares its category at the definition site."""

    @pytest.mark.parametrize(
        "import_path,class_name,expected_category",
        [
            ("datus.tools.func_tool.database", "DBFuncTool", "db_tools"),
            ("datus.tools.func_tool.semantic_tools", "SemanticTools", "semantic_tools"),
            ("datus.tools.func_tool.generation_tools", "GenerationTools", "semantic_tools"),
            ("datus.tools.func_tool.semantic_discovery_tools", "SemanticDiscoveryTools", "semantic_tools"),
            ("datus.tools.func_tool.context_search", "ContextSearchTools", "context_search_tools"),
            ("datus.tools.func_tool.bi_tools", "BIFuncTool", "bi_tools"),
            (
                "datus.tools.func_tool.reference_template_tools",
                "ReferenceTemplateTools",
                "reference_template_tools",
            ),
            ("datus.tools.func_tool.filesystem_tools", "FilesystemFuncTool", "filesystem_tools"),
            ("datus.tools.func_tool.memory_filesystem_tools", "MemoryFilesystemFuncTool", "filesystem_tools"),
            ("datus.tools.func_tool.metric_filesystem_tools", "MetricFilesystemFuncTool", "filesystem_tools"),
            ("datus.tools.func_tool.memory_tools", "MemoryFuncTool", "memory_tools"),
            ("datus.tools.func_tool.scheduler_tools", "SchedulerTools", "scheduler_tools"),
            ("datus.tools.func_tool.bash_tool", "BashTool", "bash_tools"),
            ("datus.tools.func_tool.date_parsing_tools", "DateParsingTools", "date_parsing_tools"),
            ("datus.tools.func_tool.sub_agent_task_tool", "SubAgentTaskTool", "sub_agent_tools"),
            ("datus.tools.func_tool.ask_user_tools", "AskUserTool", "tools"),
            ("datus.tools.func_tool.platform_doc_search", "PlatformDocSearchTool", "platform_doc_tools"),
            ("datus.tools.func_tool.plan_tools", "PlanTool", "tools"),
            ("datus.tools.func_tool.plan_tools", "ConfirmPlanTool", "tools"),
            ("datus.tools.func_tool.session_search_tool", "SessionSearchTool", "tools"),
            ("datus.tools.func_tool.skill_validate_tool", "SkillValidateTool", "tools"),
            ("datus.tools.func_tool.orchestrator_tools", "OrchestratorIssueTools", "tools"),
            ("datus.tools.func_tool.dashboard_artifact_tools", "DashboardArtifactTools", "artifact_tools"),
            ("datus.tools.func_tool.report_artifact_tools", "ReportArtifactTools", "artifact_tools"),
            ("datus.tools.skill_tools.skill_func_tool", "SkillFuncTool", "skills"),
        ],
    )
    def test_class_declares_category(self, import_path, class_name, expected_category):
        module = __import__(import_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        assert cls.permission_category == expected_category

    def test_artifact_filesystem_subclasses_inherit_filesystem_category(self):
        """Artifact/report fs tools inherit ``filesystem_tools`` so the
        PathZone gate in ``PermissionHooks._handle_filesystem_zone`` engages."""
        from datus.tools.func_tool.dashboard_artifact_tools import DashboardFilesystemFuncTool
        from datus.tools.func_tool.report_artifact_tools import ReportFilesystemFuncTool

        assert DashboardFilesystemFuncTool.permission_category == "filesystem_tools"
        assert ReportFilesystemFuncTool.permission_category == "filesystem_tools"

    def test_base_tool_defaults_to_catch_all(self):
        from datus.tools.base import BaseTool

        assert BaseTool.permission_category == "tools"


class TestPopulateToolRegistry:
    def test_registers_groups_under_declared_categories(self):
        node = _bare_node()
        node.db_func_tool = _group("db_tools", "read_query", "execute_ddl", with_all_tools_name=True)
        node.filesystem_func_tool = _group("filesystem_tools", "read_file", "write_file")
        node.ask_user_tool = _group("tools", "ask_user")

        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry["read_query"] == "db_tools"
        assert registry["execute_ddl"] == "db_tools"
        assert registry["read_file"] == "filesystem_tools"
        assert registry["write_file"] == "filesystem_tools"
        assert registry["ask_user"] == "tools"

    def test_task_tool_registered_outside_chat(self):
        """The historical bug: only chat declared ``sub_agent_tools``, so the
        ``task`` tool fell into the catch-all (default ASK) on gen_sql,
        gen_report, feedback and the visual artifact nodes. Declaration-site
        categories make the mapping node-independent."""
        node = _bare_node()
        node.sub_agent_task_tool = _group("sub_agent_tools", "task")

        node._populate_tool_registry()
        assert node.tool_registry.to_dict()["task"] == "sub_agent_tools"

    def test_prefers_all_tools_name_superset(self):
        """``all_tools_name()`` wins over ``available_tools()`` so method-level
        wrappers (e.g. gen_job mounting ``DBFuncTool.execute_write`` directly)
        and conditionally-hidden tools are still classified."""
        node = _bare_node()
        group = SimpleNamespace(
            permission_category="db_tools",
            available_tools=lambda: [_named("read_query")],
            all_tools_name=lambda: ["read_query", "execute_write", "transfer_query_result"],
        )
        node.db_func_tool = group

        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry["execute_write"] == "db_tools"
        assert registry["transfer_query_result"] == "db_tools"

    def test_falls_back_to_available_tools_when_all_tools_name_raises(self):
        node = _bare_node()

        def _boom():
            raise RuntimeError("no introspection")

        group = SimpleNamespace(
            permission_category="scheduler_tools",
            available_tools=lambda: [_named("list_scheduler_jobs")],
            all_tools_name=_boom,
        )
        node.scheduler_tools = group

        node._populate_tool_registry()
        assert node.tool_registry.to_dict()["list_scheduler_jobs"] == "scheduler_tools"

    def test_skips_groups_without_string_category(self):
        """Mock auto-attributes are not strings, so bare mocks (common in node
        tests) never pollute the registry with bogus categories."""
        node = _bare_node()
        node.some_mock = MagicMock()
        node.real_group = _group("memory_tools", "add_memory")

        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry == {"add_memory": "memory_tools"}

    def test_skips_none_and_class_attributes(self):
        node = _bare_node()
        node.db_func_tool = None
        node.tool_cls = SimpleNamespace  # a type, not an instance

        node._populate_tool_registry()
        assert node.tool_registry.to_dict() == {}

    def test_same_group_referenced_twice_registered_once(self):
        """Aliased attributes (e.g. ``semantic_func_tool`` aliasing
        ``semantic_tools`` on gen_semantic_model) must not double-register."""
        node = _bare_node()
        group = _group("semantic_tools", "list_metrics")
        node.semantic_tools = group
        node.semantic_func_tool = group

        node._populate_tool_registry()
        assert node.tool_registry.to_dict() == {"list_metrics": "semantic_tools"}

    def test_real_classes_register_via_class_level_introspection(self):
        """End-to-end over real classes: ``all_tools_name()`` is class-level
        on the core tools, so bare instances classify their full surface."""
        from datus.tools.func_tool.database import DBFuncTool
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        node = _bare_node()
        node.db_func_tool = DBFuncTool.__new__(DBFuncTool)
        node.filesystem_func_tool = FilesystemFuncTool.__new__(FilesystemFuncTool)

        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry["read_query"] == "db_tools"
        assert registry["execute_ddl"] == "db_tools"
        assert registry["execute_write"] == "db_tools"
        assert registry["read_file"] == "filesystem_tools"
        assert registry["write_file"] == "filesystem_tools"
        assert registry["delete_file"] == "filesystem_tools"


class TestFeedbackMemoryRegistration:
    """The feedback node mounts ``add_memory`` / ``edit_memory`` for its
    caller; the scanner must classify them under ``memory_tools`` so the
    ``memory_tools.*`` ALLOW rule governs them rather than the catch-all."""

    def test_feedback_registers_memory_tools(self):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode.__new__(FeedbackAgenticNode)
        node.tool_registry = ToolRegistry()
        node.memory_func_tool = _group("memory_tools", "add_memory", "edit_memory")

        node._populate_tool_registry()
        registry = node.tool_registry.to_dict()
        assert registry["add_memory"] == "memory_tools"
        assert registry["edit_memory"] == "memory_tools"

    def test_memory_entries_absent_when_tool_absent(self):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode.__new__(FeedbackAgenticNode)
        node.tool_registry = ToolRegistry()
        node.memory_func_tool = None

        node._populate_tool_registry()
        assert "memory_tools" not in node.tool_registry.to_dict().values()
