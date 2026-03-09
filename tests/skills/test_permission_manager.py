# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for PermissionManager.

Tests permission checking, filtering, and node-specific overrides.
"""

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_manager import PermissionManager


class TestPermissionManagerBasic:
    """Basic tests for PermissionManager."""

    def test_manager_creation_with_defaults(self):
        """Test creating a PermissionManager with default config."""
        config = PermissionConfig()
        manager = PermissionManager(global_config=config)
        assert manager.global_config == config
        assert manager.node_overrides == {}

    def test_manager_creation_with_overrides(self):
        """Test creating a PermissionManager with node overrides."""
        config = PermissionConfig()
        overrides = {
            "chatbot": PermissionConfig(
                rules=[PermissionRule(tool="skills", pattern="*", permission=PermissionLevel.DENY)]
            )
        }
        manager = PermissionManager(global_config=config, node_overrides=overrides)
        assert "chatbot" in manager.node_overrides


class TestPermissionManagerCheckPermission:
    """Tests for PermissionManager.check_permission()."""

    def test_check_permission_default_allow(self):
        """Test that default permission is returned when no rules match."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config)

        result = manager.check_permission("db_tools", "execute_sql", "chatbot")
        assert result == PermissionLevel.ALLOW

    def test_check_permission_default_deny(self):
        """Test that default deny permission is returned when no rules match."""
        config = PermissionConfig(default_permission=PermissionLevel.DENY)
        manager = PermissionManager(global_config=config)

        result = manager.check_permission("db_tools", "execute_sql", "chatbot")
        assert result == PermissionLevel.DENY

    def test_check_permission_matching_rule(self):
        """Test that matching rule permission is returned."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="execute_sql", permission=PermissionLevel.ASK),
            ],
        )
        manager = PermissionManager(global_config=config)

        result = manager.check_permission("db_tools", "execute_sql", "chatbot")
        assert result == PermissionLevel.ASK

    def test_check_permission_wildcard_pattern(self):
        """Test permission check with wildcard pattern."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="dangerous-*", permission=PermissionLevel.DENY),
            ],
        )
        manager = PermissionManager(global_config=config)

        assert manager.check_permission("skills", "dangerous-script", "chatbot") == PermissionLevel.DENY
        assert manager.check_permission("skills", "safe-script", "chatbot") == PermissionLevel.ALLOW

    def test_check_permission_last_match_wins(self):
        """Test that the last matching rule wins."""
        config = PermissionConfig(
            default_permission=PermissionLevel.DENY,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ALLOW),
                PermissionRule(tool="db_tools", pattern="execute_sql", permission=PermissionLevel.ASK),
            ],
        )
        manager = PermissionManager(global_config=config)

        # execute_sql matches both rules, last one (ASK) should win
        assert manager.check_permission("db_tools", "execute_sql", "chatbot") == PermissionLevel.ASK
        # list_tables only matches first rule
        assert manager.check_permission("db_tools", "list_tables", "chatbot") == PermissionLevel.ALLOW

    def test_check_permission_node_override(self):
        """Test that node-specific overrides take precedence."""
        global_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="dangerous-*", permission=PermissionLevel.DENY),
            ],
        )
        node_overrides = {
            "sql_expert": PermissionConfig(
                rules=[
                    PermissionRule(tool="skills", pattern="dangerous-*", permission=PermissionLevel.ALLOW),
                ],
            ),
        }
        manager = PermissionManager(global_config=global_config, node_overrides=node_overrides)

        # Regular node should use global config (DENY)
        assert manager.check_permission("skills", "dangerous-script", "chatbot") == PermissionLevel.DENY

        # sql_expert node has override (ALLOW)
        assert manager.check_permission("skills", "dangerous-script", "sql_expert") == PermissionLevel.ALLOW

    def test_check_permission_node_override_with_dict(self):
        """Test node override with dictionary format."""
        global_config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        node_overrides = {
            "restricted": {
                "rules": [
                    {"tool": "db_tools", "pattern": "*", "permission": "deny"},
                ],
            },
        }
        manager = PermissionManager(global_config=global_config, node_overrides=node_overrides)

        assert manager.check_permission("db_tools", "execute_sql", "chatbot") == PermissionLevel.ALLOW
        assert manager.check_permission("db_tools", "execute_sql", "restricted") == PermissionLevel.DENY


class TestPermissionManagerFilterTools:
    """Tests for PermissionManager.filter_available_tools()."""

    def test_filter_tools_no_deny(self):
        """Test filtering tools when none are denied."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config)

        # Mock tool objects
        class MockTool:
            def __init__(self, name):
                self.name = name

        tools = [MockTool("execute_sql"), MockTool("list_tables"), MockTool("describe_table")]
        filtered = manager.filter_available_tools(tools, "chatbot")

        assert len(filtered) == 3

    def test_filter_tools_with_deny(self):
        """Test filtering tools when some are denied."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="execute_sql", permission=PermissionLevel.DENY),
            ],
        )
        manager = PermissionManager(global_config=config)

        class MockTool:
            def __init__(self, name):
                self.name = name

        tools = [MockTool("execute_sql"), MockTool("list_tables"), MockTool("describe_table")]
        filtered = manager.filter_available_tools(tools, "chatbot", tool_category="db_tools")

        # execute_sql should be filtered out
        assert len(filtered) == 2
        assert all(t.name != "execute_sql" for t in filtered)

    def test_filter_tools_all_denied(self):
        """Test filtering tools when all are denied."""
        config = PermissionConfig(
            default_permission=PermissionLevel.DENY,
        )
        manager = PermissionManager(global_config=config)

        class MockTool:
            def __init__(self, name):
                self.name = name

        tools = [MockTool("tool1"), MockTool("tool2")]
        filtered = manager.filter_available_tools(tools, "chatbot", tool_category="any")

        assert len(filtered) == 0

    def test_filter_tools_ask_included(self):
        """Test that ASK permission tools are included (not filtered)."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="execute_sql", permission=PermissionLevel.ASK),
            ],
        )
        manager = PermissionManager(global_config=config)

        class MockTool:
            def __init__(self, name):
                self.name = name

        tools = [MockTool("execute_sql"), MockTool("list_tables")]
        filtered = manager.filter_available_tools(tools, "chatbot", tool_category="db_tools")

        # ASK tools should be included (only DENY is filtered)
        assert len(filtered) == 2


class TestPermissionManagerFilterSkills:
    """Tests for PermissionManager.filter_available_skills()."""

    def test_filter_skills_no_deny(self):
        """Test filtering skills when none are denied."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config)

        # Mock skill metadata objects
        class MockSkillMetadata:
            def __init__(self, name):
                self.name = name

        skills = [MockSkillMetadata("sql-optimization"), MockSkillMetadata("data-analysis")]
        filtered = manager.filter_available_skills(skills, "chatbot")

        assert len(filtered) == 2

    def test_filter_skills_with_deny(self):
        """Test filtering skills when some are denied."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="internal-*", permission=PermissionLevel.DENY),
            ],
        )
        manager = PermissionManager(global_config=config)

        class MockSkillMetadata:
            def __init__(self, name):
                self.name = name

        skills = [
            MockSkillMetadata("sql-optimization"),
            MockSkillMetadata("internal-admin"),
            MockSkillMetadata("internal-debug"),
        ]
        filtered = manager.filter_available_skills(skills, "chatbot")

        # internal-* skills should be filtered out
        assert len(filtered) == 1
        assert filtered[0].name == "sql-optimization"

    def test_filter_skills_node_specific(self):
        """Test filtering skills with node-specific overrides."""
        global_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
            ],
        )
        node_overrides = {
            "admin_node": PermissionConfig(
                rules=[
                    PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.ALLOW),
                ],
            ),
        }
        manager = PermissionManager(global_config=global_config, node_overrides=node_overrides)

        class MockSkillMetadata:
            def __init__(self, name):
                self.name = name

        skills = [MockSkillMetadata("admin-tools"), MockSkillMetadata("user-tools")]

        # Regular node: admin-* denied
        filtered_regular = manager.filter_available_skills(skills, "chatbot")
        assert len(filtered_regular) == 1
        assert filtered_regular[0].name == "user-tools"

        # Admin node: admin-* allowed
        filtered_admin = manager.filter_available_skills(skills, "admin_node")
        assert len(filtered_admin) == 2


class TestPermissionManagerEdgeCases:
    """Edge case tests for PermissionManager."""

    def test_empty_tool_name(self):
        """Test permission check with empty tool name."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config)

        result = manager.check_permission("db_tools", "", "chatbot")
        assert result == PermissionLevel.ALLOW

    def test_empty_node_name(self):
        """Test permission check with empty node name."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config)

        result = manager.check_permission("db_tools", "execute_sql", "")
        assert result == PermissionLevel.ALLOW

    def test_special_characters_in_pattern(self):
        """Test permission check with special characters in pattern."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="mcp", pattern="filesystem_mcp.*", permission=PermissionLevel.ASK),
            ],
        )
        manager = PermissionManager(global_config=config)

        assert manager.check_permission("mcp", "filesystem_mcp.read_file", "chatbot") == PermissionLevel.ASK
        assert manager.check_permission("mcp", "filesystem_mcp.write_file", "chatbot") == PermissionLevel.ASK
        assert manager.check_permission("mcp", "other_mcp.read_file", "chatbot") == PermissionLevel.ALLOW

    def test_none_node_overrides(self):
        """Test with None node overrides."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW)
        manager = PermissionManager(global_config=config, node_overrides=None)

        result = manager.check_permission("db_tools", "execute_sql", "chatbot")
        assert result == PermissionLevel.ALLOW
