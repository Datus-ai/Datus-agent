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

    def test_constructor_active_profile_param_works(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="dangerous")
        assert mgr.active_profile == "dangerous"


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

    def test_filter_tools_without_category_uses_catch_all(self):
        """Without an explicit category the ``tools`` catch-all applies.

        The old name-prefix auto-detection was removed when categories moved
        to the tool classes' ``permission_category`` declarations; callers
        that know the category must pass it explicitly.
        """
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="tools", pattern="execute_sql", permission=PermissionLevel.DENY),
                PermissionRule(tool="db_tools", pattern="list_tables", permission=PermissionLevel.DENY),
            ],
        )
        manager = PermissionManager(global_config=config)

        class MockTool:
            def __init__(self, name):
                self.name = name

        tools = [MockTool("execute_sql"), MockTool("list_tables")]
        filtered = manager.filter_available_tools(tools, "chatbot")

        # ``tools.execute_sql`` DENY matches via the catch-all; the
        # ``db_tools.list_tables`` rule does NOT (no prefix guessing).
        assert [t.name for t in filtered] == ["list_tables"]


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


class TestPermissionManagerProfileSwitching:
    """switch_profile() updates global_config and clears session approvals.

    Spec decision #7: switching profiles must never leave behind prior
    'always allow' grants from a more permissive profile.
    """

    def test_active_profile_defaults_to_normal(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        assert mgr.active_profile == "normal"

    def test_active_profile_accepts_constructor_arg(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="auto")
        assert mgr.active_profile == "auto"

    def test_switch_profile_updates_active_name(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.switch_profile("auto")
        assert mgr.active_profile == "auto"

    def test_switch_profile_replaces_global_config(self):
        from datus.tools.permission.permission_config import PermissionLevel
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.switch_profile("dangerous")
        # Dangerous has default ALLOW with no rules
        assert mgr.global_config.default_permission == PermissionLevel.ALLOW
        assert len(mgr.global_config.rules) == 0

    def test_switch_profile_clears_session_approvals(self):
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        mgr.approve_for_session("db_tools", "execute_ddl")
        assert mgr._session_approvals
        mgr.switch_profile("auto")
        assert mgr._session_approvals == {}

    def test_switch_profile_with_user_overrides(self):
        from datus.tools.permission.permission_config import (
            PermissionConfig,
            PermissionLevel,
            PermissionRule,
        )
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager()
        user_overrides = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[
                PermissionRule(
                    tool="db_tools",
                    pattern="execute_ddl",
                    permission=PermissionLevel.DENY,
                )
            ],
        )
        mgr.switch_profile("auto", user_overrides=user_overrides)
        matching = [r for r in mgr.global_config.rules if r.tool == "db_tools" and r.pattern == "execute_ddl"]
        # Final matching rule's permission should be DENY (user override wins)
        final = matching[-1].permission
        final_level = PermissionLevel(final) if isinstance(final, str) else final
        assert final_level == PermissionLevel.DENY

    def test_switch_profile_unknown_raises(self):
        import pytest

        from datus.tools.permission.permission_manager import PermissionManager
        from datus.utils.exceptions import DatusException

        mgr = PermissionManager()
        with pytest.raises(DatusException, match="Unknown profile"):
            mgr.switch_profile("yolo")


class TestPermissionManagerPersistentRules:
    """``add_persistent_rule`` injects rules that survive profile switches."""

    def test_add_persistent_rule_installs_immediately(self):
        from datus.tools.permission.permission_config import PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="normal")
        rule = PermissionRule(tool="skills", pattern="exec_*", permission=PermissionLevel.ASK)
        mgr.add_persistent_rule(rule)

        assert rule in mgr._persistent_rules
        assert any(r.tool == "skills" and r.pattern == "exec_*" for r in mgr.global_config.rules)

    def test_add_persistent_rule_skips_existing_identical_rule(self):
        from datus.tools.permission.permission_config import PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="normal")
        rule = PermissionRule(tool="skills", pattern="exec_*", permission=PermissionLevel.ASK)
        before = len(mgr.global_config.rules)
        mgr.add_persistent_rule(rule)
        mid = len(mgr.global_config.rules)
        # Second call with same tool+pattern must not stack duplicates on the
        # rules list (``_persistent_rules`` bookkeeping list still appends).
        mgr.add_persistent_rule(rule)
        after = len(mgr.global_config.rules)

        assert mid == before + 1
        assert after == mid
        assert mgr._persistent_rules.count(rule) == 2

    def test_switch_profile_reapplies_persistent_rule(self):
        from datus.tools.permission.permission_config import PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="normal")
        rule = PermissionRule(tool="skills", pattern="bash_*", permission=PermissionLevel.ASK)
        mgr.add_persistent_rule(rule)

        mgr.switch_profile("dangerous")
        assert any(r.tool == "skills" and r.pattern == "bash_*" for r in mgr.global_config.rules)

    def test_switch_profile_persistent_rule_dedup_against_fresh_base(self):
        """If the rebuilt base already contains the rule, don't insert again.

        Exercises the ``if not any(...)`` branch inside ``switch_profile`` —
        the safeguard must preserve last-match-wins semantics rather than
        stacking duplicate ``skills.bash_*`` entries on every re-switch.
        """
        from datus.tools.permission.permission_config import PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager

        mgr = PermissionManager(active_profile="normal")
        rule = PermissionRule(tool="skills", pattern="*", permission=PermissionLevel.ASK)
        mgr.add_persistent_rule(rule)
        mgr.switch_profile("normal")
        # ``skills.*`` is already in the NORMAL profile; persistent-rule
        # re-application should be a no-op — assert we have exactly one.
        matching = [r for r in mgr.global_config.rules if r.tool == "skills" and r.pattern == "*"]
        assert len(matching) == 1

    def test_copy_config_isolates_persistent_rule_from_shared_profile(self):
        """Two managers with the same profile must not share a rules list.

        Regression for the profile-singleton leak: ``add_persistent_rule``
        on one manager previously mutated ``get_profile("normal").rules``
        via ``insert(0, …)``, so every subsequent manager inherited the
        custom rule. With ``_copy_config`` the rules list is independent.
        """
        from datus.tools.permission.permission_config import PermissionLevel, PermissionRule
        from datus.tools.permission.permission_manager import PermissionManager
        from datus.tools.permission.profiles import get_profile

        before = len(get_profile("normal").rules)
        mgr_a = PermissionManager(global_config=get_profile("normal"), active_profile="normal")
        mgr_a.add_persistent_rule(PermissionRule(tool="skills", pattern="only_a", permission=PermissionLevel.ASK))
        # Building another manager from the same profile must not see ``only_a``.
        mgr_b = PermissionManager(global_config=get_profile("normal"), active_profile="normal")
        assert not any(r.pattern == "only_a" for r in mgr_b.global_config.rules)
        # The shared profile itself is untouched.
        assert len(get_profile("normal").rules) == before


class TestPermissionManagerProjectBashAllows:
    """Project-scope bash allows replay across profile switches — except onto
    a profile that intentionally carries no command-level ruleset."""

    def _manager_with_grant(self):
        from unittest.mock import patch

        from datus.tools.permission.permission_manager import PermissionManager
        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(global_config=get_profile("normal"), active_profile="normal")
        with patch("datus.configuration.project_config.append_project_bash_allow"):
            mgr.add_project_bash_allow("make:*")
        assert "make:*" in mgr.global_config.bash_commands.allow
        return mgr

    def test_switch_to_dangerous_does_not_reenable_bash_gating(self):
        """``dangerous`` keeps ``bash_commands`` unset so the fine-grained
        bash gate steps aside entirely; replaying a project grant onto it
        would flip that documented zero-friction posture."""
        mgr = self._manager_with_grant()
        mgr.switch_profile("dangerous")
        assert mgr.global_config.bash_commands is None

    def test_switch_between_rule_profiles_replays_grant(self):
        mgr = self._manager_with_grant()
        mgr.switch_profile("auto")
        assert "make:*" in mgr.global_config.bash_commands.allow


class TestPermissionManagerPluginBashRules:
    """Plugin-declared bash rules survive runtime profile switches."""

    @staticmethod
    def _plugin_rules_map():
        from datus.tools.permission.bash_rules import BashCommandRules

        return {
            "normal": BashCommandRules(allow=["datus hello greet:*"], ask=["datus hello config set:*"]),
            "auto": BashCommandRules(allow=["datus hello:*"]),
        }

    def _manager(self):
        from datus.tools.permission.profiles import build_effective_config

        rules_map = self._plugin_rules_map()
        return PermissionManager(
            global_config=build_effective_config("normal", None, plugin_bash_rules=rules_map["normal"]),
            active_profile="normal",
            plugin_bash_rules=rules_map,
        )

    def test_initial_config_carries_normal_plugin_rules(self):
        mgr = self._manager()
        assert "datus hello greet:*" in mgr.global_config.bash_commands.allow

    def test_switch_to_auto_applies_auto_plugin_rules(self):
        mgr = self._manager()
        mgr.switch_profile("auto")
        assert "datus hello:*" in mgr.global_config.bash_commands.allow
        # normal-only rules are not carried across profiles.
        assert "datus hello greet:*" not in mgr.global_config.bash_commands.allow

    def test_switch_to_dangerous_stays_ungated(self):
        mgr = self._manager()
        mgr.switch_profile("dangerous")
        assert mgr.global_config.bash_commands is None

    def test_switch_back_to_normal_restores_plugin_rules(self):
        mgr = self._manager()
        mgr.switch_profile("dangerous")
        mgr.switch_profile("normal")
        assert "datus hello greet:*" in mgr.global_config.bash_commands.allow
        assert "datus hello config set:*" in mgr.global_config.bash_commands.ask

    def test_plugin_rules_coexist_with_project_bash_allows(self):
        from unittest.mock import patch

        mgr = self._manager()
        with patch("datus.configuration.project_config.append_project_bash_allow"):
            mgr.add_project_bash_allow("make:*")
        mgr.switch_profile("auto")
        assert "make:*" in mgr.global_config.bash_commands.allow
        assert "datus hello:*" in mgr.global_config.bash_commands.allow

    def test_profile_singletons_never_mutated_by_switches(self):
        from datus.tools.permission.profiles import get_profile

        before = list(get_profile("auto").bash_commands.allow)
        mgr = self._manager()
        mgr.switch_profile("auto")
        mgr.switch_profile("normal")
        assert get_profile("auto").bash_commands.allow == before

    def test_manager_without_plugin_rules_unchanged(self):
        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(global_config=get_profile("normal"), active_profile="normal")
        mgr.switch_profile("auto")
        assert all("datus hello" not in p for p in mgr.global_config.bash_commands.allow)

    def test_switch_profile_merge_order_matches_startup(self):
        """``switch_profile`` must layer plugin rules BEFORE user overrides so a
        runtime ``/profile`` switch produces the same bash_commands as startup's
        ``build_effective_config`` (plugin-then-user), not user-then-plugin."""
        from datus.tools.permission.profiles import build_effective_config

        rules_map = self._plugin_rules_map()
        user_raw = {"bash_commands": {"ask": ["datus hello config set:*"], "allow": ["git status:*"]}}
        user_overrides = PermissionConfig.from_dict(user_raw)

        mgr = PermissionManager(
            global_config=build_effective_config("normal", None, plugin_bash_rules=rules_map["normal"]),
            active_profile="normal",
            plugin_bash_rules=rules_map,
        )
        mgr.switch_profile("auto", user_overrides=user_overrides)

        # The startup path is the reference: plugin base merged first, then user.
        expected = build_effective_config("auto", user_raw, plugin_bash_rules=rules_map["auto"])
        assert mgr.global_config.bash_commands.allow == expected.bash_commands.allow
        assert mgr.global_config.bash_commands.ask == expected.bash_commands.ask


class TestProjectBashGrants:
    """Exact-match project grants that can bypass ask-rule hits."""

    def test_constructor_seeds_grants(self):
        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(
            global_config=get_profile("normal"),
            project_bash_allows=["datus hello config set:*"],
        )
        assert mgr.has_project_bash_grant("datus hello config set:*")
        assert not mgr.has_project_bash_grant("datus hello:*")
        assert not mgr.has_project_bash_grant(None)

    def test_add_project_bash_allow_registers_grant(self):
        from unittest.mock import patch

        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(global_config=get_profile("normal"))
        with patch("datus.configuration.project_config.append_project_bash_allow"):
            mgr.add_project_bash_allow("datus hello config set:*")
        assert mgr.has_project_bash_grant("datus hello config set:*")

    def test_grants_survive_profile_switches(self):
        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(
            global_config=get_profile("normal"),
            project_bash_allows=["datus hello config set:*"],
        )
        mgr.switch_profile("auto")
        assert mgr.has_project_bash_grant("datus hello config set:*")

    def test_is_plugin_ask_pattern_scoped_to_active_profile(self):
        from datus.tools.permission.bash_rules import BashCommandRules
        from datus.tools.permission.profiles import get_profile

        mgr = PermissionManager(
            global_config=get_profile("normal"),
            active_profile="normal",
            plugin_bash_rules={
                "normal": BashCommandRules(ask=["datus hello config set:*"]),
                "auto": BashCommandRules(ask=["datus hello config del:*"]),
            },
        )
        assert mgr.is_plugin_ask_pattern("datus hello config set:*")
        assert not mgr.is_plugin_ask_pattern("datus hello config del:*")  # auto-only
        assert not mgr.is_plugin_ask_pattern("docker:*")
        assert not mgr.is_plugin_ask_pattern(None)
        mgr.switch_profile("auto")
        assert mgr.is_plugin_ask_pattern("datus hello config del:*")
        assert not mgr.is_plugin_ask_pattern("datus hello config set:*")
