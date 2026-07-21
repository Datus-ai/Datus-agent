# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for permission configuration models.

Tests PermissionLevel, PermissionRule, and PermissionConfig classes.
"""

import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule


class TestPermissionLevel:
    """Tests for PermissionLevel enum."""

    def test_permission_level_values(self):
        """Test that PermissionLevel has the expected values."""
        assert PermissionLevel.ALLOW.value == "allow"
        assert PermissionLevel.DENY.value == "deny"
        assert PermissionLevel.ASK.value == "ask"

    def test_permission_level_from_string(self):
        """Test creating PermissionLevel from string values."""
        assert PermissionLevel("allow") == PermissionLevel.ALLOW
        assert PermissionLevel("deny") == PermissionLevel.DENY
        assert PermissionLevel("ask") == PermissionLevel.ASK

    def test_permission_level_invalid_value(self):
        """Test that invalid permission level raises ValueError."""
        with pytest.raises(ValueError):
            PermissionLevel("invalid")

    def test_permission_level_string_representation(self):
        """Test string representation of PermissionLevel."""
        assert str(PermissionLevel.ALLOW) == "PermissionLevel.ALLOW"
        assert PermissionLevel.ALLOW.value == "allow"


class TestPermissionRule:
    """Tests for PermissionRule model."""

    def test_permission_rule_creation(self):
        """Test creating a PermissionRule."""
        rule = PermissionRule(
            tool="db_tools",
            pattern="execute_sql",
            permission=PermissionLevel.ASK,
        )
        assert rule.tool == "db_tools"
        assert rule.pattern == "execute_sql"
        assert rule.permission == PermissionLevel.ASK

    def test_permission_rule_with_wildcard_pattern(self):
        """Test creating a PermissionRule with wildcard pattern."""
        rule = PermissionRule(
            tool="skills",
            pattern="dangerous-*",
            permission=PermissionLevel.DENY,
        )
        assert rule.tool == "skills"
        assert rule.pattern == "dangerous-*"
        assert rule.permission == PermissionLevel.DENY

    def test_permission_rule_with_string_permission(self):
        """Test creating a PermissionRule with string permission value."""
        rule = PermissionRule(
            tool="mcp",
            pattern="*",
            permission="allow",
        )
        assert rule.permission == PermissionLevel.ALLOW

    def test_permission_rule_all_wildcards(self):
        """Test creating a PermissionRule with all wildcards."""
        rule = PermissionRule(
            tool="*",
            pattern="*",
            permission=PermissionLevel.ALLOW,
        )
        assert rule.tool == "*"
        assert rule.pattern == "*"

    def test_permission_rule_matches_basic(self):
        """Test basic pattern matching."""
        rule = PermissionRule(
            tool="db_tools",
            pattern="execute_sql",
            permission=PermissionLevel.ASK,
        )
        assert rule.matches("db_tools", "execute_sql") is True
        assert rule.matches("db_tools", "list_tables") is False
        assert rule.matches("mcp", "execute_sql") is False

    def test_permission_rule_matches_wildcard_pattern(self):
        """Test wildcard pattern matching."""
        rule = PermissionRule(
            tool="skills",
            pattern="sql-*",
            permission=PermissionLevel.ALLOW,
        )
        assert rule.matches("skills", "sql-optimization") is True
        assert rule.matches("skills", "sql-debugging") is True
        assert rule.matches("skills", "data-analysis") is False

    def test_permission_rule_matches_tool_wildcard(self):
        """Test tool wildcard matching."""
        rule = PermissionRule(
            tool="*",
            pattern="dangerous-*",
            permission=PermissionLevel.DENY,
        )
        assert rule.matches("skills", "dangerous-script") is True
        assert rule.matches("db_tools", "dangerous-query") is True
        assert rule.matches("mcp", "safe-tool") is False

    def test_permission_rule_matches_all_wildcards(self):
        """Test matching with all wildcards."""
        rule = PermissionRule(
            tool="*",
            pattern="*",
            permission=PermissionLevel.ALLOW,
        )
        assert rule.matches("any_tool", "any_pattern") is True
        assert rule.matches("db_tools", "execute_sql") is True


class TestPermissionConfig:
    """Tests for PermissionConfig model."""

    def test_permission_config_default(self):
        """Test PermissionConfig with default values."""
        config = PermissionConfig()
        assert config.default_permission == PermissionLevel.ALLOW
        assert config.rules == []

    def test_permission_config_with_rules(self):
        """Test PermissionConfig with rules."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ALLOW),
                PermissionRule(tool="skills", pattern="dangerous-*", permission=PermissionLevel.DENY),
            ],
        )
        assert config.default_permission == PermissionLevel.ASK
        assert len(config.rules) == 2

    def test_permission_config_from_dict(self):
        """Test creating PermissionConfig from dictionary."""
        config_dict = {
            "default_permission": "deny",
            "rules": [
                {"tool": "db_tools", "pattern": "*", "permission": "allow"},
                {"tool": "skills", "pattern": "internal-*", "permission": "ask"},
            ],
        }
        config = PermissionConfig.from_dict(config_dict)
        assert config.default_permission == PermissionLevel.DENY
        assert len(config.rules) == 2
        assert config.rules[0].tool == "db_tools"
        assert config.rules[0].permission == PermissionLevel.ALLOW
        assert config.rules[1].tool == "skills"
        assert config.rules[1].permission == PermissionLevel.ASK

    def test_permission_config_from_dict_empty(self):
        """Test creating PermissionConfig from empty dictionary."""
        config = PermissionConfig.from_dict({})
        assert config.default_permission == PermissionLevel.ALLOW
        assert config.rules == []

    def test_permission_config_from_dict_only_default(self):
        """Test creating PermissionConfig with only default permission."""
        config = PermissionConfig.from_dict({"default_permission": "ask"})
        assert config.default_permission == PermissionLevel.ASK
        assert config.rules == []

    def test_permission_config_from_dict_only_rules(self):
        """Test creating PermissionConfig with only rules."""
        config = PermissionConfig.from_dict(
            {
                "rules": [
                    {"tool": "mcp", "pattern": "*", "permission": "deny"},
                ],
            }
        )
        assert config.default_permission == PermissionLevel.ALLOW
        assert len(config.rules) == 1

    def test_permission_config_serialization(self):
        """Test PermissionConfig serialization."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ALLOW),
            ],
        )
        data = config.model_dump()
        assert data["default_permission"] == PermissionLevel.ASK
        assert len(data["rules"]) == 1
        assert data["rules"][0]["tool"] == "db_tools"

    def test_permission_config_merge_with_none(self):
        """Test merge_with returns self when override is None."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            ],
        )
        merged = config.merge_with(None)
        assert merged is config

    def test_permission_config_merge_with_default_only(self):
        """Test merge_with uses override's default_permission even without rules."""
        base_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            ],
        )
        override_config = PermissionConfig(
            default_permission=PermissionLevel.DENY,
            rules=[],  # No rules, just override the default
        )
        merged = base_config.merge_with(override_config)

        # Override's default_permission should be used even without rules
        assert merged.default_permission == PermissionLevel.DENY
        # Base rules should still be present
        assert len(merged.rules) == 1
        assert merged.rules[0].tool == "db_tools"

    def test_permission_config_merge_with_rules(self):
        """Test merge_with combines rules with override taking precedence."""
        base_config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ALLOW),
            ],
        )
        override_config = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[
                PermissionRule(tool="skills", pattern="*", permission=PermissionLevel.DENY),
            ],
        )
        merged = base_config.merge_with(override_config)

        assert merged.default_permission == PermissionLevel.ASK
        # Rules should be combined: base rules first, then override rules
        assert len(merged.rules) == 2
        assert merged.rules[0].tool == "db_tools"  # Base rule
        assert merged.rules[1].tool == "skills"  # Override rule (evaluated later)


class TestClassifySqlKind:
    """Kind → class mapping used by the execute_sql permission gate."""

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("select", "read"),
            ("metadata", "read"),
            ("explain", "read"),
            ("insert", "write"),
            ("create", "write"),
            ("ddl", "write"),
            ("context_set", "write"),
            ("update", "destructive"),
            ("delete", "destructive"),
            ("merge", "destructive"),
            ("drop", "destructive"),
            ("truncate", "destructive"),
            ("alter", "destructive"),
            ("replace", "destructive"),
            ("unknown", "unknown"),
        ],
    )
    def test_known_kinds(self, kind, expected):
        from datus.tools.permission.permission_config import classify_sql_kind

        assert classify_sql_kind(kind) == expected

    def test_unmapped_kind_fails_safe_to_unknown(self):
        from datus.tools.permission.permission_config import classify_sql_kind

        assert classify_sql_kind("frobnicate") == "unknown"
        assert classify_sql_kind("") == "unknown"


class TestSqlStatementRules:
    """Per-class SQL statement rules model."""

    def test_defaults_match_normal_posture(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        rules = SqlStatementRules()
        assert rules.level_for_class("read") == PermissionLevel.ALLOW
        assert rules.level_for_class("write") == PermissionLevel.ASK
        assert rules.level_for_class("destructive") == PermissionLevel.ASK
        assert rules.level_for_class("unknown") == PermissionLevel.ASK

    def test_level_for_class_returns_permission_level(self):
        """use_enum_values stores strings; level_for_class must coerce back."""
        from datus.tools.permission.permission_config import SqlStatementRules

        level = SqlStatementRules().level_for_class("read")
        assert isinstance(level, PermissionLevel)

    def test_level_for_unmapped_class_falls_to_unknown(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        rules = SqlStatementRules(unknown=PermissionLevel.DENY)
        assert rules.level_for_class("nonsense") == PermissionLevel.DENY

    def test_from_dict_none_returns_none(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        assert SqlStatementRules.from_dict(None) is None

    def test_from_dict_tracks_explicit_fields_only(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        rules = SqlStatementRules.from_dict({"write": "allow"})
        assert rules.model_fields_set == {"write"}
        assert rules.level_for_class("write") == PermissionLevel.ALLOW
        assert rules.level_for_class("destructive") == PermissionLevel.ASK

    def test_from_dict_empty_dict_has_no_explicit_fields(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        rules = SqlStatementRules.from_dict({})
        assert isinstance(rules, SqlStatementRules)
        assert rules.model_fields_set == set()

    def test_from_dict_rejects_non_mapping(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        with pytest.raises(ValueError, match="mapping"):
            SqlStatementRules.from_dict(["read"])

    def test_from_dict_rejects_invalid_level(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        with pytest.raises(ValueError):
            SqlStatementRules.from_dict({"destructive": "yolo"})

    def test_merge_with_override_if_set(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        base = SqlStatementRules(write=PermissionLevel.ALLOW)
        override = SqlStatementRules.from_dict({"destructive": "deny"})
        merged = base.merge_with(override)
        # Explicit override field wins; unset override fields keep base values.
        assert merged.level_for_class("destructive") == PermissionLevel.DENY
        assert merged.level_for_class("write") == PermissionLevel.ALLOW
        assert merged.level_for_class("read") == PermissionLevel.ALLOW

    def test_merge_with_none_returns_self(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        base = SqlStatementRules(write=PermissionLevel.ALLOW)
        assert base.merge_with(None) is base

    def test_merge_does_not_mutate_base(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        base = SqlStatementRules()
        base.merge_with(SqlStatementRules.from_dict({"write": "deny"}))
        assert base.level_for_class("write") == PermissionLevel.ASK


class TestPermissionConfigSqlStatements:
    """sql_statements carried through PermissionConfig from_dict / merge_with."""

    def test_from_dict_parses_sql_statements(self):
        config = PermissionConfig.from_dict(
            {"default": "ask", "sql_statements": {"write": "allow", "destructive": "ask"}}
        )
        assert config.sql_statements.level_for_class("write") == PermissionLevel.ALLOW
        assert config.sql_statements.level_for_class("destructive") == PermissionLevel.ASK

    def test_from_dict_without_sql_statements_is_none(self):
        config = PermissionConfig.from_dict({"default": "ask"})
        assert config.sql_statements is None

    def test_merge_with_layers_sql_statements(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        base = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            sql_statements=SqlStatementRules(write=PermissionLevel.ALLOW),
        )
        override = PermissionConfig.from_dict({"default": "ask", "sql_statements": {"destructive": "deny"}})
        merged = base.merge_with(override)
        assert merged.sql_statements.level_for_class("write") == PermissionLevel.ALLOW
        assert merged.sql_statements.level_for_class("destructive") == PermissionLevel.DENY

    def test_merge_with_no_base_takes_override(self):
        override = PermissionConfig.from_dict({"default": "ask", "sql_statements": {"write": "allow"}})
        merged = PermissionConfig(default_permission=PermissionLevel.ASK).merge_with(override)
        assert merged.sql_statements.level_for_class("write") == PermissionLevel.ALLOW

    def test_merge_with_override_lacking_sql_statements_keeps_base(self):
        from datus.tools.permission.permission_config import SqlStatementRules

        base = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            sql_statements=SqlStatementRules(write=PermissionLevel.ALLOW),
        )
        merged = base.merge_with(PermissionConfig(default_permission=PermissionLevel.ASK))
        assert merged.sql_statements.level_for_class("write") == PermissionLevel.ALLOW
