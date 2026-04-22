"""Tests for predefined permission profiles."""

import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel
from datus.tools.permission.profiles import (
    AUTO,
    DANGEROUS,
    NORMAL,
    PROFILE_NAMES,
    get_profile,
)


class TestProfileRegistry:
    def test_three_profiles_exist(self):
        assert PROFILE_NAMES == ("normal", "auto", "dangerous")

    def test_get_profile_returns_expected_instance(self):
        assert get_profile("normal") is NORMAL
        assert get_profile("auto") is AUTO
        assert get_profile("dangerous") is DANGEROUS

    def test_get_profile_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("yolo")

    def test_normal_default_is_ask(self):
        assert NORMAL.default_permission == PermissionLevel.ASK

    def test_auto_default_is_ask(self):
        assert AUTO.default_permission == PermissionLevel.ASK

    def test_dangerous_default_is_allow(self):
        assert DANGEROUS.default_permission == PermissionLevel.ALLOW


class TestNormalProfile:
    def test_read_tools_allowed(self):
        """Normal allows context search, date parsing, and DB/BI/FS reads."""
        config = NORMAL
        assert _resolve(config, "context_search_tools", "search_metrics") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "read_query") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "list_tables") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "list_dashboards") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "read_file") == PermissionLevel.ALLOW

    def test_writes_ask(self):
        """Normal ASKs on all write-ish tools via default_permission."""
        config = NORMAL
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ASK
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ASK
        assert _resolve(config, "tools", "todo_write") == PermissionLevel.ASK

    def test_named_destructive_denied(self):
        """Normal DENYs named destructive BI and scheduler tools."""
        config = NORMAL
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.DENY
        assert _resolve(config, "bi_tools", "delete_chart") == PermissionLevel.DENY
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.DENY

    def test_mcp_and_skills_ask(self):
        config = NORMAL
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ASK


class TestAutoProfile:
    def test_inherits_normal_reads(self):
        config = AUTO
        assert _resolve(config, "db_tools", "read_query") == PermissionLevel.ALLOW
        assert _resolve(config, "context_search_tools", "search_metrics") == PermissionLevel.ALLOW

    def test_workspace_writes_allowed(self):
        config = AUTO
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "edit_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "create_directory") == PermissionLevel.ALLOW

    def test_bi_write_allowed_but_delete_denied(self):
        config = AUTO
        assert _resolve(config, "bi_tools", "create_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "update_chart") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.DENY

    def test_scheduler_trigger_still_asks(self):
        config = AUTO
        assert _resolve(config, "scheduler_tools", "submit_sql_job") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "trigger_scheduler_job") == PermissionLevel.ASK

    def test_db_writes_still_ask(self):
        """No env detection in MVP — all DB writes always ASK."""
        config = AUTO
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "execute_write") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "transfer_query_result") == PermissionLevel.ASK

    def test_mcp_and_skills_still_ask(self):
        config = AUTO
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ASK


class TestDangerousProfile:
    def test_everything_allowed_by_default(self):
        config = DANGEROUS
        assert _resolve(config, "db_tools", "execute_ddl") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.ALLOW
        assert _resolve(config, "mcp.anything", "whatever") == PermissionLevel.ALLOW
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ALLOW


def _resolve(config: PermissionConfig, category: str, pattern: str) -> PermissionLevel:
    """Walk the rules last-match-wins, returning the final PermissionLevel.

    Uses the production ``PermissionRule.matches()`` so a future change to
    the matcher (e.g., glob semantics) automatically reflects in tests.
    """
    result = config.default_permission
    for rule in config.rules:
        if rule.matches(category, pattern):
            result = PermissionLevel(rule.permission) if isinstance(rule.permission, str) else rule.permission
    return result
