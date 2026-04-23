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

    def test_sub_agent_delegation_allowed(self):
        """``task()`` delegation is ALLOW — the subagent's own hooks gate its calls."""
        config = NORMAL
        assert _resolve(config, "sub_agent_tools", "task") == PermissionLevel.ALLOW

    def test_ask_user_always_allowed(self):
        """``ask_user`` is the user-interaction channel itself — must be ALLOW.

        Asking the permission broker "may I ask the user?" is absurd UX
        (it is the user, already present). Regression guard.
        """
        config = NORMAL
        assert _resolve(config, "tools", "ask_user") == PermissionLevel.ALLOW

    def test_tools_bucket_read_patterns_allowed(self):
        """Benign read-only helpers in the ``tools`` catch-all follow the
        ``list_*`` / ``search_*`` / ``get_*`` convention and should ALLOW."""
        config = NORMAL
        assert _resolve(config, "tools", "list_document_nav") == PermissionLevel.ALLOW
        assert _resolve(config, "tools", "search_document") == PermissionLevel.ALLOW
        assert _resolve(config, "tools", "get_anything") == PermissionLevel.ALLOW
        # Writes still ASK via default.
        assert _resolve(config, "tools", "todo_write") == PermissionLevel.ASK


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

    def test_bi_write_allowed_delete_asks(self):
        """Auto downgrades NORMAL's DENY on destructives to ASK — user is
        already in productive mode, force-switch to Dangerous would be hostile."""
        config = AUTO
        assert _resolve(config, "bi_tools", "create_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "update_chart") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.ASK
        assert _resolve(config, "bi_tools", "delete_chart") == PermissionLevel.ASK
        assert _resolve(config, "bi_tools", "delete_dataset") == PermissionLevel.ASK

    def test_scheduler_trigger_still_asks(self):
        config = AUTO
        assert _resolve(config, "scheduler_tools", "submit_sql_job") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "trigger_scheduler_job") == PermissionLevel.ASK
        # destructive also downgraded from DENY to ASK
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.ASK

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


class TestBuildEffectiveConfig:
    def test_no_user_raw_returns_profile_base(self):
        from datus.tools.permission.profiles import AUTO, build_effective_config

        effective = build_effective_config("auto", None)
        # Merging with None should return the base itself (identity via merge_with)
        assert effective is AUTO or (
            effective.default_permission == AUTO.default_permission and len(effective.rules) == len(AUTO.rules)
        )

    def test_empty_user_raw_returns_profile_base(self):
        from datus.tools.permission.profiles import AUTO, build_effective_config

        effective = build_effective_config("auto", {})
        assert effective.default_permission == AUTO.default_permission

    def test_user_rules_preserve_profile_default(self):
        from datus.tools.permission.permission_config import PermissionLevel
        from datus.tools.permission.profiles import build_effective_config

        effective = build_effective_config(
            "auto",
            {"rules": [{"tool": "db_tools", "pattern": "execute_ddl", "permission": "deny"}]},
        )
        # Auto's default is ASK; user didn't set default, so it stays ASK
        assert effective.default_permission == PermissionLevel.ASK

    def test_user_explicit_default_wins(self):
        from datus.tools.permission.permission_config import PermissionLevel
        from datus.tools.permission.profiles import build_effective_config

        effective = build_effective_config(
            "normal",
            {"default": "allow", "rules": []},
        )
        assert effective.default_permission == PermissionLevel.ALLOW

    def test_unknown_profile_raises(self):
        import pytest

        from datus.tools.permission.profiles import build_effective_config

        with pytest.raises(ValueError, match="Unknown profile"):
            build_effective_config("yolo", {})
