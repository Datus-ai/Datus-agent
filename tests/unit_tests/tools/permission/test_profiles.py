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
        # ``execute_sql`` has no static rule — its read-vs-write gating lives in
        # ``PermissionHooks._handle_sql_permission`` (read auto-allow, write ASK),
        # so statically it resolves to the profile default (ASK). ``verify_sql``
        # remains an explicit read ALLOW.
        assert _resolve(config, "db_tools", "verify_sql") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "execute_sql") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "list_tables") == PermissionLevel.ALLOW
        assert _resolve(config, "db_tools", "search_table") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "list_dashboards") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "read_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "glob") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "grep") == PermissionLevel.ALLOW
        # Both plan-read tools are ALLOW in NORMAL — they only inspect the
        # in-memory todolist and cannot affect external systems.
        assert _resolve(config, "tools", "todo_list") == PermissionLevel.ALLOW
        assert _resolve(config, "tools", "todo_read") == PermissionLevel.ALLOW

    def test_writes_ask(self):
        """Normal ASKs on all write-ish tools via default_permission."""
        config = NORMAL
        assert _resolve(config, "db_tools", "execute_sql") == PermissionLevel.ASK
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ASK
        assert _resolve(config, "tools", "todo_write") == PermissionLevel.ASK

    def test_named_destructive_denied(self):
        """Normal DENYs named destructive BI and scheduler tools."""
        config = NORMAL
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.DENY
        assert _resolve(config, "bi_tools", "delete_chart") == PermissionLevel.DENY
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.DENY

    def test_mcp_asks_skill_loading_allowed(self):
        config = NORMAL
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ALLOW
        assert _resolve(config, "skills", "load_skill") == PermissionLevel.ALLOW

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
        assert _resolve(config, "tools", "validate_skill") == PermissionLevel.ALLOW
        # Writes still ASK via default.
        assert _resolve(config, "tools", "todo_write") == PermissionLevel.ASK

    def test_generation_helpers_allowed(self):
        """GenerationTools helpers ride the ``semantic_tools`` category and
        should not trigger permission prompts in normal mode."""
        config = NORMAL
        assert _resolve(config, "semantic_tools", "check_semantic_object_exists") == PermissionLevel.ALLOW
        assert _resolve(config, "semantic_tools", "generate_sql_summary_id") == PermissionLevel.ALLOW
        assert _resolve(config, "semantic_tools", "end_semantic_model_generation") == PermissionLevel.ALLOW
        assert _resolve(config, "semantic_tools", "end_metric_generation") == PermissionLevel.ALLOW

    def test_all_semantic_tools_allowed(self):
        config = NORMAL
        assert _resolve(config, "semantic_tools", "validate_semantic") == PermissionLevel.ALLOW
        assert _resolve(config, "semantic_tools", "attribution_analyze") == PermissionLevel.ALLOW
        assert _resolve(config, "semantic_tools", "future_semantic_tool") == PermissionLevel.ALLOW

    def test_reference_template_tools_allowed(self):
        """All reference-template helpers are read-only end to end —
        ``execute_reference_template`` renders Jinja then runs
        ``db_tools.read_query``."""
        config = NORMAL
        assert _resolve(config, "reference_template_tools", "search_reference_template") == PermissionLevel.ALLOW
        assert _resolve(config, "reference_template_tools", "get_reference_template") == PermissionLevel.ALLOW
        assert _resolve(config, "reference_template_tools", "render_reference_template") == PermissionLevel.ALLOW
        assert _resolve(config, "reference_template_tools", "execute_reference_template") == PermissionLevel.ALLOW

    def test_artifact_tools_allowed(self):
        """Artifact authoring helpers are subagent-internal state mutations;
        the user reviews the artifact as a whole via the rendered preview."""
        config = NORMAL
        assert _resolve(config, "artifact_tools", "start_new_report") == PermissionLevel.ALLOW
        assert _resolve(config, "artifact_tools", "bind_existing_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "artifact_tools", "save_query_template") == PermissionLevel.ALLOW
        assert _resolve(config, "artifact_tools", "validate_render") == PermissionLevel.ALLOW

    def test_platform_doc_reads_and_web_tool_allowed(self):
        """Doc lookups are local reads; ``web_tool`` is read-only retrieval
        (web_fetch hardened against SSRF) and provider-native web tools bypass
        local hooks anyway, so both are ALLOW rather than ASK."""
        config = NORMAL
        assert _resolve(config, "platform_doc_tools", "list_document_nav") == PermissionLevel.ALLOW
        assert _resolve(config, "platform_doc_tools", "get_document") == PermissionLevel.ALLOW
        assert _resolve(config, "platform_doc_tools", "search_document") == PermissionLevel.ALLOW
        assert _resolve(config, "web_tool", "web_search") == PermissionLevel.ALLOW
        assert _resolve(config, "web_tool", "web_fetch") == PermissionLevel.ALLOW


class TestAutoProfile:
    def test_inherits_normal_reads(self):
        config = AUTO
        assert _resolve(config, "db_tools", "verify_sql") == PermissionLevel.ALLOW
        assert _resolve(config, "context_search_tools", "search_metrics") == PermissionLevel.ALLOW

    def test_workspace_writes_allowed(self):
        config = AUTO
        # ``write_file`` / ``edit_file`` / ``delete_file`` are the full set of
        # write tools ``FilesystemFuncTool`` exposes today (``create_directory``
        # / ``move_file`` were removed in the #561 refactor and used to live
        # here as dead rules — see ``test_dead_filesystem_rules_absent``).
        assert _resolve(config, "filesystem_tools", "write_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "edit_file") == PermissionLevel.ALLOW
        assert _resolve(config, "filesystem_tools", "delete_file") == PermissionLevel.ALLOW

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
        """No env detection in MVP — all DB writes always ASK. ``execute_sql``
        writes resolve to the profile default ASK (the hook auto-allows only
        reads); the standalone transfer tool keeps its explicit ASK rule."""
        config = AUTO
        assert _resolve(config, "db_tools", "execute_sql") == PermissionLevel.ASK
        assert _resolve(config, "db_tools", "transfer_query_result") == PermissionLevel.ASK

    def test_mcp_still_asks_skill_loading_allowed(self):
        config = AUTO
        assert _resolve(config, "mcp.filesystem", "read_file") == PermissionLevel.ASK
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ALLOW
        assert _resolve(config, "skills", "load_skill") == PermissionLevel.ALLOW


class TestDangerousProfile:
    def test_everything_allowed_by_default(self):
        config = DANGEROUS
        assert _resolve(config, "db_tools", "execute_sql") == PermissionLevel.ALLOW
        assert _resolve(config, "bi_tools", "delete_dashboard") == PermissionLevel.ALLOW
        assert _resolve(config, "scheduler_tools", "delete_job") == PermissionLevel.ALLOW
        assert _resolve(config, "mcp.anything", "whatever") == PermissionLevel.ALLOW
        assert _resolve(config, "skills", "any-skill") == PermissionLevel.ALLOW


class TestFilesystemRuleSurface:
    """Filesystem rules must match the tool surface exposed by
    ``FilesystemFuncTool.available_tools``.

    ``#561`` reduced the toolset to five methods (``read_file``,
    ``write_file``, ``edit_file``, ``glob``, ``grep``) but the rule tables
    kept five stale patterns until this PR. The asserts below lock in that
    cleanup so a future refactor doesn't silently grow another dead rule.
    """

    _DEAD_PATTERNS = ("list_*", "directory_tree", "search_files", "create_directory", "move_file")
    _LIVE_PATTERNS = ("read_*", "glob", "grep", "write_file", "edit_file", "delete_file")

    def test_dead_filesystem_rules_absent(self):
        for cfg in (NORMAL, AUTO):
            fs_patterns = {r.pattern for r in cfg.rules if r.tool == "filesystem_tools"}
            for dead in self._DEAD_PATTERNS:
                assert dead not in fs_patterns, (
                    f"Dead filesystem rule '{dead}' resurfaced in profile rules. "
                    "FilesystemFuncTool no longer exposes this method — see "
                    "datus/tools/permission/profiles.py docstring."
                )

    def test_live_filesystem_rules_cover_actual_tool_surface(self):
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        fs_tool = FilesystemFuncTool(root_path="/tmp")
        tool_names = {t.name for t in fs_tool.available_tools()}
        # AUTO contains every NORMAL rule by construction; check there.
        fs_patterns = {r.pattern for r in AUTO.rules if r.tool == "filesystem_tools"}
        assert fs_patterns == set(self._LIVE_PATTERNS), (
            f"Expected live patterns {sorted(self._LIVE_PATTERNS)}; got {sorted(fs_patterns)}"
        )
        # Every live tool name must be resolvable through the AUTO rules
        # (either matched by an exact name or by a wildcard like ``read_*``).
        for name in tool_names:
            assert _resolve(AUTO, "filesystem_tools", name) in {
                PermissionLevel.ALLOW,
                PermissionLevel.ASK,
            }, f"AUTO rule lookup returned an unexpected level for filesystem_tools.{name}"


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
            {"rules": [{"tool": "db_tools", "pattern": "execute_sql", "permission": "deny"}]},
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
