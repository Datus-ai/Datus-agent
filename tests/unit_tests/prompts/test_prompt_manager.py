# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment

from datus.prompts.prompt_manager import (
    PromptManager,
    get_prompt_manager,
)
from datus.utils.path_manager import DatusPathManager, reset_path_manager, set_current_path_manager


def _write_template(directory: Path, template_name: str, version: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{template_name}_{version}.j2"
    path.write_text(content, encoding="utf-8")
    return path


def _make_manager(
    tmp_path: Path,
    *,
    path_manager: DatusPathManager | None = None,
    agent_config: object | None = None,
) -> PromptManager:
    manager = PromptManager(path_manager=path_manager, agent_config=agent_config)
    manager.default_templates_dir = tmp_path / "default_templates"
    manager.default_templates_dir.mkdir(parents=True, exist_ok=True)
    return manager


@pytest.fixture(autouse=True)
def reset_context_home():
    reset_path_manager()
    PromptManager.clear_env_cache()
    yield
    reset_path_manager()
    PromptManager.clear_env_cache()


class TestPromptManager:
    def test_user_templates_dir_uses_explicit_path_manager(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)

        assert manager.user_templates_dir == path_manager.template_dir

    def test_user_templates_dir_uses_agent_config(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        agent_config = SimpleNamespace(path_manager=path_manager)
        manager = _make_manager(tmp_path, agent_config=agent_config)

        assert manager.user_templates_dir == path_manager.template_dir

    def test_get_env_is_cached_per_home(self, tmp_path):
        manager = _make_manager(tmp_path)

        outer_token = set_current_path_manager(tmp_path / "home_a")
        env_a = manager._get_env()
        reset_path_manager(outer_token)

        inner_token = set_current_path_manager(tmp_path / "home_b")
        env_b = manager._get_env()
        reset_path_manager(inner_token)

        assert env_a is not env_b
        assert len(manager._env_cache) == 2

    def test_get_template_path_prefers_user_template(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        user_path = _write_template(manager.user_templates_dir, "greet", "1.0", "Hello from user")
        _write_template(manager.default_templates_dir, "greet", "1.0", "Hello from default")

        assert manager._get_template_path("greet", "1.0") == user_path

    def test_get_template_path_falls_back_to_default(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        default_path = _write_template(manager.default_templates_dir, "greet", "1.0", "Hello from default")

        assert manager._get_template_path("greet", "1.0") == default_path

    def test_render_template_and_get_raw_template(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "Hello {{ name }}")

        assert manager.render_template("greet", "1.0", name="Ada") == "Hello Ada"
        assert manager.get_raw_template("greet", "1.0") == "Hello {{ name }}"

    def test_list_templates_merges_user_and_default_templates(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "default_only", "1.0", "default")
        _write_template(manager.user_templates_dir, "user_only", "1.0", "user")
        _write_template(manager.user_templates_dir, "shared", "1.0", "user shared")
        _write_template(manager.default_templates_dir, "shared", "1.0", "default shared")
        (manager.default_templates_dir / "invalid_name.j2").write_text("ignored", encoding="utf-8")

        assert manager.list_templates() == ["default_only", "shared", "user_only"]

    def test_list_template_versions_merges_and_sorts_versions(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "2.0", "default 2.0")
        _write_template(manager.default_templates_dir, "greet", "1.0", "default 1.0")
        _write_template(manager.user_templates_dir, "greet", "1.1", "user 1.1")
        _write_template(manager.user_templates_dir, "greet", "2.0", "user 2.0")

        assert manager.list_template_versions("greet") == ["1.0", "1.1", "2.0"]

    def test_get_latest_version_raises_when_missing(self, tmp_path):
        manager = _make_manager(tmp_path)

        with pytest.raises(FileNotFoundError, match="No versions found"):
            manager.get_latest_version("missing")

    def test_create_template_version_copies_from_latest_default(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.default_templates_dir, "greet", "1.1", "v1.1")

        manager.create_template_version("greet", "1.2")

        assert (manager.user_templates_dir / "greet_1.2.j2").read_text(encoding="utf-8") == "v1.1"

    def test_create_template_version_rejects_existing_version(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.user_templates_dir, "greet", "1.1", "existing")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_template_version("greet", "1.1", base_version="1.0")

    def test_template_exists_handles_present_and_missing_templates(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "hello")

        assert manager.template_exists("greet", "1.0") is True
        assert manager.template_exists("missing", "1.0") is False

    def test_get_template_info_reports_versions(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "v1")
        _write_template(manager.default_templates_dir, "greet", "2.0", "v2")

        assert manager.get_template_info("greet") == {
            "name": "greet",
            "available_versions": ["1.0", "2.0"],
            "latest_version": "2.0",
            "total_versions": 2,
        }

    def test_copy_to_creates_user_dir_and_respects_overwrite(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant_home")
        manager = _make_manager(tmp_path, path_manager=path_manager)
        _write_template(manager.default_templates_dir, "greet", "1.0", "default")

        copied_path = Path(manager.copy_to("greet", "greet_copy", "1.0"))
        copied_path.write_text("customized", encoding="utf-8")

        manager.copy_to("greet", "greet_copy", "1.0", overwrite=False)
        assert copied_path.read_text(encoding="utf-8") == "customized"

        manager.copy_to("greet", "greet_copy", "1.0", overwrite=True)
        assert copied_path.read_text(encoding="utf-8") == "default"

    def test_env_cache_is_class_level_shared_across_instances(self, tmp_path):
        pm_a = DatusPathManager(tmp_path / "home_a")
        pm_b = DatusPathManager(tmp_path / "home_b")
        manager_a = _make_manager(tmp_path, path_manager=pm_a)
        manager_b = _make_manager(tmp_path, path_manager=pm_b)

        manager_a._get_env()
        manager_b._get_env()

        # Both instances share the class-level cache
        assert len(PromptManager._env_cache) == 2
        assert manager_a._env_cache is manager_b._env_cache

    def test_env_cache_evicts_lru_when_full(self, tmp_path):
        original_max = PromptManager._MAX_ENV_CACHE_SIZE
        PromptManager._MAX_ENV_CACHE_SIZE = 3
        try:
            managers = []
            for i in range(4):
                pm = DatusPathManager(tmp_path / f"home_{i}")
                m = _make_manager(tmp_path, path_manager=pm)
                m._get_env()
                managers.append(m)

            # Cache should have evicted the oldest (home_0)
            assert len(PromptManager._env_cache) == 3
            assert str(managers[0].user_templates_dir) not in PromptManager._env_cache
            for m in managers[1:]:
                assert str(m.user_templates_dir) in PromptManager._env_cache
        finally:
            PromptManager._MAX_ENV_CACHE_SIZE = original_max

    def test_env_cache_lru_access_refreshes_entry(self, tmp_path):
        original_max = PromptManager._MAX_ENV_CACHE_SIZE
        PromptManager._MAX_ENV_CACHE_SIZE = 3
        try:
            managers = []
            for i in range(3):
                pm = DatusPathManager(tmp_path / f"home_{i}")
                m = _make_manager(tmp_path, path_manager=pm)
                m._get_env()
                managers.append(m)

            # Access home_0 to refresh it (make it most-recently-used)
            managers[0]._get_env()

            # Adding a 4th should evict home_1 (now the LRU), not home_0
            pm_new = DatusPathManager(tmp_path / "home_new")
            m_new = _make_manager(tmp_path, path_manager=pm_new)
            m_new._get_env()

            assert str(managers[0].user_templates_dir) in PromptManager._env_cache
            assert str(managers[1].user_templates_dir) not in PromptManager._env_cache
        finally:
            PromptManager._MAX_ENV_CACHE_SIZE = original_max

    def test_invalidate_env_removes_single_entry(self, tmp_path):
        pm_a = DatusPathManager(tmp_path / "home_a")
        pm_b = DatusPathManager(tmp_path / "home_b")
        manager_a = _make_manager(tmp_path, path_manager=pm_a)
        manager_b = _make_manager(tmp_path, path_manager=pm_b)
        manager_a._get_env()
        manager_b._get_env()

        PromptManager.invalidate_env(str(manager_a.user_templates_dir))

        assert len(PromptManager._env_cache) == 1
        assert str(manager_a.user_templates_dir) not in PromptManager._env_cache
        assert str(manager_b.user_templates_dir) in PromptManager._env_cache

    def test_invalidate_env_noop_for_missing_key(self):
        before = dict(PromptManager._env_cache)
        PromptManager.invalidate_env("/nonexistent/path")
        assert PromptManager._env_cache == before


class TestGetPromptManager:
    def test_returns_from_agent_config_prompt_manager_attr(self, tmp_path):
        pm = PromptManager(path_manager=DatusPathManager(tmp_path / "config"))
        agent_config = SimpleNamespace(prompt_manager=pm, path_manager=None)
        result = get_prompt_manager(agent_config=agent_config)

        assert result is pm

    def test_builds_from_agent_config_path_manager(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant")
        agent_config = SimpleNamespace(path_manager=path_manager)
        result = get_prompt_manager(agent_config=agent_config)

        assert isinstance(result, PromptManager)
        assert result.user_templates_dir == path_manager.template_dir

    def test_agent_config_without_prompt_manager_attr(self, tmp_path):
        path_manager = DatusPathManager(tmp_path / "tenant")
        # agent_config has no prompt_manager attribute at all
        agent_config = SimpleNamespace(path_manager=path_manager)

        result = get_prompt_manager(agent_config=agent_config)

        assert isinstance(result, PromptManager)
        assert result.user_templates_dir == path_manager.template_dir

    def test_agent_config_prompt_manager_takes_priority(self, tmp_path):
        pm = PromptManager(path_manager=DatusPathManager(tmp_path / "pm"))
        # Both prompt_manager and path_manager set — prompt_manager wins
        agent_config = SimpleNamespace(
            prompt_manager=pm,
            path_manager=DatusPathManager(tmp_path / "other"),
        )

        result = get_prompt_manager(agent_config=agent_config)

        assert result is pm

    def test_falls_back_to_default_when_no_agent_config(self):
        result = get_prompt_manager()
        assert isinstance(result, PromptManager)


class TestEnvCacheBehavior:
    def test_cache_hit_returns_same_environment_object(self, tmp_path):
        pm = DatusPathManager(tmp_path / "home")
        manager = _make_manager(tmp_path, path_manager=pm)

        env1 = manager._get_env()
        env2 = manager._get_env()

        assert env1 is env2

    def test_clear_env_cache_empties_all_entries(self, tmp_path):
        for i in range(3):
            pm = DatusPathManager(tmp_path / f"home_{i}")
            _make_manager(tmp_path, path_manager=pm)._get_env()

        assert len(PromptManager._env_cache) == 3
        PromptManager.clear_env_cache()
        assert len(PromptManager._env_cache) == 0

    def test_no_eviction_at_exact_max_size(self, tmp_path):
        original_max = PromptManager._MAX_ENV_CACHE_SIZE
        PromptManager._MAX_ENV_CACHE_SIZE = 3
        try:
            managers = []
            for i in range(3):
                pm = DatusPathManager(tmp_path / f"home_{i}")
                m = _make_manager(tmp_path, path_manager=pm)
                m._get_env()
                managers.append(m)

            # At exact max — no eviction
            assert len(PromptManager._env_cache) == 3
            for m in managers:
                assert str(m.user_templates_dir) in PromptManager._env_cache
        finally:
            PromptManager._MAX_ENV_CACHE_SIZE = original_max

    def test_invalidate_then_reinsert_creates_fresh_env(self, tmp_path):
        pm = DatusPathManager(tmp_path / "home")
        manager = _make_manager(tmp_path, path_manager=pm)

        env_before = manager._get_env()
        PromptManager.invalidate_env(str(manager.user_templates_dir))
        env_after = manager._get_env()

        assert env_before is not env_after


class TestPlanModeSystemTemplate:
    """Render the real packaged plan_mode_system template (interactive vs auto-execute)."""

    def _render(self, tmp_path, **kwargs):
        # Keep the packaged default_templates_dir so the real template resolves.
        manager = PromptManager(path_manager=DatusPathManager(tmp_path / "tenant_home"))
        return manager.render_template("plan_mode_system", None, plan_file_path="./.datus/plans/x.md", **kwargs)

    def test_full_workflow_interactive_requires_user_approval(self, tmp_path):
        rendered = self._render(tmp_path, workflow_prompt_sent=False, auto_execute_plan=False)
        assert "request the user's approval" in rendered
        assert "`ask_user`" in rendered
        assert "auto-approved" not in rendered

    def test_full_workflow_auto_execute_forbids_ask_user(self, tmp_path):
        rendered = self._render(tmp_path, workflow_prompt_sent=False, auto_execute_plan=True)
        assert "auto-approved" in rendered
        assert "Do NOT call `ask_user`" in rendered
        assert "request the user's approval" not in rendered

    def test_reminder_interactive_keeps_ask_user_rule(self, tmp_path):
        rendered = self._render(tmp_path, workflow_prompt_sent=True, auto_execute_plan=False)
        assert "Plan mode is still active" in rendered
        assert "`ask_user` (if you still need information)" in rendered

    def test_reminder_auto_execute_auto_approves(self, tmp_path):
        rendered = self._render(tmp_path, workflow_prompt_sent=True, auto_execute_plan=True)
        assert "Plan mode is still active" in rendered
        assert "auto-approved" in rendered
        assert "Do NOT call `ask_user`" in rendered


class TestPromptTemplateSandbox:
    """The user template dir is writable AND takes precedence over the built-in
    templates, so PromptManager renders untrusted input. These pin the sandbox
    and, just as importantly, the three settings that must NOT change with it.
    """

    def test_env_is_sandboxed(self, tmp_path):
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))

        assert isinstance(manager._get_env(), SandboxedEnvironment)

    def test_chained_unsafe_attribute_raises_security_error(self, tmp_path):
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(
            manager.user_templates_dir,
            "evil",
            "1.0",
            "{{ ''.__class__.__mro__[1].__subclasses__() }}",
        )

        with pytest.raises(SecurityError):
            manager.render_template("evil", "1.0")

    def test_globals_walk_on_context_object_raises_security_error(self, tmp_path):
        """The escape that matters in this process: reaching a module's globals
        from an object the caller handed the template. Those globals hold live
        datasource and model credentials.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(manager.user_templates_dir, "evil_globals", "1.0", "{{ ctx.__init__.__globals__ }}")

        with pytest.raises(SecurityError):
            manager.render_template("evil_globals", "1.0", ctx=SimpleNamespace(safe="ok"))

    def test_attr_filter_escape_is_neutralized(self, tmp_path):
        """``|attr`` (CVE-2025-27516's vector) does not raise under the lenient
        Undefined this env keeps — each hop returns an undefined rather than the
        real object. It yields nothing either way, which is what matters; assert
        the neutralization rather than an exception that never comes.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(manager.user_templates_dir, "evil_attr", "1.0", "[{{ ''|attr('__class__') }}]")

        assert manager.render_template("evil_attr", "1.0") == "[]"

    def test_single_unsafe_attribute_renders_empty_instead_of_leaking(self, tmp_path):
        """Under the *lenient* Undefined this env keeps, one unsafe attribute
        yields an undefined that stringifies to "" rather than raising. Nothing
        leaks either way; this pins the interaction so a future ``undefined=``
        change is caught by a test instead of in production.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(manager.user_templates_dir, "probe", "1.0", "[{{ ''.__class__ }}]")

        assert manager.render_template("probe", "1.0") == "[]"

    def test_lenient_undefined_is_preserved(self, tmp_path):
        """Guard against over-tightening: the shipped templates reference
        variables their callers do not always pass, so a missing variable must
        keep rendering empty rather than raising.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(manager.default_templates_dir, "loose", "1.0", "a{{ never_passed }}b")

        assert manager.render_template("loose", "1.0") == "ab"

    def test_trim_and_lstrip_blocks_are_preserved(self, tmp_path):
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(
            manager.default_templates_dir,
            "blocks",
            "1.0",
            "start\n    {% if flag %}\nkept\n    {% endif %}\nend",
        )

        assert manager.render_template("blocks", "1.0", flag=True) == "start\nkept\nend"

    def test_macro_import_still_resolves(self, tmp_path):
        """The loader must survive the swap: three shipped templates use
        ``{% import %}`` to call macros from _osi_dialect_map.j2.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        (manager.default_templates_dir / "_macros.j2").write_text(
            "{% macro shout(word) %}{{ word|upper }}{% endmacro %}", encoding="utf-8"
        )
        _write_template(
            manager.default_templates_dir,
            "uses_macro",
            "1.0",
            "{% import '_macros.j2' as m %}{{ m.shout('hi') }}",
        )

        assert manager.render_template("uses_macro", "1.0") == "HI"

    def test_safe_context_object_methods_still_work(self, tmp_path):
        """Shipped templates call .items()/.get()/.strftime() on context objects;
        the sandbox must not block those.
        """
        manager = _make_manager(tmp_path, path_manager=DatusPathManager(tmp_path / "tenant_home"))
        _write_template(
            manager.default_templates_dir,
            "safe_calls",
            "1.0",
            "{% for k, v in mapping.items() %}{{ k }}={{ v }};{% endfor %}{{ row.get('x', 'na') }}",
        )

        rendered = manager.render_template("safe_calls", "1.0", mapping={"a": 1}, row={"x": "ok"})
        assert rendered == "a=1;ok"
