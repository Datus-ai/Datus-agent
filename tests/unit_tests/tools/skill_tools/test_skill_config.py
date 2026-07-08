# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for skill configuration models.

Tests SkillConfig and SkillMetadata classes.
"""

from pathlib import Path

import pytest

from datus.tools.skill_tools.skill_config import (
    SkillConfig,
    SkillMetadata,
    _builtin_skills_dir,
    _default_skill_directories,
    _entry_point_skill_directories,
)


class _FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name, loaded, *, raises=False):
        self.name = name
        self.group = "datus.skills"
        self._loaded = loaded
        self._raises = raises

    def load(self):
        if self._raises:
            raise ImportError("boom")
        return self._loaded


class _FakeEntryPoints:
    """Mimics the ``select(group=...)`` API of modern importlib.metadata."""

    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group, name=None):
        out = [ep for ep in self._eps if ep.group == group]
        if name is not None:
            out = [ep for ep in out if ep.name == name]
        return out


def _patch_entry_points(monkeypatch, eps):
    """Patch ``importlib.metadata.entry_points`` (resolved lazily inside the
    discovery helper) to return our fake set."""
    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: _FakeEntryPoints(eps))


class TestSkillConfig:
    """Tests for SkillConfig model."""

    def test_skill_config_defaults(self):
        """Default directories include the user-facing dirs and the packaged built-ins."""
        config = SkillConfig()
        # First two entries are the documented user-facing locations.
        assert config.directories[:2] == ["./.datus/skills", "~/.datus/skills"]
        # Packaged built-ins ship with the wheel, so the resolver should
        # produce a path that exists on this checkout — and it must come last
        # so user overrides win.
        builtin = _builtin_skills_dir()
        assert isinstance(builtin, str), "datus/resources/skills must resolve in the test env"
        assert config.directories[-1] == builtin
        assert config.warn_duplicates is True
        assert config.whitelist_from_compaction is True

    def test_skill_config_custom_directories(self):
        """Explicit constructor directories are kept verbatim (no auto-append)."""
        config = SkillConfig(directories=["/custom/skills", "./project/skills"])
        assert config.directories == ["/custom/skills", "./project/skills"]

    def test_skill_config_disable_warnings(self):
        """Test SkillConfig with warnings disabled."""
        config = SkillConfig(warn_duplicates=False)
        assert config.warn_duplicates is False

    def test_skill_config_from_dict(self):
        """from_dict should append the packaged built-ins after user-supplied dirs."""
        config_dict = {
            "directories": ["/my/skills"],
            "warn_duplicates": False,
            "whitelist_from_compaction": False,
        }
        config = SkillConfig.from_dict(config_dict)
        builtin = _builtin_skills_dir()
        assert config.directories[0] == "/my/skills"
        assert config.directories[-1] == builtin
        assert config.warn_duplicates is False
        assert config.whitelist_from_compaction is False

    def test_skill_config_from_dict_does_not_double_add_builtin(self):
        """If the user already includes the packaged dir, from_dict must not duplicate it."""
        builtin = _builtin_skills_dir()
        assert isinstance(builtin, str)
        config = SkillConfig.from_dict({"directories": [builtin, "/other"]})
        assert config.directories.count(builtin) == 1

    def test_skill_config_from_dict_does_not_double_add_builtin_equivalent_path(self):
        """Equivalent path spellings of the packaged dir should not be appended again."""
        builtin = _builtin_skills_dir()
        assert isinstance(builtin, str)
        equivalent_builtin = str(Path(builtin) / ".." / Path(builtin).name)

        config = SkillConfig.from_dict({"directories": [equivalent_builtin, "/other"]})

        builtin_paths = [
            directory
            for directory in config.directories
            if Path(directory).expanduser().resolve() == Path(builtin).expanduser().resolve()
        ]
        assert builtin_paths == [equivalent_builtin]

    def test_skill_config_from_dict_empty(self):
        """Empty config dict falls back to the packaged-aware default."""
        config = SkillConfig.from_dict({})
        assert config.directories == _default_skill_directories()
        assert config.warn_duplicates is True

    def test_skill_config_from_dict_partial(self):
        """Partial config dict still gets the packaged-aware default for directories."""
        config = SkillConfig.from_dict({"warn_duplicates": False})
        assert config.directories == _default_skill_directories()
        assert config.warn_duplicates is False


class TestEntryPointSkillDirectories:
    """Tests for ``datus.skills`` entry-point discovery."""

    def test_discovers_str_path_entry_point(self, monkeypatch, tmp_path):
        """A string directory returned by an entry point is discovered."""
        skill_dir = tmp_path / "adapter_skills"
        skill_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(skill_dir))])
        result = _entry_point_skill_directories()
        assert result == [str(skill_dir)]

    def test_discovers_path_object_entry_point(self, monkeypatch, tmp_path):
        """A ``Path`` object (not just str) is coerced and discovered."""
        skill_dir = tmp_path / "adapter_skills"
        skill_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", skill_dir)])
        result = _entry_point_skill_directories()
        assert result == [str(skill_dir)]

    def test_discovers_callable_entry_point(self, monkeypatch, tmp_path):
        """A zero-arg callable returning a directory is invoked and discovered."""
        skill_dir = tmp_path / "adapter_skills"
        skill_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", lambda: str(skill_dir))])
        result = _entry_point_skill_directories()
        assert result == [str(skill_dir)]

    def test_skips_nonexistent_directory(self, monkeypatch, tmp_path):
        """An entry point pointing at a missing directory is silently skipped."""
        missing = tmp_path / "does_not_exist"
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(missing))])
        assert _entry_point_skill_directories() == []

    def test_skips_file_path(self, monkeypatch, tmp_path):
        """A path that is a file (not a directory) is skipped."""
        f = tmp_path / "a_file.md"
        f.write_text("x")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(f))])
        assert _entry_point_skill_directories() == []

    def test_swallows_load_exception(self, monkeypatch, tmp_path):
        """A broken entry point ``load()`` does not break discovery of others."""
        good = tmp_path / "good"
        good.mkdir()
        _patch_entry_points(
            monkeypatch,
            [
                _FakeEntryPoint("broken", None, raises=True),
                _FakeEntryPoint("hello", str(good)),
            ],
        )
        assert _entry_point_skill_directories() == [str(good)]

    def test_dedupes_duplicate_directories(self, monkeypatch, tmp_path):
        """Two entry points resolving to the same directory collapse to one."""
        skill_dir = tmp_path / "adapter_skills"
        skill_dir.mkdir()
        _patch_entry_points(
            monkeypatch,
            [
                _FakeEntryPoint("a", str(skill_dir)),
                _FakeEntryPoint("b", str(skill_dir)),
            ],
        )
        assert _entry_point_skill_directories() == [str(skill_dir)]

    def test_lookup_failure_returns_empty(self, monkeypatch):
        """A failure resolving entry points returns an empty list, never raises."""
        import importlib.metadata as importlib_metadata

        def _boom():
            raise RuntimeError("metadata broken")

        monkeypatch.setattr(importlib_metadata, "entry_points", _boom)
        assert _entry_point_skill_directories() == []

    def test_default_directories_ordering(self, monkeypatch, tmp_path):
        """Scan order is project → user → entry-point dirs → packaged builtin (last)."""
        ep_dir = tmp_path / "adapter_skills"
        ep_dir.mkdir()
        builtin_dir = tmp_path / "builtin_skills"
        builtin_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(ep_dir))])
        monkeypatch.setattr("datus.tools.skill_tools.skill_config._builtin_skills_dir", lambda: str(builtin_dir))
        dirs = _default_skill_directories()
        assert dirs[0] == "./.datus/skills"
        assert dirs[1] == "~/.datus/skills"
        # entry-point dir comes before the packaged builtin, builtin is last
        assert dirs.index(str(ep_dir)) < dirs.index(str(builtin_dir))
        assert dirs[-1] == str(builtin_dir)

    def test_from_dict_appends_entry_point_dirs(self, monkeypatch, tmp_path):
        """Explicit ``directories`` still get adapter entry-point dirs appended."""
        ep_dir = tmp_path / "adapter_skills"
        ep_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(ep_dir))])
        config = SkillConfig.from_dict({"directories": ["/my/skills"]})
        assert config.directories[0] == "/my/skills"
        assert str(ep_dir) in config.directories

    def test_default_dirs_dedupe_entry_point_against_listed(self, monkeypatch, tmp_path):
        """An entry-point dir already present is not appended twice."""
        ep_dir = tmp_path / "adapter_skills"
        ep_dir.mkdir()
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("hello", str(ep_dir))])
        config = SkillConfig.from_dict({"directories": [str(ep_dir), "/other"]})
        assert config.directories.count(str(ep_dir)) == 1

    def test_skill_config_serialization(self):
        """Test SkillConfig serialization."""
        config = SkillConfig(directories=["/test"])
        data = config.model_dump()
        assert data["directories"] == ["/test"]
        assert data["warn_duplicates"] is True


class TestPluginSkillDirectories:
    """``datus.plugins`` skill dirs merge into the scan order before legacy
    ``datus.skills`` adapters."""

    def test_plugin_dir_included_before_legacy_and_builtin(self, monkeypatch, tmp_path):
        plugin_dir = tmp_path / "plugin_skills"
        plugin_dir.mkdir()
        legacy_dir = tmp_path / "legacy_skills"
        legacy_dir.mkdir()
        builtin_dir = tmp_path / "builtin_skills"
        builtin_dir.mkdir()
        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", lambda: [str(plugin_dir)])
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("mwaa", str(legacy_dir))])
        monkeypatch.setattr("datus.tools.skill_tools.skill_config._builtin_skills_dir", lambda: str(builtin_dir))

        dirs = _default_skill_directories()
        assert dirs[0] == "./.datus/skills"
        assert dirs[1] == "~/.datus/skills"
        # plugin dir precedes the legacy datus.skills dir.
        assert dirs.index(str(plugin_dir)) < dirs.index(str(legacy_dir))
        assert dirs[-1] == str(builtin_dir)

    def test_plugin_dir_deduped_against_legacy(self, monkeypatch, tmp_path):
        shared = tmp_path / "shared_skills"
        shared.mkdir()
        # Same dir contributed by both a plugin and a legacy adapter appears once.
        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", lambda: [str(shared)])
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("mwaa", str(shared))])
        assert _default_skill_directories().count(str(shared)) == 1

    def test_from_dict_appends_plugin_dirs(self, monkeypatch, tmp_path):
        plugin_dir = tmp_path / "plugin_skills"
        plugin_dir.mkdir()
        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", lambda: [str(plugin_dir)])
        _patch_entry_points(monkeypatch, [])
        config = SkillConfig.from_dict({"directories": ["/my/skills"]})
        assert config.directories[0] == "/my/skills"
        assert str(plugin_dir) in config.directories

    def test_plugin_discovery_failure_is_swallowed(self, monkeypatch):
        def boom():
            raise RuntimeError("registry import exploded")

        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", boom)
        _patch_entry_points(monkeypatch, [])
        # Must not raise; falls back to the non-plugin default order.
        dirs = _default_skill_directories()
        assert dirs[:2] == ["./.datus/skills", "~/.datus/skills"]


class _FakeManager:
    def __init__(self, data):
        self.data = data


class TestPluginsEnabledGate:
    """``agent.plugins_enabled: false`` removes plugin skill dirs everywhere."""

    def _disable(self, monkeypatch):
        from datus.configuration import agent_config_loader

        monkeypatch.setattr(agent_config_loader, "CONFIGURATION_MANAGER", _FakeManager({"plugins_enabled": False}))

    def test_disabled_excludes_plugin_dirs_from_default(self, monkeypatch, tmp_path):
        plugin_dir = tmp_path / "plugin_skills"
        plugin_dir.mkdir()
        legacy_dir = tmp_path / "legacy_skills"
        legacy_dir.mkdir()
        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", lambda: [str(plugin_dir)])
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("mwaa", str(legacy_dir))])
        self._disable(monkeypatch)

        dirs = _default_skill_directories()
        assert str(plugin_dir) not in dirs
        # Legacy ``datus.skills`` adapters are not governed by the switch.
        assert str(legacy_dir) in dirs

    def test_disabled_excludes_plugin_dirs_from_from_dict(self, monkeypatch, tmp_path):
        plugin_dir = tmp_path / "plugin_skills"
        plugin_dir.mkdir()
        monkeypatch.setattr("datus.plugins.registry.plugin_skill_directories", lambda: [str(plugin_dir)])
        _patch_entry_points(monkeypatch, [])
        self._disable(monkeypatch)

        config = SkillConfig.from_dict({"directories": ["/my/skills"]})
        assert str(plugin_dir) not in config.directories

    @pytest.mark.parametrize("value,expected", [(None, True), (True, True), ("false", False), ("off", False)])
    def test_enabled_flag_coercion(self, monkeypatch, value, expected):
        from datus.configuration import agent_config_loader
        from datus.tools.skill_tools.skill_config import _plugins_system_enabled

        monkeypatch.setattr(agent_config_loader, "CONFIGURATION_MANAGER", _FakeManager({"plugins_enabled": value}))
        assert _plugins_system_enabled() is expected

    def test_no_loaded_config_defaults_to_enabled(self, monkeypatch):
        from datus.configuration import agent_config_loader
        from datus.tools.skill_tools.skill_config import _plugins_system_enabled

        monkeypatch.setattr(agent_config_loader, "CONFIGURATION_MANAGER", None)
        assert _plugins_system_enabled() is True


class TestSkillMetadata:
    """Tests for SkillMetadata model."""

    def test_skill_metadata_required_fields(self):
        """Test SkillMetadata with required fields only."""
        metadata = SkillMetadata(
            name="test-skill",
            description="A test skill",
            location=Path("/skills/test-skill"),
        )
        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill"
        assert metadata.location == Path("/skills/test-skill")

    def test_skill_metadata_all_fields(self):
        """Test SkillMetadata with all fields."""
        metadata = SkillMetadata(
            name="advanced-skill",
            description="An advanced skill",
            location=Path("/skills/advanced"),
            tags=["sql", "optimization"],
            version="1.0.0",
            disable_model_invocation=False,
            user_invocable=True,
            context="fork",
            agent="Explore",
        )
        assert metadata.name == "advanced-skill"
        assert metadata.tags == ["sql", "optimization"]
        assert metadata.version == "1.0.0"
        assert metadata.context == "fork"
        assert metadata.agent == "Explore"

    def test_skill_metadata_defaults(self):
        """Test SkillMetadata default values."""
        metadata = SkillMetadata(
            name="test",
            description="test",
            location=Path("/test"),
        )
        assert metadata.tags == []
        assert metadata.version is None
        assert metadata.disable_model_invocation is False
        assert metadata.user_invocable is True
        assert metadata.context is None
        assert metadata.agent is None
        assert metadata.content is None

    def test_skill_metadata_is_model_invocable(self):
        """Test is_model_invocable method."""
        # Model invocable by default
        metadata_default = SkillMetadata(
            name="default",
            description="Default",
            location=Path("/test"),
        )
        assert metadata_default.is_model_invocable() is True

        # Explicitly disabled
        metadata_disabled = SkillMetadata(
            name="disabled",
            description="Disabled",
            location=Path("/test"),
            disable_model_invocation=True,
        )
        assert metadata_disabled.is_model_invocable() is False

    def test_skill_metadata_from_frontmatter(self):
        """Test creating SkillMetadata from frontmatter dict."""
        frontmatter = {
            "name": "sql-optimization",
            "description": "SQL query optimization techniques",
            "tags": ["sql", "performance"],
            "version": "1.0.0",
        }
        metadata = SkillMetadata.from_frontmatter(frontmatter, Path("/skills/sql-optimization"))
        assert metadata.name == "sql-optimization"
        assert metadata.description == "SQL query optimization techniques"
        assert metadata.tags == ["sql", "performance"]
        assert metadata.version == "1.0.0"
        assert metadata.location == Path("/skills/sql-optimization")

    def test_skill_metadata_from_frontmatter_rejects_string_tags(self):
        """Tags must be a list, not a comma-separated string."""
        from pydantic import ValidationError

        frontmatter = {
            "name": "bad-tags",
            "description": "Skill with string tags",
            "tags": "sql, performance, optimization",
        }
        with pytest.raises(ValidationError, match="tags"):
            SkillMetadata.from_frontmatter(frontmatter, Path("/skills/bad-tags"))

    def test_skill_metadata_from_frontmatter_minimal(self):
        """Test creating SkillMetadata from minimal frontmatter."""
        frontmatter = {
            "name": "simple",
            "description": "Simple skill",
        }
        metadata = SkillMetadata.from_frontmatter(frontmatter, Path("/skills/simple"))
        assert metadata.name == "simple"
        assert metadata.description == "Simple skill"
        assert metadata.tags == []

    def test_skill_metadata_serialization(self):
        """Test SkillMetadata serialization."""
        metadata = SkillMetadata(
            name="test",
            description="Test skill",
            location=Path("/test"),
            tags=["tag1"],
        )
        data = metadata.model_dump()
        assert data["name"] == "test"
        assert data["description"] == "Test skill"
        assert data["tags"] == ["tag1"]
        # Path should be converted to string
        assert str(data["location"]) == "/test"

    def test_skill_metadata_content_lazy_loaded(self):
        """Test that content is lazy loaded (initially None)."""
        metadata = SkillMetadata(
            name="test",
            description="Test",
            location=Path("/test"),
        )
        assert metadata.content is None

        # Content can be set later
        metadata.content = "# Test Skill\n\nContent here"
        assert metadata.content == "# Test Skill\n\nContent here"


class TestAllowedAgents:
    """Tests for the allowed_agents frontmatter field on SkillMetadata."""

    def test_defaults_to_empty_list(self):
        metadata = SkillMetadata(
            name="open",
            description="open-access skill",
            location=Path("/test"),
        )
        assert metadata.allowed_agents == []

    def test_from_frontmatter_parses_list(self):
        metadata = SkillMetadata.from_frontmatter(
            {
                "name": "scoped",
                "description": "only for one agent",
                "allowed_agents": ["gen_dashboard", "gen_table"],
            },
            Path("/skills/scoped"),
        )
        assert metadata.allowed_agents == ["gen_dashboard", "gen_table"]

    def test_from_frontmatter_defaults_when_absent(self):
        metadata = SkillMetadata.from_frontmatter(
            {"name": "open", "description": "no scoping"},
            Path("/skills/open"),
        )
        assert metadata.allowed_agents == []

    def test_to_dict_roundtrips_allowed_agents(self):
        metadata = SkillMetadata(
            name="scoped",
            description="scoped",
            location=Path("/test"),
            allowed_agents=["scheduler"],
        )
        data = metadata.to_dict()
        assert data["allowed_agents"] == ["scheduler"]

    def test_is_allowed_for_empty_list_allows_everyone(self):
        metadata = SkillMetadata(
            name="open",
            description="open",
            location=Path("/test"),
        )
        assert metadata.is_allowed_for("chat") is True
        assert metadata.is_allowed_for("gen_dashboard") is True

    def test_is_allowed_for_hit(self):
        metadata = SkillMetadata(
            name="scoped",
            description="scoped",
            location=Path("/test"),
            allowed_agents=["gen_dashboard"],
        )
        assert metadata.is_allowed_for("gen_dashboard") is True

    def test_is_allowed_for_miss(self):
        metadata = SkillMetadata(
            name="scoped",
            description="scoped",
            location=Path("/test"),
            allowed_agents=["gen_dashboard"],
        )
        assert metadata.is_allowed_for("chat") is False

    def test_is_allowed_for_accepts_alias_plus_class(self):
        """A custom alias should still match when its class name is whitelisted."""
        metadata = SkillMetadata(
            name="scoped",
            description="scoped",
            location=Path("/test"),
            allowed_agents=["gen_dashboard"],
        )
        # Alias not in whitelist, but class is → allowed.
        assert metadata.is_allowed_for("my_dashboard", "gen_dashboard") is True
        # Neither alias nor class is whitelisted → denied.
        assert metadata.is_allowed_for("my_dashboard", "gen_table") is False

    def test_is_allowed_for_ignores_none_identifiers(self):
        metadata = SkillMetadata(
            name="scoped",
            description="scoped",
            location=Path("/test"),
            allowed_agents=["gen_dashboard"],
        )
        assert metadata.is_allowed_for("gen_dashboard", None) is True
        assert metadata.is_allowed_for(None, "chat") is False
