# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.registry`` (``datus.plugins`` discovery)."""

import importlib.metadata as importlib_metadata

from datus.plugins import registry


class _FakeEntryPoint:
    def __init__(self, name, obj, *, raises=False, group="datus.plugins"):
        self.name = name
        self.group = group
        self._obj = obj
        self._raises = raises

    def load(self):
        if self._raises:
            raise ImportError("cannot import plugin")
        return self._obj


class _FakeEntryPoints:
    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group, name=None):
        out = [ep for ep in self._eps if ep.group == group]
        if name is not None:
            out = [ep for ep in out if ep.name == name]
        return out


def _patch(monkeypatch, eps):
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: _FakeEntryPoints(eps))


class _Plugin:
    """A minimal well-formed plugin class."""

    def __init__(self, profile=None):
        self.profile = profile or {}

    @classmethod
    def skills_dir(cls):  # overridden per-test via monkeypatch when a real dir is needed
        return None

    def run_cli(self, argv):
        return 0


def test_load_plugin_class_matching(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _Plugin)])
    assert registry.load_plugin_class("hello") is _Plugin


def test_load_plugin_class_unknown_returns_none(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _Plugin)])
    assert registry.load_plugin_class("mystery") is None


def test_load_plugin_class_load_failure_returns_none(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _Plugin, raises=True)])
    assert registry.load_plugin_class("hello") is None


def test_load_plugin_class_multiple_uses_first(monkeypatch):
    class _Other(_Plugin):
        pass

    _patch(monkeypatch, [_FakeEntryPoint("hello", _Plugin), _FakeEntryPoint("hello", _Other)])
    assert registry.load_plugin_class("hello") is _Plugin


def test_iter_plugin_entry_points(monkeypatch):
    eps = [_FakeEntryPoint("hello", _Plugin), _FakeEntryPoint("dagster", _Plugin)]
    _patch(monkeypatch, eps)
    assert {ep.name for ep in registry.iter_plugin_entry_points()} == {"hello", "dagster"}


def test_plugin_skill_directories_collects_existing(monkeypatch, tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()

    class _WithSkills(_Plugin):
        @classmethod
        def skills_dir(cls):
            return str(skill_dir)

    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithSkills)])
    assert registry.plugin_skill_directories() == [str(skill_dir)]


def test_plugin_skill_directories_skips_missing_dir(monkeypatch):
    class _BadDir(_Plugin):
        @classmethod
        def skills_dir(cls):
            return "/no/such/dir/at/all"

    _patch(monkeypatch, [_FakeEntryPoint("hello", _BadDir)])
    assert registry.plugin_skill_directories() == []


def test_plugin_skill_directories_skips_plugin_without_skills(monkeypatch):
    class _NoSkills:
        def run_cli(self, argv):
            return 0

    _patch(monkeypatch, [_FakeEntryPoint("hello", _NoSkills)])
    assert registry.plugin_skill_directories() == []


def test_plugin_skill_directories_dedup(monkeypatch, tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()

    class _WithSkills(_Plugin):
        @classmethod
        def skills_dir(cls):
            return str(skill_dir)

    _patch(monkeypatch, [_FakeEntryPoint("a", _WithSkills), _FakeEntryPoint("b", _WithSkills)])
    # Same directory contributed by two plugins is de-duplicated.
    assert registry.plugin_skill_directories() == [str(skill_dir)]


def test_plugin_skill_directories_survives_bad_plugin(monkeypatch, tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()

    class _WithSkills(_Plugin):
        @classmethod
        def skills_dir(cls):
            return str(skill_dir)

    _patch(
        monkeypatch,
        [_FakeEntryPoint("broken", _Plugin, raises=True), _FakeEntryPoint("hello", _WithSkills)],
    )
    # A broken plugin is skipped; the good one still contributes.
    assert registry.plugin_skill_directories() == [str(skill_dir)]


def test_lookup_never_raises_on_entry_points_failure(monkeypatch):
    def boom():
        raise RuntimeError("entry_points exploded")

    monkeypatch.setattr(importlib_metadata, "entry_points", boom)
    assert registry.load_plugin_class("hello") is None
    assert registry.iter_plugin_entry_points() == []
    assert registry.plugin_skill_directories() == []
    assert registry.plugin_system_prompt_sections(_FakeConfig({})) == []


class _FakeConfig:
    """Stand-in for AgentConfig exposing only ``plugin_services``."""

    def __init__(self, plugin_services):
        self.plugin_services = plugin_services


def test_system_prompt_sections_collects_and_passes_profiles(monkeypatch):
    captured = {}

    class _WithPrompt(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            captured["profiles"] = profiles
            return "## Hello\nManage DAGs."

    profiles = {"local": {"name": "local", "api_base_url": "http://h"}}
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithPrompt)])
    cfg = _FakeConfig({"hello": profiles})

    sections = registry.plugin_system_prompt_sections(cfg)

    # A datus-owned config-location preamble is prepended to plugin sections.
    assert len(sections) == 2
    assert sections[0].startswith("## Plugins")
    assert sections[1] == "## Hello\nManage DAGs."
    # datus indexes plugin_services by the entry-point name and passes it verbatim.
    assert captured["profiles"] is profiles


def test_system_prompt_sections_skips_plugin_without_hook(monkeypatch):
    class _NoPrompt:
        def run_cli(self, argv):
            return 0

    _patch(monkeypatch, [_FakeEntryPoint("hello", _NoPrompt)])
    assert registry.plugin_system_prompt_sections(_FakeConfig({"hello": {"p": {}}})) == []


def test_system_prompt_sections_filters_empty_and_none(monkeypatch):
    class _Empty(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            return "   "

    class _NoneRet(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            return None

    _patch(monkeypatch, [_FakeEntryPoint("a", _Empty), _FakeEntryPoint("b", _NoneRet)])
    assert registry.plugin_system_prompt_sections(_FakeConfig({})) == []


def test_system_prompt_sections_survives_raising_plugin(monkeypatch):
    class _Boom(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            raise RuntimeError("nope")

    class _Good(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            return "## Good"

    _patch(monkeypatch, [_FakeEntryPoint("boom", _Boom), _FakeEntryPoint("good", _Good)])
    # A raising plugin is skipped; the good one still contributes (after the preamble).
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert sections[0].startswith("## Plugins")
    assert sections[1:] == ["## Good"]


def test_system_prompt_sections_defaults_missing_profiles_to_empty(monkeypatch):
    captured = {}

    class _WithPrompt(_Plugin):
        @classmethod
        def system_prompt(cls, profiles):
            captured["profiles"] = profiles
            return None

    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithPrompt)])
    # No ``hello`` key in plugin_services -> the plugin receives an empty dict.
    registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert captured["profiles"] == {}


class _WithStaticPrompt(_Plugin):
    @classmethod
    def system_prompt(cls, profiles):
        return "## Hello\nManage DAGs."


def test_preamble_names_config_file_location(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithStaticPrompt)])
    monkeypatch.setattr(registry, "_agent_config_location", lambda: "/srv/conf/agent.yml")

    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))

    assert "`/srv/conf/agent.yml`" in sections[0]
    assert "agent.plugins.<plugin>.<profile>" in sections[0]


def test_preamble_degrades_without_config_path(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithStaticPrompt)])
    monkeypatch.setattr(registry, "_agent_config_location", lambda: None)

    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))

    assert sections[0].startswith("## Plugins")
    assert "agent.plugins.<plugin>.<profile>" in sections[0]
    assert "agent.yml" in sections[0]  # generic wording instead of a path


def test_agent_config_location_prefers_loaded_manager(monkeypatch):
    from datus.configuration import agent_config_loader

    class _Mgr:
        config_path = "/opt/datus/agent.yml"

    monkeypatch.setattr(agent_config_loader, "CONFIGURATION_MANAGER", _Mgr())
    assert registry._agent_config_location() == "/opt/datus/agent.yml"


def test_agent_config_location_none_when_unresolvable(monkeypatch):
    from datus.configuration import agent_config_loader

    monkeypatch.setattr(agent_config_loader, "CONFIGURATION_MANAGER", None)

    def boom(*args, **kwargs):
        raise RuntimeError("no config anywhere")

    monkeypatch.setattr(agent_config_loader, "parse_config_path", boom)
    assert registry._agent_config_location() is None


# ---------------------------------------------------------------------------
# collect_plugin_cli_permissions
# ---------------------------------------------------------------------------


class _WithCliPerms(_Plugin):
    @classmethod
    def cli_permissions(cls):
        return {
            "normal": {"allow": ["greet:*", "version"], "ask": ["config set:*"], "deny": ["config wipe:*"]},
            "auto": {"allow": [":*"]},
        }


def test_cli_permissions_prefixing_and_shapes(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithCliPerms)])
    rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal", "auto"}
    # ``greet:*`` -> prefix rule; ``version`` (no colon) -> exact match.
    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]
    assert rules["normal"].ask == ["datus hello config set:*"]
    assert rules["normal"].deny == ["datus hello config wipe:*"]
    # ``:*`` covers the whole namespace.
    assert rules["auto"].allow == ["datus hello:*"]
    assert rules["auto"].ask == []


def test_cli_permissions_never_set_scalar_fields(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithCliPerms)])
    rules = registry.collect_plugin_cli_permissions()
    # ``default`` / ``classifier`` must stay unset so the profile posture and
    # merge_with scalar semantics are untouched by plugin declarations.
    for ruleset in rules.values():
        assert "default" not in ruleset.model_fields_set
        assert "classifier" not in ruleset.model_fields_set


def test_cli_permissions_dangerous_and_unknown_profiles_dropped(monkeypatch, caplog):
    class _Overreaching(_Plugin):
        @classmethod
        def cli_permissions(cls):
            return {
                "dangerous": {"deny": ["config wipe:*"]},
                "paranoid": {"ask": ["greet:*"]},
                "normal": {"allow": ["greet:*"]},
            }

    _patch(monkeypatch, [_FakeEntryPoint("hello", _Overreaching)])
    with caplog.at_level("WARNING"):
        rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal"}
    assert "dangerous" in caplog.text
    assert "paranoid" in caplog.text


def test_cli_permissions_malformed_entries_skipped(monkeypatch):
    class _Malformed(_Plugin):
        @classmethod
        def cli_permissions(cls):
            return {
                "normal": {
                    "allow": ["greet:*", 42, "", "   "],  # non-str / empty entries dropped
                    "ask": "not-a-list",  # non-list action dropped
                    "grant": ["greet:*"],  # unknown action dropped
                },
                "auto": ["not-a-dict"],  # non-dict profile dropped
            }

    _patch(monkeypatch, [_FakeEntryPoint("hello", _Malformed)])
    rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal"}
    assert rules["normal"].allow == ["datus hello greet:*"]
    assert rules["normal"].ask == []
    assert rules["normal"].deny == []


def test_cli_permissions_non_dict_hook_ignored(monkeypatch):
    class _WrongType(_Plugin):
        @classmethod
        def cli_permissions(cls):
            return ["normal"]

    _patch(monkeypatch, [_FakeEntryPoint("hello", _WrongType)])
    assert registry.collect_plugin_cli_permissions() == {}


def test_cli_permissions_plain_dict_attribute_accepted(monkeypatch):
    class _AttrStyle(_Plugin):
        cli_permissions = {"normal": {"allow": ["greet:*"]}}

    _patch(monkeypatch, [_FakeEntryPoint("hello", _AttrStyle)])
    rules = registry.collect_plugin_cli_permissions()
    assert rules["normal"].allow == ["datus hello greet:*"]


def test_cli_permissions_raising_hook_skips_only_that_plugin(monkeypatch):
    class _Boom(_Plugin):
        @classmethod
        def cli_permissions(cls):
            raise RuntimeError("nope")

    _patch(monkeypatch, [_FakeEntryPoint("boom", _Boom), _FakeEntryPoint("hello", _WithCliPerms)])
    rules = registry.collect_plugin_cli_permissions()
    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]


def test_cli_permissions_plugin_without_hook_skipped(monkeypatch):
    class _NoHook:
        def run_cli(self, argv):
            return 0

    _patch(monkeypatch, [_FakeEntryPoint("hello", _NoHook)])
    assert registry.collect_plugin_cli_permissions() == {}


def test_cli_permissions_duplicate_entry_point_first_wins(monkeypatch, caplog):
    class _Second(_Plugin):
        @classmethod
        def cli_permissions(cls):
            return {"normal": {"allow": ["other:*"]}}

    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithCliPerms), _FakeEntryPoint("hello", _Second)])
    with caplog.at_level("WARNING"):
        rules = registry.collect_plugin_cli_permissions()

    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]
    assert "Duplicate" in caplog.text


def test_cli_permissions_unsafe_entry_point_name_skipped(monkeypatch, caplog):
    _patch(
        monkeypatch,
        [_FakeEntryPoint("evil name", _WithCliPerms), _FakeEntryPoint("*", _WithCliPerms)],
    )
    with caplog.at_level("WARNING"):
        assert registry.collect_plugin_cli_permissions() == {}
    assert "not a safe CLI token" in caplog.text


def test_cli_permissions_broken_load_skipped(monkeypatch):
    _patch(
        monkeypatch,
        [_FakeEntryPoint("broken", _WithCliPerms, raises=True), _FakeEntryPoint("hello", _WithCliPerms)],
    )
    rules = registry.collect_plugin_cli_permissions()
    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]


def test_cli_permissions_entry_points_failure_returns_empty(monkeypatch):
    def boom():
        raise RuntimeError("entry_points exploded")

    monkeypatch.setattr(importlib_metadata, "entry_points", boom)
    assert registry.collect_plugin_cli_permissions() == {}


# ---------------------------------------------------------------------------
# Tests: collect_plugin_tool_transformers
# ---------------------------------------------------------------------------


def _sql_transformer(tool_name, args, context):
    return args


def _audit_transformer(tool_name, args, context):
    return args


class _WithTransformers(_Plugin):
    @classmethod
    def tool_transformers(cls):
        return {"db_tools.execute_sql": _sql_transformer}


class _WithTransformerList(_Plugin):
    @classmethod
    def tool_transformers(cls):
        return {"execute_sql": [_audit_transformer, _sql_transformer]}


def test_tool_transformers_collects_single_callable(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithTransformers)])
    collected = registry.collect_plugin_tool_transformers()
    assert collected == {"db_tools.execute_sql": [_sql_transformer]}


def test_tool_transformers_collects_callable_list(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _WithTransformerList)])
    collected = registry.collect_plugin_tool_transformers()
    assert collected == {"execute_sql": [_audit_transformer, _sql_transformer]}


def test_tool_transformers_accumulates_across_plugins(monkeypatch):
    _patch(
        monkeypatch,
        [_FakeEntryPoint("a", _WithTransformers), _FakeEntryPoint("b", _WithTransformers)],
    )
    collected = registry.collect_plugin_tool_transformers()
    assert collected == {"db_tools.execute_sql": [_sql_transformer, _sql_transformer]}


def test_tool_transformers_plugin_without_hook_skipped(monkeypatch):
    _patch(monkeypatch, [_FakeEntryPoint("hello", _Plugin)])
    assert registry.collect_plugin_tool_transformers() == {}


def test_tool_transformers_non_dict_declaration_skipped(monkeypatch, caplog):
    class _BadShape(_Plugin):
        @classmethod
        def tool_transformers(cls):
            return ["not", "a", "dict"]

    _patch(monkeypatch, [_FakeEntryPoint("hello", _BadShape)])
    with caplog.at_level("WARNING"):
        assert registry.collect_plugin_tool_transformers() == {}
    assert "must return a dict" in caplog.text


def test_tool_transformers_raising_declaration_skipped(monkeypatch, caplog):
    class _Raises(_Plugin):
        @classmethod
        def tool_transformers(cls):
            raise RuntimeError("boom")

    _patch(monkeypatch, [_FakeEntryPoint("hello", _Raises)])
    with caplog.at_level("WARNING"):
        assert registry.collect_plugin_tool_transformers() == {}
    assert "tool_transformers() failed" in caplog.text


def test_tool_transformers_non_callable_entries_skipped(monkeypatch, caplog):
    class _Mixed(_Plugin):
        @classmethod
        def tool_transformers(cls):
            return {"execute_sql": [_sql_transformer, "not-callable"], "other_tool": "also-not-callable"}

    _patch(monkeypatch, [_FakeEntryPoint("hello", _Mixed)])
    with caplog.at_level("WARNING"):
        collected = registry.collect_plugin_tool_transformers()
    assert collected == {"execute_sql": [_sql_transformer]}
    assert "non-callable" in caplog.text


def test_tool_transformers_invalid_pattern_skipped(monkeypatch, caplog):
    class _BadPattern(_Plugin):
        @classmethod
        def tool_transformers(cls):
            return {"": _sql_transformer, 42: _sql_transformer, "  ok  ": _sql_transformer}

    _patch(monkeypatch, [_FakeEntryPoint("hello", _BadPattern)])
    with caplog.at_level("WARNING"):
        collected = registry.collect_plugin_tool_transformers()
    assert collected == {"ok": [_sql_transformer]}
    assert "invalid pattern" in caplog.text


def test_tool_transformers_broken_plugin_skipped(monkeypatch):
    _patch(
        monkeypatch,
        [_FakeEntryPoint("broken", _WithTransformers, raises=True), _FakeEntryPoint("ok", _WithTransformers)],
    )
    collected = registry.collect_plugin_tool_transformers()
    assert collected == {"db_tools.execute_sql": [_sql_transformer]}


def test_tool_transformers_entry_points_failure_returns_empty(monkeypatch):
    def boom():
        raise RuntimeError("entry_points exploded")

    monkeypatch.setattr(importlib_metadata, "entry_points", boom)
    assert registry.collect_plugin_tool_transformers() == {}
