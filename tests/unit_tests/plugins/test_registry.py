# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.plugins.registry`` (manifest discovery + collection).

Uses the ``plugin_env`` fixture (conftest) which builds REAL importable
packages with a ``datus-plugin.yml`` and fake ``datus.plugins`` entry points,
so these tests exercise the actual ``find_spec``-without-import path.
"""

import importlib
import importlib.metadata as importlib_metadata

from datus.plugins import registry

MINIMAL = "manifest_version: 1\n"

TRANSFORMER_MODULE = "def passthrough(tool_name, args, context):\n    return args\n\nNOT_CALLABLE = 42\n"


class _FakeConfig:
    """Stand-in for AgentConfig exposing only ``plugin_services``.

    ``config_mutable`` is only set when supplied so the default case also
    exercises the ``getattr``-fallback path in the registry.
    """

    def __init__(self, plugin_services, config_mutable=None):
        self.plugin_services = plugin_services
        if config_mutable is not None:
            self.config_mutable = config_mutable


# ---------------------------------------------------------------------------
# Discovery / load_plugin_manifest
# ---------------------------------------------------------------------------


def test_load_plugin_manifest_matching(plugin_env):
    plugin_env("hello", MINIMAL + "description: Hi.\n")
    manifest = registry.load_plugin_manifest("hello")
    assert manifest.name == "hello"
    assert manifest.description == "Hi."


def test_load_plugin_manifest_unknown_returns_none(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.load_plugin_manifest("mystery") is None


def test_load_plugin_manifest_missing_manifest_returns_none(plugin_env, caplog):
    plugin_env("hello", manifest_yaml=None)  # importable package, no yml
    with caplog.at_level("WARNING"):
        assert registry.load_plugin_manifest("hello") is None
    assert "datus-plugin.yml" in caplog.text


def test_load_plugin_manifest_legacy_class_entry_rejected(plugin_env, caplog):
    pkg = plugin_env("hello", MINIMAL, value="OVERRIDDEN")
    # Simulate the old ``pkg.plugin:Class`` entry-point style.
    plugin_env("legacy", MINIMAL, value=f"{pkg.name}:HelloPlugin")
    with caplog.at_level("WARNING"):
        assert registry.load_plugin_manifest("legacy") is None
    assert "legacy class-based contract" in caplog.text


def test_load_plugin_manifest_dotted_entry_point_rejected(plugin_env, caplog):
    pkg = plugin_env("hello", MINIMAL)
    # A dotted entry-point value would make ``find_spec`` import the parent
    # package (executing plugin code) before the manifest is read; discovery
    # must reject it and only accept bare top-level package names.
    plugin_env("dotted", MINIMAL, value=f"{pkg.name}.submodule")
    with caplog.at_level("WARNING"):
        assert registry.load_plugin_manifest("dotted") is None
    assert "bare top-level package" in caplog.text


def test_load_plugin_manifest_multiple_uses_first(plugin_env, caplog):
    first = plugin_env("hello", MINIMAL + "description: first\n")
    plugin_env("hello", MINIMAL + "description: second\n")
    with caplog.at_level("WARNING"):
        manifest = registry.load_plugin_manifest("hello")
    assert manifest.description == "first"
    assert manifest.package_dir == first
    assert "Multiple" in caplog.text


def test_iter_plugin_entry_points(plugin_env):
    plugin_env("hello", MINIMAL)
    plugin_env("dagster", MINIMAL)
    assert {ep.name for ep in registry.iter_plugin_entry_points()} == {"hello", "dagster"}


COMMANDS_YAML = (
    "manifest_version: 1\n"
    "commands:\n"
    "  - name: sync\n"
    "    description: Sync tables\n"
    "    args:\n"
    "      - {name: table, required: true}\n"
    "      - {name: --limit}\n"
)


def test_iter_plugin_manifests_lists_valid(plugin_env):
    plugin_env("hello", MINIMAL + "description: Hi.\n")
    plugin_env("dagster", MINIMAL)
    assert {name for name, _ in registry.iter_plugin_manifests()} == {"hello", "dagster"}


def test_iter_plugin_manifests_skips_missing_manifest(plugin_env):
    plugin_env("hello", MINIMAL)
    plugin_env("broken", manifest_yaml=None)  # importable, but no datus-plugin.yml
    assert {name for name, _ in registry.iter_plugin_manifests()} == {"hello"}


def test_plugin_commands_parsed(plugin_env):
    plugin_env("hello", COMMANDS_YAML)
    cmds = registry.plugin_commands("hello")
    assert [c.name for c in cmds] == ["sync"]
    assert cmds[0].args[0].name == "table"
    assert cmds[0].args[0].required is True
    assert cmds[0].args[1].name == "--limit"


def test_plugin_commands_unknown_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_commands("mystery") == []


def test_lookup_never_raises_on_entry_points_failure(monkeypatch):
    def boom():
        raise RuntimeError("entry_points exploded")

    monkeypatch.setattr(importlib_metadata, "entry_points", boom)
    assert registry.load_plugin_manifest("hello") is None
    assert registry.iter_plugin_entry_points() == []
    assert registry.plugin_skill_directories() == []
    assert registry.plugin_system_prompt_sections(_FakeConfig({})) == []
    assert registry.collect_plugin_cli_permissions() == {}
    assert registry.collect_plugin_tool_transformers() == {}


def test_unimportable_package_skipped(plugin_env, caplog):
    plugin_env("hello", MINIMAL, value="no_such_package_anywhere_xyz")
    with caplog.at_level("WARNING"):
        assert registry.load_plugin_manifest("hello") is None
    assert "not importable" in caplog.text


# ---------------------------------------------------------------------------
# The no-code-execution property
# ---------------------------------------------------------------------------


def test_manifest_surfaces_never_execute_plugin_code(plugin_env):
    """skills / permissions / prompt / schema collect fine even when importing
    the package would explode — proof that no plugin code runs."""
    pkg = plugin_env(
        "hello",
        MINIMAL
        + "cli: {pkg}.cli:main\n"
        + "skills: skills\n"
        + "system_prompt: prompt.md.j2\n"
        + "permissions:\n  normal:\n    allow: ['greet:*']\n"
        + "config_schema:\n  type: object\n  properties:\n    api_key: {type: string}\n",
        init_body="raise AssertionError('plugin package was imported')",
        files={"prompt.md.j2": "## hello section"},
    )
    (pkg / "skills").mkdir()

    assert registry.plugin_skill_directories() == [str(pkg / "skills")]
    rules = registry.collect_plugin_cli_permissions()
    assert rules["normal"].allow == ["datus hello greet:*"]
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert "## hello section" in sections
    assert registry.plugin_config_schema("hello") == [
        {"name": "api_key", "description": "", "required": False, "secret": False}
    ]
    # Only an actual code ref resolution imports the package — and the
    # exploding __init__ surfaces as a warn-and-None, never a crash.
    assert registry.resolve_code_ref(f"{pkg.name}.cli:main", "hello") is None


def test_transformer_import_is_lazy(plugin_env):
    """The plugin module is imported by collect_plugin_tool_transformers, not
    by manifest loading or the other collectors."""
    pkg = plugin_env(
        "hello",
        MINIMAL + "tool_transformers:\n  execute_sql: {pkg}.tf:passthrough\nskills: skills\n",
        init_body="import pathlib\npathlib.Path(__file__).parent.joinpath('IMPORTED').touch()\n",
        files={"tf.py": TRANSFORMER_MODULE},
    )
    (pkg / "skills").mkdir()
    marker = pkg / "IMPORTED"

    registry.load_plugin_manifest("hello")
    registry.plugin_skill_directories()
    registry.collect_plugin_cli_permissions()
    assert not marker.exists()

    collected = registry.collect_plugin_tool_transformers()
    assert marker.exists()
    assert len(collected["execute_sql"]) == 1


# ---------------------------------------------------------------------------
# plugin_skill_directories
# ---------------------------------------------------------------------------


def test_skill_directories_collects_existing(plugin_env):
    pkg = plugin_env("hello", MINIMAL + "skills: skills\n")
    (pkg / "skills").mkdir()
    assert registry.plugin_skill_directories() == [str(pkg / "skills")]


def test_skill_directories_skips_missing_dir(plugin_env):
    plugin_env("hello", MINIMAL + "skills: skills\n")  # declared but not created
    assert registry.plugin_skill_directories() == []


def test_skill_directories_skips_plugin_without_skills(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_skill_directories() == []


def test_skill_directories_dedup(plugin_env):
    pkg = plugin_env("a", MINIMAL + "skills: skills\n")
    (pkg / "skills").mkdir()
    # A second entry point resolving to the same package contributes the same dir.
    plugin_env("b", manifest_yaml=None, value=pkg.name)
    assert registry.plugin_skill_directories() == [str(pkg / "skills")]


def test_skill_directories_escape_rejected(plugin_env, caplog):
    pkg = plugin_env("hello", MINIMAL + "skills: ../evil\n")
    (pkg.parent / "evil").mkdir()  # exists, but outside the package
    with caplog.at_level("WARNING"):
        assert registry.plugin_skill_directories() == []
    assert "escapes the package directory" in caplog.text


def test_skill_directories_survives_bad_plugin(plugin_env):
    plugin_env("broken", manifest_yaml=None)
    pkg = plugin_env("hello", MINIMAL + "skills: skills\n")
    (pkg / "skills").mkdir()
    assert registry.plugin_skill_directories() == [str(pkg / "skills")]


# ---------------------------------------------------------------------------
# plugin_system_prompt_sections
# ---------------------------------------------------------------------------

ECHO_TEMPLATE = "PROFILES={{ profiles.keys()|sort|join(',') }}"


def _prompt_plugin(plugin_env, name="hello", template="## Hello\nManage DAGs.", schema_props=None):
    schema = ""
    if schema_props:
        props = "\n".join(f"    {p}: {{type: string}}" for p in schema_props)
        schema = f"config_schema:\n  type: object\n  properties:\n{props}\n"
    return plugin_env(name, MINIMAL + "system_prompt: prompt.md.j2\n" + schema, files={"prompt.md.j2": template})


def test_system_prompt_sections_renders_template_with_preamble(plugin_env):
    _prompt_plugin(plugin_env)
    sections = registry.plugin_system_prompt_sections(_FakeConfig({"hello": {}}))
    assert len(sections) == 2
    assert sections[0].startswith("## Plugins")
    assert sections[1] == "## Hello\nManage DAGs."


def test_system_prompt_sections_passes_stripped_profiles(plugin_env):
    _prompt_plugin(plugin_env, template=ECHO_TEMPLATE, schema_props=["base_url"])
    profiles = {"local": {"base_url": "http://h", "api_key": "sk-REAL"}}
    sections = registry.plugin_system_prompt_sections(_FakeConfig({"hello": profiles}))
    assert "PROFILES=local" in sections
    assert all("sk-REAL" not in s for s in sections)


def test_system_prompt_sections_skips_plugin_without_template(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_system_prompt_sections(_FakeConfig({"hello": {"p": {}}})) == []


def test_system_prompt_sections_survives_broken_template(plugin_env):
    _prompt_plugin(plugin_env, name="boom", template="{% if %}")
    _prompt_plugin(plugin_env, name="good", template="## Good")
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert sections[0].startswith("## Plugins")
    assert sections[1:] == ["## Good"]


def test_system_prompt_sections_defaults_missing_profiles_to_empty(plugin_env):
    _prompt_plugin(plugin_env, template="{% if profiles %}configured{% else %}unconfigured{% endif %}")
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert "unconfigured" in sections


def test_preamble_names_config_file_location(plugin_env, monkeypatch):
    _prompt_plugin(plugin_env)
    monkeypatch.setattr(registry, "_agent_config_location", lambda: "/srv/conf/agent.yml")
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert "`/srv/conf/agent.yml`" in sections[0]
    assert "agent.plugins.<plugin>.<profile>" in sections[0]


def test_preamble_degrades_without_config_path(plugin_env, monkeypatch):
    _prompt_plugin(plugin_env)
    monkeypatch.setattr(registry, "_agent_config_location", lambda: None)
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}))
    assert sections[0].startswith("## Plugins")
    assert "agent.yml" in sections[0]  # generic wording instead of a path


def test_preamble_immutable_omits_config_path_and_edit_guidance(plugin_env, monkeypatch):
    """Read-only mode: the preamble must not leak the server config path nor
    describe the profile shape — it defers to the administrator instead."""
    _prompt_plugin(plugin_env)
    monkeypatch.setattr(registry, "_agent_config_location", lambda: "/srv/conf/agent.yml")
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}, config_mutable=False))
    assert sections[0].startswith("## Plugins")
    assert "/srv/conf/agent.yml" not in sections[0]
    assert "agent.plugins.<plugin>.<profile>" not in sections[0]
    assert "administrator" in sections[0]
    assert "read-only" in sections[0]
    # The managed bridge accepts one plugin invocation per bash call; telling
    # the model up front avoids rejected commands it has to retry.
    assert "one `datus <plugin>` command per bash call" in sections[0]


def test_sections_pass_config_mutable_to_template(plugin_env):
    _prompt_plugin(plugin_env, template="{% if config_mutable %}CAN-EDIT{% else %}NO-EDIT{% endif %}")
    assert "NO-EDIT" in registry.plugin_system_prompt_sections(_FakeConfig({}, config_mutable=False))
    assert "CAN-EDIT" in registry.plugin_system_prompt_sections(_FakeConfig({}, config_mutable=True))


def test_immutable_withholds_config_path_from_template(plugin_env, monkeypatch):
    _prompt_plugin(plugin_env, template="PATH={{ config_path }}")
    monkeypatch.setattr(registry, "_agent_config_location", lambda: "/srv/conf/agent.yml")
    sections = registry.plugin_system_prompt_sections(_FakeConfig({}, config_mutable=False))
    assert "PATH=None" in sections
    mutable_sections = registry.plugin_system_prompt_sections(_FakeConfig({}, config_mutable=True))
    assert "PATH=/srv/conf/agent.yml" in mutable_sections


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


def _profiles_cfg(active_profiles):
    class _Cfg:
        plugin_services = {"hello": {"prod": {"name": "prod"}, "staging": {"name": "staging"}}}

        def active_plugin_names(self):
            return {"hello"}

        def active_plugin_profiles(self, name):
            return active_profiles

    return _Cfg()


def test_system_prompt_narrows_to_active_profiles(plugin_env):
    """Only the project-pinned profiles reach the LLM, not every environment."""
    _prompt_plugin(plugin_env, template=ECHO_TEMPLATE)
    sections = registry.plugin_system_prompt_sections(_profiles_cfg(["staging"]))
    assert "PROFILES=staging" in sections
    assert "PROFILES=prod,staging" not in sections


def test_system_prompt_no_pin_passes_all_profiles(plugin_env):
    _prompt_plugin(plugin_env, template=ECHO_TEMPLATE)
    assert "PROFILES=prod,staging" in registry.plugin_system_prompt_sections(_profiles_cfg(None))


def test_system_prompt_stale_pin_falls_back_to_all(plugin_env):
    """A pin that matches no configured profile surfaces everything rather than
    blanking the plugin out of the prompt."""
    _prompt_plugin(plugin_env, template=ECHO_TEMPLATE)
    assert "PROFILES=prod,staging" in registry.plugin_system_prompt_sections(_profiles_cfg(["deleted"]))


def test_system_prompt_sections_respect_active_names(plugin_env):
    # Distinct per-plugin output so the assertion proves the *right* plugin was
    # kept — identical text would pass even if the filter let plugin "b" through.
    _prompt_plugin(plugin_env, name="a", template="## Plugin A section")
    _prompt_plugin(plugin_env, name="b", template="## Plugin B section")

    class _Cfg:
        plugin_services = {}

        def active_plugin_names(self):
            return {"a"}

    sections = registry.plugin_system_prompt_sections(_Cfg())
    assert any("## Plugin A section" in s for s in sections)
    assert not any("## Plugin B section" in s for s in sections)


# ---------------------------------------------------------------------------
# collect_plugin_cli_permissions
# ---------------------------------------------------------------------------

PERMS = (
    "permissions:\n"
    "  normal:\n"
    "    allow: ['greet:*', 'version']\n"
    "    ask: ['config set:*']\n"
    "    deny: ['config wipe:*']\n"
    "  auto:\n"
    "    allow: [':*']\n"
)


def test_cli_permissions_prefixing_and_shapes(plugin_env):
    plugin_env("hello", MINIMAL + PERMS)
    rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal", "auto"}
    # ``greet:*`` -> prefix rule; ``version`` (no colon) -> exact match.
    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]
    assert rules["normal"].ask == ["datus hello config set:*"]
    assert rules["normal"].deny == ["datus hello config wipe:*"]
    # ``:*`` covers the whole namespace.
    assert rules["auto"].allow == ["datus hello:*"]
    assert rules["auto"].ask == []


def test_cli_permissions_never_set_scalar_fields(plugin_env):
    plugin_env("hello", MINIMAL + PERMS)
    rules = registry.collect_plugin_cli_permissions()
    # ``default`` / ``classifier`` must stay unset so the profile posture and
    # merge_with scalar semantics are untouched by plugin declarations.
    for ruleset in rules.values():
        assert "default" not in ruleset.model_fields_set
        assert "classifier" not in ruleset.model_fields_set


def test_cli_permissions_dangerous_and_unknown_profiles_dropped(plugin_env, caplog):
    plugin_env(
        "hello",
        MINIMAL
        + "permissions:\n"
        + "  dangerous:\n    deny: ['config wipe:*']\n"
        + "  paranoid:\n    ask: ['greet:*']\n"
        + "  normal:\n    allow: ['greet:*']\n",
    )
    with caplog.at_level("WARNING"):
        rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal"}
    assert "dangerous" in caplog.text
    assert "paranoid" in caplog.text


def test_cli_permissions_malformed_entries_skipped(plugin_env):
    plugin_env(
        "hello",
        MINIMAL
        + "permissions:\n"
        + "  normal:\n"
        + "    allow: ['greet:*', 42, '', '   ']\n"  # non-str / empty entries dropped
        + "    ask: not-a-list\n"  # non-list action dropped
        + "    grant: ['greet:*']\n"  # unknown action dropped
        + "  auto: [not-a-dict]\n",  # non-dict profile dropped
    )
    rules = registry.collect_plugin_cli_permissions()

    assert set(rules) == {"normal"}
    assert rules["normal"].allow == ["datus hello greet:*"]
    assert rules["normal"].ask == []
    assert rules["normal"].deny == []


def test_cli_permissions_non_dict_declaration_ignored(plugin_env):
    plugin_env("hello", MINIMAL + "permissions: [normal]\n")
    assert registry.collect_plugin_cli_permissions() == {}


def test_cli_permissions_plugin_without_permissions_skipped(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.collect_plugin_cli_permissions() == {}


def test_cli_permissions_duplicate_entry_point_first_wins(plugin_env, caplog):
    plugin_env("hello", MINIMAL + PERMS)
    plugin_env("hello", MINIMAL + "permissions:\n  normal:\n    allow: ['other:*']\n")
    with caplog.at_level("WARNING"):
        rules = registry.collect_plugin_cli_permissions()

    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]
    assert "Duplicate" in caplog.text


def test_cli_permissions_unsafe_entry_point_name_skipped(plugin_env, caplog):
    plugin_env("evil name", MINIMAL + PERMS)
    plugin_env("*", MINIMAL + PERMS)
    with caplog.at_level("WARNING"):
        assert registry.collect_plugin_cli_permissions() == {}
    assert "not a safe CLI token" in caplog.text


def test_cli_permissions_broken_plugin_skipped(plugin_env):
    plugin_env("broken", manifest_yaml=None)
    plugin_env("hello", MINIMAL + PERMS)
    rules = registry.collect_plugin_cli_permissions()
    assert rules["normal"].allow == ["datus hello greet:*", "datus hello version"]


# ---------------------------------------------------------------------------
# collect_plugin_tool_transformers
# ---------------------------------------------------------------------------


def _tf_plugin(plugin_env, name="hello", patterns="  db_tools.execute_sql: {pkg}.tf:passthrough\n"):
    return plugin_env(name, MINIMAL + "tool_transformers:\n" + patterns, files={"tf.py": TRANSFORMER_MODULE})


def test_tool_transformers_collects_single_ref(plugin_env):
    pkg = _tf_plugin(plugin_env)
    collected = registry.collect_plugin_tool_transformers()
    expected = importlib.import_module(f"{pkg.name}.tf").passthrough
    assert collected == {"db_tools.execute_sql": [expected]}


def test_tool_transformers_collects_ref_list(plugin_env):
    pkg = _tf_plugin(
        plugin_env,
        patterns="  execute_sql:\n    - {pkg}.tf:passthrough\n    - {pkg}.tf:passthrough\n",
    )
    collected = registry.collect_plugin_tool_transformers()
    expected = importlib.import_module(f"{pkg.name}.tf").passthrough
    assert collected == {"execute_sql": [expected, expected]}


def test_tool_transformers_accumulates_across_plugins(plugin_env):
    _tf_plugin(plugin_env, name="a")
    _tf_plugin(plugin_env, name="b")
    collected = registry.collect_plugin_tool_transformers()
    assert len(collected["db_tools.execute_sql"]) == 2
    assert all(callable(t) for t in collected["db_tools.execute_sql"])


def test_tool_transformers_plugin_without_declaration_skipped(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.collect_plugin_tool_transformers() == {}


def test_tool_transformers_import_failure_skips_only_that_ref(plugin_env, caplog):
    _tf_plugin(
        plugin_env,
        patterns="  execute_sql:\n    - {pkg}.missing_module:fn\n    - {pkg}.tf:passthrough\n",
    )
    with caplog.at_level("WARNING"):
        collected = registry.collect_plugin_tool_transformers()
    assert len(collected["execute_sql"]) == 1
    assert "failed to load code ref" in caplog.text


def test_tool_transformers_non_callable_ref_skipped(plugin_env, caplog):
    _tf_plugin(plugin_env, patterns="  execute_sql: {pkg}.tf:NOT_CALLABLE\n")
    with caplog.at_level("WARNING"):
        assert registry.collect_plugin_tool_transformers() == {}
    assert "not callable" in caplog.text


def test_tool_transformers_broken_plugin_skipped(plugin_env):
    plugin_env("broken", manifest_yaml=None)
    _tf_plugin(plugin_env, name="ok")
    collected = registry.collect_plugin_tool_transformers()
    assert len(collected["db_tools.execute_sql"]) == 1


def test_tool_transformers_resolution_memoized(plugin_env, monkeypatch):
    _tf_plugin(plugin_env)
    calls = []
    original = registry.resolve_code_ref

    def counting(ref, plugin_name):
        calls.append(ref)
        return original(ref, plugin_name)

    monkeypatch.setattr(registry, "resolve_code_ref", counting)
    registry.collect_plugin_tool_transformers()
    registry.collect_plugin_tool_transformers()
    assert len(calls) == 1  # second collection served from the memo


# ---------------------------------------------------------------------------
# active_names filtering (per-project activation whitelist)
# ---------------------------------------------------------------------------


def _two_full_plugins(plugin_env):
    dirs = []
    for name in ("a", "b"):
        pkg = plugin_env(
            name,
            MINIMAL
            + "skills: skills\n"
            + "tool_transformers:\n  execute_sql: {pkg}.tf:passthrough\n"
            + "permissions:\n  normal:\n    allow: ['greet:*']\n",
            files={"tf.py": TRANSFORMER_MODULE},
        )
        (pkg / "skills").mkdir()
        dirs.append(str(pkg / "skills"))
    return dirs


def test_skill_directories_active_names_none_includes_all(plugin_env):
    a_dir, b_dir = _two_full_plugins(plugin_env)
    assert registry.plugin_skill_directories(active_names=None) == [a_dir, b_dir]


def test_skill_directories_active_names_filters(plugin_env):
    a_dir, _b_dir = _two_full_plugins(plugin_env)
    assert registry.plugin_skill_directories(active_names={"a"}) == [a_dir]


def test_skill_directories_empty_active_names_excludes_all(plugin_env):
    _two_full_plugins(plugin_env)
    assert registry.plugin_skill_directories(active_names=set()) == []


def test_tool_transformers_active_names_filters(plugin_env):
    _two_full_plugins(plugin_env)
    assert len(registry.collect_plugin_tool_transformers(active_names={"a"})["execute_sql"]) == 1
    assert registry.collect_plugin_tool_transformers(active_names=set()) == {}


def test_cli_permissions_active_names_filters(plugin_env):
    _two_full_plugins(plugin_env)
    rules = registry.collect_plugin_cli_permissions(active_names={"b"})
    assert rules["normal"].allow == ["datus b greet:*"]
    assert registry.collect_plugin_cli_permissions(active_names=set()) == {}


# ---------------------------------------------------------------------------
# plugin_config_schema (JSON Schema -> TUI field specs)
# ---------------------------------------------------------------------------

SCHEMA_YAML = (
    "config_schema:\n"
    "  type: object\n"
    "  required: [api_key]\n"
    "  properties:\n"
    "    api_key:\n"
    "      type: string\n"
    "      description: key\n"
    "      x-secret: true\n"
    "    base_url:\n"
    "      type: string\n"
    "      description: url\n"
    "      default: https://x\n"
)


def test_config_schema_derives_field_specs(plugin_env):
    plugin_env("hello", MINIMAL + SCHEMA_YAML)
    assert registry.plugin_config_schema("hello") == [
        {"name": "api_key", "description": "key", "required": True, "secret": True},
        {"name": "base_url", "description": "url", "required": False, "secret": False, "default": "https://x"},
    ]


def test_config_schema_preserves_property_order(plugin_env):
    plugin_env(
        "hello",
        MINIMAL
        + "config_schema:\n  type: object\n  properties:\n"
        + "    zeta: {type: string}\n    alpha: {type: string}\n",
    )
    assert [f["name"] for f in registry.plugin_config_schema("hello")] == ["zeta", "alpha"]


def test_config_schema_absent_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_config_schema("hello") == []


def test_config_schema_unknown_plugin_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_config_schema("mystery") == []


def test_config_schema_without_properties_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL + "config_schema:\n  type: object\n")
    assert registry.plugin_config_schema("hello") == []


NESTED_SCHEMA_YAML = (
    "config_schema:\n"
    "  type: object\n"
    "  required: [api_base_url, s3]\n"
    "  properties:\n"
    "    api_base_url:\n"
    "      type: string\n"
    "      description: REST endpoint\n"
    "    s3:\n"
    "      type: object\n"
    "      required: [secret_access_key]\n"
    "      properties:\n"
    "        region:\n"
    "          type: string\n"
    "          default: us-east-1\n"
    "        secret_access_key:\n"
    "          type: string\n"
    "          description: S3 secret key\n"
    "          x-secret: true\n"
)


def test_config_schema_flattens_nested_objects(plugin_env):
    plugin_env("hello", MINIMAL + NESTED_SCHEMA_YAML)
    assert registry.plugin_config_schema("hello") == [
        {"name": "api_base_url", "description": "REST endpoint", "required": True, "secret": False},
        {"name": "s3.region", "description": "", "required": False, "secret": False, "default": "us-east-1"},
        {"name": "s3.secret_access_key", "description": "S3 secret key", "required": True, "secret": True},
    ]


def test_config_schema_nested_required_needs_required_ancestors(plugin_env):
    """A required leaf inside an optional object is not form-required."""
    plugin_env(
        "hello",
        MINIMAL
        + "config_schema:\n"
        + "  type: object\n"
        + "  properties:\n"
        + "    s3:\n"
        + "      type: object\n"
        + "      required: [key]\n"
        + "      properties:\n"
        + "        key: {type: string}\n",
    )
    assert registry.plugin_config_schema("hello") == [
        {"name": "s3.key", "description": "", "required": False, "secret": False}
    ]


def test_config_schema_block_level_secret_marks_all_leaves(plugin_env):
    plugin_env(
        "hello",
        MINIMAL
        + "config_schema:\n"
        + "  type: object\n"
        + "  properties:\n"
        + "    s3:\n"
        + "      type: object\n"
        + "      x-secret: true\n"
        + "      properties:\n"
        + "        region: {type: string}\n"
        + "        key: {type: string}\n",
    )
    specs = registry.plugin_config_schema("hello")
    assert [s["name"] for s in specs] == ["s3.region", "s3.key"]
    assert all(s["secret"] for s in specs)


def test_config_schema_object_without_properties_stays_flat(plugin_env):
    """A ``type: object`` property without nested ``properties`` stays one field."""
    plugin_env("hello", MINIMAL + "config_schema:\n  type: object\n  properties:\n    extras: {type: object}\n")
    assert [s["name"] for s in registry.plugin_config_schema("hello")] == ["extras"]


def test_config_schema_flattens_three_levels(plugin_env):
    plugin_env(
        "hello",
        MINIMAL
        + "config_schema:\n"
        + "  type: object\n"
        + "  properties:\n"
        + "    a:\n"
        + "      type: object\n"
        + "      properties:\n"
        + "        b:\n"
        + "          type: object\n"
        + "          properties:\n"
        + "            c: {type: string}\n",
    )
    assert [s["name"] for s in registry.plugin_config_schema("hello")] == ["a.b.c"]


def test_config_schema_nesting_depth_capped(plugin_env, caplog):
    """Fields nested beyond the flattening cap are dropped with a warning."""
    lines = ["config_schema:", "  type: object"]
    indent = "  "
    for i in range(10):
        lines.append(f"{indent}properties:")
        lines.append(f"{indent}  n{i}:")
        lines.append(f"{indent}    type: object")
        indent += "    "
    lines.append(f"{indent}properties:")
    lines.append(f"{indent}  leaf: {{type: string}}")
    plugin_env("hello", MINIMAL + "\n".join(lines) + "\n")
    with caplog.at_level("WARNING"):
        assert registry.plugin_config_schema("hello") == []
    assert "nests deeper" in caplog.text


# ---------------------------------------------------------------------------
# plugin_validate_profile (jsonschema)
# ---------------------------------------------------------------------------

VALIDATION_SCHEMA = (
    "config_schema:\n"
    "  type: object\n"
    "  required: [api_key]\n"
    "  properties:\n"
    "    api_key:\n"
    "      type: string\n"
    "      pattern: '^sk-'\n"
    "    region:\n"
    "      type: string\n"
    "      enum: [us, eu]\n"
)


def test_validate_profile_reports_missing_required(plugin_env):
    plugin_env("hello", MINIMAL + VALIDATION_SCHEMA)
    errors = registry.plugin_validate_profile("hello", {})
    assert len(errors) == 1
    assert errors[0].startswith("profile:")
    assert "api_key" in errors[0]


def test_validate_profile_reports_field_violations_with_path(plugin_env):
    plugin_env("hello", MINIMAL + VALIDATION_SCHEMA)
    errors = registry.plugin_validate_profile("hello", {"api_key": "wrong", "region": "mars"})
    assert any(e.startswith("api_key:") for e in errors)
    assert any(e.startswith("region:") for e in errors)


def test_validate_profile_valid_profile_passes(plugin_env):
    plugin_env("hello", MINIMAL + VALIDATION_SCHEMA)
    assert registry.plugin_validate_profile("hello", {"api_key": "sk-1", "region": "us"}) == []


def test_validate_profile_env_placeholders_are_opaque(plugin_env):
    """``${ENV_VAR}`` values are shape-only: pattern/enum violations on them are
    suppressed, but a missing required field still fires."""
    plugin_env("hello", MINIMAL + VALIDATION_SCHEMA)
    assert registry.plugin_validate_profile("hello", {"api_key": "${STATSIG_KEY}"}) == []
    errors = registry.plugin_validate_profile("hello", {"region": "${REGION}"})
    assert len(errors) == 1  # required api_key missing; placeholder region passes
    assert "api_key" in errors[0]


def test_validate_profile_no_schema_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL)
    assert registry.plugin_validate_profile("hello", {}) == []


def test_validate_profile_unknown_plugin_returns_empty(plugin_env):
    plugin_env("hello", MINIMAL + VALIDATION_SCHEMA)
    assert registry.plugin_validate_profile("mystery", {}) == []
