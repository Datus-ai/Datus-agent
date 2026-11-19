# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Registration contracts shared by factory, discovery and delegation consumers."""

from types import SimpleNamespace

import pytest

from datus.agent.agent_registry import list_configurable_agents, resolve_configurable_agent
from datus.utils.exceptions import DatusException
from tests.unit_tests.plugins.conftest import plugin_env as plugin_env


def _config(nodes=None, plugins=()):
    return SimpleNamespace(agentic_nodes=nodes or {}, plugins_enabled=True, active_plugin_names=lambda: set(plugins))


def test_plain_configuration_and_discovery_resolve_the_same_definition(plugin_env):
    config = _config(
        {
            "support": {
                "node_class": "configurable",
                "instructions": "Answer using the configured references.",
                "tools": ["db_tools.list_tables"],
                "mcp": [{"connection_ref": "documents", "tools": ["search"]}],
                "skills": ["support-guidelines"],
            },
            "legacy_sql": {"node_class": "gen_sql", "tools": "db_tools.*"},
        }
    )

    resolved = resolve_configurable_agent("support", config)
    catalog = list_configurable_agents(config)

    assert set(catalog) == {"support"}
    assert catalog["support"] == resolved
    assert resolved.spec.skills == ["support-guidelines"]
    assert resolved.spec.mcp[0].tools == ["search"]
    assert resolve_configurable_agent("legacy_sql", config) is None


def test_installed_plugin_definitions_are_available_without_copying_node_configuration(plugin_env):
    plugin_env(
        "research",
        "manifest_version: 1\nagents:\n  researcher:\n    instructions: Consult the selected business documents.\n",
    )
    config = _config(plugins=("research",))

    definition = resolve_configurable_agent("researcher", config)

    assert definition.spec.name == "researcher"
    assert definition.plugin.plugin_name == "research"
    assert list_configurable_agents(config)["researcher"] == definition
    assert config.agentic_nodes == {}


def test_project_instance_binds_its_own_resources_without_hardcoding_project_names_in_the_package(plugin_env):
    plugin_env("research", "manifest_version: 1\nagents:\n  researcher:\n    instructions: Review evidence.\n")
    config = _config(
        {
            "sales_review": {
                "node_class": "configurable",
                "source": {"plugin": "research", "agent": "researcher"},
                "tools": ["db_tools.execute_sql"],
                "scoped_context": {"datasource": "tenant_sales"},
            }
        },
        plugins=("research",),
    )
    resolved = resolve_configurable_agent("sales_review", config)
    assert resolved.spec.tools == ["db_tools.execute_sql"]
    assert resolved.spec.scoped_context.datasource == "tenant_sales"
    assert resolved.plugin.spec.scoped_context is None


def test_explicit_plugin_source_preserves_the_local_instance_name_and_validates_business_parameters(plugin_env):
    plugin_env(
        "research",
        """manifest_version: 1
agents:
  researcher:
    instructions: Consult the selected business documents.
    config_schema:
      type: object
      required: [collection]
      properties:
        collection: {type: string}
      additionalProperties: false
""",
    )
    config = _config(
        {
            "sales_support": {
                "node_class": "configurable",
                "source": {"plugin": "research", "agent": "researcher"},
                "runtime_config": {"collection": "sales"},
            }
        },
        plugins=("research",),
    )

    definition = resolve_configurable_agent("sales_support", config)
    definition.validate_runtime_config()

    assert definition.spec.name == "sales_support"
    assert definition.plugin.spec.name == "researcher"
    assert definition.runtime_config == {"collection": "sales"}


@pytest.mark.parametrize("source", [{"plugin": "research"}, {"plugin": "research", "agent": []}])
def test_malformed_plugin_reference_does_not_fall_back_to_a_plain_agent(plugin_env, source):
    config = _config({"support": {"node_class": "configurable", "source": source}})

    with pytest.raises(DatusException, match="source requires"):
        resolve_configurable_agent("support", config)


def test_disabled_plugin_reference_does_not_fall_back_to_a_plain_agent(plugin_env):
    plugin_env("research", "manifest_version: 1\nagents:\n  researcher: {}\n")
    config = _config(
        {"support": {"node_class": "configurable", "source": {"plugin": "research", "agent": "researcher"}}}
    )

    with pytest.raises(DatusException, match="missing or disabled"):
        resolve_configurable_agent("support", config)


def test_configured_and_plugin_name_collision_requires_an_explicit_source(plugin_env):
    plugin_env("research", "manifest_version: 1\nagents:\n  researcher: {}\n")
    config = _config({"researcher": {"node_class": "configurable"}}, plugins=("research",))

    with pytest.raises(DatusException, match="conflicts with a plugin agent"):
        resolve_configurable_agent("researcher", config)
    assert "researcher" not in list_configurable_agents(config)


@pytest.mark.parametrize("name", ["analytics_explorer", "ask_dashboard", "ask_report", "chat", "explore", "gen_sql"])
def test_plugin_can_replace_a_builtin_only_with_an_explicit_source(plugin_env, name):
    plugin_env("research", f"manifest_version: 1\nagents:\n  {name}: {{}}\n  researcher: {{}}\n")
    config = _config(plugins=("research",))

    assert resolve_configurable_agent(name, config) is None
    assert set(list_configurable_agents(config)) == {"researcher"}

    config.agentic_nodes[name] = {
        "node_class": "configurable",
        "source": {"plugin": "research", "agent": name},
    }
    assert list_configurable_agents(config)[name].plugin.plugin_name == "research"


def test_invalid_plugin_agent_does_not_hide_other_agents_or_block_builtins(plugin_env):
    plugin_env(
        "research",
        "manifest_version: 1\nagents:\n  bad_agent:\n    protocol_version: 2\n  researcher: {}\n",
    )
    config = _config({"support": {"node_class": "configurable"}}, plugins=("research",))

    assert set(list_configurable_agents(config)) == {"support", "researcher"}
    assert resolve_configurable_agent("support", config).plugin is None
    assert resolve_configurable_agent("gen_sql", config) is None
    assert resolve_configurable_agent("researcher", config).plugin.plugin_name == "research"
    with pytest.raises(DatusException, match="protocol_version"):
        resolve_configurable_agent("bad_agent", config)


def test_runtime_configuration_error_does_not_echo_a_secret_value(plugin_env):
    plugin_env(
        "research",
        """manifest_version: 1
agents:
  researcher:
    config_schema:
      type: object
      properties:
        credential: {type: integer}
""",
    )
    config = _config(
        {
            "support": {
                "node_class": "configurable",
                "source": {"plugin": "research", "agent": "researcher"},
                "runtime_config": {"credential": "secret-canary-value"},
            }
        },
        plugins=("research",),
    )

    with pytest.raises(DatusException, match="credential") as exc:
        resolve_configurable_agent("support", config).validate_runtime_config()

    assert "secret-canary-value" not in str(exc.value)


def test_required_instance_parameters_do_not_prevent_manifest_discovery(plugin_env):
    plugin_env(
        "research",
        """manifest_version: 1
agents:
  researcher:
    config_schema:
      type: object
      required: [collection]
""",
    )
    definitions = list_configurable_agents(_config(plugins=("research",)))

    assert definitions["researcher"].plugin.config_schema["required"] == ["collection"]
    with pytest.raises(DatusException, match="required validation"):
        definitions["researcher"].validate_runtime_config()


def test_user_node_cannot_supply_an_executable_factory_reference(plugin_env):
    config = _config({"support": {"node_class": "configurable", "runtime_factory": "os:system"}})

    with pytest.raises(DatusException, match="registered plugin source"):
        resolve_configurable_agent("support", config)
