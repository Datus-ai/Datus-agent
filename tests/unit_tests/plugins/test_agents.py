# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Behavioral contracts for manifest agents and plain Python runtime preparations."""

import json
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from datus.plugins.agents import PluginAgentDefinition, RuntimePreparation
from datus.plugins.base import parse_manifest
from datus.plugins.registry import collect_plugin_agents, resolve_agent_runtime_factory
from datus.schemas.agent_models import ScopedContext
from datus.utils.exceptions import DatusException


def _config(*plugins):
    return SimpleNamespace(plugins_enabled=True, active_plugin_names=lambda: set(plugins))


def test_agent_discovery_does_not_execute_package_or_factory(plugin_env, tmp_path):
    marker = tmp_path / "imported"
    package = plugin_env(
        "research",
        """manifest_version: 1
agents:
  researcher:
    instructions: Answer the question.
    runtime_factory: "{pkg}.runtime:create_runtime"
""",
        init_body=f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        files={"runtime.py": "def create_runtime(config, host):\n    return {'config': dict(config), 'host': host}\n"},
    )

    definition = collect_plugin_agents(_config("research"))["researcher"]

    # Discovery is an explicit no-execution contract, including package __init__.
    assert not marker.exists()
    assert definition.package_dir == package
    factory = resolve_agent_runtime_factory(definition, _config("research"))
    first = factory({"request": "first"}, "host")
    second = factory({"request": "second"}, "host")
    first["config"]["request"] = "changed"
    assert marker.read_text() == "imported"
    assert second == {"config": {"request": "second"}, "host": "host"}


def test_only_project_enabled_agents_are_discovered(plugin_env):
    manifest = "manifest_version: 1\nagents:\n  {name}:\n    instructions: Answer the question.\n"
    plugin_env("research", manifest.replace("{name}", "researcher"))
    plugin_env("private", manifest.replace("{name}", "private_assistant"))

    assert set(collect_plugin_agents(_config("research"))) == {"researcher"}
    assert collect_plugin_agents(_config()) == {}
    disabled = _config("research", "private")
    disabled.plugins_enabled = False
    assert collect_plugin_agents(disabled) == {}


def test_factory_resolution_rechecks_project_activation(plugin_env):
    plugin_env(
        "research",
        "manifest_version: 1\nagents:\n  researcher:\n    runtime_factory: '{pkg}.runtime:create_runtime'\n",
        files={"runtime.py": "raise RuntimeError('inactive runtime was imported')\n"},
    )
    definition = collect_plugin_agents(_config("research"))["researcher"]

    with pytest.raises(DatusException, match="not enabled"):
        resolve_agent_runtime_factory(definition, _config())


def test_failed_project_activation_lookup_cannot_enable_private_agents(plugin_env, tmp_path):
    marker = tmp_path / "private-imported"
    plugin_env(
        "private",
        "manifest_version: 1\nagents:\n  private_assistant:\n    runtime_factory: '{pkg}:create_runtime'\n",
        init_body=f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
    )

    def broken_activation():
        raise RuntimeError("secret-activation-canary")

    config = SimpleNamespace(plugins_enabled=True, active_plugin_names=broken_activation)
    with pytest.raises(DatusException, match="Unable to resolve enabled") as error:
        collect_plugin_agents(config)
    assert "secret-activation-canary" not in str(error.value)
    assert not marker.exists()


def test_duplicate_agent_names_fail_instead_of_picking_one_plugin(plugin_env):
    manifest = "manifest_version: 1\nagents:\n  researcher:\n    instructions: Answer the question.\n"
    plugin_env("first", manifest)
    plugin_env("second", manifest)

    with pytest.raises(DatusException, match="multiple enabled plugins"):
        collect_plugin_agents(_config("first", "second"))


def test_invalid_protocol_preserves_existing_plugin_features_but_cannot_activate(plugin_env):
    plugin_env(
        "research",
        """manifest_version: 1
cli: "{pkg}.cli:main"
agents:
  researcher:
    protocol_version: 100
""",
    )

    from datus.plugins.registry import load_plugin_manifest

    manifest = load_plugin_manifest("research")
    assert manifest.cli.endswith(".cli:main")
    assert "researcher" in manifest.agent_errors
    with pytest.raises(DatusException, match="protocol_version"):
        collect_plugin_agents(_config("research"))


def test_manifest_agent_uses_common_list_configuration(tmp_path):
    metadata = yaml.safe_load(
        """manifest_version: 1
agents:
  researcher:
    description: Search the selected documents.
    tools:
      - db_tools.list_tables
    mcp:
      - connection_ref: documents
        tools:
          - search
    skills:
      - research-guidelines
    scoped_context:
      datasource: warehouse
      tables_by_datasource:
        warehouse: orders,customers
        crm: customers
      metrics: Finance.Revenue
      sqls: Finance.Reports
      project_files:
        - reports/**
"""
    )

    manifest = parse_manifest(metadata, "research", tmp_path)
    spec = manifest.agents["researcher"].spec

    assert spec.tools == ["db_tools.list_tables"]
    assert spec.mcp[0].connection_ref == "documents"
    assert spec.mcp[0].tools == ["search"]
    assert spec.skills == ["research-guidelines"]
    assert spec.scoped_context.model_dump(exclude_unset=True) == metadata["agents"]["researcher"]["scoped_context"]


def test_instructions_resource_cannot_escape_the_plugin_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (tmp_path / "outside.md").write_text("not a plugin resource")

    with pytest.raises(ValidationError, match="inside the plugin package"):
        PluginAgentDefinition.from_manifest("research", package, "researcher", {"instructions_file": "../outside.md"})


@pytest.mark.parametrize(
    "changes",
    [
        {"input_schema": {"type": "array"}},
        {"tool_id": "read_file"},
        {"access": "readonly"},
    ],
)
def test_plugin_tool_contract_is_validated_before_loading_code(tmp_path, changes):
    tool = {
        "tool_id": "research_tools.search",
        "description": "Search business documents.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "access": "read",
        **changes,
    }

    with pytest.raises(ValueError):
        PluginAgentDefinition.from_manifest(
            "research",
            tmp_path,
            "researcher",
            {"runtime_factory": "research.runtime:create_runtime", "tool_definitions": [tool]},
        )


def test_preparation_accepts_bound_methods_without_framework_decorators():
    class Runtime:
        def __init__(self):
            self.results = []

        def remember(self, value):
            self.results.append(value)
            return list(self.results)

    runtime = Runtime()
    prepared = RuntimePreparation.model_validate(
        {"context": {"source": "business"}, "tool_bindings": {"research_tools.remember": runtime.remember}}
    )

    assert prepared.tool_bindings["research_tools.remember"]("first") == ["first"]
    assert runtime.results == ["first"]
    assert prepared.context == {"source": "business"}


def test_preparation_cannot_replace_native_tool_configuration():
    with pytest.raises(ValidationError, match="tools"):
        RuntimePreparation.model_validate({"tools": ["db_tools.execute_sql"]})


def test_runtime_scope_round_trips_using_the_existing_context_model():
    scoped_context = {
        "datasource": "warehouse",
        "tables_by_datasource": {"warehouse": "orders,customers", "crm": "customers"},
        "metrics": "Finance.Revenue",
        "sqls": "Finance.Reports",
        "project_files": ["reports/**"],
    }
    prepared = RuntimePreparation(scoped_context=scoped_context)
    assert isinstance(prepared.scoped_context, ScopedContext)
    assert json.loads(prepared.model_dump_json(exclude_unset=True))["scoped_context"] == scoped_context


def test_runtime_scope_rejects_a_misspelled_business_constraint():
    with pytest.raises(ValidationError, match="Unknown scoped_context fields"):
        RuntimePreparation(scoped_context={"project_file": ["private/**"]})
