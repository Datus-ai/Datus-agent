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


def test_duplicate_publisher_identity_is_rejected_before_agent_name_filtering(tmp_path, monkeypatch):
    from datus.plugins import registry

    one = parse_manifest({"manifest_version": 1, "agents": {"research": {}}}, "publisher", tmp_path)
    two = parse_manifest({"manifest_version": 1, "agents": {"report": {}}}, "publisher", tmp_path)
    monkeypatch.setattr(registry, "_loaded_manifests", lambda: [("publisher", one), ("publisher", two)])

    with pytest.raises(DatusException, match="Multiple installed packages"):
        collect_plugin_agents(_config("publisher"), "research")
    assert collect_plugin_agents(_config("publisher"), strict=False) == {}


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


def test_nonstring_schema_dialect_leaves_the_plugin_cli_available(plugin_env):
    from datus.plugins.registry import load_plugin_manifest

    plugin_env(
        "research",
        """manifest_version: 1
cli: "{pkg}.cli:main"
agents:
  research:
    config_schema: {$schema: 7}
  report: {}
""",
    )

    manifest = load_plugin_manifest("research")
    assert manifest.cli.endswith(".cli:main")
    assert "dialect must be a string" in manifest.agent_errors["research"]
    assert set(manifest.agents) == {"report"}


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
    "tool_id",
    [
        "semantic_tools.review_model",
        "skills.review_model",
        "research_tools.load_skill",
        "research_tools.mcp__documents__search",
    ],
)
def test_plugin_tool_cannot_claim_a_reserved_permission_identity(tmp_path, tool_id):
    declaration = {
        "runtime_factory": "research.runtime:create_runtime",
        "tool_definitions": [
            {
                "tool_id": tool_id,
                "description": "Review a model.",
                "input_schema": {"type": "object"},
                "access": "write",
            }
        ],
    }

    with pytest.raises(ValidationError, match="reserved"):
        PluginAgentDefinition.from_manifest("research", tmp_path, "researcher", declaration)


def test_profile_grants_for_ontology_tools_are_reserved_for_the_official_plugin(tmp_path):
    declaration = {
        "runtime_factory": "publisher.runtime:create_runtime",
        "tool_definitions": [
            {
                "tool_id": "ontology_tools.describe_concept",
                "description": "Describe the bound concept.",
                "input_schema": {"type": "object"},
                "access": "read",
            }
        ],
    }

    with pytest.raises(ValidationError, match="reserved for datus-semantic-plugin"):
        PluginAgentDefinition.from_manifest("research", tmp_path, "researcher", declaration)
    official = PluginAgentDefinition.from_manifest("datus-semantic-plugin", tmp_path, "semantic_modeling", declaration)
    assert official.tool_definitions[0].definition.tool_id == "ontology_tools.describe_concept"


def test_unresolvable_instructions_preserve_the_rest_of_the_manifest(tmp_path, monkeypatch):
    from pathlib import Path

    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.name == "loop.md":
            raise RuntimeError("Symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    manifest = parse_manifest(
        {
            "manifest_version": 1,
            "cli": "research.cli:main",
            "agents": {"researcher": {"instructions_file": "loop.md"}},
        },
        "research",
        tmp_path,
    )

    assert manifest.cli == "research.cli:main"
    assert "instructions_file" in manifest.agent_errors["researcher"]
    assert "researcher" not in manifest.agents


@pytest.mark.parametrize(
    "dialect, error",
    [("http://json-schema.org/draft-04/schema#", "required"), ("urn:unknown-draft", "unsupported dialect")],
)
def test_invalid_tool_schema_is_rejected_in_its_declared_dialect(tmp_path, dialect, error):
    manifest = parse_manifest(
        {
            "manifest_version": 1,
            "agents": {
                "researcher": {
                    "runtime_factory": "research.runtime:create_runtime",
                    "tool_definitions": [
                        {
                            "tool_id": "research_tools.search",
                            "description": "Search the records.",
                            "input_schema": {
                                "$schema": dialect,
                                "type": "object",
                                "required": [],
                            },
                            "access": "read",
                        }
                    ],
                }
            },
        },
        "research",
        tmp_path,
    )

    assert error in manifest.agent_errors["researcher"]
    assert "researcher" not in manifest.agents


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"input_schema": {"type": "array"}}, "input_schema must describe an object"),
        ({"tool_id": "read_file"}, "canonical family.method identity"),
        ({"access": "readonly"}, "access"),
    ],
)
def test_plugin_tool_contract_is_validated_before_loading_code(tmp_path, changes, message):
    tool = {
        "tool_id": "research_tools.search",
        "description": "Search business documents.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "access": "read",
        **changes,
    }

    with pytest.raises(ValueError, match=message):
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
